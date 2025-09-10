"""
Core RAG logic for the Medical Document Assistant.

This module encapsulates model initialization, document ingestion, vectorstore
management, and answering logic. It is designed to be imported by web or CLI
entry points (e.g., Flask app).
"""

import os
import csv
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


# Load .env from project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))


def initialize_models():
    """Initialize LLM and embeddings with OpenAI as primary, Mistral AI as fallback."""
    llm = None
    embeddings = None
    model_provider = "Unknown"

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
            # smoke test
            _ = llm.invoke([HumanMessage(content="Hello")])
            model_provider = "OpenAI"
        except Exception:
            llm = None
            embeddings = None

    if not llm or not embeddings:
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise RuntimeError("No API keys found. Set OPENAI_API_KEY or MISTRAL_API_KEY in .env")
        try:
            llm = ChatMistralAI(model="mistral-small-latest", temperature=0, api_key=mistral_api_key)
            embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
            _ = llm.invoke([HumanMessage(content="Hello")])
            model_provider = "Mistral AI"
        except Exception as e:
            raise RuntimeError(f"Both OpenAI and Mistral initialization failed: {e}")

    return llm, embeddings, model_provider


def load_documents_from_path(file_path: str) -> List[Document]:
    """Load a file into a list of Documents. Supports .pdf, .docx, .soap, .csv."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    if ext == ".docx":
        return Docx2txtLoader(file_path).load()
    if ext == ".soap":
        return TextLoader(file_path, encoding="utf-8").load()
    if ext == ".csv":
        lines = []
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                lines.append(", ".join(row))
        content = "\n".join(lines)
        return [Document(page_content=content, metadata={"source": os.path.basename(file_path)})]

    raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .docx, .soap, .csv")


def build_vectorstore(documents: List[Document], embeddings, persist_directory: str, collection_name: str = "medical_documents") -> Chroma:
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return vectorstore


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def make_tools_from_retriever(retriever):
    from langchain_core.tools import tool

    @tool
    def retriever_tool(query: str) -> str:
        """
        Retrieve grounded context from the ingested medical document(s)
        to answer clinical and biomedical questions.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the provided medical document(s)."
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}")
        return "\n\n".join(results)

    return [retriever_tool]


def create_graph(llm, tools):
    tools_dict = {t.name: t for t in tools}

    system_prompt = (
        "You are a Medical Document Assistant. Answer questions strictly based on the ingested medical documents "
        "(PDF, DOCX, SOAP/txt, CSV). \n\n"
        "IMPORTANT: You MUST use the retriever_tool to search the medical documents before answering any questions. "
        "If you cannot find the information using the retriever_tool, say \"I do not have sufficient information in the provided documents.\""
    )

    def should_continue(state: AgentState):
        result = state["messages"][-1]
        return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

    def call_llm(state: AgentState) -> AgentState:
        messages = list(state["messages"])
        messages = [SystemMessage(content=system_prompt)] + messages
        try:
            message = llm.invoke(messages)
            return {"messages": [message]}
        except Exception as e:
            user_query = None
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    user_query = m.content
                    break
            if not user_query:
                return {"messages": [AIMessage(content="I encountered an error and could not identify the question.")]} 
            try:
                retrieved_context = tools_dict["retriever_tool"].invoke(user_query)
            except Exception:
                retrieved_context = ""
            if retrieved_context and "no relevant information" not in retrieved_context.lower():
                fallback_answer = answer_with_context(llm, user_query, retrieved_context)
            else:
                fallback_answer = "I do not have sufficient information in the provided documents."
            return {"messages": [AIMessage(content=fallback_answer)]}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            if t["name"] not in tools_dict:
                result = "Incorrect Tool Name."
            else:
                result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            results.append(
                AIMessage(content=str(result))
            )
        return {"messages": results}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    return graph.compile()


def answer_with_context(llm, query: str, context: str) -> str:
    qa_system_prompt = (
        "You are a medical RAG assistant. Answer ONLY using the provided CONTEXT. "
        "If the answer is not present in the CONTEXT, say: 'I do not have sufficient information in the provided documents.' "
        "Respond concisely and extract explicit fields when present (e.g., Patient Name, Age, Disease)."
    )
    messages = [
        SystemMessage(content=qa_system_prompt),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer succinctly."),
    ]
    try:
        resp = llm.invoke(messages)
        return resp.content
    except Exception as e:
        return f"Error while processing your request: {str(e)}"


class RagService:
    """High-level service for managing documents, vectorstore, and chat sessions."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.llm, self.embeddings, self.model_provider = initialize_models()
        self.persist_directory = persist_directory or os.path.join(PROJECT_ROOT, "chroma_store")
        self.vectorstore = None
        self.retriever = None
        self.tools = None
        self.agent = None

    def ingest_files(self, file_paths: List[str]):
        documents: List[Document] = []
        for path in file_paths:
            documents.extend(load_documents_from_path(path))
        self.vectorstore = build_vectorstore(documents, self.embeddings, self.persist_directory)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.tools = make_tools_from_retriever(self.retriever)
        try:
            self.llm = self.llm.bind_tools(self.tools)
        except Exception:
            pass
        self.agent = create_graph(self.llm, self.tools)

    def ask(self, conversation_messages: List[BaseMessage], user_input: str) -> str:
        if not self.agent and self.retriever is None:
            raise RuntimeError("No documents ingested. Please upload documents first.")
        messages = conversation_messages + [HumanMessage(content=user_input)]
        result = self.agent.invoke({"messages": messages})
        # fallback deterministic answer using context to ensure grounded output
        try:
            retrieved_context = self.tools[0].invoke(user_input)
        except Exception:
            retrieved_context = ""
        final_answer = result["messages"][-1].content
        if retrieved_context and "no relevant information" not in retrieved_context.lower():
            final_answer = answer_with_context(self.llm, user_input, retrieved_context)
        return final_answer


