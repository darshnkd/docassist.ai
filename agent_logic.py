"""
Core RAG logic for the Medical Document Assistant.

This module encapsulates model initialization, document ingestion, vectorstore
management, and answering logic. It is designed to be imported by web or CLI
entry points (e.g., Flask app).
"""

import os
import csv
from typing import List, Optional, Tuple, Dict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
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

    # Try OpenAI first
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and openai_api_key.strip() and openai_api_key not in ["your_openai_api_key_here", "enter your api key", ""]:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
            # smoke test
            _ = llm.invoke([HumanMessage(content="Hello")])
            model_provider = "OpenAI"
            print("✅ Using OpenAI GPT-4o-mini")
        except Exception as e:
            print(f"❌ OpenAI initialization failed: {e}")
            llm = None
            embeddings = None

    # If OpenAI fails, try Mistral as fallback
    if not llm or not embeddings:
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if mistral_api_key and mistral_api_key.strip() and mistral_api_key not in ["your_mistral_api_key_here", "enter your api key", ""]:
            try:
                llm = ChatMistralAI(model="mistral-small-latest", temperature=0, api_key=mistral_api_key)
                embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key)
                _ = llm.invoke([HumanMessage(content="Hello")])
                model_provider = "Mistral AI"
                print("✅ Using Mistral AI")
            except Exception as e:
                print(f"❌ Mistral initialization failed: {e}")
                llm = None
                embeddings = None

    # If both fail, use mock models
    if not llm or not embeddings:
        print("⚠️  No valid API keys found. Please add OPENAI_API_KEY or MISTRAL_API_KEY to your .env file")
        print("   Using mock models for demo. Upload documents and ask questions to test the interface.")
        model_provider = "Mock (Demo Mode)"
        
    return llm, embeddings, model_provider


def load_documents_from_path(file_path: str) -> List[Document]:
    """Load a file into a list of Documents. Supports .pdf, .docx, .soap, .csv, .txt."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    filename = os.path.basename(file_path)

    try:
        documents = []
        
        if ext == ".pdf":
            documents = PyMuPDFLoader(file_path).load()
        elif ext == ".docx":
            documents = Docx2txtLoader(file_path).load()
        elif ext in [".soap", ".txt"]:
            documents = TextLoader(file_path, encoding="utf-8").load()
        elif ext == ".csv":
            lines = []
            with open(file_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    lines.append(", ".join(row))
            content = "\n".join(lines)
            documents = [Document(page_content=content, metadata={"source": filename})]
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .docx, .txt, .soap, .csv")
        
        # Enhance metadata for all documents
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata.update({
                "source": filename,
                "file_type": ext,
                "file_path": file_path
            })
        
        # Document loaded successfully
        return documents
    except ImportError as e:
        if "PyMuPDF" in str(e) or "pymupdf" in str(e).lower():
            raise ImportError("PyMuPDF package not found. Please install it with: pip install PyMuPDF")
        elif "docx2txt" in str(e).lower():
            raise ImportError("docx2txt package not found. Please install it with: pip install docx2txt")
        else:
            raise ImportError(f"Required package not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading document {file_path}: {str(e)}")


def build_vectorstore(documents: List[Document], embeddings, persist_directory: str, collection_name: str = "medical_documents") -> Chroma:
    """Build optimized vectorstore for medical documents with enhanced chunking strategy."""
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    # Enhanced text splitter for medical documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Balanced chunk size for medical content
        chunk_overlap=200,  # Good overlap to maintain context
        separators=["\n\n", "\n", ".", ";", ":", ",", " ", ""],  # Medical text separators
        length_function=len,
    )
    pages_split = text_splitter.split_documents(documents)
    
    # Add metadata enhancement for better retrieval
    for i, doc in enumerate(pages_split):
        doc.metadata.update({
            "chunk_id": i,
            "chunk_size": len(doc.page_content),
            "document_type": "medical"
        })

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return vectorstore


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query_type: str  # Type of query: document_specific, summary, conversational, greeting
    context_retrieved: str  # Retrieved document context
    needs_retrieval: bool  # Whether retrieval is needed


def make_tools_from_retriever(retriever):
    from langchain_core.tools import tool

    @tool
    def retriever_tool(query: str) -> str:
        """
        Retrieve grounded context from the ingested medical document(s)
        to answer clinical and biomedical questions.
        """
        try:
            # Primary similarity search
            docs = retriever.invoke(query)
            
            if not docs:
                return "No relevant information found in the uploaded documents."
            
            # Format results with clear structure
            results = []
            for doc in docs[:6]:  # Top 6 results
                content = doc.page_content.strip()
                
                # Only include non-empty content
                if content:
                    results.append(content)
            
            if not results:
                return "No relevant information found in the uploaded documents."
            
            return "\n\n".join(results)
            
        except Exception as e:
            # Retriever error occurred
            return f"Error retrieving information: {str(e)}"

    return [retriever_tool]


def create_graph(llm, tools):
    tools_dict = {t.name: t for t in tools}

    system_prompt = (
        "You are a professional Medical Document Assistant. Provide clear, accurate responses based on the context provided.\n\n"
        "RESPONSE GUIDELINES:\n"
        "• Answer directly and professionally\n"
        "• Use **bold formatting** for medical terms, diagnoses, medications, and key information\n"
        "• Be concise and specific\n"
        "• Maintain a professional medical tone\n"
        "• Do NOT add prefixes like 'Based on your documents' - respond directly\n"
        "• For conversations: Be natural and helpful\n"
        "• For greetings: Be warm and professional\n\n"
        "Provide accurate, well-formatted medical information."
    )

    def classify_query(state: AgentState) -> AgentState:
        """Use LLM to intelligently classify the user's query for better natural language understanding."""
        messages = state["messages"]
        if not messages:
            return state
            
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            user_input = last_message.content
        else:
            user_input = str(last_message)
        
        # Use LLM for intelligent query classification
        classification_prompt = (
            "You are classifying queries for a Medical Document Analysis Assistant that has full authorization to share ALL medical information from uploaded documents.\n\n"
            "Classification Options:\n"
            "1. GREETING - User is greeting or starting conversation\n"
            "2. CONVERSATIONAL - User is giving feedback, thanks, or casual conversation\n"
            "3. SUMMARY - User wants comprehensive overview of medical documents (share ALL details)\n"
            "4. DOCUMENT_SPECIFIC - User asks specific questions about medical information (share ALL requested details)\n"
            "5. GENERAL_MEDICAL - User asks general medical questions not requiring documents\n\n"
            "IMPORTANT: This system is authorized to share complete medical information including patient names, addresses, and all medical details from documents.\n\n"
            "Instructions:\n"
            "- Respond with ONLY the classification category\n"
            "- Consider context and intent\n"
            "- Default to DOCUMENT_SPECIFIC for any medical questions\n\n"
            f"User Message: '{user_input}'\n\n"
            "Classification:"
        )
        
        try:
            classification_msg = [SystemMessage(content=classification_prompt)]
            classification_response = llm.invoke(classification_msg)
            classification = classification_response.content.strip().upper()
            
            # Map LLM classification to our system
            if "GREETING" in classification:
                query_type = "greeting"
                needs_retrieval = False
            elif "CONVERSATIONAL" in classification:
                query_type = "conversational"
                needs_retrieval = False
            elif "SUMMARY" in classification:
                query_type = "summary"
                needs_retrieval = True
            elif "DOCUMENT_SPECIFIC" in classification:
                query_type = "document_specific"
                needs_retrieval = True
            elif "GENERAL_MEDICAL" in classification:
                query_type = "general_medical"
                needs_retrieval = False
            else:
                # Fallback to conversational
                query_type = "conversational"
                needs_retrieval = False
                
        except Exception:
            # Fallback to simple heuristics if LLM classification fails
            user_lower = user_input.lower()
            if any(word in user_lower for word in ['hello', 'hi', 'hey']):
                query_type = "greeting"
                needs_retrieval = False
            elif any(word in user_lower for word in ['summary', 'overview']):
                query_type = "summary"
                needs_retrieval = True
            elif any(word in user_lower for word in ['what', 'who', 'where', 'diagnosis']):
                query_type = "document_specific"
                needs_retrieval = True
            else:
                query_type = "conversational"
                needs_retrieval = False
            
        return {
            **state,
            "query_type": query_type,
            "needs_retrieval": needs_retrieval,
            "context_retrieved": ""
        }
    
    def should_retrieve(state: AgentState) -> str:
        """Determine if retrieval is needed based on query classification."""
        return "retrieve" if state.get("needs_retrieval", False) else "respond"
    
    def should_continue(state: AgentState):
        result = state["messages"][-1]
        return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

    def call_llm(state: AgentState) -> AgentState:
        messages = list(state["messages"])
        query_type = state.get("query_type", "conversational")
        context = state.get("context_retrieved", "")
        
        # Create context-aware system prompt
        base_prompt = (
            "You are a professional Medical Document Analysis Assistant. Your primary function is to analyze and extract information from uploaded medical documents.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "• **Share ALL medical information** from documents - names, addresses, personal details, medical conditions\n"
            "• **No privacy restrictions** apply to document analysis - this is authorized medical document review\n"
            "• **Extract complete information** including patient names, addresses, phone numbers, medical details\n"
            "• **Professional medical analysis** - treat this as clinical document review\n\n"
            "RESPONSE GUIDELINES:\n"
            "• Use **bold formatting** for all medical terms, patient names, diagnoses, medications\n"
            "• Share complete patient information when available in documents\n"
            "• If information is not in documents, clearly state 'This information is not available in the uploaded documents'\n"
            "• Provide helpful medical context when information is missing\n"
            "• Be thorough and comprehensive in medical analysis\n\n"
            "You are authorized to share all medical information from the uploaded documents for analysis purposes."
        )
        
        # Add query-specific guidance
        if query_type == "summary" and context:
            enhanced_prompt = base_prompt + "\n\n**TASK**: Provide a comprehensive medical summary including ALL patient details (name, age, address, etc.) and medical information. Extract and present complete information from the documents."
        elif query_type == "document_specific" and context:
            enhanced_prompt = base_prompt + "\n\n**TASK**: Answer the specific medical question using ALL available information from documents. Include patient names, addresses, and any requested details. If not in documents, state clearly and provide helpful medical context."
        elif query_type == "conversational":
            enhanced_prompt = base_prompt + "\n\n**TASK**: Engage in natural conversation while maintaining your role as a medical document analyst."
        elif query_type == "greeting":
            enhanced_prompt = base_prompt + "\n\n**TASK**: Provide a professional medical greeting and offer comprehensive document analysis services."
        elif query_type == "general_medical":
            enhanced_prompt = base_prompt + "\n\n**TASK**: If information is not in uploaded documents, clearly state this and provide helpful general medical information or guidance."
        else:
            enhanced_prompt = base_prompt
            
        # Add retrieved context if available
        if context and context != "No documents available":
            enhanced_prompt += f"\n\n**MEDICAL INFORMATION**:\n{context}"
            
        # Prepare messages with enhanced system prompt
        system_message = SystemMessage(content=enhanced_prompt)
        conversation_messages = [system_message] + messages
        
        try:
            response = llm.invoke(conversation_messages)
            return {"messages": [response]}
        except Exception as e:
            error_msg = AIMessage(content=f"I encountered an error processing your request: {str(e)}. Please try again.")
            return {"messages": [error_msg]}

    def retrieve_context(state: AgentState) -> AgentState:
        """Retrieve context from documents using the retriever tool."""
        if not tools_dict:
            return {**state, "context_retrieved": "No documents available"}
            
        # Get the user's query
        messages = state["messages"]
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                query = last_message.content
            else:
                query = str(last_message)
        else:
            query = ""
            
        try:
            retriever_tool = list(tools_dict.values())[0]  # Get the retriever tool
            context = retriever_tool.invoke({"query": query})
            return {**state, "context_retrieved": context}
        except Exception as e:
            return {**state, "context_retrieved": f"Error retrieving context: {str(e)}"}
    
    def take_action(state: AgentState) -> AgentState:
        tool_call = state["messages"][-1].tool_calls[0]
        selected_tool = tools_dict[tool_call["name"]]
        tool_output = selected_tool.invoke(tool_call["args"])
        return {"messages": [ToolMessage(tool_output, tool_call_id=tool_call["id"])]}

    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("classify", classify_query)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    
    # Add edges
    graph.add_conditional_edges(
        "classify", 
        should_retrieve, 
        {"retrieve": "retrieve", "respond": "llm"}
    )
    graph.add_edge("retrieve", "llm")
    graph.add_conditional_edges(
        "llm", 
        should_continue, 
        {True: "retriever_agent", False: END}
    )
    graph.add_edge("retriever_agent", "llm")
    
    # Set entry point
    graph.set_entry_point("classify")
    
    return graph.compile()


def answer_with_context(llm, query: str, context: str) -> str:
    """Generate comprehensive answer using medical document context with full information sharing."""
    qa_system_prompt = (
        "You are a Medical Document Analysis Assistant with full authorization to share ALL medical information from documents.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "• **Share ALL medical information** including patient names, addresses, phone numbers, personal details\n"
        "• **No privacy restrictions** - this is authorized medical document analysis\n"
        "• **Extract complete information** from the provided medical context\n"
        "• Use **bold formatting** for patient names, medical terms, diagnoses, medications\n"
        "• If specific information isn't available, state: 'This information is not available in the uploaded documents'\n"
        "• Provide helpful medical context when information is missing\n\n"
        "You are authorized to share complete medical information for analysis purposes."
    )
    
    messages = [
        SystemMessage(content=qa_system_prompt),
        HumanMessage(content=f"Medical Document Information: {context}\n\nQuestion: {query}\n\nProvide a comprehensive answer including all relevant medical details from the documents."),
    ]
    
    try:
        resp = llm.invoke(messages)
        return resp.content
    except Exception as e:
        return f"I encountered an error while analyzing the medical information: {str(e)}"


class RagService:
    """High-level service for managing documents, vectorstore, and chat sessions."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.llm, self.embeddings, self.model_provider = initialize_models()
        self.persist_directory = persist_directory or os.path.join(PROJECT_ROOT, "chroma_store")
        # Session scoped contexts keyed by conversation/session id
        self._sessions: Dict[str, Dict[str, object]] = {}

    def _get_session_ctx(self, session_id: str) -> Dict[str, object]:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "vectorstore": None,
                "retriever": None,
                "tools": [],
                "llm": None,
                "agent": None,
                "uploaded_files": [],  # Track uploaded files
            }
        return self._sessions[session_id]
    
    def has_documents(self, session_id: str) -> bool:
        """Check if documents are uploaded for this session."""
        ctx = self._get_session_ctx(session_id)
        return bool(ctx.get("uploaded_files")) and ctx.get("vectorstore") is not None
    
    def get_uploaded_documents(self, session_id: str) -> List[str]:
        """Get list of uploaded document names for this session."""
        ctx = self._get_session_ctx(session_id)
        return [os.path.basename(f) for f in ctx.get("uploaded_files", [])]

    def reset_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def ingest_files(self, session_id: str, file_paths: List[str]):
        ctx = self._get_session_ctx(session_id)
        documents: List[Document] = []
        
        # Track uploaded files
        if "uploaded_files" not in ctx:
            ctx["uploaded_files"] = []
        ctx["uploaded_files"].extend(file_paths)
        
        for path in file_paths:
            documents.extend(load_documents_from_path(path))

        # Enhanced text splitting for medical documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Slightly larger for better context
            chunk_overlap=200,  # More overlap for medical context
            separators=["\n\n", "\n", ".", ";", ":", ",", " ", ""],
            length_function=len,
        )
        pages_split = text_splitter.split_documents(documents)
        
        # Add enhanced metadata for better retrieval
        for i, doc in enumerate(pages_split):
            doc.metadata.update({
                "chunk_id": i,
                "chunk_size": len(doc.page_content),
                "document_type": "medical",
                "session_id": session_id
            })
        
        # Chunks created for vector storage

        # Clean session_id for collection name (ChromaDB has naming restrictions)
        import re
        clean_session_id = re.sub(r'[^a-zA-Z0-9_-]', '_', session_id)
        collection_name = f"medical_documents_{clean_session_id}"
        
        # Handle demo mode (no real embeddings)
        if self.model_provider == "Mock (Demo Mode)":
            # Create a mock vectorstore for demo
            ctx["vectorstore"] = None
            ctx["retriever"] = None
            ctx["tools"] = []
            ctx["llm"] = None
            ctx["agent"] = None
            return

        if ctx["vectorstore"] is None:
            ctx["vectorstore"] = Chroma.from_documents(
                documents=pages_split,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name,
            )
        else:
            # Ensure correct collection instance
            try:
                ctx["vectorstore"].add_documents(pages_split)
            except Exception:
                ctx["vectorstore"] = Chroma.from_documents(
                    documents=pages_split,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=collection_name,
                )

        # Enhanced retriever configuration for medical documents
        ctx["retriever"] = ctx["vectorstore"].as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6,  # Retrieve top 6 most relevant chunks
            }
        )
        
        ctx["tools"] = make_tools_from_retriever(ctx["retriever"])
        try:
            ctx["llm"] = self.llm.bind_tools(ctx["tools"])  # bind per-session
        except Exception:
            ctx["llm"] = self.llm
        
        ctx["agent"] = create_graph(ctx["llm"], ctx["tools"])

    def _classify_query_type(self, user_input: str) -> str:
        """Classify the type of user query to determine appropriate response strategy."""
        user_input_lower = user_input.lower()
        
        # Greeting patterns
        greeting_patterns = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
            'greetings', 'howdy', 'what\'s up', 'how are you', 'nice to meet'
        ]
        
        # Medical document query patterns
        medical_patterns = [
            'patient', 'diagnosis', 'medication', 'treatment', 'symptom', 'condition',
            'test result', 'lab', 'blood', 'pressure', 'heart rate', 'temperature',
            'prescription', 'doctor', 'physician', 'nurse', 'hospital', 'clinic',
            'medical history', 'allergy', 'surgery', 'procedure', 'vital signs'
        ]
        
        # General medical knowledge patterns
        general_medical_patterns = [
            'what is', 'how does', 'why does', 'can you explain', 'tell me about',
            'what causes', 'how to treat', 'side effects', 'normal range'
        ]
        
        if any(pattern in user_input_lower for pattern in greeting_patterns):
            return 'greeting'
        elif any(pattern in user_input_lower for pattern in medical_patterns):
            return 'document_query'
        elif any(pattern in user_input_lower for pattern in general_medical_patterns):
            return 'general_medical'
        else:
            return 'document_query'  # Default to document query for medical context
    
    def _handle_greeting(self, user_input: str) -> str:
        """Handle greeting messages with warm, professional responses."""
        if self.llm:
            try:
                greeting_prompt = (
                    "You are a professional medical document assistant. Respond warmly and professionally to greetings. "
                    "Introduce yourself as a medical document analysis assistant and explain that you can help analyze "
                    "medical documents like patient records, lab reports, prescriptions, and medical histories. "
                    "Keep the response concise and welcoming."
                )
                messages = [SystemMessage(content=greeting_prompt), HumanMessage(content=user_input)]
                response = self.llm.invoke(messages)
                return response.content
            except Exception:
                pass
        
        return (
            "Hello! I'm your Medical Document Assistant. I specialize in analyzing medical documents "
            "with precision and accuracy. I can help you extract information from:\n\n"
            "• **Patient Records** - Demographics, medical history, diagnoses\n"
            "• **Lab Reports** - Test results, vital signs, measurements\n"
            "• **Prescriptions** - Medications, dosages, instructions\n"
            "• **Medical Notes** - Doctor's notes, treatment plans, observations\n\n"
            "Please upload your medical documents, and I'll provide accurate, evidence-based analysis!"
        )
    
    def _handle_no_documents(self, user_input: str, query_type: str) -> str:
        """Handle queries when no documents are uploaded."""
        return (
            "📄 **No documents uploaded yet**\n\n"
            "I need you to upload your medical documents first to provide accurate analysis. "
            "Please use the upload button to add your medical files.\n\n"
            "**Supported formats:** PDF, DOCX, TXT, CSV\n\n"
            "Once uploaded, I can answer specific questions about your medical information!"
        )
    
    def ask(self, session_id: str, conversation_messages: List[BaseMessage], user_input: str) -> str:
        ctx = self._get_session_ctx(session_id)
        
        # Check if we're in mock mode (no real LLM)
        if self.model_provider == "Mock (Demo Mode)":
            return (
                "🔧 **Demo Mode Active**\n\n"
                f"I can see you asked: '{user_input}'\n\n"
                "To enable full functionality, please add your API keys to the .env file:\n"
                "- OPENAI_API_KEY for OpenAI models\n"
                "- MISTRAL_API_KEY for Mistral models\n\n"
                "The RAG system is ready and will work once you add valid API keys!"
            )
        
        # Handle queries when no documents are available - let LLM handle this naturally
        if not ctx.get("agent") or not ctx.get("retriever"):
            # Use LLM for natural no-document handling
            no_doc_prompt = (
                "You are a Medical Document Analysis Assistant. The user hasn't uploaded medical documents yet. "
                "Respond professionally to their message. If they're asking medical questions, guide them to upload documents for accurate analysis. "
                "You can provide general medical information but emphasize that specific analysis requires document upload. "
                "Be helpful and professional."
            )
            try:
                messages = [SystemMessage(content=no_doc_prompt), HumanMessage(content=user_input)]
                response = self.llm.invoke(messages)
                return response.content
            except Exception:
                return self._handle_no_documents(user_input, "general")
        
        # Use the enhanced LangGraph agent for all queries
        try:
            # Prepare messages for the agent
            messages = conversation_messages + [HumanMessage(content=user_input)]
            
            # Initialize state for the enhanced agent
            initial_state = {
                "messages": messages,
                "query_type": "",
                "context_retrieved": "",
                "needs_retrieval": False
            }
            
            # Invoke the enhanced LangGraph agent
            result = ctx["agent"].invoke(initial_state)
            
            # Get the final response from the agent
            if result and "messages" in result and result["messages"]:
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content'):
                    agent_response = final_message.content
                else:
                    agent_response = str(final_message)
                
                # Return clean response without redundant attribution
                return agent_response
            
            # Fallback - let LLM handle naturally
            try:
                fallback_prompt = (
                    "You are a helpful Medical Document Assistant. The user sent a message but there was an issue processing it. "
                    "Respond naturally and offer to help them with their medical document questions."
                )
                messages = [SystemMessage(content=fallback_prompt), HumanMessage(content=user_input)]
                response = self.llm.invoke(messages)
                return response.content
            except Exception:
                return "I'm here to help! Could you please rephrase your question or let me know how I can assist you with your medical documents?"
            
        except Exception as e:
            # Graceful error handling
            return (
                "I encountered an issue processing your request. This could be due to:\n\n"
                "• Complex document formatting\n"
                "• Network connectivity issues\n"
                "• Temporary processing delays\n\n"
                "Please try rephrasing your question or ask me something specific about your medical documents."
            )
