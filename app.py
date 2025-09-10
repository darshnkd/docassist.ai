import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from logic import RagService


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rag_service = RagService()


def get_conversation_key():
    if "conversation_id" not in session:
        session["conversation_id"] = os.urandom(8).hex()
    return f"conv_{session['conversation_id']}"


def get_history_list():
    return session.get("history_list", [])


def get_conversation_messages():
    key = get_conversation_key()
    return session.get(key, [])


def set_conversation_messages(msgs):
    key = get_conversation_key()
    session[key] = msgs


@app.route("/")
def index():
    session_finished = session.get("session_finished", False)
    return render_template(
        "doc_assist/index.html",
        history_list=get_history_list(),
        current_messages=get_conversation_messages(),
        session_finished=session_finished,
    )


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    saved_paths = []
    for f in files:
        if not f or f.filename == "":
            continue
        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        saved_paths.append(path)
    if not saved_paths:
        return jsonify({"ok": False, "error": "No files uploaded"}), 400
    try:
        rag_service.ingest_files(saved_paths)
        return jsonify({"ok": True, "message": "Documents ingested successfully"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"ok": False, "error": "Empty message"}), 400
    # restore conversation
    messages = get_conversation_messages()
    lc_messages = []
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    try:
        answer = rag_service.ask(lc_messages, user_text)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    # append and persist
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": answer})
    set_conversation_messages(messages)

    # ensure sidebar history contains current conversation as soon as chatting starts
    history = get_history_list()
    if not any(h.get("id") == get_conversation_key() for h in history):
        history.append({"id": get_conversation_key(), "title": user_text[:40] or "New chat"})
        session["history_list"] = history

    return jsonify({"ok": True, "answer": answer, "messages": messages, "history": history})


@app.route("/new_chat", methods=["POST"])
def new_chat():
    session.pop(get_conversation_key(), None)
    session["conversation_id"] = os.urandom(8).hex()
    session["session_finished"] = False
    return jsonify({"ok": True})


@app.route("/load_chat", methods=["POST"])
def load_chat():
    # For now, conversations are kept in session only. Hook persistent storage here later.
    return jsonify({"ok": True, "messages": get_conversation_messages()})


@app.route("/end_session", methods=["POST"])
def end_session():
    messages = get_conversation_messages()
    if messages:
        title_source = next((m for m in messages if m.get("role") == "user"), None)
        title = (title_source.get("content")[:40] if title_source else "Conversation") or "Conversation"
        history = get_history_list()
        history.append({"id": get_conversation_key(), "title": title})
        session["history_list"] = history
    session["session_finished"] = True
    return jsonify({"ok": True, "history": get_history_list()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=True)


