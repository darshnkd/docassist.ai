import os
import json
import uuid
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from email_validator import validate_email, EmailNotValidError

from agent_logic import RagService
from user_models import db, User, Document, ChatSession, ChatMessage, UserLoginLog, init_db, migrate_existing_users


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///docassist.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# User data directory
USER_DATA_DIR = os.path.join(os.path.dirname(__file__), "UserData")
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Simple user storage (in production, use a proper database)
USERS_FILE = os.path.join(USER_DATA_DIR, "users.json")

# Initialize database
init_db(app)

# Migrate existing users from JSON to SQL database
with app.app_context():
    migrate_existing_users(USERS_FILE)

# Make get_user_conversation_key available in templates
@app.template_global()
def get_user_conversation_key():
    """Get user-specific conversation key"""
    username = session.get('username')
    if not username:
        return None
    if "conversation_id" not in session:
        session["conversation_id"] = str(uuid.uuid4())
    return f"{username}_{session['conversation_id']}"

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def get_user_data_dir(username):
    """Get user-specific data directory"""
    user = User.query.filter_by(username=username).first()
    if user:
        return user.get_user_data_dir()
    else:
        # Fallback for backward compatibility
        user_dir = os.path.join(USER_DATA_DIR, f"{username}_Upload")
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

def get_user_chroma_dir(username):
    """Get user-specific Chroma directory"""
    user = User.query.filter_by(username=username).first()
    if user:
        return user.get_user_chroma_dir()
    else:
        # Fallback for backward compatibility
        user_dir = get_user_data_dir(username)
        chroma_dir = os.path.join(user_dir, "chroma_store")
        os.makedirs(chroma_dir, exist_ok=True)
        return chroma_dir

def get_user_history_file(username):
    """Get user-specific history file"""
    user_dir = get_user_data_dir(username)
    return os.path.join(user_dir, "history.json")

def load_user_history(username):
    """Load user's chat history"""
    history_file = get_user_history_file(username)
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []

def save_user_history(username, history):
    """Save user's chat history"""
    history_file = get_user_history_file(username)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

# Global RAG service instances per user
user_rag_services = {}

def get_user_rag_service(username):
    """Get or create user-specific RAG service"""
    if username not in user_rag_services:
        user_chroma_dir = get_user_chroma_dir(username)
        user_rag_services[username] = RagService(persist_directory=user_chroma_dir)
    return user_rag_services[username]

def require_login(f):
    """Decorator to require login"""
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def get_conversation_key():
    if "conversation_id" not in session:
        session["conversation_id"] = str(uuid.uuid4())
    return f"conv_{session['conversation_id']}"

def save_chat_to_database(username, conversation_id, messages):
    """Save chat messages to database"""
    try:
        # Clear existing messages for this conversation
        ChatMessage.query.filter_by(
            user_id=username,
            conversation_id=conversation_id
        ).delete()
        
        # Save new messages
        for msg in messages:
            chat_message = ChatMessage(
                user_id=username,
                conversation_id=conversation_id,
                role=msg['role'],
                message=msg['content']
            )
            db.session.add(chat_message)
        
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error saving chat to database: {e}")
        return False

def get_history_list():
    """Get chat history list from database"""
    username = session.get('username')
    if not username:
        return []
    
    try:
        # Get unique conversations with their first message as title
        conversations = db.session.query(
            ChatMessage.conversation_id,
            db.func.min(ChatMessage.message).label('title'),
            db.func.min(ChatMessage.timestamp).label('created_at')
        ).filter_by(
            user_id=username,
            role='user'
        ).group_by(
            ChatMessage.conversation_id
        ).order_by(
            db.func.min(ChatMessage.timestamp).desc()
        ).all()
        
        history = []
        for conv in conversations:
            title = conv.title[:40] if conv.title else "New chat"
            history.append({
                "id": conv.conversation_id,
                "title": title
            })
        
        return history
    except Exception as e:
        print(f"Error loading history from database: {e}")
        # Fallback to JSON file
        return load_user_history(username)

def get_conversation_messages():
    """Get conversation messages from database"""
    username = session.get('username')
    if not username:
        return []
    
    conversation_id = get_user_conversation_key()
    
    try:
        messages = ChatMessage.query.filter_by(
            user_id=username,
            conversation_id=conversation_id
        ).order_by(ChatMessage.timestamp.asc()).all()
        
        return [{
            "role": msg.role,
            "content": msg.message
        } for msg in messages]
    except Exception as e:
        print(f"Error loading conversation from database: {e}")
        # Fallback to session
        key = get_user_conversation_key()
        return session.get(key, [])

def set_conversation_messages(msgs):
    """Set conversation messages in database and session"""
    username = session.get('username')
    if not username:
        return
    
    conversation_id = get_user_conversation_key()
    
    # Save to database
    save_chat_to_database(username, conversation_id, msgs)
    
    # Also keep in session for backward compatibility
    key = get_user_conversation_key()
    session[key] = msgs


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("doc_assist/login.html")
    
    data = request.get_json()
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "")
    is_register = data.get("is_register", False)
    
    if not username or not password:
        return jsonify({"ok": False, "error": "Username and password are required"}), 400
    
    if is_register:
        # Registration flow
        if not email:
            return jsonify({"ok": False, "error": "Email is required for registration"}), 400
        
        # Validate email format
        try:
            validate_email(email)
        except EmailNotValidError:
            return jsonify({"ok": False, "error": "Invalid email format"}), 400
        
        # Check if user already exists
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            return jsonify({"ok": False, "error": "Username or email already exists"}), 400
        
        # Create new user
        try:
            new_user = User(username=username, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            
            session["username"] = username
            session["user_session_id"] = str(uuid.uuid4())
            return jsonify({"ok": True, "message": "Account created successfully"})
        except Exception as e:
            db.session.rollback()
            return jsonify({"ok": False, "error": f"Registration failed: {str(e)}"}), 500
    
    else:
        # Login flow - allow login by username or email
        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()
        
        # Determine login method
        login_method = 'email' if '@' in username else 'username'
        
        # Get client information
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        user_agent = request.headers.get('User-Agent', 'unknown')
        
        if user and user.check_password(password):
            # Successful login
            user.update_last_login()
            session["username"] = user.username  # Always use the actual username
            session["user_session_id"] = str(uuid.uuid4())
            
            # Log successful login
            login_log = UserLoginLog(
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent,
                login_method=login_method,
                success=True
            )
            db.session.add(login_log)
            db.session.commit()
            
            return jsonify({"ok": True, "message": "Login successful"})
        else:
            # Failed login attempt
            failure_reason = "Invalid credentials"
            if user:
                failure_reason = "Wrong password"
            else:
                failure_reason = "User not found"
            
            # Log failed login attempt
            login_log = UserLoginLog(
                user_id=user.id if user else None,
                ip_address=ip_address,
                user_agent=user_agent,
                login_method=login_method,
                success=False,
                failure_reason=failure_reason
            )
            db.session.add(login_log)
            db.session.commit()
            
            return jsonify({"ok": False, "error": "Invalid credentials"}), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/")
@require_login
def index():
    username = session.get('username')
    session_finished = session.get("session_finished", False)
    
    # Check if documents are uploaded
    session_id = get_user_conversation_key()
    user_rag_service = get_user_rag_service(username)
    has_docs = user_rag_service.has_documents(session_id)
    uploaded_docs = user_rag_service.get_uploaded_documents(session_id)
    
    return render_template(
        "doc_assist/index.html",
        username=username,
        history_list=get_history_list(),
        current_messages=get_conversation_messages(),
        session_finished=session_finished,
        has_documents=has_docs,
        uploaded_documents=uploaded_docs,
    )


@app.route("/upload", methods=["POST"])
@require_login
def upload():
    username = session.get('username')
    files = request.files.getlist("files")
    saved_paths = []
    
    # Create user-specific upload directory
    user_upload_dir = get_user_data_dir(username)
    
    for f in files:
        if not f or f.filename == "":
            continue
        filename = secure_filename(f.filename)
        path = os.path.join(user_upload_dir, filename)
        f.save(path)
        saved_paths.append(path)
    
    if not saved_paths:
        return jsonify({"ok": False, "error": "No files uploaded"}), 400
    
    try:
        session_id = get_user_conversation_key()
        user_rag_service = get_user_rag_service(username)
        user_rag_service.ingest_files(session_id, saved_paths)
        
        # Get document status for frontend
        has_docs = user_rag_service.has_documents(session_id)
        uploaded_docs = user_rag_service.get_uploaded_documents(session_id)
        
        return jsonify({
            "ok": True, 
            "message": f"Successfully uploaded and processed {len(saved_paths)} document(s)",
            "has_documents": has_docs,
            "uploaded_documents": uploaded_docs
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
@require_login
def chat():
    username = session.get('username')
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
        session_id = get_user_conversation_key()
        user_rag_service = get_user_rag_service(username)
        answer = user_rag_service.ask(session_id, lc_messages, user_text)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    # append and persist
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": answer})
    set_conversation_messages(messages)

    # ensure sidebar history contains current conversation as soon as chatting starts
    history = get_history_list()
    if not any(h.get("id") == get_user_conversation_key() for h in history):
        history.append({"id": get_user_conversation_key(), "title": user_text[:40] or "New chat"})
        save_user_history(username, history)

    # Check document status for frontend
    has_docs = user_rag_service.has_documents(session_id)
    uploaded_docs = user_rag_service.get_uploaded_documents(session_id)
    
    return jsonify({
        "ok": True, 
        "answer": answer, 
        "messages": messages, 
        "history": history,
        "has_documents": has_docs,
        "uploaded_documents": uploaded_docs
    })


@app.route("/new_chat", methods=["POST"])
@require_login
def new_chat():
    username = session.get('username')
    # reset conversation messages
    session.pop(get_user_conversation_key(), None)
    # reset session-scoped vectorstore/agent
    try:
        user_rag_service = get_user_rag_service(username)
        user_rag_service.reset_session(get_user_conversation_key())
    except Exception:
        pass
    session["conversation_id"] = str(uuid.uuid4())
    session["session_finished"] = False
    return jsonify({"ok": True})

@app.route("/load_chat", methods=["POST"])
@require_login
def load_chat():
    """Load a specific chat conversation from database"""
    username = session.get('username')
    data = request.get_json()
    chat_id = data.get("chat_id")
    
    if not chat_id:
        return jsonify({"ok": False, "error": "Chat ID required"}), 400
    
    try:
        # Load messages from database
        messages = ChatMessage.query.filter_by(
            user_id=username,
            conversation_id=chat_id
        ).order_by(ChatMessage.timestamp.asc()).all()
        
        # Convert to format expected by frontend
        chat_messages = [{
            "role": msg.role,
            "content": msg.message
        } for msg in messages]
        
        # Set this as the current conversation
        # Extract conversation ID from the full chat_id
        if chat_id.startswith(f"{username}_"):
            session["conversation_id"] = chat_id.replace(f"{username}_", "")
        else:
            session["conversation_id"] = chat_id
        
        return jsonify({
            "ok": True, 
            "messages": chat_messages,
            "chat_id": chat_id
        })
    except Exception as e:
        print(f"Error loading chat {chat_id}: {e}")
        return jsonify({"ok": False, "error": "Failed to load chat"}), 500

@app.route("/end_session", methods=["POST"])
@require_login
def end_session():
    username = session.get('username')
    messages = get_conversation_messages()
    if messages:
        title_source = next((m for m in messages if m.get("role") == "user"), None)
        title = (title_source.get("content")[:40] if title_source else "Conversation") or "Conversation"
        history = get_history_list()
        history.append({"id": get_user_conversation_key(), "title": title})
        save_user_history(username, history)
    session["session_finished"] = True
    return jsonify({"ok": True, "history": get_history_list()})

@app.route("/delete_chat", methods=["POST"])
@require_login
def delete_chat():
    """Delete a chat conversation from database"""
    username = session.get('username')
    data = request.get_json()
    chat_id = data.get("chat_id")
    
    if not chat_id:
        return jsonify({"ok": False, "error": "Chat ID required"}), 400
    
    try:
        # Delete messages from database
        ChatMessage.query.filter_by(
            user_id=username,
            conversation_id=chat_id
        ).delete()
        
        db.session.commit()
        
        # Get updated history
        history = get_history_list()
        
        return jsonify({"ok": True, "history": history})
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting chat {chat_id}: {e}")
        return jsonify({"ok": False, "error": "Failed to delete chat"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5006)), debug=True)


