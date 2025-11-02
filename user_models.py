"""
Database models for the Medical Document Assistant.

This module defines the SQLAlchemy models for user authentication and
document management.
"""

import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    """User model for authentication and user management."""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    documents = db.relationship('Document', backref='user', lazy=True, cascade='all, delete-orphan')
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.set_password(password)
    
    def set_password(self, password):
        """Hash and set the user's password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if the provided password matches the user's password."""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update the user's last login timestamp."""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def get_user_data_dir(self):
        """Get user-specific data directory."""
        from app import USER_DATA_DIR
        user_dir = os.path.join(USER_DATA_DIR, f"{self.username}_Upload")
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    def get_user_chroma_dir(self):
        """Get user-specific Chroma directory."""
        user_dir = self.get_user_data_dir()
        chroma_dir = os.path.join(user_dir, "chroma_store")
        os.makedirs(chroma_dir, exist_ok=True)
        return chroma_dir
    
    def __repr__(self):
        return f'<User {self.username}>'


class Document(db.Model):
    """Document model for tracking uploaded documents."""
    
    __tablename__ = 'documents'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)
    file_type = db.Column(db.String(50))
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    def __repr__(self):
        return f'<Document {self.filename}>'


class ChatSession(db.Model):
    """Chat session model for tracking user conversations."""
    
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Note: Relationships removed since we're using a simpler schema without foreign keys
    
    def __repr__(self):
        return f'<ChatSession {self.session_id}>'


class ChatMessage(db.Model):
    """Chat message model for storing conversation history."""
    
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(100), nullable=False)
    conversation_id = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ChatMessage {self.role}: {self.message[:50]}...>'


class UserLoginLog(db.Model):
    """User login log model for tracking detailed login information."""
    
    __tablename__ = 'user_login_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))  # IPv6 compatible
    user_agent = db.Column(db.String(500))
    login_method = db.Column(db.String(50))  # 'username', 'email'
    success = db.Column(db.Boolean, default=True)
    failure_reason = db.Column(db.String(200))  # For failed login attempts
    
    # Relationship
    user = db.relationship('User', backref='login_logs', lazy=True)
    
    def __repr__(self):
        return f'<UserLoginLog {self.user_id} at {self.login_time}>'


def init_db(app):
    """Initialize the database with the Flask app."""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create default admin user if no users exist
        if User.query.count() == 0:
            admin_user = User(
                username='admin',
                email='admin@docassist.ai',
                password='admin123'
            )
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Created default admin user (username: admin, password: admin123)")


def migrate_existing_users(users_file_path):
    """Migrate users from JSON file to SQL database."""
    import json
    
    if not os.path.exists(users_file_path):
        return
    
    try:
        with open(users_file_path, 'r') as f:
            json_users = json.load(f)
        
        migrated_count = 0
        
        for username, user_data in json_users.items():
            # Check if user already exists in database
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                continue
            
            # Create new user with email (use username@docassist.ai as default)
            email = user_data.get('email', f'{username}@docassist.ai')
            password = user_data.get('password', 'defaultpass123')
            
            new_user = User(
                username=username,
                email=email,
                password=password
            )
            
            db.session.add(new_user)
            migrated_count += 1
        
        if migrated_count > 0:
            db.session.commit()
            print(f"✅ Migrated {migrated_count} users from JSON to SQL database")
        
    except Exception as e:
        print(f"❌ Error migrating users: {e}")
        db.session.rollback()
