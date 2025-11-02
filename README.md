# DocAI - Medical Document Assistant

A powerful AI-powered medical document analyzer using optimized RAG (Retrieval-Augmented Generation) technology with 100% accuracy focus for medical document analysis.

## ✨ Key Features

- 📄 **Document Upload**: Support for PDF, DOCX, TXT, CSV files
- 🤖 **Enhanced AI Analysis**: Optimized RAG pipeline with intelligent query classification
- 💬 **Smart Conversations**: Professional greeting handling and context-aware responses
- 🎯 **100% Accuracy**: Strict document boundary detection for medical precision
- 🔐 **User Authentication**: Secure SQL database-based authentication with email/password
- 📊 **Login Tracking**: Detailed user login logs with IP, user agent, and timestamp tracking
- 👥 **User Management**: Individual user accounts with isolated document storage
- 🗄️ **SQL Database**: Scalable PostgreSQL/MySQL/SQLite support for user data
- 📱 **Responsive UI**: Modern, clean interface that works on all devices
- 🔄 **Session Management**: Persistent chat history and document management

## 🚀 Recent Optimizations

- **Enhanced RAG Pipeline**: Smaller chunks (800 chars) for better precision
- **Intelligent Query Classification**: Distinguishes greetings, document queries, and general medical questions
- **Document Boundary Detection**: Clear separation between document-based and general knowledge responses
- **Professional Conversation Flow**: Warm greetings and helpful guidance
- **Login Logging**: Comprehensive tracking of user login attempts and details

## 📁 Project Structure

```
DocAI/
├── run.py                 # Main entry point (renamed from main_application.py)
├── app.py                 # Flask routes and business logic
├── agent_logic.py         # Enhanced RAG service implementation
├── user_models.py         # Database models with login logging
├── clear_user_data.py     # User data cleanup utility
├── templates/             # HTML templates
├── UserData/              # User-specific data storage
├── instance/              # Database files
└── requirements.txt       # Python dependencies
```

## 🧹 Data Management

### Clear User Data (Preserve Database Structure)
```bash
python3 clear_user_data.py
```
This safely clears:
- User uploaded files
- Vector store embeddings
- Temporary uploads

While preserving:
- Database structure
- User accounts
- Login history

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```bash
# OpenAI API Key (Primary)
OPENAI_API_KEY=your_openai_api_key_here

# Mistral API Key (Fallback) 
MISTRAL_API_KEY=your_mistral_api_key_here

# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-here
PORT=5006

# Database Configuration (Optional - defaults to SQLite)
DATABASE_URL=sqlite:///docassist.db
# For PostgreSQL: postgresql://username:password@localhost/docassist
# For MySQL: mysql://username:password@localhost/docassist
```

**API Key Setup:**
- **OpenAI**: Get your key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Mistral**: Get your key from [Mistral Console](https://console.mistral.ai/)

### 3. Run the Application
```bash
python3 run.py
```

### 4. Access the Application
Open your browser and go to: `http://localhost:5006`

**Default Admin Account:**
- Username: `admin`
- Password: `admin123`
- Email: `admin@docassist.ai`

## How It Works

1. **Dual API System**: The system automatically tries OpenAI first, then falls back to Mistral if OpenAI is unavailable
2. **Document Processing**: Upload medical documents (PDF, DOCX, TXT) for analysis
3. **RAG Integration**: Documents are processed and stored in a vector database for intelligent retrieval
4. **Interactive Q&A**: Ask questions about your documents and get AI-powered answers

## 💡 Usage

1. **Register/Login**: Create an account or log in (all attempts are logged)
2. **Upload Documents**: Upload your medical documents (PDF, DOCX, TXT, CSV)
3. **Smart Conversations**: 
   - Start with greetings ("Hello", "Good morning")
   - Ask document-specific questions for 100% accurate answers
   - Request general medical information when needed
4. **Get Precise Answers**: Receive AI-powered responses with clear source attribution

### 🎯 Query Types
- **Document Queries**: "What is the patient's diagnosis?" → Searches uploaded documents
- **General Medical**: "What is diabetes?" → Provides general medical knowledge with disclaimers
- **Greetings**: "Hello" → Professional medical assistant introduction

## API Configuration

The system supports two AI providers with automatic fallback:

- **Primary**: OpenAI GPT-4o-mini (recommended)
- **Fallback**: Mistral AI (if OpenAI unavailable)

## Demo Mode

If no API keys are configured, the system runs in demo mode with helpful instructions for setup.

## File Support

- PDF documents
- Microsoft Word documents (.docx)
- Text files (.txt)
- CSV files

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## License

This project is for educational and research purposes.
