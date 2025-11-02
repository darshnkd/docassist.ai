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

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/darshnkd/docassist.ai.git
cd docassist.ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy the example environment file and configure your API keys:
```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
```bash
# API Keys for DocAI
OPENAI_API_KEY=your_openai_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here

# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-here
PORT=5005

# Database Configuration
DATABASE_URL=sqlite:///docassist.db
```

**🔑 API Key Setup:**
- **OpenAI**: Get your key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Mistral**: Get your key from [Mistral Console](https://console.mistral.ai/)

### 4. Run the Application
```bash
python3 run.py
```

The application will automatically:
- Create the SQLite database if it doesn't exist
- Set up all required tables
- Create a default admin user

### 5. Access the Application
Open your browser and go to: `http://localhost:5005`

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

## 🔄 Development Workflow

For contributors and team members, follow this professional workflow:

### Setting up Development Environment
```bash
# 1. Ensure you're on main and pull latest changes
git checkout main
git pull origin main

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes and test locally

# 4. Commit and push your changes
git add .
git commit -m "Add: your feature description"
git push -u origin feature/your-feature-name

# 5. Create a Pull Request on GitHub
# 6. After review and approval, merge into main
```

### Branch Naming Convention
- `feature/feature-name` - New features
- `bugfix/issue-description` - Bug fixes
- `hotfix/critical-fix` - Critical production fixes
- `docs/documentation-update` - Documentation updates

## 📸 Screenshots

### Login Interface
*Coming soon - Login page with user registration*

### Document Upload
*Coming soon - Drag and drop document upload interface*

### Chat Interface
*Coming soon - AI-powered chat with document analysis*

### User Dashboard
*Coming soon - User management and document history*

## 🔐 Security Features

- **Environment Variables**: All sensitive data stored in `.env` files (not committed to git)
- **Password Hashing**: Secure password storage using Werkzeug
- **Session Management**: Secure Flask sessions with configurable secret keys
- **Input Validation**: Email validation and secure file uploads
- **SQL Injection Protection**: SQLAlchemy ORM with parameterized queries
- **Login Tracking**: Comprehensive audit trail of all login attempts

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team

## License

This project is for educational and research purposes.
