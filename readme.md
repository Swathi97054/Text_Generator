# 🧠 LLM API with Multi-Modal Processing

A comprehensive FastAPI-based backend application for interacting with Large Language Models (LLMs) through LM Studio, featuring multi-modal input processing, logging, and advanced text analysis capabilities.

---

## 📦 Features

### Core LLM Functionality
- ✅ **Text Generation**: Generate responses from LLM models via LM Studio
- 🔍 **Multi-Modal Input Processing**: Support for text, images, PDFs, Word documents, and audio files
- 🎵 **Audio Processing**: Convert audio to text using Whisper, text-to-speech capabilities
- 📄 **Document Processing**: Extract and process content from PDFs, Word documents, and Excel files
- 🖼️ **Image Analysis**: Process and analyze image content
- 📊 **Data Export**: Export processed data to various formats

### Advanced Features
- 🗑️ **Soft Delete Logging**: Comprehensive request/response logging with soft delete capability
- 🔒 **Content Filtering**: Profanity detection and content moderation
- 📈 **Performance Monitoring**: Request timing and response analysis
- 🧩 **Modular Architecture**: Easily extensible and maintainable codebase
- 🆔 **UUID-based Tracking**: Unique identification for all requests and responses

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- LM Studio running locally (for LLM interactions)
- Docker and Docker Compose (for containerized deployment)
- FFmpeg (for audio processing)

### Option 1: Local Development

#### 1. Clone the Repository
```bash
git clone https://github.com/Swathi97054/Text_Generator.git
cd LLM
```

#### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Start LM Studio
- Launch LM Studio application
- Load your preferred LLM model
- Ensure it's running on the default port (usually 1234)

#### 5. Run the FastAPI Application
```bash
uvicorn textgen:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker Deployment

#### 1. Build and Run with Docker Compose
```bash
docker-compose up --build
```

This will:
- Build the application container
- Start a PostgreSQL database
- Expose the API on port 8000
- Set up proper networking between services

#### 2. Access the Application
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Database: localhost:5432

---

## 🧰 Tech Stack

### Backend Framework
- **FastAPI** - Modern, fast web framework for building APIs
- **Uvicorn** - ASGI server for running FastAPI applications

### AI & ML Libraries
- **Transformers** - Hugging Face transformers library
- **PyTorch** - Deep learning framework
- **Whisper** - OpenAI's speech recognition model
- **LM Studio** - Local LLM inference

### Data Processing
- **Pandas** - Data manipulation and analysis
- **Pillow (PIL)** - Image processing
- **PyPDF2** - PDF text extraction
- **python-docx** - Word document processing
- **openpyxl** - Excel file handling

### Audio Processing
- **pydub** - Audio file manipulation
- **soundfile** - Audio file I/O
- **gTTS** - Google Text-to-Speech

### Database & Utilities
- **SQLAlchemy** - SQL toolkit and ORM
- **Pydantic** - Data validation using Python type annotations
- **PostgreSQL** - Primary database (via Docker)

---

## 🔧 Configuration

### Environment Variables
```bash
# LM Studio Configuration
LM_STUDIO_HOST=localhost
LM_STUDIO_PORT=1234
LM_STUDIO_PROTOCOL=http

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/llm_logs

# Logging
LOG_LEVEL=info
```

### LM Studio Setup
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download your preferred LLM model
3. Start the local server in LM Studio
4. Ensure the API endpoint is accessible at the configured host/port

---

## 📚 API Endpoints

### Core LLM Endpoints
- `POST /generate` - Generate text responses from LLM
- `POST /chat` - Interactive chat with LLM
- `POST /process-file` - Process uploaded files (PDF, Word, Excel, images, audio)

### Utility Endpoints
- `GET /health` - Health check endpoint
- `GET /logs` - Retrieve application logs
- `POST /logs` - Create new log entries

### File Processing
- **Text Files**: Direct text processing
- **PDFs**: Text extraction and analysis
- **Word Documents**: Content extraction and processing
- **Excel Files**: Data parsing and analysis
- **Images**: Image analysis and description
- **Audio Files**: Speech-to-text conversion

---

## 🔒 Security & Features

- **Content Moderation**: Built-in profanity detection
- **Input Validation**: Comprehensive request validation using Pydantic
- **Error Handling**: Graceful error handling with detailed logging
- **CORS Support**: Cross-origin resource sharing enabled
- **Rate Limiting**: Configurable request rate limiting
- **Audit Trail**: Complete request/response logging for compliance

---

## 🚀 Deployment

### Production Considerations
- Use proper environment variables for sensitive configuration
- Implement proper authentication and authorization
- Set up monitoring and alerting
- Configure proper logging levels
- Use production-grade database (PostgreSQL recommended)
- Implement health checks and monitoring

### Scaling
- The application is designed to be stateless
- Can be deployed behind a load balancer
- Database connections are properly managed
- Background tasks are handled asynchronously

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add proper docstrings and type hints
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🆘 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check the `/docs` endpoint for interactive API documentation
- **Logs**: Application logs are stored in the `logs/` directory with rotation

---

## 🔄 Changelog

### Recent Updates
- Multi-modal file processing support
- Audio processing with Whisper integration
- Enhanced logging and monitoring
- Docker containerization
- PostgreSQL database integration
- Comprehensive error handling and validation
