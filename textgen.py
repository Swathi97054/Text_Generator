import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import docx
import lmstudio as lms
import pandas as pd
import PyPDF2
import requests

# Text-to-speech imports
# from TTS.api import TTS
import soundfile as sf
import whisper
from better_profanity import profanity
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    logger,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from pydub import AudioSegment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# import time


# Update the logging setup section in log_llm.py

# Ensure logs directory and file exist
try:
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create log filename with timestamp
    log_filename = os.path.join(logs_dir, f"llm_api_{datetime.now().strftime('%Y-%m-%d')}.log")

    # Test if we can write to the log file
    with open(log_filename, "a") as f:
        f.write(f"Application started at {datetime.now().isoformat()}\n")

    print(f"Log file ready at: {os.path.abspath(log_filename)}")

    # Configure logging with rotation
    from logging.handlers import RotatingFileHandler

    # Set up formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create file handler with rotation (10MB max size, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
    )
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Create app-specific logger
    logger = logging.getLogger("log_llm")
    logger.info(f"Logging initialized - logs will be stored in: {os.path.abspath(log_filename)}")

    # Log system information
    import platform
    import sys

    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python version: {sys.version}")

except Exception as e:
    print(f"ERROR setting up logging: {str(e)}")
    # Set up a fallback logger without file output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("log_llm")
    logger.error(f"Failed to initialize file logging: {str(e)}")

"""# Define Docker-compatible LM Studio endpoint
LM_STUDIO_HOST = os.environ.get("LM_STUDIO_HOST", "host.docker.internal")
LM_STUDIO_PORT = os.environ.get("LM_STUDIO_PORT", "1234")
# Change from WebSocket to HTTP
LM_STUDIO_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1"

# Set the LM Studio API endpoint before using it
os.environ["LMSTUDIO_API_URL"] = LM_STUDIO_URL"""


# Initialize profanity filter
profanity.load_censor_words()

# Supported languages with names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
}

# Language prompts to guide model responses
LANGUAGE_PROMPTS = {
    "en": "Respond in English.",
    "fr": "Veuillez répondre en français.",
    "es": "Por favor, responde en español.",
    "de": "Bitte antworten Sie auf Deutsch.",
    "it": "Per favore, rispondi in italiano.",
    "zh": "请用中文回答。",
    "ja": "日本語で回答してください。",
    "ru": "Пожалуйста, ответьте на русском языке.",
}
USER_INTENTS = {
    "question": {
        "description": "User is asking a factual question",
        "examples": ["What is the capital of France?", "How does photosynthesis work?"],
    },
    "command": {
        "description": "User is requesting the system to perform a task",
        "examples": ["Translate this text", "Summarize this article"],
    },
    "conversation": {
        "description": "User is engaging in casual conversation",
        "examples": ["How are you today?", "Tell me a joke"],
    },
    "creative": {
        "description": "User is requesting creative content generation",
        "examples": ["Write a poem about autumn", "Create a short story"],
    },
    "clarification": {
        "description": "User is asking for clarification on a previous response",
        "examples": ["Can you explain that in simpler terms?", "What do you mean by that?"],
    },
    "comparison": {
        "description": "User is asking to compare things",
        "examples": ["What's the difference between X and Y?", "Compare these two options"],
    },
    "opinion": {
        "description": "User is asking for opinions or recommendations",
        "examples": ["What do you think about this?", "Which option is better?"],
    },
    "technical": {
        "description": "User is asking for technical information or assistance",
        "examples": ["How do I fix this error?", "Explain how this code works"],
    },
    "translation": {
        "description": "User is asking for translation assistance",
        "examples": ["Translate 'hello' to French", "How do you say 'thank you' in Spanish?"],
    },
}


# Available models configuration with language support
AVAILABLE_MODELS = {
    "llava-7b": {
        "path": "llava-v1.5-7b",
        "description": "LLaVA v1.5 Vision Model (7B)",
        "active": True,
        "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ru"],  # Supports English primarily
    },
    "minicpm": {
        "path": "minicpm-o-2_6",
        "description": "MiniCPM-O 2.6 Model",
        "active": True,
        "languages": ["en", "zh"],
    },
    "deepseek-qwen": {
        "path": "deepseek-r1-distill-qwen-7b",
        "description": "DeepSeek R1 Distilled from Qwen 7B",
        "active": True,
        "languages": ["en", "zh", "fr", "es", "de", "ja"],
    },
    "gemma-3b": {
        "path": "gemma-3-1b-it",
        "description": "Gemma 3B Instruct Tuned",
        "active": True,
        "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ru"],
    },
    "gemma-7b": {
        "path": "gemma-7b-it",
        "description": "Gemma 7B Instruct Tuned",
        "active": True,
        "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ru"],
    },
    "llama-3": {
        "path": "llama-3.2-1b-instruct",
        "description": "Llama 3.2 1B Instruct",
        "active": True,
        "languages": ["en", "fr", "es", "de", "it", "zh", "ja", "ru"],
    },
}

# Model cache to avoid reloading
model_cache: Dict[str, any] = {}


# Safety checker class
class SafetyChecker:
    def check_input(self, text: str) -> bool:
        if profanity.contains_profanity(text):
            logger.warning(f"Unsafe input detected: {text}")
            return False
        return True

    def check_output(self, text: str) -> bool:
        if profanity.contains_profanity(text):
            logger.warning(f"Unsafe output detected: {text}")
            return False
        return True


checker = SafetyChecker()


# Memory-related classes
class Message:
    def __init__(self, role: str, content: str, timestamp=None, intent=None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.intent = intent  # Store the detected intent

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
        }


class Conversation:
    def __init__(self, max_history=10):
        self.id = str(uuid.uuid4())
        self.messages = deque(maxlen=max_history)  # Limit history size
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        self.language = "en"  # Default language

    def add_message(self, role: str, content: str, intent=None):
        message = Message(role, content, intent=intent)
        self.messages.append(message)
        self.last_updated = datetime.now()
        return message

    def get_history(self, as_dict=False):
        if as_dict:
            return [msg.to_dict() for msg in self.messages]
        return list(self.messages)

    def get_context_string(self, max_length=None):
        """Convert conversation history to a string summary for context"""
        context = []
        for msg in self.messages:
            prefix = "User: " if msg.role == "user" else "Assistant: "
            context.append(f"{prefix}{msg.content}")

        result = "\n".join(context)
        if max_length and len(result) > max_length:
            # Truncate if needed, keeping most recent context
            return "..." + result[-max_length:]
        return result

    def set_language(self, language):
        """Set the primary language for this conversation"""
        self.language = language


class MemoryManager:
    def __init__(self, max_conversations=100, max_history=10):
        self.conversations = {}  # Map from conversation_id to Conversation
        self.max_conversations = max_conversations
        self.max_history = max_history

    def create_conversation(self, language="en"):
        """Create a new conversation and return its ID"""
        # Clean up old conversations if limit reached
        if len(self.conversations) >= self.max_conversations:
            oldest_id = min(self.conversations.items(), key=lambda x: x[1].last_updated)[0]
            del self.conversations[oldest_id]

        conversation = Conversation(max_history=self.max_history)
        conversation.set_language(language)
        self.conversations[conversation.id] = conversation
        return conversation.id

    def get_conversation(self, conversation_id):
        """Get a conversation by ID"""
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id, role, content, intent=None):
        """Add a message to a conversation"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            # Create a new conversation if ID doesn't exist
            conversation_id = self.create_conversation()
            conversation = self.get_conversation(conversation_id)

        conversation.add_message(role, content, intent)
        return conversation_id

    def get_conversation_history(self, conversation_id, as_dict=False):
        """Get the message history for a conversation"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        return conversation.get_history(as_dict=as_dict)

    def get_conversation_context(self, conversation_id, max_length=None):
        """Get the conversation history as a context string"""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return ""
        return conversation.get_context_string(max_length=max_length)

    def list_conversations(self):
        """Get a list of all conversation IDs with timestamps"""
        return [
            {
                "id": conv_id,
                "created_at": conv.created_at.isoformat(),
                "last_updated": conv.last_updated.isoformat(),
                "message_count": len(conv.messages),
                "language": conv.language,
            }
            for conv_id, conv in self.conversations.items()
        ]


# Model configuration class with properties
class ModelConfig(BaseModel):
    context_length: int = Field(
        default=4096, ge=1, le=32768, description="Maximum length of input context"
    )
    eval_batch_size: int = Field(
        default=1, ge=1, le=128, description="Batch size for model evaluation"
    )
    rope_freq_base: float = Field(
        default=10000.0,
        ge=1.0,
        le=1000000.0,
        description="Base frequency for RoPE (Rotary Position Embedding)",
    )
    rope_freq_scale: float = Field(
        default=1.0, ge=0.1, le=100.0, description="Scaling factor for RoPE frequency"
    )
    temperature: float = Field(
        default=0.7, ge=0, le=2.0, description="Controls randomness in generation"
    )
    top_p: float = Field(default=0.9, ge=0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=40, ge=0, description="Top-k sampling parameter")


# Create a directory for temporary file storage
TEMP_DIR = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# Modify the Whisper model loading section at the top of your file
# Initialize TTS model
# Replace the TTS imports and initialization with this:
try:
    from gtts import gTTS

    logger.info("gTTS module loaded successfully")
    TTS_AVAILABLE = True
except ImportError as import_error:
    logger.error(f"gTTS module not found: {str(import_error)}")
    TTS_AVAILABLE = False
except Exception as e:
    logger.error(f"Failed to initialize gTTS: {str(e)}")
    TTS_AVAILABLE = False
# Replace the existing try-except block with this one
try:
    # First check if whisper is properly imported
    import whisper

    # Try to load a small model by default for faster inference
    try:
        WHISPER_MODEL = whisper.load_model("base")
        logger.info("Whisper model loaded successfully: base")
    except Exception as model_error:
        logger.warning(f"Could not load Whisper base model: {str(model_error)}")
        try:
            # Fallback to tiny model if base fails
            WHISPER_MODEL = whisper.load_model("tiny")
            logger.info("Whisper model loaded successfully: tiny (fallback)")
        except Exception as tiny_error:
            logger.error(f"Failed to load Whisper tiny model: {str(tiny_error)}")
            WHISPER_MODEL = None

    # Set ffmpeg path explicitly for pydub if needed
    ffmpeg_path = os.environ.get("FFMPEG_PATH", "ffmpeg")
    AudioSegment.converter = ffmpeg_path
    logger.info(f"Setting ffmpeg path to: {ffmpeg_path}")

except ImportError as import_error:
    logger.error(f"Whisper module not found: {str(import_error)}")
    WHISPER_MODEL = None
except Exception as e:
    logger.error(f"Failed to initialize Whisper: {str(e)}")
    WHISPER_MODEL = None


def process_audio_file(file_content, file_ext):
    """Process audio file and return extracted text using Whisper with progress indicators"""
    try:
        start_time = time.time()
        print(f"\nProcessing audio file ({len(file_content)/1024:.1f} KB)...", end="")

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(file_content)
        temp_file.close()

        # Check if Whisper model is available
        if WHISPER_MODEL is None:
            print(" error: Whisper model not available")
            logger.error("Audio transcription failed: Whisper model not available")
            return "[Error: Whisper model not available. Please check server logs.]"

        # Process different audio formats
        if file_ext.lower() in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]:
            try:
                # For formats that may need conversion, use pydub
                if file_ext.lower() not in [".wav", ".mp3", ".flac"]:
                    try:
                        print(" converting format...", end="")
                        logger.info(f"Converting {file_ext} to WAV format for processing")
                        audio = AudioSegment.from_file(temp_file.name, format=file_ext[1:])
                        wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        wav_file.close()
                        audio.export(wav_file.name, format="wav")
                        audio_path = wav_file.name
                    except Exception as conv_error:
                        print(" conversion error!")
                        logger.error(f"Error converting audio format: {str(conv_error)}")
                        return f"[Audio conversion error: {str(conv_error)}. Try uploading a WAV file instead.]"
                else:
                    audio_path = temp_file.name

                # Use Whisper to transcribe
                print(" transcribing...", end="")
                logger.info(f"Transcribing audio file with Whisper: {audio_path}")
                result = WHISPER_MODEL.transcribe(audio_path)
                transcription = result["text"]

                # Log performance
                processing_time = time.time() - start_time
                print(f" complete! ({processing_time:.2f}s) ✓")
                logger.info(f"Audio transcription completed in {processing_time:.2f} seconds")

                # Clean up temporary files
                try:
                    os.unlink(temp_file.name)
                    if audio_path != temp_file.name:
                        os.unlink(audio_path)
                except Exception as clean_error:
                    logger.warning(f"Error cleaning up temp files: {str(clean_error)}")

                return transcription

            except Exception as e:
                print(" error!")
                logger.error(f"Error in audio transcription: {str(e)}")
                return f"[Audio transcription error: {str(e)}]"
        else:
            print(f" error: unsupported format {file_ext}")
            return f"[Unsupported audio format: {file_ext}]"
    except Exception as e:
        print(" error!")
        logger.error(f"Error processing audio: {str(e)}")
        return f"[Error processing audio file: {str(e)}]"


def extract_audio_features(file_path):
    """Extract audio features for analysis"""
    try:
        # Load the audio file using pydub for feature extraction
        audio = AudioSegment.from_file(file_path)

        # Extract basic features
        return {
            "duration_seconds": len(audio) / 1000,  # pydub uses milliseconds
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "bitrate": audio.frame_width * 8 * audio.frame_rate,
        }
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
        return {"error": str(e)}


def text_to_speech(
    text: str, output_format: str = "mp3", language: str = "en", voice: str = "default"
):
    """Convert text to speech using gTTS and return audio data"""
    try:
        # Show a simple progress indicator
        print(f"\nGenerating speech ({len(text)} chars)...", end="")
        start_time = time.time()

        if not TTS_AVAILABLE:
            raise Exception("Text-to-speech is not available")

        # Create a temporary file for the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}")
        temp_file.close()

        # Use appropriate language code for gTTS
        gtts_lang = language
        if language not in ["en", "fr", "es", "de", "it", "zh", "ja", "ru"]:
            logger.warning(
                f"Language {language} might not be supported by gTTS, falling back to English"
            )
            gtts_lang = "en"

        print(".", end="")  # Progress indicator

        # Generate speech with gTTS
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(temp_file.name)

        print(".", end="")  # Progress indicator

        # Read the generated audio file
        with open(temp_file.name, "rb") as f:
            audio_data = f.read()

        # Clean up
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logger.warning(f"Error removing temporary file: {str(e)}")

        # Log performance
        processing_time = time.time() - start_time
        logger.info(f"TTS completed in {processing_time:.2f} seconds")

        # Show completion
        print(f" complete! ({processing_time:.2f}s, {len(audio_data)/1024:.1f} KB) ✓")

        # If user requested wav but gTTS only does mp3, note that in the result
        actual_format = "mp3" if output_format != "mp3" else output_format
        if actual_format != output_format:
            logger.warning(
                f"Requested format {output_format} not supported by gTTS, using {actual_format} instead"
            )

        return {
            "audio_data": audio_data,
            "format": actual_format,
            "sample_rate": 22050,  # Default for gTTS
            "duration": 0,  # We can't easily determine duration without additional libraries
            "channels": 1,
        }
    except Exception as e:
        print(" error!")
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise


def extract_text_from_pdf(file_content):
    """Extract text from a PDF file"""
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        # Extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"

            # Add page number for context
            text += f"[End of page {page_num + 1}]\n\n"

            # Limit text extraction to prevent huge inputs
            if len(text) > 10000:
                text += f"\n\n[Document truncated, {len(pdf_reader.pages) - page_num - 1} more pages not shown]"
                break

        return text
    except Exception as e:
        return f"[Error extracting PDF text: {str(e)}]"


def extract_text_from_docx(file_content):
    """Extract text from a DOCX file"""
    try:
        docx_file = BytesIO(file_content)
        doc = docx.Document(docx_file)

        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)

            # Limit text extraction
            if len("\n".join(full_text)) > 10000:
                full_text.append("\n[Document truncated due to length]")
                break

        return "\n".join(full_text)
    except Exception as e:
        return f"[Error extracting DOCX text: {str(e)}]"


def encode_image_to_base64(img_content):
    """Properly encode image content to base64"""
    return base64.b64encode(img_content).decode("utf-8")


def process_file(file: UploadFile) -> dict:
    """Process different file types and return their content/metadata"""
    file_content = file.file.read()
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
        # Process image
        try:
            img = Image.open(io.BytesIO(file_content))
            width, height = img.size
            # Convert to base64 for vision models
            img_b64 = encode_image_to_base64(file_content)
            return {
                "type": "image",
                "filename": file.filename,
                "size": len(file_content),
                "dimensions": f"{width}x{height}",
                "format": file_ext.replace(".", ""),
                "base64": img_b64,
                "description": f"An image file ({file.filename}) with dimensions {width}x{height}",
            }
        except Exception as e:
            return {"type": "error", "filename": file.filename, "error": str(e)}

    elif file_ext in [".csv", ".xlsx", ".xls"]:
        # Process table/spreadsheet
        try:
            if file_ext == ".csv":
                df = pd.read_csv(io.BytesIO(file_content))
            else:
                df = pd.read_excel(io.BytesIO(file_content))

            # Get basic info about the table
            rows, cols = df.shape
            headers = df.columns.tolist()

            # Convert small tables to text
            if rows <= 20 and cols <= 10:
                table_text = df.to_string()
            else:
                # For larger tables, just provide a summary
                table_text = (
                    f"Table with {rows} rows and {cols} columns. Headers: {', '.join(headers[:10])}"
                )
                if len(headers) > 10:
                    table_text += f" and {len(headers) - 10} more columns"

            return {
                "type": "table",
                "filename": file.filename,
                "size": len(file_content),
                "rows": rows,
                "columns": cols,
                "headers": headers,
                "sample": df.head(5).to_dict(),
                "table_text": table_text,
            }
        except Exception as e:
            return {"type": "error", "filename": file.filename, "error": str(e)}

    elif file_ext in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
        try:
            # Save the audio temporarily
            temp_file_id = str(uuid.uuid4())
            temp_file_path = os.path.join(TEMP_DIR, f"{temp_file_id}{file_ext}")
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            # Extract text from audio using Whisper
            logger.info(f"Processing audio file: {file.filename}")
            transcribed_text = process_audio_file(file_content, file_ext)

            # Get audio features
            audio_features = extract_audio_features(temp_file_path)

            return {
                "type": "audio",
                "filename": file.filename,
                "size": len(file_content),
                "temp_path": temp_file_path,
                "transcription": transcribed_text,
                "duration": audio_features.get("duration_seconds", "unknown"),
                "sample_rate": audio_features.get("sample_rate", "unknown"),
                "temp_id": temp_file_id,
            }
        except Exception as e:
            logger.error(f"Error processing audio file {file.filename}: {str(e)}")
            return {"type": "error", "filename": file.filename, "error": str(e)}

    elif file_ext in [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml"]:
        # Process text file
        try:
            text_content = file_content.decode("utf-8")
            lines = text_content.count("\n") + 1

            return {
                "type": "text",
                "filename": file.filename,
                "size": len(file_content),
                "lines": lines,
                "content": (
                    text_content
                    if len(text_content) < 10000
                    else text_content[:10000] + "...(truncated)"
                ),
            }
        except Exception as e:
            return {"type": "error", "filename": file.filename, "error": str(e)}

    elif file_ext == ".pdf":
        # Process PDF file
        try:
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(file_content)

            # Save the PDF temporarily for possible future use
            temp_file_id = str(uuid.uuid4())
            temp_file_path = os.path.join(TEMP_DIR, f"{temp_file_id}{file_ext}")
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            return {
                "type": "document",
                "subtype": "pdf",
                "filename": file.filename,
                "size": len(file_content),
                "temp_path": temp_file_path,
                "content": pdf_text,
                "temp_id": temp_file_id,
            }
        except Exception as e:
            return {"type": "error", "filename": file.filename, "error": str(e)}

    elif file_ext in [".doc", ".docx"]:
        # Process Word document
        try:
            # Extract text from DOCX
            if file_ext == ".docx":
                doc_text = extract_text_from_docx(file_content)
            else:
                # For older .doc format, we'll just indicate it's not directly supported
                doc_text = "[DOC format not directly supported for text extraction]"

            # Save the document temporarily
            temp_file_id = str(uuid.uuid4())
            temp_file_path = os.path.join(TEMP_DIR, f"{temp_file_id}{file_ext}")
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            return {
                "type": "document",
                "subtype": "doc" if file_ext == ".doc" else "docx",
                "filename": file.filename,
                "size": len(file_content),
                "temp_path": temp_file_path,
                "content": doc_text,
                "temp_id": temp_file_id,
            }
        except Exception as e:
            return {"type": "error", "filename": file.filename, "error": str(e)}

    else:
        # Generic file handling
        return {
            "type": "unknown",
            "filename": file.filename,
            "size": len(file_content),
            "description": f"An uploaded file ({file.filename}) of size {len(file_content) // 1024} KB",
        }


# Default configuration
DEFAULT_CONFIG = ModelConfig()

# LLM Wrapper class to handle configuration
# ...existing code...


# LLM Wrapper class to handle configuration
class LLMWrapper:
    def __init__(self, model_path: str, config: ModelConfig):
        self.url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        self.model_path = model_path
        self.config = config

        # Create session with retry logic
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def respond(self, prompt: str, language: str = "en") -> Tuple[str, int, int]:
        """Generate response with context length limits and language instruction"""
        try:
            print("\nGenerating response...", end="")
            start_time = time.time()

            # Add language instruction if specified
            if language != "en":
                language_instruction = LANGUAGE_PROMPTS.get(language, "")
                full_prompt = f"{prompt}\n\n{language_instruction}"
            else:
                full_prompt = prompt

            # Truncate input prompt to context length
            truncated_prompt = full_prompt[: self.config.context_length]
            input_length = len(truncated_prompt)

            # Use HTTP API with OpenAI-compatible endpoint
            payload = {
                "model": self.model_path,
                "messages": [{"role": "user", "content": truncated_prompt}],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": min(4096, self.config.context_length),
                "stream": False,  # Ensure we're not using streaming
            }

            # Test connection before trying full request
            try:
                test_response = self.session.get(f"{self.url}/models", timeout=5)
                if test_response.status_code != 200:
                    raise Exception(f"LM Studio API not available: {test_response.status_code}")
                print(".", end="")
            except Exception as conn_error:
                raise Exception(f"Cannot connect to LM Studio API: {str(conn_error)}")

            # Make the actual request with longer timeout
            print(".", end="")
            response = self.session.post(
                f"{self.url}/chat/completions",
                json=payload,
                timeout=120,  # Increase timeout to 2 minutes
            )

            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code}, {response.text}")

            response_data = response.json()
            response_str = response_data["choices"][0]["message"]["content"]

            # Truncate output if it exceeds context length
            truncated_response = response_str[: self.config.context_length]
            output_length = len(truncated_response)

            # Calculate completion time and show it
            elapsed = time.time() - start_time
            print(f" complete! ({elapsed:.2f}s, {output_length} chars) ✓")

            return truncated_response, input_length, output_length
        except Exception as e:
            print(" error!")
            raise Exception(f"Generation failed: {str(e)}")

    def respond_with_progress(self, prompt: str, language: str = "en") -> Tuple[str, int, int]:
        """Generate response with streaming and progress bar"""
        try:
            # Add language instruction if specified
            if language != "en":
                language_instruction = LANGUAGE_PROMPTS.get(language, "")
                full_prompt = f"{prompt}\n\n{language_instruction}"
            else:
                full_prompt = prompt

            # Truncate input prompt to context length
            truncated_prompt = full_prompt[: self.config.context_length]
            input_length = len(truncated_prompt)

            # Set up streaming payload
            payload = {
                "model": self.model_path,
                "messages": [{"role": "user", "content": truncated_prompt}],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": min(4096, self.config.context_length),
                "stream": True,  # Enable streaming
            }

            # Test connection before trying full request
            try:
                test_response = self.session.get(f"{self.url}/models", timeout=5)
                if test_response.status_code != 200:
                    raise Exception(f"LM Studio API not available: {test_response.status_code}")
                logger.info(
                    f"API connection successful, found models: {[m['id'] for m in test_response.json()['data']]}"
                )
            except Exception as conn_error:
                raise Exception(f"Cannot connect to LM Studio API: {str(conn_error)}")

            # Make the streaming request
            logger.info(f"Starting streaming request to {self.url}/chat/completions")

            # Initialize progress tracking
            full_text = ""
            start_time = time.time()
            last_update_time = start_time
            update_interval = 0.2  # Update progress every 0.2 seconds
            # progress_chars = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
            # progress_idx = 0
            token_count = 0
            estimated_total = 200  # Initial estimate of total tokens
            bar_length = 30  # Length of the progress bar in characters

            # Make the streaming request
            with self.session.post(
                f"{self.url}/chat/completions", json=payload, timeout=120, stream=True
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"API error: {response.status_code}, {response.text}")

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")

                        # Skip the "data: " prefix and handle special messages
                        if line_text.startswith("data: "):
                            line_text = line_text[6:]

                            # Check for the "[DONE]" message that indicates the end of the stream
                            if line_text == "[DONE]":
                                break

                            try:
                                # Parse the JSON data
                                data = json.loads(line_text)

                                # Extract the content delta
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_text += content
                                        token_count += 1

                                        # Dynamically adjust estimated total
                                        if token_count > estimated_total * 0.8:
                                            estimated_total = int(estimated_total * 1.5)

                                        # Update progress if enough time has passed
                                        current_time = time.time()
                                        if current_time - last_update_time >= update_interval:
                                            elapsed = current_time - start_time
                                            speed = token_count / elapsed if elapsed > 0 else 0

                                            # Calculate progress percentage (capped at 100%)
                                            progress = min(
                                                100, (token_count / estimated_total) * 100
                                            )

                                            # Create progress bar
                                            filled_length = int(bar_length * progress // 100)
                                            bar = "█" * filled_length + "░" * (
                                                bar_length - filled_length
                                            )

                                            # Print progress bar
                                            print(
                                                f"\r[{bar}] {token_count} tokens | {speed:.1f} t/s | {elapsed:.1f}s",
                                                end="",
                                            )
                                            last_update_time = current_time
                            except json.JSONDecodeError:
                                # Skip lines that aren't valid JSON
                                continue

            # Print completion
            print("\rResponse generation complete! ✓                                    ")

            # Calculate final statistics
            elapsed = time.time() - start_time
            output_length = len(full_text)
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

            # Print completion with final stats
            filled_bar = "█" * bar_length
            print(
                f"\r[{filled_bar}] {token_count} tokens | {tokens_per_sec:.1f} t/s | {elapsed:.1f}s ✓"
            )
            logger.info(
                f"Generated {token_count} tokens in {elapsed:.2f} seconds ({tokens_per_sec:.1f} tokens/sec)"
            )

            # Truncate output if it exceeds context length
            truncated_response = full_text[: self.config.context_length]

            return truncated_response, input_length, output_length

        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            raise Exception(f"Generation failed: {str(e)}")


# Add a respond_with_vision method to LLMWrapper (simplified implementation)
# Ensure wrapper is defined before checking its attributes
wrapper = None  # Initialize wrapper or replace with actual initialization logic
if not hasattr(wrapper, "respond_with_vision"):
    # Add this to your LLMWrapper class
    def respond_with_vision(self, prompt, image_data, language):
        """Implementation for vision model using HTTP API"""
        try:
            logger.info(f"Processing vision request with {len(image_data)} images")

            # Format images for the API
            formatted_images = []
            for img in image_data:
                formatted_images.append({"data": img["base64"], "type": f"image/{img['format']}"})

            # Create payload for the vision model
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            # Add image to the content
            for img in formatted_images:
                messages[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img['type']};base64,{img['data']}"},
                    }
                )

            # Add language instruction if needed
            if language != "en":
                language_instruction = LANGUAGE_PROMPTS.get(language, "")
                if language_instruction:
                    messages[0]["content"][0]["text"] += f"\n\n{language_instruction}"

            # Log the request details
            logger.info(f"Sending vision request to model: {self.model_path}")

            # Make API request
            payload = {
                "model": self.model_path,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": min(4096, self.config.context_length),
                "stream": False,
            }

            # Send request with longer timeout for vision processing
            response = self.session.post(
                f"{self.url}/chat/completions",
                json=payload,
                timeout=180,  # 3 minutes timeout for vision processing
            )

            if response.status_code != 200:
                error_detail = f"Vision API error: {response.status_code}"
                try:
                    error_detail += f", {response.json()}"
                except:
                    error_detail += f", {response.text}"
                logger.error(error_detail)
                raise Exception(error_detail)

            response_data = response.json()
            response_str = response_data["choices"][0]["message"]["content"]
            input_length = len(prompt)
            output_length = len(response_str)

            logger.info(f"Vision response received: {len(response_str)} characters")
            return response_str, input_length, output_length

        except Exception as e:
            logger.error(f"Vision processing failed: {str(e)}")
            # Fall back to text-only if vision fails
            logger.warning("Falling back to text-only mode due to vision processing failure")
            return self.respond(
                f"{prompt} [Note: Image processing failed. Responding based on text only.]",
                language,
            )

    def respond_with_vision_progress(self, prompt, image_data, language):
        """Implementation for vision model using HTTP API with progress bar"""
        try:
            logger.info(f"Processing vision request with {len(image_data)} images (with progress)")

            # Format images for the API
            formatted_images = []
            for img in image_data:
                formatted_images.append({"data": img["base64"], "type": f"image/{img['format']}"})

            # Create payload for the vision model
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            # Add image to the content
            for img in formatted_images:
                messages[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img['type']};base64,{img['data']}"},
                    }
                )

            # Add language instruction if needed
            if language != "en":
                language_instruction = LANGUAGE_PROMPTS.get(language, "")
                if language_instruction:
                    messages[0]["content"][0]["text"] += f"\n\n{language_instruction}"

            # Log the request details
            logger.info(f"Sending vision request to model with streaming: {self.model_path}")

            # Make API request with streaming
            payload = {
                "model": self.model_path,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": min(4096, self.config.context_length),
                "stream": True,
            }

            # Initialize progress tracking
            full_text = ""
            start_time = time.time()
            last_update_time = start_time
            update_interval = 0.2  # Update progress every 0.2 seconds
            token_count = 0
            estimated_total = 300  # Initial estimate of total tokens (vision may generate more)
            bar_length = 30  # Length of the progress bar in characters

            # Make the streaming request
            with self.session.post(
                f"{self.url}/chat/completions",
                json=payload,
                timeout=180,  # 3 minutes for vision
                stream=True,
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"Vision API error: {response.status_code}, {response.text}")

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")

                        # Skip the "data: " prefix and handle special messages
                        if line_text.startswith("data: "):
                            line_text = line_text[6:]

                            # Check for the "[DONE]" message
                            if line_text == "[DONE]":
                                break

                            try:
                                # Parse the JSON data
                                data = json.loads(line_text)

                                # Extract the content delta
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_text += content
                                        token_count += 1

                                        # Dynamically adjust estimated total
                                        if token_count > estimated_total * 0.8:
                                            estimated_total = int(estimated_total * 1.5)

                                        # Update progress if enough time has passed
                                        current_time = time.time()
                                        if current_time - last_update_time >= update_interval:
                                            elapsed = current_time - start_time
                                            speed = token_count / elapsed if elapsed > 0 else 0

                                            # Calculate progress percentage
                                            progress = min(
                                                100, (token_count / estimated_total) * 100
                                            )

                                            # Create progress bar
                                            filled_length = int(bar_length * progress // 100)
                                            bar = "█" * filled_length + "░" * (
                                                bar_length - filled_length
                                            )

                                            # Print progress bar
                                            print(
                                                f"\r[{bar}] {token_count} tokens | {speed:.1f} t/s | {elapsed:.1f}s (Vision)",
                                                end="",
                                            )
                                            last_update_time = current_time
                            except json.JSONDecodeError:
                                # Skip lines that aren't valid JSON
                                continue

            # Calculate final statistics
            elapsed = time.time() - start_time
            output_length = len(full_text)
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

            # Print completion with final stats
            filled_bar = "█" * bar_length
            print(
                f"\r[{filled_bar}] {token_count} tokens | {tokens_per_sec:.1f} t/s | {elapsed:.1f}s (Vision) ✓"
            )
            logger.info(
                f"Generated {token_count} tokens for vision request in {elapsed:.2f} seconds ({tokens_per_sec:.1f} tokens/sec)"
            )

            return full_text, len(prompt), output_length

        except Exception as e:
            logger.error(f"Vision processing with progress failed: {str(e)}")
            # Fall back to regular vision processing
            return self.respond_with_vision(prompt, image_data, language)


class IntentClassifier:
    def __init__(self):
        self.intents = USER_INTENTS

    def classify_intent(self, text: str, model_wrapper: LLMWrapper) -> str:
        """
        Classify the user's intent by prompting the LLM
        Returns the classified intent category
        """
        # Build a prompt for zero-shot classification
        intent_options = ", ".join(self.intents.keys())

        prompt = f"""Classify the following user input into exactly one of these intent categories: {intent_options}.
        
Respond with only the category name, nothing else.

User input: "{text}"

Intent category:"""

        try:
            # Use the LLM to classify the intent
            result, _, _ = model_wrapper.respond(prompt, "en")

            # Clean and normalize the result
            result = result.strip().lower()

            # Extract just the category if there's additional content
            for intent in self.intents.keys():
                if intent in result:
                    return intent

            # Default to conversation if no clear intent is detected
            return "conversation"
        except Exception as e:
            print(f"Intent classification failed: {str(e)}")
            return "conversation"  # Default fallback

    def get_intent_prompt(self, intent: str, original_prompt: str) -> str:
        """
        Enhance the original prompt with intent-specific instructions
        """
        intent_prompts = {
            "question": "Please provide a factual and accurate answer to this question: ",
            "command": "I'll help you with this task. ",
            "conversation": "Let's have a friendly conversation. ",
            "creative": "I'll create something creative based on your request. ",
            "clarification": "Let me clarify this for you. ",
            "comparison": "I'll compare these items for you. ",
            "opinion": "Here's my perspective on this topic. ",
            "technical": "Here's the technical information you requested. ",
        }

        # Get the appropriate prefix for the intent
        prefix = intent_prompts.get(intent, "")

        # Return the enhanced prompt
        return f"{prefix}{original_prompt}"


def verify_available_models(silent=False):
    """Check which models are actually available in LM Studio using HTTP API"""
    try:
        # Updated LM_STUDIO_URL configuration
        LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        print(f"Attempting to connect to LM Studio at {LM_STUDIO_URL}")
        # Get models from LM Studio API
        # Updated LM_STUDIO_URL configuration
        # Updated LM_STUDIO_URL configuration
        # Updated LM_STUDIO_URL configuration
        # Default to localhost if not set
        LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        # LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234")  # Default to localhost if not set
        # LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234")  # Default to localhost if not set
        # LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234")  # Default to localhost if not set
        # LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234")  # Default to localhost if not set
        response = requests.get(f"{LM_STUDIO_URL}/models", timeout=10)
        if response.status_code == 200:
            available_models_data = response.json()
            available_model_ids = [model["id"] for model in available_models_data["data"]]

            if not silent:
                print(f"Successfully connected to LM Studio. Found models: {available_model_ids}")

            for model_name, info in AVAILABLE_MODELS.items():
                # Check if model is in available models list
                if info["path"] in available_model_ids:
                    AVAILABLE_MODELS[model_name]["active"] = True
                    if not silent:
                        print(f"✓ Model {model_name} ({info['path']}) is available")
                else:
                    AVAILABLE_MODELS[model_name]["active"] = False
                    if not silent:
                        print(
                            f"✗ Model {model_name} ({info['path']}) is not available in LM Studio"
                        )
        else:
            # If API call fails, mark all models as inactive
            for model_name in AVAILABLE_MODELS:
                AVAILABLE_MODELS[model_name]["active"] = False
            if not silent:
                print(f"✗ Failed to get models from LM Studio API: {response.status_code}")

    except Exception as e:
        # If connection fails, mark all models as inactive
        for model_name in AVAILABLE_MODELS:
            AVAILABLE_MODELS[model_name]["active"] = False
        if not silent:
            print(f"✗ Failed to connect to LM Studio: {str(e)}")


# Define lifespan context manager
intent_statistics = {intent: 0 for intent in USER_INTENTS.keys()}
# Initialize memory manager
memory_manager = MemoryManager(max_conversations=100, max_history=20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup operations
    verify_available_models(silent=True)
    global intent_statistics
    intent_statistics = {intent: 0 for intent in USER_INTENTS.keys()}

    # Create temp directory for file uploads
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Start a background task to clean up old temp files
    background_tasks = BackgroundTasks()

    # Define cleanup function
    async def cleanup_temp_files():
        while True:
            try:
                # Get current time
                now = time.time()

                # Remove files older than 24 hours
                if os.path.exists(TEMP_DIR):
                    for filename in os.listdir(TEMP_DIR):
                        file_path = os.path.join(TEMP_DIR, filename)
                        if os.path.isfile(file_path):
                            # If file is older than 24 hours, delete it
                            if now - os.path.getctime(file_path) > 24 * 3600:
                                try:
                                    os.remove(file_path)
                                    logger.info(f"Cleaned up old temp file: {filename}")
                                except Exception as e:
                                    logger.error(f"Error deleting temp file: {str(e)}")
            except Exception as e:
                logger.error(f"Error in temp file cleanup: {str(e)}")

            # Sleep for 1 hour
            await asyncio.sleep(3600)

    # Start the background task
    asyncio.create_task(cleanup_temp_files())

    yield

    # Shutdown operations
    model_cache.clear()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Multilingual Multi-Model LLM API",
    description="API for accessing multiple language models with configurable parameters and multilingual support",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Enhanced request model with model selection, configuration, and language
# Enhanced request model with model selection, configuration, language, and memory
class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for the model")
    model_name: str = Field(..., description="Name of the model to use")
    language: str = Field(default="en", description="Language code for response (e.g., en, fr, es)")
    config: Optional[ModelConfig] = Field(None, description="Optional model configuration")
    conversation_id: Optional[str] = Field(None, description="ID of the conversation to continue")
    use_memory: bool = Field(True, description="Whether to use conversation history as context")

    # Existing validators remain the same

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if v not in AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v, values):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language code. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}"
            )

        # Check if the selected model supports this language
        if "model_name" in values.data:
            model_name = values.data["model_name"]
            if v not in AVAILABLE_MODELS[model_name]["languages"]:
                raise ValueError(f"Model {model_name} does not support {SUPPORTED_LANGUAGES[v]}")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Tell me a joke",
                    "model_name": "llava-7b",
                    "language": "en",
                    "config": {"temperature": 0.8, "top_p": 0.95, "context_length": 2048},
                },
                {
                    "prompt": "Raconte-moi une blague",
                    "model_name": "llava-7b",
                    "language": "fr",
                    "config": {"temperature": 0.7},
                },
            ]
        }
    }


# Enhanced response model with memory
class LLMResponse(BaseModel):
    response: str = Field(..., description="Generated response text")
    status: str = Field(..., description="Status of the generation request")
    model_used: str = Field(..., description="Name of the model used for generation")
    language: str = Field(..., description="Language code of the response")
    input_length: Optional[int] = Field(None, description="Length of the input prompt")
    output_length: Optional[int] = Field(None, description="Length of the generated output")
    intent: str = Field("conversation", description="Detected user intent category")
    conversation_id: Optional[str] = Field(None, description="ID of the conversation")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "This is a sample response.",
                    "status": "success",
                    "model_used": "llava-7b",
                    "language": "en",
                    "input_length": 120,
                    "output_length": 85,
                    "intent": "question",
                    "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                }
            ]
        }
    }


# ...existing code...


def load_model(model_name: str):
    """Check if model exists in LM Studio via HTTP API"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available")

    # No need to cache models when using HTTP API
    try:
        # Verify model is available
        LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        logger.info(f"Connecting to LM Studio at {LM_STUDIO_URL}")
        response = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)

        if response.status_code != 200:
            raise Exception(f"Failed to get models from LM Studio API: {response.status_code}")

        # Parse the response more safely
        try:
            response_data = response.json()

            # LM Studio should return OpenAI-compatible format with 'data' array
            if "data" not in response_data:
                error_msg = f"Unexpected API response format: {response_data}"
                logger.error(error_msg)
                raise Exception(error_msg)

            available_model_ids = [model["id"] for model in response_data["data"]]
            logger.info(f"Available models in LM Studio: {available_model_ids}")

            model_path = AVAILABLE_MODELS[model_name]["path"]
            if model_path not in available_model_ids:
                AVAILABLE_MODELS[model_name]["active"] = False
                raise Exception(f"Model {model_path} not found in LM Studio")

            AVAILABLE_MODELS[model_name]["active"] = True
            logger.info(f"Model loaded: {model_name} → {model_path}")
            return model_path

        except json.JSONDecodeError:
            error_msg = f"Invalid JSON response from API: {response.text[:100]}..."
            logger.error(error_msg)
            raise Exception(error_msg)

    except Exception as e:
        AVAILABLE_MODELS[model_name]["active"] = False
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Failed to load model {model_name}: {str(e)}")


classifier = IntentClassifier()


@app.post("/generate", response_model=LLMResponse)
async def generate_response(
    request: PromptRequest,
    show_progress: bool = Query(True, description="Show generation progress"),
):
    # Existing code...
    """
    Generate a response using the selected LLM model with configuration, intent classification, and memory.

    Parameters:
        - request: PromptRequest containing the prompt, model, language, and memory options

    Returns:
        - LLMResponse with generated text, status, model used, language, and conversation context
    """
    try:
        if not checker.check_input(request.prompt):
            raise HTTPException(status_code=400, detail="Unsafe input detected.")

        # Get configuration (use default if not provided)
        config = request.config or DEFAULT_CONFIG

        # Get model path and verify availability
        model_path = load_model(request.model_name)

        # Create wrapper with model and configuration
        wrapper = LLMWrapper(model_path, config)

        # Classify the user's intent
        intent = classifier.classify_intent(request.prompt, wrapper)

        # Update intent statistics
        global intent_statistics
        intent_statistics[intent] = intent_statistics.get(intent, 0) + 1

        # Use the proper logger
        logger.info(f"Detected intent: {intent} for prompt: '{request.prompt[:50]}...'")

        # Handle conversation memory
        conversation_id = request.conversation_id

        # If no conversation ID provided or ID doesn't exist, create a new conversation
        if not conversation_id or not memory_manager.get_conversation(conversation_id):
            conversation_id = memory_manager.create_conversation(language=request.language)
            logger.info(f"Created new conversation: {conversation_id}")

        # Store the user message in memory with intent
        memory_manager.add_message(conversation_id, "user", request.prompt, intent)

        # Get conversation history to use as context if requested
        context = ""
        if request.use_memory and len(memory_manager.get_conversation_history(conversation_id)) > 1:
            context = memory_manager.get_conversation_context(
                conversation_id,
                max_length=config.context_length // 2,  # Use half of context for history
            )
            logger.info(f"Using conversation context with {len(context)} characters")

        # Enhance the prompt based on the detected intent
        enhanced_prompt = classifier.get_intent_prompt(intent, request.prompt)

        # Add conversation history if available
        if context:
            enhanced_prompt = (
                f"Previous conversation:\n{context}\n\nCurrent request: {enhanced_prompt}"
            )

        # Generate response with context length limits and language
        # result, input_length, output_length = wrapper.respond(enhanced_prompt, request.language)
        if show_progress:
            logger.info("Generating response with progress bar")
            result, input_length, output_length = wrapper.respond_with_progress(
                enhanced_prompt, request.language
            )
        else:
            # Generate response with regular method
            result, input_length, output_length = wrapper.respond(enhanced_prompt, request.language)

        if not checker.check_output(result):
            raise HTTPException(status_code=400, detail="Generated output flagged as unsafe.")

        # Store the assistant's response in memory
        memory_manager.add_message(conversation_id, "assistant", result, "response")

        return LLMResponse(
            response=result,
            status="success",
            model_used=request.model_name,
            language=request.language,
            input_length=input_length,
            output_length=output_length,
            intent=intent,
            conversation_id=conversation_id,
        )
    except Exception as e:
        # Log errors too
        logger.error(f"Error in generate_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


import json

from fastapi.responses import StreamingResponse


@app.post("/llm/stream")
async def stream_llm_response(request: PromptRequest, background_tasks: BackgroundTasks):
    """
    Stream a response from the LLM model chunk by chunk as it's generated.

    Parameters:
        - request: PromptRequest containing the prompt, model, language, and memory options

    Returns:
        - StreamingResponse with generated text chunks
    """
    try:
        # Input validation
        if not checker.check_input(request.prompt):
            raise HTTPException(status_code=400, detail="Unsafe input detected.")

        # Get configuration
        config = request.config or DEFAULT_CONFIG

        # Get model path and verify availability
        model_path = load_model(request.model_name)

        # Classify the user's intent
        wrapper = LLMWrapper(model_path, config)
        intent = classifier.classify_intent(request.prompt, wrapper)

        # Update intent statistics
        global intent_statistics
        intent_statistics[intent] = intent_statistics.get(intent, 0) + 1

        # Handle conversation memory
        conversation_id = request.conversation_id
        if not conversation_id or not memory_manager.get_conversation(conversation_id):
            conversation_id = memory_manager.create_conversation(language=request.language)

        # Store the user message in memory with intent
        memory_manager.add_message(conversation_id, "user", request.prompt, intent)

        # Get conversation history to use as context
        context = ""
        if request.use_memory and len(memory_manager.get_conversation_history(conversation_id)) > 1:
            context = memory_manager.get_conversation_context(
                conversation_id, max_length=config.context_length // 2
            )

        # Enhance the prompt based on the detected intent
        enhanced_prompt = classifier.get_intent_prompt(intent, request.prompt)

        # Add conversation history if available
        if context:
            enhanced_prompt = (
                f"Previous conversation:\n{context}\n\nCurrent request: {enhanced_prompt}"
            )

        # Create streaming generator function
        async def generate_stream():
            nonlocal conversation_id

            # Add language instruction if specified
            if request.language != "en":
                language_instruction = LANGUAGE_PROMPTS.get(request.language, "")
                full_prompt = f"{enhanced_prompt}\n\n{language_instruction}"
            else:
                full_prompt = enhanced_prompt

            # Truncate input prompt to context length
            truncated_prompt = full_prompt[: config.context_length]

            # Set up streaming payload
            payload = {
                "model": model_path,
                "messages": [{"role": "user", "content": truncated_prompt}],
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": min(4096, config.context_length),
                "stream": True,
            }

            # Test connection before starting stream
            try:
                session = requests.Session()
                test_response = session.get(
                    f"{os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1')}/models",
                    timeout=5,
                )
                if test_response.status_code != 200:
                    yield f"Error: LM Studio API not available ({test_response.status_code})\n"
                    return
            except Exception as conn_error:
                yield f"Error: Cannot connect to LM Studio API: {str(conn_error)}\n"
                return

            # Store complete response for memory
            full_response = ""

            # Make the streaming request
            with session.post(
                f"{os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1')}/chat/completions",
                json=payload,
                timeout=120,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status code {response.status_code}\n"
                    return

                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode("utf-8")

                        # Skip the "data: " prefix and handle special messages
                        if line_text.startswith("data: "):
                            line_text = line_text[6:]

                            # Check for the "[DONE]" message
                            if line_text == "[DONE]":
                                break

                            try:
                                # Parse the JSON data
                                data = json.loads(line_text)

                                # Extract the content delta
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        # Add to full response for memory
                                        full_response += content
                                        # Yield the content chunk
                                        yield content
                            except json.JSONDecodeError:
                                # Skip lines that aren't valid JSON
                                continue

            # Store the complete response in memory
            if full_response:
                background_tasks.add_task(
                    memory_manager.add_message,
                    conversation_id,
                    "assistant",
                    full_response,
                    "response",
                )

        # Return streaming response
        return StreamingResponse(generate_stream(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in stream_llm_response: {str(e)}")
        return StreamingResponse(
            iter([f"Error: {str(e)}"]), media_type="text/plain", status_code=500
        )


@app.get("/translate")
async def translate(
    text: str = Query(..., description="Text to translate"),
    source_lang: str = Query(..., description="Source language code"),
    target_lang: str = Query(..., description="Target language code"),
    model_name: str = Query("llava-7b", description="Model to use for translation"),
    temperature: float = Query(0.7, description="Temperature for generation"),
    conversation_id: Optional[str] = Query(
        None, description="Conversation ID to store translation history"
    ),
    use_memory: bool = Query(True, description="Whether to use translation history as context"),
    show_progress: bool = Query(True, description="Show generation progress"),
):
    """Translate text with memory to keep track of previous translations"""
    try:
        # Validate language codes first
        if source_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported source language: {source_lang}"
            )

        if target_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported target language: {target_lang}"
            )

        # Validate model
        if model_name not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model_name}")

        # Input safety check
        if not checker.check_input(text):
            raise HTTPException(status_code=400, detail="Unsafe input detected.")

        # Create custom config for translation
        config = ModelConfig(temperature=temperature)

        # Get model path and verify availability
        model_path = load_model(model_name)

        # Create wrapper with model and configuration
        wrapper = LLMWrapper(model_path, config)

        # Classify the user's intent (should be "translation" in most cases)
        intent = classifier.classify_intent(
            f"Translate from {source_lang} to {target_lang}: {text}", wrapper
        )

        # Update intent statistics
        global intent_statistics
        intent_statistics[intent] = intent_statistics.get(intent, 0) + 1

        # Handle conversation memory
        if not conversation_id or not memory_manager.get_conversation(conversation_id):
            conversation_id = memory_manager.create_conversation(language=target_lang)

        # Store the translation request in memory
        memory_manager.add_message(
            conversation_id,
            "user",
            f"Translate from {source_lang} to {target_lang}: {text}",
            intent,
        )

        source_name = SUPPORTED_LANGUAGES[source_lang]
        target_name = SUPPORTED_LANGUAGES[target_lang]

        # Get previous translations if use_memory is True
        context = ""
        if use_memory and len(memory_manager.get_conversation_history(conversation_id)) > 1:
            context = memory_manager.get_conversation_context(conversation_id, max_length=2000)

        # Enhance prompt with intent-specific instructions and context
        base_prompt = f"Translate the following {source_name} text to {target_name}:\n\n{text}"
        enhanced_prompt = classifier.get_intent_prompt(intent, base_prompt)

        # Add context if available
        if context:
            enhanced_prompt = (
                f"Previous translations:\n{context}\n\nNew translation request: {enhanced_prompt}"
            )

        # Generate translation with appropriate progress indication
        try:
            if show_progress:
                logger.info(f"Translating with progress bar from {source_lang} to {target_lang}")
                result, input_length, output_length = wrapper.respond_with_progress(
                    enhanced_prompt, target_lang
                )
            else:
                logger.info(f"Translating without progress bar from {source_lang} to {target_lang}")
                result, input_length, output_length = wrapper.respond(enhanced_prompt, target_lang)
        except Exception as translation_error:
            logger.error(f"Translation process failed: {str(translation_error)}")
            raise HTTPException(
                status_code=500, detail=f"Translation failed: {str(translation_error)}"
            )

        # Apply safety check to output
        if not checker.check_output(result):
            raise HTTPException(status_code=400, detail="Generated translation flagged as unsafe.")

        # Store the translation result in memory
        memory_manager.add_message(conversation_id, "assistant", result, "translation_result")

        # Return enhanced response with intent and conversation information
        return {
            "original_text": text,
            "translated_text": result,
            "source_language": {"code": source_lang, "name": source_name},
            "target_language": {"code": target_lang, "name": target_name},
            "model_used": model_name,
            "intent": intent,
            "input_length": input_length,
            "output_length": output_length,
            "conversation_id": conversation_id,
        }
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions as-is
        raise http_ex
    except Exception as e:
        # Log the error with full traceback
        import traceback

        logger.error(f"Translation error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


# Add intent tracking data structure
intent_statistics = {intent: 0 for intent in USER_INTENTS.keys()}


@app.post("/generate-multimodal", response_model=LLMResponse)
async def generate_multimodal_response(
    text: str = Form(..., description="Text prompt for the model"),
    model_name: str = Form(..., description="Name of the model to use"),
    language: str = Form("en", description="Language code for response"),
    conversation_id: Optional[str] = Form(None, description="ID of the conversation to continue"),
    use_memory: bool = Form(True, description="Whether to use conversation history"),
    files: List[UploadFile] = File(None, description="Files to process alongside text"),
    temperature: float = Form(0.7, description="Temperature for generation"),
    show_progress: bool = Form(True, description="Show generation progress"),
):
    """Generate response based on text and uploaded files (images, documents, tables, etc.)"""
    try:
        # Validate the model and language
        if model_name not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Available models: {list(AVAILABLE_MODELS.keys())}",
            )

        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported language")

        # Check text input for safety
        if not checker.check_input(text):
            raise HTTPException(status_code=400, detail="Unsafe input detected.")

        # Create configuration
        config = ModelConfig(temperature=temperature)

        # Process files
        file_descriptions = []
        file_info = []
        image_data = []
        document_texts = []
        audio_transcriptions = []

        if files:
            for file in files:
                if not file.filename:  # Skip empty file entries
                    continue

                try:
                    # Process the file based on its type
                    processed = process_file(file)
                    file_info.append(processed)

                    # Handle specific file types
                    if processed["type"] == "image":
                        logger.info(f"Processing image file: {processed['filename']}")
                        image_data.append(processed)
                        file_descriptions.append(
                            f"[Image: {processed['filename']} - {processed['dimensions']}]"
                        )

                    elif processed["type"] == "document":
                        logger.info(f"Processing document file: {processed['filename']}")
                        # Add document content to prompt
                        if processed.get("content"):
                            doc_text = (
                                f"[Document: {processed['filename']}]\n{processed['content']}"
                            )
                            document_texts.append(doc_text)
                            file_descriptions.append(
                                f"[Document: {processed['filename']} - {len(processed['content'])} characters]"
                            )

                    elif processed["type"] == "table":
                        logger.info(f"Processing table file: {processed['filename']}")
                        if processed.get("table_text"):
                            file_descriptions.append(
                                f"[Table: {processed['filename']} - {processed['rows']}x{processed['columns']}]"
                            )
                            file_descriptions.append(processed["table_text"])

                    elif processed["type"] == "text":
                        logger.info(f"Processing text file: {processed['filename']}")
                        if processed.get("content"):
                            file_descriptions.append(
                                f"[Text file: {processed['filename']} - {processed['lines']} lines]"
                            )
                            file_descriptions.append(processed["content"])

                    elif processed["type"] == "audio":
                        logger.info(f"Processing audio file: {processed['filename']}")
                        if processed.get("transcription"):
                            transcription = processed["transcription"]
                            audio_transcriptions.append(transcription)
                            file_descriptions.append(
                                f"[Audio: {processed['filename']} - Duration: {processed.get('duration', 'unknown')} seconds]"
                            )
                            file_descriptions.append(f"Transcription: {transcription}")

                    elif processed["type"] == "error":
                        logger.error(
                            f"Error processing file: {processed['filename']} - {processed['error']}"
                        )
                        file_descriptions.append(
                            f"[Error processing file {processed['filename']}: {processed['error']}]"
                        )

                    else:
                        file_descriptions.append(
                            f"[Unknown file: {processed['filename']} - {processed['size'] // 1024} KB]"
                        )

                except Exception as file_error:
                    logger.error(f"Error processing file {file.filename}: {str(file_error)}")
                    file_descriptions.append(
                        f"[Error processing file {file.filename}: {str(file_error)}]"
                    )

        # Get model path and create wrapper
        model_path = load_model(model_name)
        wrapper = LLMWrapper(model_path, config)

        # Classify the user's intent
        intent = classifier.classify_intent(text, wrapper)

        # Update intent statistics
        intent_statistics[intent] = intent_statistics.get(intent, 0) + 1
        logger.info(f"Multimodal request with intent: {intent}, files: {len(file_info)}")

        # Handle conversation memory
        if not conversation_id or not memory_manager.get_conversation(conversation_id):
            conversation_id = memory_manager.create_conversation(language=language)

        # Combine text with document content
        full_prompt = text

        # Add document texts first (they're most important for context)
        if document_texts:
            document_content = "\n\n".join(document_texts)
            full_prompt = f"{text}\n\nContent from uploaded documents:\n{document_content}"

        # Add other file descriptions
        if file_descriptions and (
            not document_texts or len(file_descriptions) > len(document_texts)
        ):
            other_descriptions = [
                desc for desc in file_descriptions if not desc.startswith("[Document:")
            ]
            if other_descriptions:
                file_context = "\n\n".join(other_descriptions)
                full_prompt = f"{full_prompt}\n\nOther uploaded files:\n{file_context}"

        if audio_transcriptions:
            transcription_text = "\n\n".join(audio_transcriptions)
            full_prompt = f"{full_prompt}\n\nTranscription from audio files:\n{transcription_text}"

        # For images, we'll handle them separately if using a vision model
        has_images = len(image_data) > 0

        # Store the user message in memory
        memory_manager.add_message(conversation_id, "user", full_prompt, intent)

        # Get conversation history if requested
        context = ""
        if use_memory and len(memory_manager.get_conversation_history(conversation_id)) > 1:
            context = memory_manager.get_conversation_context(
                conversation_id, max_length=config.context_length // 2
            )

        # Enhance the prompt based on intent
        enhanced_prompt = classifier.get_intent_prompt(intent, full_prompt)

        # Add conversation history if available
        if context:
            enhanced_prompt = (
                f"Previous conversation:\n{context}\n\nCurrent request: {enhanced_prompt}"
            )

        # Log prompt size for debugging
        logger.info(f"Enhanced prompt length: {len(enhanced_prompt)} characters")

        # Generate response based on input type
        if has_images and "llava" in model_path.lower():
            # If model supports vision and we have images
            try:
                # Dynamically add the respond_with_vision method if it doesn't exist
                if not hasattr(wrapper, "respond_with_vision"):
                    import types

                    wrapper.respond_with_vision = types.MethodType(respond_with_vision, wrapper)
                    logger.info("Added respond_with_vision method to wrapper")

                # Use vision capabilities
                logger.info(f"Using vision model to process {len(image_data)} images")
                result, input_length, output_length = wrapper.respond_with_vision(
                    enhanced_prompt, image_data, language
                )
            except Exception as vision_error:
                logger.error(f"Vision processing error: {str(vision_error)}")
                # Fall back to text-only with progress bar if requested
                logger.warning("Falling back to text-only response due to vision error")
                if show_progress:
                    result, input_length, output_length = wrapper.respond_with_progress(
                        enhanced_prompt + " [Note: Unable to process attached images]", language
                    )
                else:
                    result, input_length, output_length = wrapper.respond(
                        enhanced_prompt + " [Note: Unable to process attached images]", language
                    )
        else:
            # Text-only response
            # result, input_length, output_length = wrapper.respond(enhanced_prompt, language)
            if show_progress:
                result, input_length, output_length = wrapper.respond_with_progress(
                    enhanced_prompt, language
                )
            else:
                result, input_length, output_length = wrapper.respond(enhanced_prompt, language)

        # Check output safety
        if not checker.check_output(result):
            raise HTTPException(status_code=400, detail="Generated output flagged as unsafe.")

        # Store the assistant's response in memory
        memory_manager.add_message(conversation_id, "assistant", result, "response")

        return LLMResponse(
            response=result,
            status="success",
            model_used=model_name,
            language=language,
            input_length=input_length,
            output_length=output_length,
            intent=intent,
            conversation_id=conversation_id,
        )
    except Exception as e:
        logger.error(f"Error in generate_multimodal_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test-vision")
async def test_vision(
    file: UploadFile = File(..., description="Image file to analyze"),
    text: str = Form("Describe this image in detail", description="Prompt for image analysis"),
    model_name: str = Form("llava-7b", description="Vision model to use"),
    show_progress: bool = Form(True, description="Show generation progress"),
):
    """Test vision capabilities with a single image"""
    try:
        # Check file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Please upload an image."
            )

        # Process image
        file_content = await file.read()
        processed = process_file(UploadFile(file=io.BytesIO(file_content), filename=file.filename))

        if processed.get("type") != "image":
            raise HTTPException(status_code=400, detail="Failed to process image.")

        # Get model path and create wrapper
        model_path = load_model(model_name)
        wrapper = LLMWrapper(model_path, ModelConfig(temperature=0.7))

        # Manually add vision method
        if not hasattr(wrapper, "respond_with_vision"):
            import types

            wrapper.respond_with_vision = types.MethodType(respond_with_vision, wrapper)

        # Call vision API with progress if requested
        if show_progress and hasattr(wrapper, "respond_with_vision_progress"):
            # If we have a streaming vision method
            result, input_length, output_length = wrapper.respond_with_vision_progress(
                text, [processed], "en"
            )
        else:
            # Use standard vision method
            result, input_length, output_length = wrapper.respond_with_vision(
                text, [processed], "en"
            )

        return {
            "description": result,
            "model_used": model_name,
            "image_details": {
                "filename": processed["filename"],
                "dimensions": processed["dimensions"],
                "format": processed["format"],
            },
        }
    except Exception as e:
        logger.error(f"Error in test-vision: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/check-vision-models")
async def check_vision_models():
    """Check if vision models are available in LM Studio"""
    try:
        LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        # Get models from LM Studio API
        response = requests.get(f"{LM_STUDIO_URL}/models", timeout=10)

        if response.status_code != 200:
            return {
                "status": "error",
                "message": f"Failed to connect to LM Studio: {response.status_code}",
            }

        available_models_data = response.json()
        available_model_ids = [model["id"] for model in available_models_data["data"]]

        # Check specifically for vision models
        vision_models = []
        for model_name, info in AVAILABLE_MODELS.items():
            if "llava" in model_name.lower() or "vision" in model_name.lower():
                vision_models.append(
                    {
                        "name": model_name,
                        "path": info["path"],
                        "available": info["path"] in available_model_ids,
                    }
                )

        if not any(model["available"] for model in vision_models):
            return {
                "status": "warning",
                "message": "No vision models are currently available in LM Studio",
                "models": vision_models,
            }

        return {
            "status": "success",
            "message": "Vision models are available",
            "models": vision_models,
        }
    except Exception as e:
        return {"status": "error", "message": f"Error checking vision models: {str(e)}"}


# Add this after your other endpoints
@app.get("/intent-stats")
async def get_intent_statistics():
    """Get statistics on detected intents"""
    # Ensure global reference
    global intent_statistics
    total = sum(intent_statistics.values()) or 1  # Avoid division by zero

    intent_data = []
    for intent, count in intent_statistics.items():
        intent_data.append(
            {
                "intent": intent,
                "count": count,
                "percentage": round((count / total) * 100, 2),
                "description": USER_INTENTS[intent]["description"],
            }
        )

    # Sort by count, descending
    intent_data.sort(key=lambda x: x["count"], reverse=True)

    return {"total_requests": total, "intents": intent_data}


@app.post("/debug-vision-request")
async def debug_vision_request(
    file: UploadFile = File(..., description="Image file to analyze"),
    text: str = Form("Describe this image in detail", description="Prompt for image analysis"),
):
    """Debug endpoint to check what's being sent to the vision model"""
    try:
        # Process image
        file_content = await file.read()
        processed = process_file(UploadFile(file=io.BytesIO(file_content), filename=file.filename))

        if processed.get("type") != "image":
            raise HTTPException(status_code=400, detail="Failed to process image.")

        # Create formatted message
        formatted_image = {"data": processed["base64"], "type": f"image/{processed['format']}"}

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{formatted_image['type']};base64,{formatted_image['data'][:30]}..."  # Truncated for display
                        },
                    },
                ],
            }
        ]

        return {
            "request_structure": {
                "endpoint": f"{os.environ.get('LM_STUDIO_URL', 'http://localhost:1234/v1')}/chat/completions",
                "model": "llava-7b",
                "messages_structure": messages,
                "image_info": {
                    "format": processed["format"],
                    "dimensions": processed["dimensions"],
                    "base64_length": len(processed["base64"]),
                    "base64_sample": processed["base64"][:30] + "...",  # Truncated for display
                },
            },
            "validation": {
                "is_base64_valid": all(
                    c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
                    for c in processed["base64"]
                ),
                "format_valid": processed["format"] in ["jpg", "jpeg", "png", "gif", "bmp"],
            },
        }
    except Exception as e:
        logger.error(f"Error in debug-vision-request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List all available models with details"""
    try:
        models_info = []
        for name, info in AVAILABLE_MODELS.items():
            supported_langs = []

            # Safely process languages
            for code in info.get("languages", []):
                try:
                    # Check if language code exists in SUPPORTED_LANGUAGES
                    if code in SUPPORTED_LANGUAGES:
                        supported_langs.append({"code": code, "name": SUPPORTED_LANGUAGES[code]})
                    else:
                        # Handle unknown language codes gracefully
                        supported_langs.append({"code": code, "name": f"Unknown ({code})"})
                except Exception as lang_error:
                    # Log the error but continue processing
                    logger.error(
                        f"Error processing language {code} for model {name}: {str(lang_error)}"
                    )

            # Create model info with safe defaults
            model_info = {
                "name": name,
                "description": info.get("description", "No description"),
                "active": info.get("active", False),
                "supported_languages": supported_langs,
            }
            models_info.append(model_info)

        return {"models": models_info}

    except Exception as e:
        # Log the error and return a meaningful error response
        logger.error(f"Error in list_models: {str(e)}")
        return {"error": "Failed to retrieve model information", "message": str(e)}


@app.get("/languages")
async def list_languages():
    """List all supported languages"""
    languages = []
    for code, name in SUPPORTED_LANGUAGES.items():
        # Find which models support this language
        supported_by = [
            model
            for model, info in AVAILABLE_MODELS.items()
            if code in info["languages"] and info["active"]
        ]

        languages.append({"code": code, "name": name, "supported_by": supported_by})
    return {"languages": languages}


@app.post("/text-to-speech")
async def generate_speech(
    text: str = Form(..., description="Text to convert to speech"),
    output_format: str = Form("mp3", description="Audio format (mp3 supported)"),
    language: str = Form("en", description="Language code"),
    voice: str = Form("default", description="Voice to use (if available)"),
):
    """Convert text to speech and return audio file"""
    try:
        # Check if TTS is available
        if not TTS_AVAILABLE:
            return {
                "status": "error",
                "message": "Text-to-speech not available. Check server logs for details.",
                "setup_help": "Make sure gTTS is installed with 'pip install gtts'",
            }

        # Generate speech
        result = text_to_speech(text, "mp3", language, voice)

        # Create response
        content_disposition = f"attachment; filename=speech.{result['format']}"

        # Return audio file
        return Response(
            content=result["audio_data"],
            media_type=f"audio/{result['format']}",
            headers={"Content-Disposition": content_disposition},
        )
    except Exception as e:
        logger.error(f"Error in generate_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")


@app.post("/transcribe-audio")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model_size: str = Form(
        "base", description="Whisper model size (tiny, base, small, medium, large)"
    ),
    language: Optional[str] = Form(
        None, description="Language code (optional, auto-detect if None)"
    ),
    analyze: bool = Form(False, description="Analyze audio content after transcription"),
):
    """Transcribe audio file to text using OpenAI's Whisper model"""
    try:
        # Check if Whisper is available
        if WHISPER_MODEL is None:
            return {
                "status": "error",
                "message": "Whisper model not available. Check server logs for details.",
                "setup_help": "Make sure OpenAI Whisper is installed with 'pip install openai-whisper'",
            }

        # Check if file is an audio file
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Please upload an audio file."
            )

        # Read file content
        file_content = await file.read()

        # Process the audio file
        processed = process_file(UploadFile(file=io.BytesIO(file_content), filename=file.filename))

        if processed.get("type") == "error":
            raise HTTPException(status_code=500, detail=processed.get("error"))

        transcription = processed.get("transcription", "")

        # Check if transcription failed
        if not transcription or transcription.startswith("[Error:"):
            return {
                "status": "error",
                "message": (
                    transcription if transcription else "Transcription failed with unknown error"
                ),
                "filename": file.filename,
            }

        # Prepare response
        response = {
            "status": "success",
            "filename": file.filename,
            "transcription": transcription,
            "audio_details": {
                "duration_seconds": processed.get("duration", 0),
                "sample_rate": processed.get("sample_rate", 0),
            },
        }

        # Analyze content if requested
        if analyze and transcription and len(transcription) > 0:
            try:
                # Create configuration
                config = ModelConfig(temperature=0.7)

                # Get model and create wrapper
                model_name = "gemma-3b"  # Use a default model for analysis
                model_path = load_model(model_name)
                wrapper = LLMWrapper(model_path, config)

                # Create analysis prompt
                analysis_prompt = f"""Analyze the following transcribed audio content:

"{transcription}"

Please provide:
1. A summary of the main topics discussed
2. Key points or information mentioned
3. Any questions or requests that need addressing
4. The overall tone or sentiment of the audio"""

                # Generate analysis
                analysis, _, _ = wrapper.respond(analysis_prompt, "en")

                # Add analysis to response
                response["analysis"] = analysis

            except Exception as analysis_error:
                logger.error(f"Error analyzing transcription: {str(analysis_error)}")
                response["analysis_error"] = str(analysis_error)

        return response

    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio-to-response")
async def audio_to_response(
    file: UploadFile = File(..., description="Audio file containing the user's query"),
    model_name: str = Form("llava-7b", description="Model to use for response generation"),
    language: str = Form("en", description="Language code for response"),
    generate_audio_response: bool = Form(
        False, description="Whether to generate audio for the response"
    ),
    temperature: float = Form(0.7, description="Temperature for generation"),
    conversation_id: Optional[str] = Form(None, description="ID of conversation to continue"),
    show_progress: bool = Form(True, description="Show generation progress"),
):
    """Process audio file, transcribe it, and generate a response to the transcribed content"""
    try:
        # Check if Whisper is available for transcription
        if WHISPER_MODEL is None:
            return {
                "status": "error",
                "message": "Whisper model not available for transcription. Check server logs for details.",
                "setup_help": "Make sure OpenAI Whisper is installed with 'pip install openai-whisper'",
            }

        # Check if the file is an audio file
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Please upload an audio file."
            )

        # Validate model and language
        if model_name not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")

        if language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        # Read file content
        file_content = await file.read()

        # Step 1: Transcribe the audio
        logger.info(f"Transcribing audio file: {file.filename}")
        processed = process_file(UploadFile(file=io.BytesIO(file_content), filename=file.filename))

        if processed.get("type") == "error":
            raise HTTPException(status_code=500, detail=processed.get("error"))

        transcription = processed.get("transcription", "")

        # Check if transcription failed
        if not transcription or transcription.startswith("[Error:"):
            return {
                "status": "error",
                "message": (
                    transcription if transcription else "Transcription failed with unknown error"
                ),
                "filename": file.filename,
            }

        logger.info(f"Transcription successful: {len(transcription)} characters")

        # Step 2: Generate response to the transcription
        logger.info(f"Generating response to transcribed content using model: {model_name}")

        # Create configuration and load model
        config = ModelConfig(temperature=temperature)
        model_path = load_model(model_name)
        wrapper = LLMWrapper(model_path, config)

        # Classify intent
        intent = classifier.classify_intent(transcription, wrapper)
        global intent_statistics
        intent_statistics[intent] = intent_statistics.get(intent, 0) + 1

        # Handle conversation memory
        if not conversation_id or not memory_manager.get_conversation(conversation_id):
            conversation_id = memory_manager.create_conversation(language=language)
            logger.info(f"Created new conversation: {conversation_id}")

        # Store the transcribed content as user message
        memory_manager.add_message(conversation_id, "user", transcription, intent)

        # Get conversation context
        context = ""
        if len(memory_manager.get_conversation_history(conversation_id)) > 1:
            context = memory_manager.get_conversation_context(
                conversation_id, max_length=config.context_length // 2
            )

        # Enhance the prompt
        enhanced_prompt = classifier.get_intent_prompt(intent, transcription)

        # Add conversation history if available
        if context:
            enhanced_prompt = (
                f"Previous conversation:\n{context}\n\nCurrent request: {enhanced_prompt}"
            )

        # Generate response with progress bar if requested
        if show_progress:
            logger.info("Generating response with progress bar")
            result, input_length, output_length = wrapper.respond_with_progress(
                enhanced_prompt, language
            )
        else:
            result, input_length, output_length = wrapper.respond(enhanced_prompt, language)

        # Store assistant response
        memory_manager.add_message(conversation_id, "assistant", result, "response")

        # Prepare response object
        response_data = {
            "status": "success",
            "transcribed_audio": transcription,
            "response": result,
            "model_used": model_name,
            "language": language,
            "intent": intent,
            "audio_details": {
                "duration_seconds": processed.get("duration", 0),
                "sample_rate": processed.get("sample_rate", 0),
                "filename": file.filename,
            },
            "conversation_id": conversation_id,
        }

        # Step 3: Generate audio response if requested
        if generate_audio_response and TTS_AVAILABLE:
            try:
                logger.info("Generating audio response")
                audio_result = text_to_speech(result, "mp3", language)

                # Convert audio to base64 for inclusion in response
                audio_b64 = base64.b64encode(audio_result["audio_data"]).decode("utf-8")

                # Add audio to response
                response_data["audio_response"] = {
                    "format": audio_result["format"],
                    "audio_base64": audio_b64,
                    "player_html": f'<audio controls src="data:audio/{audio_result["format"]};base64,{audio_b64}"></audio>',
                }
            except Exception as tts_error:
                logger.error(f"Error generating audio response: {str(tts_error)}")
                response_data["tts_error"] = str(tts_error)

        return response_data

    except Exception as e:
        logger.error(f"Error in audio_to_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Update the audio_response_only endpoint
@app.post("/audio-response-only")
async def audio_response_only(
    file: UploadFile = File(..., description="Audio file containing the user's query"),
    model_name: str = Form("llava-7b", description="Model to use for response generation"),
    language: str = Form("en", description="Language code for response"),
    temperature: float = Form(0.7, description="Temperature for generation"),
    show_progress: bool = Form(True, description="Show generation progress"),
):
    """Process audio and return an audio response directly (for voice assistants)"""
    try:
        # Check both required services
        if WHISPER_MODEL is None:
            raise HTTPException(status_code=503, detail="Speech-to-text service not available")

        if not TTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Text-to-speech service not available")

        # Read file content
        file_content = await file.read()

        # Transcribe the audio
        print(f"\nTranscribing audio file: {file.filename}...")
        transcribe_start = time.time()
        processed = process_file(UploadFile(file=io.BytesIO(file_content), filename=file.filename))
        transcribe_time = time.time() - transcribe_start

        if processed.get("type") == "error" or processed.get("type") != "audio":
            raise HTTPException(status_code=400, detail="Failed to process audio file")

        transcription = processed.get("transcription", "")
        print(f"Transcription complete ({transcribe_time:.2f}s, {len(transcription)} chars) ✓")

        # Generate text response
        config = ModelConfig(temperature=temperature)
        model_path = load_model(model_name)
        wrapper = LLMWrapper(model_path, config)

        # Generate response with or without progress bar
        if show_progress:
            result, input_length, output_length = wrapper.respond_with_progress(
                transcription, language
            )
        else:
            result, input_length, output_length = wrapper.respond(transcription, language)

        # Convert to audio with simple progress bar
        audio_result = text_to_speech(result, "mp3", language)

        # Return audio directly as a downloadable file
        return Response(
            content=audio_result["audio_data"],
            media_type=f"audio/{audio_result['format']}",
            headers={
                "Content-Disposition": f"attachment; filename=response.{audio_result['format']}"
            },
        )

    except Exception as e:
        logger.error(f"Error in audio_response_only: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts-setup-help")
async def tts_setup_help():
    """Get help with setting up text-to-speech"""
    return {
        "tts_status": "Available" if TTS_AVAILABLE is not None else "Not available",
        "installation_instructions": [
            "1. Install gTTS: pip install gtts",
            "2. Install additional Python packages: pip install soundfile",
            "3. Restart the API server after installation",
        ],
        "troubleshooting": [
            "- Make sure you have enough disk space and memory for model loading",
            "- If you encounter CUDA errors, try installing the CPU-only version",
            "- For detailed errors, check the API logs",
        ],
        "test_endpoint": "/test-tts-system",
    }


@app.get("/test-audio-system")
async def test_audio_system():
    """Test the audio processing system"""
    try:
        results = {
            "whisper_model": "Loaded" if WHISPER_MODEL is not None else "Not loaded",
            "whisper_version": (
                whisper.__version__ if hasattr(whisper, "__version__") else "Unknown"
            ),
            "temp_dir_exists": os.path.exists(TEMP_DIR),
            "temp_dir_writable": (
                os.access(TEMP_DIR, os.W_OK) if os.path.exists(TEMP_DIR) else False
            ),
        }

        # Check for ffmpeg using pydub's converter
        try:
            import subprocess

            ffmpeg_path = AudioSegment.converter
            ffmpeg_result = subprocess.run(
                [ffmpeg_path, "-version"], capture_output=True, text=True, timeout=5
            )
            if ffmpeg_result.returncode == 0:
                results["ffmpeg_status"] = "Available"
                results["ffmpeg_version"] = ffmpeg_result.stdout.split("\n")[0]
            else:
                results["ffmpeg_status"] = "Error"
                results["ffmpeg_error"] = ffmpeg_result.stderr
        except Exception as e:
            results["ffmpeg_status"] = "Not found or not working"
            results["ffmpeg_error"] = str(e)

        # Test pydub
        try:
            results["pydub_converter"] = AudioSegment.converter
            # Create a silent 1-second audio segment
            silent = AudioSegment.silent(duration=1000)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav.close()
                silent.export(temp_wav.name, format="wav")
                results["pydub_test"] = "Success"
                os.unlink(temp_wav.name)
        except Exception as e:
            results["pydub_test"] = "Failed"
            results["pydub_error"] = str(e)

        return results
    except Exception as e:
        return {"error": f"Error testing audio system: {str(e)}"}


@app.get("/test-tts-system")
async def test_tts_system():
    """Test the text-to-speech processing system"""
    try:
        results = {
            "tts_available": TTS_AVAILABLE,
            "tts_type": "Google Text-to-Speech (gTTS)",
            "supported_formats": ["mp3"],
            "temp_dir_exists": os.path.exists(TEMP_DIR),
            "temp_dir_writable": (
                os.access(TEMP_DIR, os.W_OK) if os.path.exists(TEMP_DIR) else False
            ),
        }

        # Test TTS with a short text if available
        if TTS_AVAILABLE:
            try:
                test_text = "This is a test of the text to speech system."
                result = text_to_speech(test_text, "mp3")

                results["tts_test"] = "Success"
                results["test_details"] = {
                    "audio_size_bytes": len(result["audio_data"]),
                    "format": result["format"],
                }
            except Exception as e:
                results["tts_test"] = "Failed"
                results["tts_error"] = str(e)

        return results
    except Exception as e:
        return {"error": f"Error testing TTS system: {str(e)}"}


@app.get("/test-tts-system")
async def test_tts_system():
    """Test the text-to-speech processing system"""
    try:
        results = {
            "tts_available": TTS_AVAILABLE,
            "tts_type": "Google Text-to-Speech (gTTS)",
            "supported_formats": ["mp3"],
            "temp_dir_exists": os.path.exists(TEMP_DIR),
            "temp_dir_writable": (
                os.access(TEMP_DIR, os.W_OK) if os.path.exists(TEMP_DIR) else False
            ),
        }

        # Test TTS with a short text if available
        if TTS_AVAILABLE:
            try:
                test_text = "This is a test of the text to speech system."
                result = text_to_speech(test_text, "mp3")

                results["tts_test"] = "Success"
                results["test_details"] = {
                    "audio_size_bytes": len(result["audio_data"]),
                    "format": result["format"],
                }
            except Exception as e:
                results["tts_test"] = "Failed"
                results["tts_error"] = str(e)

        return results
    except Exception as e:
        return {"error": f"Error testing TTS system: {str(e)}"}


@app.get("/audio-troubleshooting")
async def audio_troubleshooting():
    """Get troubleshooting information for audio processing"""
    return {
        "common_issues": [
            {
                "problem": "FFmpeg not found",
                "solution": "Install FFmpeg on your system and ensure it's in the PATH variable",
            },
            {
                "problem": "Audio transcription fails with conversion error",
                "solution": "Try using WAV files directly, or install pydub and ffmpeg-python",
            },
            {
                "problem": "Whisper model not available",
                "solution": "Check logs for error messages during model loading",
            },
        ],
        "supported_formats": [".wav", ".mp3", ".ogg", ".flac", ".m4a"],
        "recommended_format": ".wav or .mp3 (most reliable)",
        "system_test_endpoint": "/test-audio-system",
    }


@app.get("/audio-setup-help")
async def audio_setup_help():
    """Get help with setting up audio processing"""
    return {
        "whisper_status": "Available" if WHISPER_MODEL is not None else "Not available",
        "installation_instructions": [
            "1. Install OpenAI Whisper: pip install openai-whisper",
            "2. Install FFmpeg (system dependency):",
            "   - Windows: Download from https://ffmpeg.org/download.html and add to PATH",
            "   - Linux (Ubuntu/Debian): sudo apt update && sudo apt install ffmpeg",
            "   - macOS: brew install ffmpeg",
            "3. Install additional Python packages: pip install pydub",
            "4. Restart the API server after installation",
        ],
        "troubleshooting": [
            "- Check that FFmpeg is properly installed and in your system PATH",
            "- Make sure you have enough disk space and memory for model loading",
            "- For detailed errors, check the API logs",
        ],
        "test_endpoint": "/test-audio-system",
    }


@app.get("/conversations")
async def list_conversations():
    """List all active conversations"""
    return {"conversations": memory_manager.list_conversations()}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get the history of a specific conversation"""
    conversation = memory_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "created_at": conversation.created_at.isoformat(),
        "last_updated": conversation.last_updated.isoformat(),
        "language": conversation.language,
        "messages": conversation.get_history(as_dict=True),
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation from memory"""
    if conversation_id not in memory_manager.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")

    del memory_manager.conversations[conversation_id]
    return {"status": "success", "message": f"Conversation {conversation_id} deleted"}


@app.post("/conversations")
async def create_conversation(
    language: str = Query("en", description="Primary language for the conversation")
):
    """Create a new empty conversation"""
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

    conversation_id = memory_manager.create_conversation(language=language)
    return {"conversation_id": conversation_id, "language": language}


@app.get("/test-intent")
async def test_intent(intent: str = Query(..., description="Intent to increment")):
    """Test endpoint to manually increment intent counts"""
    global intent_statistics

    if intent not in USER_INTENTS:
        return {"error": f"Invalid intent. Available intents: {list(USER_INTENTS.keys())}"}

    # Increment the intent count
    intent_statistics[intent] = intent_statistics.get(intent, 0) + 1

    return {
        "message": f"Incremented count for intent: {intent}",
        "current_count": intent_statistics[intent],
        "all_intents": intent_statistics,
    }


@app.get("/config/default")
async def get_default_config():
    """Get the default model configuration"""
    return DEFAULT_CONFIG


@app.get("/test_connection")
async def test_connection():
    """Test the connection to LM Studio"""
    try:
        LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        response = requests.get(f"{LM_STUDIO_URL}/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            return {
                "status": "success",
                "message": "Successfully connected to LM Studio",
                "available_models": [model["id"] for model in models_data["data"]],
                "url": LM_STUDIO_URL,
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to get models from LM Studio API: {response.status_code}",
                "url": LM_STUDIO_URL,
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to connect to LM Studio: {str(e)}",
            "url": LM_STUDIO_URL,
            "host": os.environ.get("LM_STUDIO_HOST", "localhost"),
            "port": os.environ.get("LM_STUDIO_PORT", 1234),  # Default to 1234 if not set
        }


# Add this endpoint to your log_llm.py file


@app.get("/logs/status")
async def check_logs_status():
    """Check the status of log files"""
    try:
        logs_dir_path = os.path.abspath(logs_dir)
        log_file_path = os.path.abspath(log_filename)

        # Get list of all log files
        log_files = []
        if os.path.exists(logs_dir_path):
            log_files = [
                f
                for f in os.listdir(logs_dir_path)
                if f.startswith("llm_api_") and f.endswith(".log")
            ]

        # Check current log file
        current_log_exists = os.path.exists(log_file_path)
        current_log_size = os.path.getsize(log_file_path) if current_log_exists else 0

        return {
            "logs_directory": logs_dir_path,
            "directory_exists": os.path.exists(logs_dir_path),
            "all_log_files": log_files,
            "current_log_file": log_file_path,
            "current_log_exists": current_log_exists,
            "current_log_size_bytes": current_log_size,
            "last_10_lines": get_last_lines(log_file_path, 10) if current_log_exists else [],
        }
    except Exception as e:
        return {"error": f"Error checking logs: {str(e)}"}


def get_last_lines(file_path, n=10):
    """Get the last n lines from a file"""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) >= n else lines
    except Exception as e:
        return [f"Error reading file: {str(e)}"]


@app.get("/files")
async def list_temp_files():
    """List all temporary files stored on the server"""
    try:
        if not os.path.exists(TEMP_DIR):
            return {"error": "Temp directory does not exist"}

        files = []
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path):
                file_info = {
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "created": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                }
                files.append(file_info)

        return {"temp_files": files, "temp_directory": os.path.abspath(TEMP_DIR)}
    except Exception as e:
        logger.error(f"Error listing temp files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{filename}")
async def delete_temp_file(filename: str):
    """Delete a temporary file"""
    try:
        file_path = os.path.join(TEMP_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        os.remove(file_path)
        return {"status": "success", "message": f"File {filename} deleted"}
    except Exception as e:
        logger.error(f"Error deleting temp file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
async def list_logs(latest: bool = Query(False, description="Get only the latest log file")):
    """List available log files"""
    try:
        logs_dir_path = os.path.abspath(logs_dir)

        # Get all log files
        log_files = []
        if os.path.exists(logs_dir_path):
            all_files = [
                f
                for f in os.listdir(logs_dir_path)
                if f.startswith("llm_api_") and f.endswith(".log")
            ]

            # Sort by modification time, newest first
            all_files.sort(
                key=lambda x: os.path.getmtime(os.path.join(logs_dir_path, x)), reverse=True
            )

            for file in all_files:
                file_path = os.path.join(logs_dir_path, file)
                file_size = os.path.getsize(file_path)
                log_files.append(
                    {
                        "filename": file,
                        "path": file_path,
                        "size_bytes": file_size,
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat(),
                    }
                )

        if latest and log_files:
            # Return just the latest log file with content
            latest_log = log_files[0]
            with open(latest_log["path"], "r", encoding="utf-8") as f:
                latest_log["content"] = f.readlines()[-100:]  # Last 100 lines
            return {"latest_log": latest_log}

        return {
            "logs_directory": logs_dir_path,
            "log_files": log_files,
            "current_log": log_filename,
        }
    except Exception as e:
        logger.error(f"Error in list_logs: {str(e)}")
        return {"error": f"Failed to list logs: {str(e)}"}


@app.get("/")
async def root():
    """Root endpoint providing API information"""
    active_languages = list(SUPPORTED_LANGUAGES.keys())
    return {
        "message": "Multilingual Multi-Model LLM API with Multimodal Support",
        "available_models": [name for name, info in AVAILABLE_MODELS.items() if info["active"]],
        "supported_languages": active_languages,
        "supported_intents": list(USER_INTENTS.keys()),
        "endpoints": {
            "/": "This information",
            "/models": "List all available models with language support",
            "/languages": "List all supported languages",
            "/generate": "Generate text using a selected model with intent classification and memory",
            "/generate-multimodal": "Generate text from multiple inputs (text, images, documents, tables, audio)",
            "/transcribe-audio": "Transcribe audio files using Whisper and optionally analyze content",
            "/text-to-speech": "Convert text to speech audio",
            "/audio-to-response": "Process audio file, transcribe it, and generate a text/audio response",
            "/audio-response-only": "Process audio and return an audio response directly (for voice assistants)",
            "/test-tts-system": "Test TTS functionality",
            "/tts-setup-help": "Get help with TTS setup",
            "/test_connection": "Test connection to LM Studio API",
            "/logs": "List available log files",
            "/translate": "Translate text between languages with memory",
            "/files": "List temporary uploaded files",
            "/config/default": "Get default configuration values",
            "/intent-stats": "Get statistics on detected user intents",
            "/test-intent": "Test endpoint to manually increment intent counts",
            "/conversations": "List all active conversations",
            "/conversations/{id}": "Get conversation history",
            "/docs": "API documentation",
        },
        "supported_file_types": {
            "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
            "documents": [".pdf", ".doc", ".docx"],
            "tables": [".csv", ".xlsx", ".xls"],
            "text": [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml"],
            "audio": [".mp3", ".wav", ".ogg", ".flac", ".m4a"],
        },
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting LLM API service with Uvicorn")
    try:
        uvicorn.run("log_llm:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"Uvicorn startup failed: {str(e)}")
