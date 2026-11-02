"""
ResearchMate - Configuration Module
Centralized configuration settings for the RAG pipeline, vector store, and Gemini API.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# Document Processing & Chunking Parameters
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Embedding Model
# all-MiniLM-L6-v2 produces high-quality 384-dimensional dense vectors
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Vector Retrieval Settings
DEFAULT_TOP_K = 5
SIMILARITY_SCORE_THRESHOLD = 0.25  # Cosine similarity threshold for relevance

# Gemini LLM Settings
# Supported models: gemini-3.8-flash (latest high-performance model), gemini-2.5-flash
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
FALLBACK_GEMINI_MODEL = "gemini-2.5-flash"
AVAILABLE_GEMINI_MODELS = ["gemini-3.8-flash", "gemini-2.5-flash"]
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 2048


def get_gemini_api_key() -> str:
    """
    Retrieve Gemini API key with hierarchical fallback:
    1. Streamlit Secrets (st.secrets["GEMINI_API_KEY"]) if running within Streamlit
    2. Environment Variable GEMINI_API_KEY (from system or .env)
    
    Returns:
        API key string, or empty string if not configured.
    """
    # Try streamlit secrets first
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
            if key and key.strip():
                return key.strip()
    except Exception:
        pass

    # Fallback to os.environ / .env
    key = os.getenv("GEMINI_API_KEY", "")
    return key.strip()
