# config.py
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# --- Provider switches ---
# Если есть ключ Google — используем Google для эмбеддингов и генерации.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
USE_GOOGLE = bool(GOOGLE_API_KEY)

# --- Local models ---
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "google/gemma-3-4b-it")

# Включить кросс-энкодер для rerank (локально)
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Retrieval params ---
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "8"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "12"))
TOP_K_FUSED = int(os.getenv("TOP_K_FUSED", "8"))

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Paths ---
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR = DATA_DIR / "vector_store"
BM25_CORPUS_PATH = DATA_DIR / "bm25_corpus.jsonl"
HASH_STORE_PATH = DATA_DIR / "hash_storage.json"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Generation ---
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# --- Other ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_SITE = os.getenv("DEFAULT_SITE", "https://delprof.ru").strip()
DEFAULT_UPDATE_MAX_LINKS = int(os.getenv("DEFAULT_UPDATE_MAX_LINKS", "200"))
