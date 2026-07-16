"""
config.py — Central configuration for the semantic search research system.

All paths, model identifiers, and tunable parameters are defined here.
Change values in this file rather than modifying individual modules.
"""

import os
from pathlib import Path

# ── Base directory (project root) ────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# ── Data and output directories ──────────────────────────────────────────────
DATA_DIR    = BASE_DIR / "data"       # PDF document collection
MODELS_DIR  = BASE_DIR / "models"    # Cached sentence-transformer models
INDEXES_DIR = BASE_DIR / "indexes"   # Saved FAISS indexes
RESULTS_DIR = BASE_DIR / "results"   # Search result exports
LOGS_DIR    = BASE_DIR / "logs"      # Experiment CSV logs

# Ensure all directories exist on import
for _d in [DATA_DIR, MODELS_DIR, INDEXES_DIR, RESULTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Document registry ────────────────────────────────────────────────────────
REGISTRY_FILE = BASE_DIR / "indexes" / "document_registry.json"

# ── Embedding models ─────────────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "minilm": {
        "name":        "all-MiniLM-L6-v2",
        "description": "Fast, lightweight model (384-dim)",
        "dimension":   384,
    },
    "mpnet": {
        "name":        "all-mpnet-base-v2",
        "description": "High-quality general-purpose model (768-dim)",
        "dimension":   768,
    },
    "scibert": {
        "name":        "allenai/scibert_scivocab_uncased",
        "description": "Scientific/academic domain model (768-dim)",
        "dimension":   768,
    },
}

# Model key used when none is explicitly specified
DEFAULT_MODEL = "minilm"

# Directory where sentence-transformers will cache downloaded models
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR)

# ── OCR / extraction settings ─────────────────────────────────────────────────
# Pages whose PyMuPDF text is shorter than this trigger an OCR fallback
OCR_MIN_TEXT_LENGTH = 50

# Full path to the Tesseract executable.
# Set to None to use the system PATH, or provide an absolute path, e.g.:
#   TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = None

OCR_DPI      = 300   # Resolution for page→image conversion
OCR_LANGUAGE = "eng" # Tesseract language pack
# Minimum ratio of alphanumeric+whitespace characters in PyMuPDF output.
# Text scoring below this is treated as garbled and triggers OCR fallback.
OCR_MIN_TEXT_QUALITY = 0.6

# ── Text preprocessing ───────────────────────────────────────────────────────
USE_CHUNKING   = True   # True = chunk-based embeddings; False = full-document
CHUNK_SIZE     = 300    # Target chunk size in words
CHUNK_OVERLAP  = 50     # Word overlap between consecutive chunks

# ── Search defaults ──────────────────────────────────────────────────────────
DEFAULT_TOP_K          = 10    # Number of results to return
SNIPPET_CONTEXT_CHARS  = 250   # Characters of context shown around a keyword hit

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_TOP_K_VALUES  = [1, 3, 5, 10]   # K values for Precision@K / Recall@K
EVAL_QUERIES_FILE  = BASE_DIR / "evaluation" / "queries.json"
EVAL_RESULTS_DIR   = RESULTS_DIR     # Where evaluation CSVs are written

# ── Logging ──────────────────────────────────────────────────────────────────
EXPERIMENT_LOG_FILE = LOGS_DIR / "experiments.csv"
SEARCH_LOG_FILE     = LOGS_DIR / "search_history.csv"
