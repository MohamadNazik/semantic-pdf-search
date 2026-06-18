"""
embeddings/embedder.py
=======================
Module 4 — Embedding Generation

EmbeddingEngine wraps sentence-transformers to:
  * Load any of the three research models (MiniLM, MPNet, SciBERT)
  * Encode a list of text strings into L2-normalised float32 vectors
  * Persist and reload embeddings as NumPy .npy files
  * Record timing and dimension metadata for experiment logging
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Generates and persists document embeddings.

    Parameters
    ----------
    model_key   : str   — one of "minilm", "mpnet", "scibert"
    models_dir  : Path  — cache directory for downloaded models
    batch_size  : int   — encoding batch size (tune to available RAM/VRAM)
    normalize   : bool  — L2-normalise vectors (required for cosine FAISS search)
    """

    # Maps short keys to HuggingFace model identifiers
    MODEL_REGISTRY: Dict[str, str] = {
        "minilm":  "all-MiniLM-L6-v2",
        "mpnet":   "all-mpnet-base-v2",
        "scibert": "allenai/scibert_scivocab_uncased",
    }

    def __init__(
        self,
        model_key: str,
        models_dir: Path,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        if model_key not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model key {model_key!r}. "
                f"Choose from: {list(self.MODEL_REGISTRY)}"
            )

        self.model_key  = model_key
        self.model_name = self.MODEL_REGISTRY[model_key]
        self.models_dir = Path(models_dir)
        self.batch_size = batch_size
        self.normalize  = normalize

        self._model       = None   # lazy-loaded
        self._dimension:  Optional[int] = None
        self.last_encode_time: float = 0.0   # seconds for the last encode call

    # ── model loading ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Download (first run) and load the sentence-transformer model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        t0 = time.perf_counter()

        self._model = SentenceTransformer(
            self.model_name,
            cache_folder=str(self.models_dir),
        )

        # Probe dimension with a dummy sentence
        probe = self._model.encode(["probe"], convert_to_numpy=True)
        self._dimension = int(probe.shape[1])

        elapsed = time.perf_counter() - t0
        logger.info(
            "Model loaded: %s  |  dim=%d  |  load_time=%.2fs",
            self.model_name, self._dimension, elapsed,
        )

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self.load_model()
        return self._dimension  # type: ignore[return-value]

    # ── encoding ─────────────────────────────────────────────────────────────

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of strings into a float32 embedding matrix.

        Returns
        -------
        np.ndarray of shape (len(texts), dimension), dtype float32
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self.load_model()
        t0 = time.perf_counter()

        embeddings = self._model.encode(
            texts,
            batch_size         = self.batch_size,
            show_progress_bar  = len(texts) > 100,
            convert_to_numpy   = True,
            normalize_embeddings = self.normalize,
        )

        embeddings = embeddings.astype(np.float32)
        self.last_encode_time = time.perf_counter() - t0

        logger.info(
            "Encoded %d text(s) with %s in %.3fs  |  shape=%s",
            len(texts), self.model_key,
            self.last_encode_time, embeddings.shape,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string; returns shape (1, dim)."""
        return self.encode([query])

    # ── persistence ──────────────────────────────────────────────────────────

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        save_path: Path,
    ) -> None:
        """Save an embedding matrix as a compressed NumPy file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), embeddings)
        logger.info("Embeddings saved to %s  (shape=%s)", save_path, embeddings.shape)

    def load_embeddings(self, load_path: Path) -> np.ndarray:
        """Load a previously saved embedding matrix."""
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {load_path}")
        embeddings = np.load(str(load_path)).astype(np.float32)
        logger.info("Embeddings loaded from %s  (shape=%s)", load_path, embeddings.shape)
        return embeddings

    # ── helpers ───────────────────────────────────────────────────────────────

    def embedding_stats(self, embeddings: np.ndarray) -> Dict:
        """Return a small stats dict for logging."""
        return {
            "model_key":    self.model_key,
            "model_name":   self.model_name,
            "num_vectors":  embeddings.shape[0],
            "dimension":    embeddings.shape[1],
            "encode_time":  round(self.last_encode_time, 3),
            "memory_mb":    round(embeddings.nbytes / 1024 ** 2, 2),
        }

    def get_embedding_path(self, indexes_dir: Path) -> Path:
        """Canonical path for this model's embedding file."""
        return Path(indexes_dir) / f"embeddings_{self.model_key}.npy"
