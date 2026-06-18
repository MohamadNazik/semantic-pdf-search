"""
indexing/faiss_index.py
========================
Module 5 — Vector Storage

FAISSIndex wraps a FAISS flat inner-product index to provide:
  * Building from pre-computed L2-normalised embeddings
  * Saving / loading index + metadata side-car files
  * Cosine-similarity nearest-neighbour search (inner product on
    normalised vectors equals cosine similarity)
  * Association of integer FAISS IDs with chunk/document metadata

One index is created per embedding model, named by model key.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Metadata record ───────────────────────────────────────────────────────────

@dataclass
class ChunkMeta:
    """
    Metadata stored alongside each FAISS vector.

    faiss_id    — row index in the FAISS index (== position in add order)
    doc_id      — document registry ID
    chunk_index — zero-based chunk position within the document
    filename    — PDF filename (for display)
    filepath    — absolute PDF path (for opening)
    text        — the chunk text (used for snippet extraction)
    """
    faiss_id:    int
    doc_id:      int
    chunk_index: int
    filename:    str
    filepath:    str
    text:        str


# ── FAISSIndex ────────────────────────────────────────────────────────────────

class FAISSIndex:
    """
    Manages a FAISS IndexFlatIP for cosine-similarity search over
    L2-normalised document/chunk embeddings.

    Parameters
    ----------
    model_key    : str  — embedding model identifier ("minilm", "mpnet", "scibert")
    indexes_dir  : Path — directory where index files are persisted
    """

    INDEX_SUFFIX    = "_faiss.index"
    METADATA_SUFFIX = "_metadata.json"

    def __init__(self, model_key: str, indexes_dir: Path):
        self.model_key   = model_key
        self.indexes_dir = Path(indexes_dir)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        self._index     = None          # faiss.IndexFlatIP
        self._metadata: List[ChunkMeta] = []
        self._built     = False
        self.build_time: float = 0.0

    # ── file paths ────────────────────────────────────────────────────────────

    @property
    def index_path(self) -> Path:
        return self.indexes_dir / f"{self.model_key}{self.INDEX_SUFFIX}"

    @property
    def metadata_path(self) -> Path:
        return self.indexes_dir / f"{self.model_key}{self.METADATA_SUFFIX}"

    # ── building ──────────────────────────────────────────────────────────────

    def build(
        self,
        embeddings: np.ndarray,
        chunk_metas: List[ChunkMeta],
    ) -> None:
        """
        Create a new FAISS index from *embeddings* and associate each
        vector with its corresponding ChunkMeta entry.

        Parameters
        ----------
        embeddings   : float32 array of shape (N, dim) — L2-normalised
        chunk_metas  : list of N ChunkMeta objects, one per embedding row
        """
        import faiss

        if embeddings.shape[0] != len(chunk_metas):
            raise ValueError(
                f"embeddings rows ({embeddings.shape[0]}) != "
                f"chunk_metas length ({len(chunk_metas)})"
            )

        dim = embeddings.shape[1]
        t0  = time.perf_counter()

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))

        self._metadata = chunk_metas
        self._built    = True
        self.build_time = time.perf_counter() - t0

        logger.info(
            "FAISS index built: model=%s  vectors=%d  dim=%d  time=%.3fs",
            self.model_key, self._index.ntotal, dim, self.build_time,
        )

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Write the FAISS index binary and metadata JSON to disk."""
        import faiss

        if not self._built or self._index is None:
            raise RuntimeError("Index has not been built yet — call build() first.")

        faiss.write_index(self._index, str(self.index_path))

        meta_list = [asdict(m) for m in self._metadata]
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2, ensure_ascii=False)

        logger.info(
            "Index saved: %s  (%d vectors)",
            self.index_path, self._index.ntotal,
        )

    def load(self) -> bool:
        """
        Load index and metadata from disk.

        Returns True if successful, False if files do not exist.
        """
        import faiss

        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning(
                "Index files not found for model=%s", self.model_key
            )
            return False

        self._index = faiss.read_index(str(self.index_path))

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._metadata = [ChunkMeta(**item) for item in raw]

        self._built = True
        logger.info(
            "Index loaded: model=%s  vectors=%d",
            self.model_key, self._index.ntotal,
        )
        return True

    def exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Find the *top_k* most similar chunks to *query_vector*.

        Parameters
        ----------
        query_vector : float32 array of shape (1, dim) — L2-normalised
        top_k        : number of results to return

        Returns
        -------
        List of dicts, each containing:
            rank, score, doc_id, chunk_index, filename, filepath, text
        """
        if not self._built or self._index is None:
            raise RuntimeError("Index not loaded. Call build() or load() first.")

        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        scores, indices = self._index.search(
            query_vector.astype(np.float32), k
        )

        results = []
        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]), start=1
        ):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append({
                "rank":        rank,
                "score":       float(score),
                "doc_id":      meta.doc_id,
                "chunk_index": meta.chunk_index,
                "filename":    meta.filename,
                "filepath":    meta.filepath,
                "text":        meta.text,
            })
        return results

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def is_built(self) -> bool:
        return self._built
