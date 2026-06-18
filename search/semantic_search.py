"""
search/semantic_search.py
==========================
Module 6 — Semantic Search

SemanticSearch ties together EmbeddingEngine and FAISSIndex to answer
natural-language queries:

    1. Embed the query with the same model used to build the index.
    2. Search the FAISS index for the top-k nearest neighbours.
    3. Enrich each hit with a readable snippet extracted from the chunk text.
    4. Return ranked results with timing metadata.
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from embeddings.embedder import EmbeddingEngine
from indexing.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class SemanticSearch:
    """
    Semantic search over a pre-built FAISS index.

    Parameters
    ----------
    model_key       : str   — embedding model key ("minilm", "mpnet", "scibert")
    indexes_dir     : Path  — directory containing FAISS index files
    models_dir      : Path  — model cache directory
    snippet_chars   : int   — context characters around a query term for snippet
    """

    def __init__(
        self,
        model_key: str,
        indexes_dir: Path,
        models_dir: Path,
        snippet_chars: int = 250,
    ):
        self.model_key    = model_key
        self.snippet_chars = snippet_chars

        self._embedder = EmbeddingEngine(
            model_key  = model_key,
            models_dir = Path(models_dir),
        )
        self._index = FAISSIndex(
            model_key   = model_key,
            indexes_dir = Path(indexes_dir),
        )
        self.last_query_time: float = 0.0

    # ── index management ──────────────────────────────────────────────────────

    def load_index(self) -> bool:
        """Load the FAISS index from disk. Returns True on success."""
        return self._index.load()

    @property
    def is_ready(self) -> bool:
        return self._index.is_built

    @property
    def total_documents(self) -> int:
        return self._index.total_vectors

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> Dict:
        """
        Run a semantic query and return structured results.

        Parameters
        ----------
        query : natural-language search string
        top_k : number of results to return

        Returns
        -------
        dict with keys:
            query          — original query string
            model_key      — embedding model used
            top_k          — requested result count
            query_time_ms  — total query latency in milliseconds
            results        — list of result dicts (see below)

        Each result dict contains:
            rank, score, filename, filepath, snippet, chunk_index
        """
        if not self.is_ready:
            raise RuntimeError(
                "Index not loaded. Call load_index() first, or build the index."
            )

        if not query.strip():
            return self._empty_response(query, top_k)

        t0 = time.perf_counter()

        # 1. Embed query
        query_vec = self._embedder.encode_query(query.strip())

        # 2. FAISS search
        raw_results = self._index.search(query_vec, top_k=top_k)

        # 3. Enrich with snippets
        for hit in raw_results:
            hit["snippet"] = self._extract_snippet(
                hit["text"], query, self.snippet_chars
            )
            # Remove the raw chunk text from the public result
            del hit["text"]

        self.last_query_time = time.perf_counter() - t0

        return {
            "query":         query,
            "model_key":     self.model_key,
            "top_k":         top_k,
            "query_time_ms": round(self.last_query_time * 1000, 1),
            "results":       raw_results,
        }

    # ── snippet extraction ────────────────────────────────────────────────────

    @staticmethod
    def _extract_snippet(
        text: str,
        query: str,
        context_chars: int,
    ) -> str:
        """
        Return a short excerpt from *text* that contains a query term.
        Falls back to the beginning of the text if no term is found.
        """
        if not text:
            return ""

        # Build a regex from query words, case-insensitive
        terms = [re.escape(w) for w in query.split() if len(w) > 2]
        if not terms:
            return text[:context_chars * 2].strip()

        pattern = re.compile("|".join(terms), re.IGNORECASE)
        match   = pattern.search(text)

        if not match:
            return text[:context_chars * 2].strip()

        center = match.start()
        start  = max(0, center - context_chars)
        end    = min(len(text), center + context_chars)

        snippet = text[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        return snippet

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_response(query: str, top_k: int) -> Dict:
        return {
            "query":         query,
            "model_key":     "",
            "top_k":         top_k,
            "query_time_ms": 0.0,
            "results":       [],
        }
