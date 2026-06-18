"""
search/keyword_search.py
=========================
Module 7 — Keyword Search Baseline

KeywordSearch implements a TF-IDF ranked retrieval baseline that operates
over the same document corpus as the semantic search engine.

Purpose: provide a traditional retrieval baseline so that the research can
compare semantic embedding models against classic keyword matching using
the same evaluation queries and metrics.

Index is built in-memory from a list of (doc_id, filename, filepath, text)
tuples and can be rebuilt cheaply — no disk persistence needed.
"""

import logging
import math
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KeywordSearch:
    """
    TF-IDF keyword search over a flat document collection.

    Parameters
    ----------
    snippet_chars : int — context characters for snippet extraction
    """

    def __init__(self, snippet_chars: int = 250):
        self.snippet_chars = snippet_chars
        self._documents: List[Dict] = []     # {doc_id, filename, filepath, text}
        self._tf_idf_matrix: Optional[Dict] = None  # doc_id → {term: tfidf}
        self._idf: Dict[str, float] = {}
        self._is_built = False
        self.last_query_time: float = 0.0
        self.build_time: float = 0.0

    # ── index building ────────────────────────────────────────────────────────

    def build(self, documents: List[Dict]) -> None:
        """
        Build a TF-IDF index from *documents*.

        Each document dict must have:
            doc_id   : int
            filename : str
            filepath : str
            text     : str  (full document text)
        """
        if not documents:
            logger.warning("KeywordSearch.build() called with no documents.")
            return

        t0 = time.perf_counter()
        self._documents  = documents
        corpus_tokens    = [self._tokenize(d["text"]) for d in documents]

        # IDF
        N           = len(documents)
        doc_freq    = Counter()
        for tokens in corpus_tokens:
            for term in set(tokens):
                doc_freq[term] += 1

        self._idf = {
            term: math.log((N + 1) / (df + 1)) + 1  # smoothed IDF
            for term, df in doc_freq.items()
        }

        # TF-IDF per document
        self._tf_idf_matrix = {}
        for doc, tokens in zip(documents, corpus_tokens):
            tf      = Counter(tokens)
            total   = max(len(tokens), 1)
            self._tf_idf_matrix[doc["doc_id"]] = {
                term: (count / total) * self._idf.get(term, 1.0)
                for term, count in tf.items()
            }

        self._is_built  = True
        self.build_time = time.perf_counter() - t0
        logger.info(
            "Keyword index built: %d document(s)  vocab=%d  time=%.3fs",
            N, len(self._idf), self.build_time,
        )

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> Dict:
        """
        Score all documents against *query* and return top-k results.

        Scoring: sum of TF-IDF weights for each query term present.

        Returns
        -------
        dict with keys:
            query, top_k, query_time_ms, results

        Each result contains:
            rank, score, doc_id, filename, filepath, snippet
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build() first.")

        if not query.strip():
            return {"query": query, "top_k": top_k,
                    "query_time_ms": 0.0, "results": []}

        t0 = time.perf_counter()

        query_terms = self._tokenize(query)
        scored: List[Tuple[float, int]] = []

        for doc in self._documents:
            doc_tfidf = self._tf_idf_matrix.get(doc["doc_id"], {})
            score = sum(doc_tfidf.get(term, 0.0) for term in query_terms)
            if score > 0:
                scored.append((score, doc["doc_id"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        doc_map = {d["doc_id"]: d for d in self._documents}
        results = []
        for rank, (score, doc_id) in enumerate(top, start=1):
            doc = doc_map[doc_id]
            results.append({
                "rank":     rank,
                "score":    round(score, 6),
                "doc_id":   doc_id,
                "filename": doc["filename"],
                "filepath": doc["filepath"],
                "snippet":  self._extract_snippet(doc["text"], query),
            })

        self.last_query_time = time.perf_counter() - t0

        return {
            "query":         query,
            "top_k":         top_k,
            "query_time_ms": round(self.last_query_time * 1000, 1),
            "results":       results,
        }

    # ── tokenisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    # ── snippet ───────────────────────────────────────────────────────────────

    def _extract_snippet(self, text: str, query: str) -> str:
        if not text:
            return ""
        terms   = [re.escape(w) for w in query.split() if len(w) > 2]
        if not terms:
            return text[:self.snippet_chars * 2].strip()

        pattern = re.compile("|".join(terms), re.IGNORECASE)
        match   = pattern.search(text)
        if not match:
            return text[:self.snippet_chars * 2].strip()

        center  = match.start()
        start   = max(0, center - self.snippet_chars)
        end     = min(len(text), center + self.snippet_chars)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet += "…"
        return snippet

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def document_count(self) -> int:
        return len(self._documents)
