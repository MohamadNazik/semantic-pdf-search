"""
preprocessing/preprocessor.py
==============================
Module 3 — Text Preprocessing

TextPreprocessor cleans and normalises extracted document text, then
optionally segments it into overlapping word-based chunks suitable for
embedding.

The module is designed to support both:
    * Full-document embeddings  (use_chunking=False)
    * Chunk-based embeddings    (use_chunking=True)
so that the two strategies can be compared in experiments.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single text segment produced by the chunking step.

    Attributes
    ----------
    doc_id       : registry identifier of the source document
    chunk_index  : zero-based position within the document's chunk sequence
    text         : cleaned chunk text
    word_count   : number of words in *text*
    char_start   : approximate character offset in the original full text
                   (useful for snippet highlighting)
    """
    doc_id:      int
    chunk_index: int
    text:        str
    word_count:  int
    char_start:  int = 0
    metadata:    dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"Chunk(doc={self.doc_id}, idx={self.chunk_index}, "
            f"words={self.word_count}, chars={len(self.text)})"
        )


# ── TextPreprocessor ──────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Cleans raw document text and optionally splits it into chunks.

    Parameters
    ----------
    use_chunking  : bool   — if False, the entire document is one "chunk"
    chunk_size    : int    — target chunk size in words
    chunk_overlap : int    — number of words shared between consecutive chunks
    """

    def __init__(
        self,
        use_chunking: bool = True,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.use_chunking  = use_chunking
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, raw_text: str, doc_id: int) -> List[Chunk]:
        """
        Clean *raw_text* and return a list of Chunk objects.

        If use_chunking is False, returns a single Chunk containing
        the entire cleaned text.
        """
        cleaned = self.clean(raw_text)

        if not cleaned:
            logger.warning("doc_id=%d produced empty text after cleaning.", doc_id)
            return []

        if not self.use_chunking:
            return [
                Chunk(
                    doc_id      = doc_id,
                    chunk_index = 0,
                    text        = cleaned,
                    word_count  = len(cleaned.split()),
                )
            ]

        return self._chunk_text(cleaned, doc_id)

    def clean(self, text: str) -> str:
        """
        Apply cleaning steps to raw text:
        1. Normalise unicode (NFC)
        2. Replace soft hyphens and zero-width characters
        3. Collapse multiple whitespace characters
        4. Strip leading/trailing whitespace
        """
        if not text:
            return ""

        # Unicode normalisation (handles accented chars, ligatures, etc.)
        text = unicodedata.normalize("NFC", text)

        # Remove soft hyphens (­) and zero-width spaces
        text = text.replace("­", "").replace("​", "")

        # Replace form feeds and vertical tabs with spaces
        text = re.sub(r"[\f\v]", " ", text)

        # Collapse runs of blank lines (keep max 2 newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse intra-line whitespace
        text = re.sub(r"[ \t]+", " ", text)

        # Strip
        text = text.strip()

        return text

    # ── chunking ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, doc_id: int) -> List[Chunk]:
        """
        Split *text* into overlapping word-based windows.

        Strategy:
            words  → list of tokens
            stride = chunk_size - chunk_overlap
            Each window covers words[i : i+chunk_size]
        """
        words  = text.split()
        stride = self.chunk_size - self.chunk_overlap
        chunks: List[Chunk] = []

        if len(words) <= self.chunk_size:
            # Document is shorter than one chunk — return it whole
            return [
                Chunk(
                    doc_id      = doc_id,
                    chunk_index = 0,
                    text        = text,
                    word_count  = len(words),
                )
            ]

        idx = 0
        chunk_index = 0

        # Rebuild a word-start offset map so we can track char_start
        char_offsets = self._word_char_offsets(text, words)

        while idx < len(words):
            window = words[idx: idx + self.chunk_size]
            chunk_text = " ".join(window)
            char_start = char_offsets[idx]

            chunks.append(
                Chunk(
                    doc_id      = doc_id,
                    chunk_index = chunk_index,
                    text        = chunk_text,
                    word_count  = len(window),
                    char_start  = char_start,
                )
            )

            chunk_index += 1
            idx += stride

            # Stop when the remaining words are fewer than the overlap
            if idx >= len(words):
                break

        logger.debug(
            "doc_id=%d — %d word(s) → %d chunk(s) "
            "(size=%d, overlap=%d)",
            doc_id, len(words), len(chunks),
            self.chunk_size, self.chunk_overlap,
        )
        return chunks

    @staticmethod
    def _word_char_offsets(text: str, words: List[str]) -> List[int]:
        """Return the character start offset of each word in *words*."""
        offsets = []
        pos = 0
        for word in words:
            found = text.find(word, pos)
            offsets.append(found if found != -1 else pos)
            pos = found + len(word) if found != -1 else pos + len(word)
        return offsets
