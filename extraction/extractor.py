"""
extraction/extractor.py
========================
Module 1 — Dataset Ingestion
Module 2 — Hybrid Text Extraction

DocumentRegistry  : scans a folder for PDFs and maintains a metadata registry.
HybridExtractor   : extracts text per-page with PyMuPDF, falling back to
                    Tesseract OCR when extracted text is below a configurable
                    character threshold.
"""

import io
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ── DocumentRegistry ──────────────────────────────────────────────────────────

class DocumentRegistry:
    """
    Maintains a persistent SQLite database registry of all ingested PDF documents.
    Caches extracted text to speed up subsequent indexing runs and prevent redundant OCR.

    Each entry stores:
        doc_id          — primary key integer
        filename        — basename of the file
        filepath        — absolute path
        page_count      — number of pages
        file_size       — bytes
        ingested_at     — Unix timestamp
        extracted_text  — full text (cached)
    """

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.documents: List[Dict] = []
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.registry_path), check_same_thread=False)
        self._load()

    def _load(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                filepath TEXT UNIQUE,
                page_count INTEGER,
                file_size INTEGER,
                ingested_at REAL,
                extracted_text TEXT
            )
        """)
        self.conn.commit()

        # Load metadata into self.documents for iteration and retrieval
        cursor.execute("""
            SELECT doc_id, filename, filepath, page_count, file_size, ingested_at 
            FROM documents ORDER BY doc_id
        """)
        self.documents = []
        for row in cursor.fetchall():
            self.documents.append({
                "doc_id":      row[0],
                "filename":    row[1],
                "filepath":    row[2],
                "page_count":  row[3],
                "file_size":   row[4],
                "ingested_at": row[5]
            })
        logger.info(
            "Loaded registry with %d document(s) from SQLite db %s",
            len(self.documents), self.registry_path.name
        )

    def scan_folder(self, folder: Path, reset: bool = False) -> List[Dict]:
        """
        Walk *folder* recursively and register every PDF found in SQLite.
        Preserves cached text for existing files to speed up comparison runs.
        If reset=True, all existing entries are cleared before scanning.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        cursor = self.conn.cursor()

        if reset:
            cursor.execute("DELETE FROM documents")
            self.conn.commit()
            self.documents = []
            logger.info("Registry reset — all entries cleared.")

        current_pdfs = sorted(folder.rglob("*.pdf"))
        found_paths = set()
        new_docs = []

        for pdf_path in current_pdfs:
            abs_path = str(pdf_path.resolve())
            found_paths.add(abs_path)

            # Check if file is already registered
            cursor.execute(
                "SELECT doc_id, file_size FROM documents WHERE filepath = ?",
                (abs_path,)
            )
            row = cursor.fetchone()

            if row is not None:
                doc_id, old_size = row
                # If size has changed, update metadata and reset cache
                if old_size != pdf_path.stat().st_size:
                    page_count = self._get_page_count(pdf_path)
                    cursor.execute("""
                        UPDATE documents 
                        SET file_size = ?, page_count = ?, extracted_text = NULL 
                        WHERE doc_id = ?
                    """, (pdf_path.stat().st_size, page_count, doc_id))
            else:
                # Add new entry
                page_count = self._get_page_count(pdf_path)
                cursor.execute("""
                    INSERT INTO documents 
                    (filename, filepath, page_count, file_size, ingested_at, extracted_text) 
                    VALUES (?, ?, ?, ?, ?, NULL)
                """, (pdf_path.name, abs_path, page_count, pdf_path.stat().st_size, time.time()))
                new_docs.append({
                    "filename": pdf_path.name,
                    "filepath": abs_path,
                    "page_count": page_count,
                    "file_size": pdf_path.stat().st_size
                })
        self.conn.commit()

        # Delete entries of files that were removed from the folder
        cursor.execute("SELECT doc_id, filepath FROM documents")
        db_rows = cursor.fetchall()
        for doc_id, filepath in db_rows:
            if filepath not in found_paths:
                cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.conn.commit()

        # Reload documents
        self._load()
        return new_docs

    def get_extracted_text(self, doc_id: int) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT extracted_text FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def update_extracted_text(self, doc_id: int, text: str):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE documents SET extracted_text = ? WHERE doc_id = ?", (text, doc_id))
        self.conn.commit()

    @staticmethod
    def _get_page_count(pdf_path: Path) -> int:
        try:
            doc = fitz.open(str(pdf_path))
            count = len(doc)
            doc.close()
            return count
        except Exception as exc:
            logger.warning("Could not open %s: %s", pdf_path.name, exc)
            return 0

    def get_by_id(self, doc_id: int) -> Optional[Dict]:
        for doc in self.documents:
            if doc["doc_id"] == doc_id:
                return doc
        return None

    def close(self):
        self.conn.close()

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)


# ── HybridExtractor ───────────────────────────────────────────────────────────

class PageResult:
    """Holds the extraction result for a single PDF page."""

    __slots__ = (
        "page_num", "text", "method",
        "char_count", "used_ocr",
    )

    def __init__(self, page_num, text, method, char_count, used_ocr):
        self.page_num   = page_num
        self.text       = text
        self.method     = method
        self.char_count = char_count
        self.used_ocr   = used_ocr

    def __repr__(self):
        return (
            f"PageResult(page={self.page_num}, method={self.method!r}, "
            f"chars={self.char_count}, ocr={self.used_ocr})"
        )


class HybridExtractor:
    """
    Per-page hybrid extraction pipeline:

    1. Try PyMuPDF text extraction.
    2. If result is shorter than *min_text_length*, render the page as an
       image and apply Tesseract OCR.
    3. Combine page texts into one document string.

    Parameters
    ----------
    min_text_length : int
        Character threshold below which OCR is attempted.
    min_text_quality : float
        Minimum ratio of alphanumeric+whitespace characters (0.0–1.0).
        PyMuPDF output scoring below this is treated as garbled and triggers
        OCR even when the length threshold is met.
    tesseract_cmd   : str or None
        Explicit path to the Tesseract binary; None uses system PATH.
    ocr_dpi         : int
        DPI for page-to-image rendering (higher = slower but more accurate).
    ocr_language    : str
        Tesseract language code (default ``"eng"``).
    """

    def __init__(
        self,
        min_text_length: int = 50,
        min_text_quality: float = 0.6,
        tesseract_cmd: Optional[str] = None,
        ocr_dpi: int = 300,
        ocr_language: str = "eng",
    ):
        self.min_text_length  = min_text_length
        self.min_text_quality = min_text_quality
        self.ocr_dpi          = ocr_dpi
        self.ocr_language     = ocr_language
        self._ocr_available   = False

        self._init_ocr(tesseract_cmd)

    # ── OCR setup ─────────────────────────────────────────────────────────────

    def _init_ocr(self, tesseract_cmd: Optional[str]):
        try:
            import pytesseract
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            # Probe for Tesseract
            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self._ocr_available = True
            logger.info("Tesseract OCR available.")
        except Exception as exc:
            logger.warning(
                "Tesseract OCR not available (%s). "
                "OCR fallback will be disabled.",
                exc,
            )

    # ── public API ────────────────────────────────────────────────────────────

    def extract_document(self, pdf_path: str) -> Dict:
        """
        Extract all text from a PDF and return a result dict with:

            filepath        — absolute path
            full_text       — concatenated text of all pages
            pages           — list of PageResult objects
            page_count      — total pages
            ocr_page_count  — pages where OCR was used
            text_length     — total characters
            extraction_time — seconds taken
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        t_start = time.perf_counter()
        doc = fitz.open(str(pdf_path))

        page_results: List[PageResult] = []

        for page_index in range(len(doc)):
            pr = self._extract_page(doc, page_index)
            page_results.append(pr)

        doc.close()
        elapsed = time.perf_counter() - t_start

        full_text     = "\n\n".join(pr.text for pr in page_results if pr.text)
        ocr_count     = sum(1 for pr in page_results if pr.used_ocr)

        result = {
            "filepath":        str(pdf_path.resolve()),
            "filename":        pdf_path.name,
            "full_text":       full_text,
            "pages":           page_results,
            "page_count":      len(page_results),
            "ocr_page_count":  ocr_count,
            "text_length":     len(full_text),
            "extraction_time": round(elapsed, 3),
        }

        logger.info(
            "%s — %d pages, %d OCR, %d chars in %.2fs",
            pdf_path.name, len(page_results), ocr_count,
            len(full_text), elapsed,
        )
        return result

    # ── internal extraction ───────────────────────────────────────────────────

    def _extract_page(self, doc: fitz.Document, page_index: int) -> PageResult:
        page     = doc[page_index]
        page_num = page_index + 1   # 1-based for display
        text     = page.get_text("text").strip()

        too_short   = len(text) < self.min_text_length
        low_quality = (
            not too_short
            and self._text_quality(text) < self.min_text_quality
        )

        if not too_short and not low_quality:
            return PageResult(
                page_num   = page_num,
                text       = text,
                method     = "pymupdf",
                char_count = len(text),
                used_ocr   = False,
            )

        # Text too short or quality below threshold — attempt OCR
        if not self._ocr_available:
            method = "pymupdf_short" if too_short else "pymupdf_low_quality"
            return PageResult(
                page_num   = page_num,
                text       = text,
                method     = method,
                char_count = len(text),
                used_ocr   = False,
            )

        ocr_text = self._ocr_page(doc, page_index)
        final    = ocr_text if ocr_text else text

        if ocr_text:
            method = "ocr"
        elif too_short:
            method = "pymupdf_short"
        else:
            method = "pymupdf_low_quality"

        return PageResult(
            page_num   = page_num,
            text       = final,
            method     = method,
            char_count = len(final),
            used_ocr   = bool(ocr_text),
        )

    @staticmethod
    def _text_quality(text: str) -> float:
        """
        Returns the fraction of characters that are alphanumeric or whitespace.
        Garbled text from corrupt or image-based PDF layers typically scores
        below 0.6 due to high proportions of symbols and control characters.
        """
        if not text:
            return 0.0
        printable = sum(1 for c in text if c.isalnum() or c.isspace())
        return printable / len(text)

    def _ocr_page(self, doc: fitz.Document, page_index: int) -> str:
        """Render one page as an image and run Tesseract on it."""
        try:
            from PIL import Image

            page   = doc[page_index]
            matrix = fitz.Matrix(self.ocr_dpi / 72, self.ocr_dpi / 72)
            pix    = page.get_pixmap(matrix=matrix, alpha=False)
            img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            text = self._pytesseract.image_to_string(
                img, lang=self.ocr_language
            ).strip()
            return text

        except Exception as exc:
            logger.warning(
                "OCR failed for page %d: %s", page_index + 1, exc
            )
            return ""

    @property
    def ocr_available(self) -> bool:
        return self._ocr_available
