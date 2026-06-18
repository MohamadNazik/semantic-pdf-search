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
import time
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ── DocumentRegistry ──────────────────────────────────────────────────────────

class DocumentRegistry:
    """
    Maintains a persistent JSON registry of all ingested PDF documents.

    Each entry stores:
        doc_id      — zero-based integer index
        filename    — basename of the file
        filepath    — absolute path
        page_count  — number of pages
        file_size   — bytes
        ingested_at — Unix timestamp
    """

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.documents: List[Dict] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                logger.info(
                    "Loaded registry with %d document(s) from %s",
                    len(self.documents), self.registry_path,
                )
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Could not load registry (%s); starting fresh.", exc)
                self.documents = []

    def save(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)
        logger.debug("Registry saved (%d entries).", len(self.documents))

    # ── scan ─────────────────────────────────────────────────────────────────

    def scan_folder(self, folder: Path, reset: bool = False) -> List[Dict]:
        """
        Walk *folder* recursively and register every PDF found.

        Parameters
        ----------
        folder  : directory to scan
        reset   : if True, discard existing registry entries before scanning

        Returns
        -------
        list of newly added document dicts
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        if reset:
            self.documents = []

        existing_paths = {d["filepath"] for d in self.documents}
        new_docs = []

        for pdf_path in sorted(folder.rglob("*.pdf")):
            abs_path = str(pdf_path.resolve())
            if abs_path in existing_paths:
                continue

            page_count = self._get_page_count(pdf_path)
            doc_entry = {
                "doc_id":      len(self.documents),
                "filename":    pdf_path.name,
                "filepath":    abs_path,
                "page_count":  page_count,
                "file_size":   pdf_path.stat().st_size,
                "ingested_at": time.time(),
            }
            self.documents.append(doc_entry)
            new_docs.append(doc_entry)
            logger.debug("Registered: %s (%d pages)", pdf_path.name, page_count)

        self.save()
        logger.info(
            "Scan complete — %d new PDF(s) registered; total: %d",
            len(new_docs), len(self.documents),
        )
        return new_docs

    # ── helpers ───────────────────────────────────────────────────────────────

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
        tesseract_cmd: Optional[str] = None,
        ocr_dpi: int = 300,
        ocr_language: str = "eng",
    ):
        self.min_text_length = min_text_length
        self.ocr_dpi         = ocr_dpi
        self.ocr_language    = ocr_language
        self._ocr_available  = False

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
        page      = doc[page_index]
        page_num  = page_index + 1   # 1-based for display
        text      = page.get_text("text").strip()

        if len(text) >= self.min_text_length:
            return PageResult(
                page_num   = page_num,
                text       = text,
                method     = "pymupdf",
                char_count = len(text),
                used_ocr   = False,
            )

        # Text too short — attempt OCR
        if not self._ocr_available:
            return PageResult(
                page_num   = page_num,
                text       = text,
                method     = "pymupdf_short",
                char_count = len(text),
                used_ocr   = False,
            )

        ocr_text = self._ocr_page(doc, page_index)
        final    = ocr_text if ocr_text else text

        return PageResult(
            page_num   = page_num,
            text       = final,
            method     = "ocr" if ocr_text else "pymupdf_short",
            char_count = len(final),
            used_ocr   = bool(ocr_text),
        )

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
