"""extraction — PDF text extraction pipeline (PyMuPDF + Tesseract OCR)."""
from .extractor import DocumentRegistry, HybridExtractor

__all__ = ["DocumentRegistry", "HybridExtractor"]
