# Requirements

Dependencies for the Semantic PDF Search Research Prototype.

---

## System Prerequisites

### Python
- **Version**: Python 3.9, 3.10, or 3.11 (recommended: 3.10)
- Download: https://www.python.org/downloads/

### Tesseract OCR (required for scanned/image-based PDFs)
- **Windows installer**: https://github.com/UB-Mannheim/tesseract/wiki
- Recommended version: 5.x
- After installation, either:
  - Add Tesseract to your system PATH, **or**
  - Set `TESSERACT_CMD` in `config.py` to the full executable path:
    ```python
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```
- Verify: `tesseract --version`

### Poppler (required by pdf2image for OCR page rendering)
- **Windows binaries**: https://github.com/oschwartz10612/poppler-windows/releases
- Extract and add the `bin/` folder to your system PATH.
- Verify: `pdftoppm -v`

---

## Python Packages

### Core PDF Processing
| Package | Version | Purpose |
|---------|---------|---------|
| `PyMuPDF` | `>=1.23.0` | Fast text extraction from text-based PDFs |
| `pytesseract` | `>=0.3.10` | Python wrapper for Tesseract OCR engine |
| `pdf2image` | `>=1.16.3` | Convert PDF pages to PIL images for OCR |
| `Pillow` | `>=10.0.0` | Image processing (required by pdf2image and pytesseract) |

### Embedding Models
| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | `>=2.7.0` | Loads and runs MiniLM, MPNet, and SciBERT models |
| `torch` | `>=2.1.0` | PyTorch backend for sentence-transformers |
| `transformers` | `>=4.37.0` | HuggingFace model hub (used by sentence-transformers) |

### Vector Indexing
| Package | Version | Purpose |
|---------|---------|---------|
| `faiss-cpu` | `>=1.7.4` | FAISS flat inner-product index for cosine similarity search |

> **Note**: If you have an NVIDIA GPU, you may replace `faiss-cpu` with `faiss-gpu`
> for faster index building, but `faiss-cpu` is sufficient for research-scale datasets.

### Scientific Computing
| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | `>=1.24.0` | Embedding matrix storage and arithmetic |
| `scipy` | `>=1.11.0` | Optional — for additional distance metrics |

### GUI
| Package | Version | Purpose |
|---------|---------|---------|
| `tkinter` | (stdlib) | Desktop GUI — included with Python on Windows |

---

## Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 3. Install all Python dependencies
```bash
pip install PyMuPDF>=1.23.0 pytesseract>=0.3.10 pdf2image>=1.16.3 Pillow>=10.0.0 sentence-transformers>=2.7.0 faiss-cpu>=1.7.4 numpy>=1.24.0
```

Or save the following as `requirements.txt` and run `pip install -r requirements.txt`:

```
PyMuPDF>=1.23.0
pytesseract>=0.3.10
pdf2image>=1.16.3
Pillow>=10.0.0
sentence-transformers>=2.7.0
faiss-cpu>=1.7.4
numpy>=1.24.0
```

---

## First-Run Model Downloads

On the first run, `sentence-transformers` will download the model weights from
HuggingFace. After download they are cached in `models/` and used offline.

| Model key | HuggingFace ID | Approx. download size |
|-----------|----------------|-----------------------|
| `minilm`  | `all-MiniLM-L6-v2` | ~90 MB |
| `mpnet`   | `all-mpnet-base-v2` | ~420 MB |
| `scibert` | `allenai/scibert_scivocab_uncased` | ~440 MB |

---

## Verify Installation

Run the following to check all components are working:

```bash
python -c "import fitz; print('PyMuPDF OK')"
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
python -c "import faiss; print('FAISS OK, version', faiss.__version__)"
python -c "from sentence_transformers import SentenceTransformer; print('ST OK')"
```

---

## Known Compatibility Notes

- `faiss-cpu` does not have a Windows wheel for Python 3.12+ on PyPI as of 2025.
  Use Python 3.10 or 3.11 for broadest compatibility.
- `torch` CPU-only installation is sufficient; GPU is not required.
- Tesseract OCR accuracy is best with 300 DPI rendering (configured in `config.py`).
