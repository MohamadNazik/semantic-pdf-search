# Semantic PDF Search

A research prototype for semantic search over local English PDF documents on Windows. Built as a final-year undergraduate research project (IT4216) to evaluate how different embedding models affect retrieval quality on both text-based and scanned PDFs.

## Overview

The system extracts text from PDFs using a hybrid pipeline (PyMuPDF for text-based pages, Tesseract OCR as fallback for scanned/image pages), encodes the content into dense vector embeddings, stores them in a FAISS index, and retrieves the most semantically relevant chunks for a given query. A TF-IDF keyword search baseline is included for comparison.

Three embedding models are evaluated side-by-side:

| Key | Model | Dimensions |
|-----|-------|------------|
| `minilm` | `all-MiniLM-L6-v2` | 384 |
| `mpnet` | `all-mpnet-base-v2` | 768 |
| `scibert` | `allenai/scibert_scivocab_uncased` | 768 |

Retrieval quality is measured using Precision@K, Recall@K, and MAP against a labelled query set.

## System Architecture

```
data/ (PDFs)
  └─► Module 1+2: HybridExtractor  (PyMuPDF + Tesseract OCR)
        └─► Module 3: TextPreprocessor  (cleaning + chunking)
              └─► Module 4: EmbeddingEngine  (MiniLM / MPNet / SciBERT)
                    └─► Module 5: FAISSIndex  (cosine similarity)
                          └─► Module 6: SemanticSearch
                          └─► Module 7: KeywordSearch  (TF-IDF baseline)
                                └─► Module 8: Evaluator  (P@K, R@K, MAP)
                                └─► Module 9: ExperimentLogger  (CSV)
                                └─► Module 10: Tkinter GUI
```

## Prerequisites

- Python 3.10 or 3.11 (3.12+ has a `faiss-cpu` wheel issue on Windows)
- [Tesseract OCR 5.x](https://github.com/UB-Mannheim/tesseract/wiki) — add to PATH or set `TESSERACT_CMD` in `config.py`
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) — add the `bin/` folder to PATH

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install PyMuPDF>=1.23.0 pytesseract>=0.3.10 pdf2image>=1.16.3 Pillow>=10.0.0 sentence-transformers>=2.7.0 faiss-cpu>=1.7.4 numpy>=1.24.0
```

On first run, the selected model weights are downloaded from HuggingFace (~90 MB for MiniLM, ~420 MB for MPNet/SciBERT) and cached in `models/`.

See [`requirements.md`](requirements.md) for full dependency details and verification steps.

## Usage

### Desktop GUI

```bash
python main.py gui
```

### Command line

```bash
# Index a folder of PDFs
python main.py index --folder ./data --model minilm

# Run a single query
python main.py search --model minilm --query "deep learning"

# Evaluate one model against a labelled query set
python main.py evaluate --model mpnet --queries evaluation/queries.json

# Compare all three models and write a CSV summary
python main.py compare --queries evaluation/queries.json
```

## Configuration

All tunable parameters live in [`config.py`](config.py):

- `DEFAULT_MODEL` — which embedding model to use by default
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking window (words)
- `OCR_MIN_TEXT_LENGTH` — threshold below which OCR fallback triggers
- `TESSERACT_CMD` — absolute path to Tesseract if not on PATH
- `DEFAULT_TOP_K` — number of results returned per query

## Project Structure

```
├── config.py                  # Central configuration
├── main.py                    # CLI entry point
├── data/                      # PDF document collection
├── extraction/extractor.py    # Module 1+2: DocumentRegistry + HybridExtractor
├── preprocessing/preprocessor.py  # Module 3: TextPreprocessor
├── embeddings/embedder.py     # Module 4: EmbeddingEngine
├── indexing/faiss_index.py    # Module 5: FAISSIndex
├── search/semantic_search.py  # Module 6: SemanticSearch
├── search/keyword_search.py   # Module 7: KeywordSearch (TF-IDF baseline)
├── evaluation/evaluator.py    # Module 8: P@K, R@K, MAP evaluation
├── evaluation/logger.py       # Module 9: CSV experiment logger
├── evaluation/queries.json    # Labelled query set for evaluation
├── gui/app.py                 # Module 10: Tkinter desktop GUI
├── prototypes/                # Early exploratory scripts (not part of the system)
└── requirements.md            # Full dependency reference
```
