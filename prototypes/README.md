# Semantic Search Prototype - Research Implementation

This repository contains the prototype components for the research project:
**"Development and Evaluation of a Semantic Search Prototype for Local English PDF Document Retrieval on Windows Systems Using Different Embedding Models"**

## Project Components

1.  **PDF Text Extraction** (`pdf_text_extraction.py`): Extracting text from native PDF files.
2.  **OCR Extraction** (`ocr_extraction.py`): Extracting text from scanned/image-based PDFs.
3.  **Hybrid Extraction** (`hybrid_extraction.py`): Intelligent extraction handling both text and images on the same page.
4.  **Embedding Model Testing** (`embedding_test.py`): Generating vector representations using `all-MiniLM-L6-v2`.
5.  **Dataset Verification** (`dataset_check.py`): Verifying the local PDF document collection.

---

## Prerequisites & Installation

### 1. Python Environment
Ensure you have Python 3.8+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Tesseract OCR (For OCR & Hybrid Scripts)
Required for extracting text from images and scanned documents.

1.  **Download**: Get the Windows installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
2.  **Install**: Run the installer (e.g., `tesseract-ocr-w64-setup-v5.x.x.exe`).
3.  **Path Setup**:
    *   Note the installation path (usually `C:\Program Files\Tesseract-OCR`).
    *   Add this path to your system **Environment Variables** (under "Path").
4.  **Verify**: Open a terminal and type `tesseract --version`.

### 3. Poppler (For OCR Script)
Required by `pdf2image` to convert PDF pages into images.

1.  **Download**: [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases).
2.  **Extract**: Extract the ZIP to a folder (e.g., `C:\poppler`).
3.  **Path Setup**: Add `C:\poppler\Library\bin` to your system **Path** environment variable.

---

## How to Run

1.  **Verify Dataset**:
    ```bash
    python dataset_check.py
    ```

2.  **PDF Text Extraction (Standard)**:
    ```bash
    python pdf_text_extraction.py
    ```

3.  **OCR Extraction (Scanned PDFs)**:
    ```bash
    python ocr_extraction.py
    ```

4.  **Hybrid Extraction (Mixed Content)**:
    ```bash
    python hybrid_extraction.py
    ```

5.  **Embedding Model Test**:
    ```bash
    python embedding_test.py
    ```

## Notes for Presentation
*   Scripts are designed to be lightweight and fast.
*   Output is formatted for clear visibility during live demonstrations.
*   Error handling is included for missing files or dependencies.
