"""
PDF Text Extraction (Text-Based PDFs)
=====================================
This script extracts text from text-based PDF files using PyMuPDF (fitz).
It reads each page and prints the extracted text with page numbers.

Library: PyMuPDF

Prerequisites: See README.md for environment setup.

Usage:
    python pdf_text_extraction.py
"""

import fitz  # PyMuPDF
import os
import sys


# ── Configuration ──────────────────────────────────────────────────────────────
# Change this path to point to your PDF file
PDF_FILE_PATH = "./data/sample.pdf"


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a text-based PDF file page by page.

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of tuples (page_number, extracted_text).
    """

    # Step 1: Verify the file exists
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return []

    # Step 2: Open the PDF document
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"[INFO] Opened PDF: {os.path.basename(pdf_path)}")
    print(f"[INFO] Total pages: {total_pages}")
    print("=" * 60)

    results = []

    # Step 3: Loop through each page and extract text
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()  # Extract text from the page

        # Store result (page numbers are 1-indexed for readability)
        results.append((page_num + 1, text))

    # Step 4: Close the document
    doc.close()

    return results


def main():
    """Main function to demonstrate PDF text extraction."""

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       PDF TEXT EXTRACTION (Text-Based PDFs)              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Extract text from the PDF
    results = extract_text_from_pdf(PDF_FILE_PATH)

    if not results:
        print("[WARNING] No text was extracted. Check if the file exists.")
        sys.exit(1)

    # Display the extracted text for each page
    for page_num, text in results:
        print(f"\n--- Page {page_num} ---")
        if text.strip():
            # Print first 500 characters per page to keep output readable
            preview = text.strip()[:500]
            print(preview)
            if len(text.strip()) > 500:
                print(f"  ... [truncated, {len(text.strip())} chars total]")
        else:
            print("  [No text found on this page]")

    print()
    print("=" * 60)
    print(f"[DONE] Successfully extracted text from {len(results)} page(s).")
    print()


if __name__ == "__main__":
    main()
