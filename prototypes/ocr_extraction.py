"""
OCR Extraction for Scanned/Image-Based PDFs
============================================
This script uses OCR (Optical Character Recognition) to extract text
from scanned or image-based PDF files.

Libraries: pytesseract, pdf2image, Pillow

Prerequisites:
    - See README.md for Tesseract OCR and Poppler installation instructions.

Usage:
    python ocr_extraction.py
"""

import pytesseract
from pdf2image import convert_from_path
import os
import sys


# ── Configuration ──────────────────────────────────────────────────────────────
# Change this path to point to your scanned PDF file
PDF_FILE_PATH = "./data/scanned_sample.pdf"

# If Tesseract is not in your PATH, uncomment and update the line below:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text_with_ocr(pdf_path):
    """
    Extract text from a scanned/image-based PDF using OCR.

    Parameters:
        pdf_path (str): Path to the scanned PDF file.

    Returns:
        list: A list of tuples (page_number, extracted_text).
    """

    # Step 1: Verify the file exists
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return []

    print(f"[INFO] Opened PDF: {os.path.basename(pdf_path)}")

    # Step 2: Convert PDF pages to images
    # Each page becomes a PIL Image object
    print("[INFO] Converting PDF pages to images...")
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"[ERROR] Failed to convert PDF to images: {e}")
        print("[HINT] Make sure Poppler is installed and added to PATH.")
        return []

    total_pages = len(images)
    print(f"[INFO] Total pages: {total_pages}")
    print("=" * 60)

    results = []

    # Step 3: Apply OCR to each page image
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"[INFO] Running OCR on page {page_num}...")

        # Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(image, lang="eng")

        # Store result
        results.append((page_num, text))

    return results


def main():
    """Main function to demonstrate OCR extraction from scanned PDFs."""

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       OCR EXTRACTION (Scanned/Image-Based PDFs)        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Extract text using OCR
    results = extract_text_with_ocr(PDF_FILE_PATH)

    if not results:
        print("[WARNING] No text was extracted. Check file path and dependencies.")
        sys.exit(1)

    # Display the extracted text for each page
    for page_num, text in results:
        print(f"\n--- Page {page_num} (OCR) ---")
        if text.strip():
            # Print first 500 characters per page to keep output readable
            preview = text.strip()[:500]
            print(preview)
            if len(text.strip()) > 500:
                print(f"  ... [truncated, {len(text.strip())} chars total]")
        else:
            print("  [No text detected by OCR on this page]")

    print()
    print("=" * 60)
    print(f"[DONE] OCR completed for {len(results)} page(s).")
    print()


if __name__ == "__main__":
    main()
