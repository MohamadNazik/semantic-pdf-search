"""
Hybrid PDF Extraction (Text + OCR Combined)
============================================
This script handles PDFs that contain BOTH text-based and scanned content.

For each page:
  Step 1: Extract text using PyMuPDF
  Step 2: Detect if the page contains embedded images
  Step 3: If images are found, extract each image and apply OCR to it
  Step 4: Combine the text and OCR results

Libraries: PyMuPDF, pytesseract, Pillow

Prerequisites:
    - See README.md for Tesseract OCR installation instructions.

Usage:
    python hybrid_extraction.py
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import sys


# ── Configuration ──────────────────────────────────────────────────────────────
# Change this path to point to your PDF file (can be mixed text + scanned)
PDF_FILE_PATH = "./data/hybrid_sample.pdf"

# Uncomment and update this line if Tesseract is not in your system PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_images_from_page(doc, page):
    """
    Extract all images from a PDF page and return them as PIL Image objects.

    Parameters:
        doc (fitz.Document):  The opened PDF document.
        page (fitz.Page):     The page to extract images from.

    Returns:
        list: A list of PIL Image objects found on the page.
    """

    images = []

    # Get list of images on this page
    # Each entry contains: (xref, smask, width, height, bpc, colorspace, ...)
    image_list = page.get_images(full=True)

    for img_info in image_list:
        xref = img_info[0]  # Image reference number in the PDF

        try:
            # Extract the image bytes from the PDF
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert bytes to a PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            images.append(pil_image)
        except Exception as e:
            print(f"    [WARNING] Could not extract image (xref={xref}): {e}")

    return images


def ocr_images(images):
    """
    Apply OCR to a list of PIL Image objects and return combined text.

    Parameters:
        images (list): List of PIL Image objects.

    Returns:
        str: Combined OCR text from all images.
    """

    ocr_texts = []

    for i, image in enumerate(images):
        # Convert image to RGB if necessary (OCR works best with RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Run OCR on the image
        text = pytesseract.image_to_string(image, lang="eng").strip()

        if text:
            ocr_texts.append(text)

    return "\n".join(ocr_texts)


def hybrid_extract(pdf_path):
    """
    Extract text from a PDF using a hybrid approach:
      - Extract text from each page using PyMuPDF
      - Detect images on each page
      - Apply OCR to any images found
      - Combine text + OCR results

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of dicts with keys:
              page_num, text, ocr_text, method, image_count
    """

    # Step 1: Verify the file exists
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return []

    # Step 2: Open the PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"[INFO] Opened PDF: {os.path.basename(pdf_path)}")
    print(f"[INFO] Total pages: {total_pages}")
    print("=" * 60)

    results = []

    # Step 3: Process each page
    for page_num in range(total_pages):
        page = doc[page_num]
        display_num = page_num + 1  # 1-indexed for readability

        # ── Step 3a: Extract text using PyMuPDF ──
        text = page.get_text().strip()

        # ── Step 3b: Detect images on this page ──
        image_list = page.get_images(full=True)
        image_count = len(image_list)

        # ── Step 3c: If images found, extract them and apply OCR ──
        ocr_text = ""
        if image_count > 0:
            print(f"  Page {display_num}: Found {image_count} image(s) → applying OCR...")
            images = extract_images_from_page(doc, page)
            ocr_text = ocr_images(images)

        # ── Step 3d: Determine the method used ──
        if text and image_count > 0:
            method = "Text Extraction + OCR"
        elif text and image_count == 0:
            method = "Text Extraction"
        elif not text and image_count > 0:
            method = "OCR"
        else:
            method = "No Content"

        results.append({
            "page_num": display_num,
            "text": text,
            "ocr_text": ocr_text,
            "method": method,
            "image_count": image_count,
        })

    # Step 4: Close the document
    doc.close()

    return results


def main():
    """Main function to demonstrate the hybrid extraction approach."""

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       HYBRID PDF EXTRACTION (Text + OCR Combined)        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Run hybrid extraction
    results = hybrid_extract(PDF_FILE_PATH)

    if not results:
        print("[WARNING] No results. Check file path and dependencies.")
        sys.exit(1)

    # ── Summary of methods used per page ──
    print("\n--- Method Summary ---")
    for r in results:
        has_content = "✓" if (r["text"] or r["ocr_text"]) else "✗"
        images_info = f"  ({r['image_count']} image(s))" if r["image_count"] > 0 else ""
        print(f"  Page {r['page_num']} → {r['method']}{images_info}  [{has_content}]")

    # ── Display extracted content per page ──
    for r in results:
        print(f"\n--- Page {r['page_num']} [{r['method']}] ---")

        # Show extracted text (from PyMuPDF)
        if r["text"]:
            print("[Text Extraction Result]")
            preview = r["text"][:500]
            print(preview)
            if len(r["text"]) > 500:
                print(f"  ... [truncated, {len(r['text'])} chars total]")

        # Show OCR text (from images)
        if r["ocr_text"]:
            print(f"\n[OCR Result from {r['image_count']} image(s)]")
            preview = r["ocr_text"][:500]
            print(preview)
            if len(r["ocr_text"]) > 500:
                print(f"  ... [truncated, {len(r['ocr_text'])} chars total]")

        # No content at all
        if not r["text"] and not r["ocr_text"]:
            print("  [No text or images found on this page]")

    print()
    print("=" * 60)
    print(f"[DONE] Hybrid extraction completed for {len(results)} page(s).")
    print()


if __name__ == "__main__":
    main()
