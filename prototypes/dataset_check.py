"""
Dataset Verification
=====================
This script scans a folder for PDF files and verifies that the
dataset is ready for use in the research project.

No processing is performed — this script only lists and counts files.

Usage:
    python dataset_check.py
"""

import os
import sys


# ── Configuration ──────────────────────────────────────────────────────────────
# Change this path to point to your dataset folder
DATASET_FOLDER = "./data/"


def check_dataset(folder_path):
    """
    Scan a folder for PDF files and report what is found.

    Parameters:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        list: A list of PDF file names found in the folder.
    """

    # Step 1: Verify the folder exists
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return []

    if not os.path.isdir(folder_path):
        print(f"[ERROR] Path is not a directory: {folder_path}")
        return []

    # Step 2: List all files in the folder
    all_files = os.listdir(folder_path)

    # Step 3: Filter for PDF files only (case-insensitive)
    pdf_files = [
        f for f in all_files
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder_path, f))
    ]

    # Step 4: Sort alphabetically for consistent display
    pdf_files.sort()

    return pdf_files


def main():
    """Main function to verify the PDF dataset."""

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       DATASET VERIFICATION                               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Run the dataset check
    pdf_files = check_dataset(DATASET_FOLDER)

    if not pdf_files:
        print(f"[WARNING] No PDF files found in: {DATASET_FOLDER}")
        print("[HINT] Place your PDF documents in the folder and try again.")
        sys.exit(1)

    # ── Display results ──

    # Step 5: Print the list of PDF files
    print(f"Dataset folder : {os.path.abspath(DATASET_FOLDER)}")
    print(f"Total PDF files: {len(pdf_files)}")
    print()
    print("--- PDF Files Found ---")
    for i, filename in enumerate(pdf_files, start=1):
        # Get file size for additional info
        file_path = os.path.join(DATASET_FOLDER, filename)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  {i:3d}. {filename}  ({size_kb:.1f} KB)")

    # Step 6: Confirmation message
    print()
    print("=" * 60)
    print(f"[DONE] Dataset verified: {len(pdf_files)} PDF file(s) ready.")
    print()


if __name__ == "__main__":
    main()
