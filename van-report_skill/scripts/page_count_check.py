#!/usr/bin/env python3
"""
Page count checker: Count pages in a PDF and report if > 20.
"""

import sys
from pathlib import Path
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

def count_pages(pdf_path):
    """Count pages in a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        return num_pages
    except Exception as e:
        print(f"[ERROR] Could not count pages: {e}", file=sys.stderr)
        return -1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: page_count_check.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"[ERROR] PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    num_pages = count_pages(pdf_path)

    if num_pages < 0:
        sys.exit(1)

    print(f"Pages: {num_pages}")

    if num_pages > 20:
        print(f"⚠ WARNING: Report exceeds 20-page limit by {num_pages - 20} page(s)")
        sys.exit(1)
    else:
        print(f"✓ Report within 20-page limit")
        sys.exit(0)