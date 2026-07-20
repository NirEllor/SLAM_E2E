#!/usr/bin/env python3
"""
Convert report.docx to report.pdf and problem.docx to problem.pdf.
Uses LibreOffice if available, falls back to docx2pdf or similar.
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path("/mnt/c/Users/Nir/PycharmProjects/VAN_ex") if sys.platform != "win32" else Path("C:/Users/Nir/PycharmProjects/VAN_ex")

print("Converting DOCX files to PDF...")

# Try LibreOffice headless conversion
for docx_file in [REPO_ROOT / "report.docx", REPO_ROOT / "problem.docx"]:
    pdf_file = docx_file.with_suffix(".pdf")

    if not docx_file.exists():
        print(f"⚠ {docx_file.name} not found")
        continue

    print(f"\nConverting {docx_file.name}...")

    # Try libreoffice (Linux under WSL)
    try:
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(docx_file.parent), str(docx_file)],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0 and pdf_file.exists():
            print(f"✓ {pdf_file.name} (via LibreOffice)")
            continue
    except FileNotFoundError:
        pass

    # Try docx2pdf
    try:
        from docx2pdf import convert
        convert(str(docx_file), str(pdf_file))
        if pdf_file.exists():
            print(f"✓ {pdf_file.name} (via docx2pdf)")
            continue
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠ docx2pdf failed: {e}")

    # Try win32com (Windows only)
    if sys.platform == "win32":
        try:
            from win32com.client import Dispatch
            word = Dispatch("Word.Application")
            doc = word.Documents.Open(str(docx_file.absolute()))
            doc.SaveAs(str(pdf_file.absolute()), FileFormat=17)  # 17 = wdFormatPDF
            doc.Close()
            word.Quit()
            if pdf_file.exists():
                print(f"✓ {pdf_file.name} (via Word COM)")
                continue
        except Exception as e:
            print(f"⚠ Word COM conversion failed: {e}")

    print(f"⚠ {docx_file.name} → {pdf_file.name} conversion failed (no suitable converter)")

print("\n✓ Conversion complete (PDFs may not have been created; DOCX files are available)")
