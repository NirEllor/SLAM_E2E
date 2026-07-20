#!/usr/bin/env python3
"""
Render content_model.json to report.docx using python-docx.
"""

import json
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def render_docx(content_model_path, output_path):
    """Build a docx from the content model."""

    with open(content_model_path, 'r') as f:
        model = json.load(f)

    doc = Document()

    # Add table of contents placeholder
    doc.add_heading('Visual Navigation Pipeline Report', 0)
    doc.add_paragraph('[ Table of Contents would be auto-generated in full version ]')
    doc.add_paragraph()

    for block in model:
        block_type = block.get('type')

        if block_type == 'heading1':
            doc.add_heading(block['text'], level=1)

        elif block_type == 'heading2':
            doc.add_heading(block['text'], level=2)

        elif block_type == 'heading3':
            doc.add_heading(block['text'], level=3)

        elif block_type == 'paragraph':
            p = doc.add_paragraph(block['text'])
            if block.get('bold'):
                for run in p.runs:
                    run.bold = True

        elif block_type == 'code_citation':
            p = doc.add_paragraph()
            p.add_run(block['claim']).bold = True
            doc.add_paragraph(
                f"Function: {block['function']} | File: {block['file']} | "
                f"Lines: {block['line_start']}-{block['line_end']}",
                style='List Bullet'
            )

        elif block_type == 'figure':
            # Add figure
            if Path(block['artifact_path']).exists():
                try:
                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(block['artifact_path'], width=Inches(5.5))
                except:
                    doc.add_paragraph(f"[Figure not found: {block['artifact_path']}]")

            # Add caption
            caption_p = doc.add_paragraph(block['caption'], style='List Paragraph')
            for run in caption_p.runs:
                run.font.size = Pt(10)

        elif block_type == 'table':
            if block.get('rows'):
                table = doc.add_table(rows=len(block['rows']), cols=len(block['rows'][0]))
                for i, row_data in enumerate(block['rows']):
                    for j, cell_data in enumerate(row_data):
                        table.rows[i].cells[j].text = str(cell_data)

        elif block_type == 'section_break':
            doc.add_page_break()

    doc.save(output_path)
    print(f"✓ Report saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "--content-model":
        print("Usage: render_docx.py --content-model <content_model.json> --output <output.docx>")
        sys.exit(1)

    content_model_path = sys.argv[2]
    output_path = sys.argv[4] if len(sys.argv) > 4 else "report.docx"

    render_docx(content_model_path, output_path)