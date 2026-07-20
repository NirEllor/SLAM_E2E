#!/usr/bin/env python3
"""
Render content_model.json to report.pdf using reportlab.
"""

import json
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def render_pdf(content_model_path, output_path):
    """Build a PDF from the content model using reportlab."""

    with open(content_model_path, 'r') as f:
        model = json.load(f)

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER
    )

    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=10,
        spaceBefore=10
    )

    for i, block in enumerate(model):
        block_type = block.get('type')

        if block_type == 'heading1':
            story.append(Paragraph(block['text'], heading1_style))
            story.append(Spacer(1, 0.2*inch))

        elif block_type == 'heading2':
            h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceAfter=8)
            story.append(Paragraph(block['text'], h2_style))
            story.append(Spacer(1, 0.1*inch))

        elif block_type == 'paragraph':
            para_style = styles['BodyText']
            story.append(Paragraph(block['text'], para_style))
            story.append(Spacer(1, 0.1*inch))

        elif block_type == 'code_citation':
            claim_style = ParagraphStyle('Claim', parent=styles['BodyText'], fontSize=10, textColor=colors.black)
            story.append(Paragraph(f"<b>{block['claim']}</b>", claim_style))
            story.append(Paragraph(
                f"<i>File: {block['file']} (lines {block['line_start']}-{block['line_end']})</i>",
                styles['BodyText']
            ))
            story.append(Spacer(1, 0.1*inch))

        elif block_type == 'figure':
            # Add figure if it exists
            if Path(block['artifact_path']).exists():
                try:
                    img = Image(block['artifact_path'], width=5*inch, height=3.5*inch)
                    story.append(img)
                except:
                    story.append(Paragraph(f"[Figure not found: {block['artifact_path']}]", styles['Normal']))

            # Add caption
            caption_style = ParagraphStyle(
                'Caption',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                alignment=TA_LEFT
            )
            story.append(Paragraph(f"<i>{block['caption']}</i>", caption_style))
            story.append(Spacer(1, 0.15*inch))

        elif block_type == 'section_break':
            story.append(PageBreak())

    # Build PDF
    doc.build(story)
    print(f"✓ PDF saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "--content-model":
        print("Usage: render_pdf.py --content-model <content_model.json> --output <output.pdf>")
        sys.exit(1)

    content_model_path = sys.argv[2]
    output_path = sys.argv[4] if len(sys.argv) > 4 else "report.pdf"

    render_pdf(content_model_path, output_path)