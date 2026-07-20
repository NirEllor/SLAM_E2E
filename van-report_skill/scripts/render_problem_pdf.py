#!/usr/bin/env python3
"""
Render problem.pdf from TRUE_GAP/CUT-FOR-SPACE/degraded-tooling Ledger entries.
"""

import json
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

def render_problem_pdf(ledger_path, output_path):
    """Build a problem.pdf from Ledger entries with status != VERIFIED."""

    with open(ledger_path, 'r') as f:
        ledger = json.load(f)

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#d32f2f'),
    )
    story.append(Paragraph("Unverified & Missing Requirements", title_style))
    story.append(Spacer(1, 0.3*inch))

    # Intro
    story.append(Paragraph(
        "This document lists all requirements from the assignment specification "
        "that could NOT be verified, reproduced, or generated from the repository. "
        "Each item includes the exact search commands attempted and the reason for failure.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 0.2*inch))

    # Group by status
    true_gaps = [e for e in ledger if isinstance(e, dict) and e.get('status') == 'TRUE_GAP']
    cut_for_space = [e for e in ledger if isinstance(e, dict) and e.get('status') == 'CUT_FOR_SPACE']
    degraded_tooling = [e for e in ledger if isinstance(e, dict) and 'degraded' in e.get('status', '')]

    # TRUE_GAP section
    if true_gaps:
        story.append(Paragraph("Unverifiable Requirements (TRUE_GAP)", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))

        for entry in true_gaps:
            claim = entry.get('claim', entry.get('id', 'Unknown'))
            reason = entry.get('reason', 'Unknown reason')
            story.append(Paragraph(f"<b>{claim}</b>", styles['Normal']))
            story.append(Paragraph(f"Reason: {reason}", styles['Normal']))

            if entry.get('attempted_search'):
                story.append(Paragraph("Search commands attempted:", styles['Normal']))
                for cmd in entry['attempted_search']:
                    story.append(Paragraph(f"  <code>{cmd}</code>", styles['Normal']))

            story.append(Spacer(1, 0.1*inch))

    # CUT_FOR_SPACE section
    if cut_for_space:
        story.append(PageBreak())
        story.append(Paragraph("Content Cut for Space (20-page limit)", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))

        for entry in cut_for_space:
            claim = entry.get('claim', entry.get('id', 'Unknown'))
            story.append(Paragraph(f"<b>{claim}</b>", styles['Normal']))

            if entry.get('artifact_path'):
                story.append(Paragraph(f"Artifact: {entry['artifact_path']}", styles['Normal']))

            story.append(Spacer(1, 0.1*inch))

    # Degraded tooling section
    if degraded_tooling:
        story.append(PageBreak())
        story.append(Paragraph("Tooling Issues", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))

        for entry in degraded_tooling:
            story.append(Paragraph(entry.get('reason', 'Unknown tooling issue'), styles['Normal']))
            story.append(Spacer(1, 0.1*inch))

    doc.build(story)
    print(f"✓ Problem.pdf saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "--ledger":
        print("Usage: render_problem_pdf.py --ledger <evidence_ledger.json> --output <problem.pdf>")
        sys.exit(1)

    ledger_path = sys.argv[2]
    output_path = sys.argv[4] if len(sys.argv) > 4 else "problem.pdf"

    render_problem_pdf(ledger_path, output_path)