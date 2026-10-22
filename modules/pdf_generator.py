"""Executive PDF Intelligence Report Generator.

Compiles complete NLP analytical results (Sentiment, Topic, Extractive Summary,
TF-IDF Keywords, Named Entities, and Linguistic Metrics) into a publication-grade PDF document.
"""

from datetime import datetime
import io
from typing import Any, Dict, List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_pdf_report(
    buffer: io.BytesIO,
    text: str,
    sentiment: str,
    keywords: List[str],
    entities: Dict[str, List[str]],
    summary: str,
    category: str,
    confidence_sent: Optional[float] = None,
    confidence_cat: Optional[float] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Generates a styled, publication-grade PDF report from NLP analytics pipeline results.

    Args:
        buffer: In-memory byte buffer (io.BytesIO) to receive the PDF stream.
        text: Raw source text.
        sentiment: Sentiment classification ('Positive' / 'Negative' / 'Neutral').
        keywords: Top TF-IDF keyphrases.
        entities: Dict mapping entity categories to lists of entity names.
        summary: AI extractive text summary.
        category: News / topic classification label.
        confidence_sent: Optional sentiment model confidence percentage (0-100).
        confidence_cat: Optional category model confidence percentage (0-100).
        stats: Optional dictionary containing word_count, reading_time_sec, etc.
    """
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    story = []

    # Color Palette: Corporate Navy / Slate / Accent
    c_primary = colors.HexColor('#0F172A')    # Slate 900
    c_secondary = colors.HexColor('#1E3A8A')  # Blue 900
    c_accent = colors.HexColor('#2563EB')     # Blue 600
    c_muted = colors.HexColor('#64748B')      # Slate 500
    c_card_bg = colors.HexColor('#F8FAFC')    # Slate 50
    c_border = colors.HexColor('#E2E8F0')     # Slate 200

    # Sentiment dynamic color
    sent_upper = (sentiment or "").upper()
    if 'POS' in sent_upper:
        sent_color = colors.HexColor('#059669')  # Emerald
        sent_bg = colors.HexColor('#ECFDF5')
    elif 'NEG' in sent_upper:
        sent_color = colors.HexColor('#DC2626')  # Rose
        sent_bg = colors.HexColor('#FEF2F2')
    else:
        sent_color = colors.HexColor('#475569')  # Slate
        sent_bg = colors.HexColor('#F1F5F9')

    # Typography Styles
    title_style = ParagraphStyle(
        'RepTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        'RepSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_muted,
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        'RepSection',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'RepBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
    )

    badge_style = ParagraphStyle(
        'RepBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        alignment=1,  # Center
    )

    # 1. Header Section
    generated_at = datetime.now().strftime("%B %d, %Y - %H:%M UTC")
    story.append(Paragraph("NLP Text Intelligence & Analytics Report", title_style))
    story.append(Paragraph(f"Automated Linguistic Audit &bull; Generated: {generated_at} &bull; Pipeline v2.0", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=c_accent, spaceBefore=0, spaceAfter=14))

    # 2. Executive KPI Cards Table
    sent_conf_str = f" ({confidence_sent:.1f}%)" if confidence_sent else ""
    cat_conf_str = f" ({confidence_cat:.1f}%)" if confidence_cat else ""
    words_str = str(stats.get('word_count', len(text.split()))) if stats else str(len(text.split()))
    read_time = f"{stats.get('reading_time_sec', 30)}s" if stats else "N/A"

    kpi_data = [
        [
            Paragraph("<b>CLASSIFIED TOPIC</b>", ParagraphStyle('H', parent=subtitle_style, fontSize=8, spaceAfter=2)),
            Paragraph("<b>SENTIMENT POLARITY</b>", ParagraphStyle('H', parent=subtitle_style, fontSize=8, spaceAfter=2)),
            Paragraph("<b>DOCUMENT LENGTH</b>", ParagraphStyle('H', parent=subtitle_style, fontSize=8, spaceAfter=2)),
            Paragraph("<b>EST. READING TIME</b>", ParagraphStyle('H', parent=subtitle_style, fontSize=8, spaceAfter=2)),
        ],
        [
            Paragraph(f"<font color='{c_secondary.hexval()}'><b>{category.upper()}</b></font>{cat_conf_str}", badge_style),
            Paragraph(f"<font color='{sent_color.hexval()}'><b>{sentiment.upper()}</b></font>{sent_conf_str}", badge_style),
            Paragraph(f"<b>{words_str}</b> words", badge_style),
            Paragraph(f"<b>{read_time}</b>", badge_style),
        ],
    ]

    kpi_table = Table(kpi_data, colWidths=[130, 130, 125, 130])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_card_bg),
        ('BOX', (0, 0), (-1, -1), 1, c_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # 3. AI Extractive Summary Callout
    story.append(Paragraph("1. AI Extractive Summary", section_style))
    summary_p = Paragraph(f"<i>&ldquo;{summary}&rdquo;</i>", body_style)
    summary_box = Table([[summary_p]], colWidths=[515])
    summary_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('LINELEFT', (0, 0), (0, -1), 3.5, c_accent),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(summary_box)
    story.append(Spacer(1, 12))

    # 4. Statistical Key Terms (TF-IDF)
    story.append(Paragraph("2. Salient Keyphrases & Topics (TF-IDF)", section_style))
    if keywords:
        kw_items = ", ".join([f"<b>{k}</b>" for k in keywords])
        kw_content = Paragraph(f"Statistical keyphrases ranked by TF-IDF term weights: {kw_items}", body_style)
    else:
        kw_content = Paragraph("No statistically distinctive keywords extracted.", body_style)
    story.append(kw_content)
    story.append(Spacer(1, 12))

    # 5. Named Entity Recognition (NER)
    story.append(Paragraph("3. Named Entity Recognition (NER)", section_style))
    table_rows = [[
        Paragraph("<b>Entity Mention</b>", body_style),
        Paragraph("<b>Category / Type</b>", body_style),
    ]]

    entity_count = 0
    if isinstance(entities, dict):
        for cat_name, entity_list in entities.items():
            for item in entity_list:
                table_rows.append([
                    Paragraph(f"<b>{str(item)}</b>", body_style),
                    Paragraph(f"<font color='{c_accent.hexval()}'>{str(cat_name)}</font>", body_style),
                ])
                entity_count += 1

    if entity_count > 0:
        ner_table = Table(table_rows, colWidths=[260, 255])
        ner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EFF6FF')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_card_bg]),
        ]))
        story.append(ner_table)
    else:
        story.append(Paragraph("No named entities identified in the analyzed text.", body_style))

    story.append(Spacer(1, 12))

    # 6. Source Text Excerpt
    story.append(Paragraph("4. Analyzed Source Excerpt", section_style))
    clean_sample = text[:600] + ("..." if len(text) > 600 else "")
    story.append(Paragraph(f"<font color='#64748B'>{clean_sample}</font>", body_style))

    # Build the document
    doc.build(story)


if __name__ == "__main__":
    buf = io.BytesIO()
    sample = "Apple Inc. announced record profits in California under Tim Cook. AI technology is accelerating rapidly."
    generate_pdf_report(
        buf,
        text=sample,
        sentiment="Positive",
        keywords=["apple", "profits", "ai technology"],
        entities={"Organization": ["Apple Inc."], "Person": ["Tim Cook"], "Location": ["California"]},
        summary="Apple Inc. announced record profits in California under Tim Cook.",
        category="Tech",
        confidence_sent=95.4,
        confidence_cat=91.2,
        stats={"word_count": 22, "reading_time_sec": 7},
    )
    print("PDF generated successfully, byte size:", len(buf.getvalue()))
