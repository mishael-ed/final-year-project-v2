from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import APP_TITLE, REPORT_DIR


# ──────────────────────────────────────────────────────────────────────────────
# CSV
# ──────────────────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, name: str) -> Path:
    path = REPORT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# PDF
# ──────────────────────────────────────────────────────────────────────────────

_HEADER_COLOR   = colors.HexColor("#2b1055")
_HIGH_COLOR     = colors.HexColor("#f8d7da")
_MEDIUM_COLOR   = colors.HexColor("#fff3cd")
_LOW_COLOR      = colors.HexColor("#d1e7dd")
_ALT_ROW_COLOR  = colors.HexColor("#f5f5f5")


def export_pdf(df: pd.DataFrame, name: str) -> Path:
    path     = REPORT_DIR / f"{name}.pdf"
    styles   = getSampleStyleSheet()
    title_st = ParagraphStyle(
        "title", parent=styles["Heading1"],
        textColor=colors.HexColor("#2b1055"), fontSize=14, spaceAfter=6,
    )
    sub_st   = ParagraphStyle(
        "sub", parent=styles["Normal"],
        textColor=colors.grey, fontSize=9, spaceAfter=12,
    )

    preview = df.copy().astype(str)
    data    = [preview.columns.tolist()] + preview.values.tolist()

    # Colour-code rows by risk tier
    risk_col_idx = preview.columns.tolist().index("Risk Level") \
        if "Risk Level" in preview.columns else -1

    table = Table(data, repeatRows=1)

    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0), _HEADER_COLOR),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 7),
        ("GRID",        (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _ALT_ROW_COLOR]),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    if risk_col_idx >= 0:
        for row_idx, row in enumerate(preview.values.tolist(), start=1):
            tier = row[risk_col_idx]
            if tier == "High":
                style_cmds.append(("BACKGROUND", (risk_col_idx, row_idx), (risk_col_idx, row_idx), _HIGH_COLOR))
            elif tier == "Medium":
                style_cmds.append(("BACKGROUND", (risk_col_idx, row_idx), (risk_col_idx, row_idx), _MEDIUM_COLOR))
            elif tier == "Low":
                style_cmds.append(("BACKGROUND", (risk_col_idx, row_idx), (risk_col_idx, row_idx), _LOW_COLOR))

    table.setStyle(TableStyle(style_cmds))

    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=2 * cm, bottomMargin=1.5 * cm,
    )
    story = [
        Paragraph(APP_TITLE, title_st),
        Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  ·  {len(df)} student records", sub_st),
        Spacer(1, 0.3 * cm),
        table,
    ]
    doc.build(story)
    return path
