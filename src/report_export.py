"""
report_export.py
================
File exports for the Reports screen: CSV and Excel violation exports, a one-page PDF violation
notice, and a PDF summary report for supervisors.

CSV/Excel cells that start with ``=``, ``+``, ``-`` or ``@`` are prefixed with an apostrophe so a
hostile OCR read or location name cannot run as a spreadsheet formula (CSV injection).
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.config import DISPLAY_UTC_OFFSET_MINUTES, EVIDENCE_FRAMES_DIR, get_logger

logger = get_logger(__name__)

EXPORT_COLUMNS: List[str] = [
    "id", "timestamp", "plate_number", "plate_confidence", "vehicle_type", "detected_speed",
    "speed_limit", "excess_speed", "severity", "status", "camera_id", "location", "notice_generated",
]
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _safe_cell(value: Any) -> Any:
    """Neutralise spreadsheet-formula injection in text cells."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def _rows(violations: Iterable[Dict[str, Any]]) -> List[List[Any]]:
    return [[_safe_cell(v.get(col)) for col in EXPORT_COLUMNS] for v in violations]


def violations_to_csv(violations: Sequence[Dict[str, Any]]) -> bytes:
    """Serialise violations to UTF-8 CSV (with BOM so Excel opens it correctly).

    Args:
        violations: Violation row dicts.

    Returns:
        CSV file bytes.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(EXPORT_COLUMNS)
    writer.writerows(_rows(violations))
    return b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")


def violations_to_xlsx(violations: Sequence[Dict[str, Any]]) -> bytes:
    """Serialise violations to a formatted .xlsx workbook.

    Args:
        violations: Violation row dicts.

    Returns:
        XLSX file bytes.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "Violations"
    ws.append(EXPORT_COLUMNS)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1565C0")
    for row in _rows(violations):
        ws.append(row)
    ws.freeze_panes = "A2"
    for idx, col in enumerate(EXPORT_COLUMNS, start=1):
        width = max([len(col)] + [len(str(r[idx - 1])) for r in _rows(violations)[:200] if r[idx - 1] is not None])
        ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = min(width + 2, 48)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def _pdf_safe(text: Any) -> str:
    """Make text printable with the built-in PDF fonts (Latin-1): INR sign and exotic characters replaced."""
    s = str(text if text is not None else "")
    s = s.replace("\u20b9", "Rs.").replace("\u2014", "-").replace("\u2013", "-").replace("\u2018", "'").replace("\u2019", "'")
    return s.encode("latin-1", "replace").decode("latin-1")


def _para_markup(text: str) -> str:
    """Escape text for a reportlab Paragraph (which parses a small HTML subset), keeping line breaks."""
    safe = _pdf_safe(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return safe.replace("\n", "<br/>")


def _local_str(iso: Optional[str]) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        from datetime import timedelta

        return (dt + timedelta(minutes=DISPLAY_UTC_OFFSET_MINUTES)).strftime("%d %b %Y %H:%M:%S IST")
    except ValueError:
        return iso


def _evidence_image_path(violation: Dict[str, Any]) -> Optional[Path]:
    raw = violation.get("evidence_frame_path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = EVIDENCE_FRAMES_DIR / path.name
    try:
        path.resolve().relative_to(EVIDENCE_FRAMES_DIR.resolve())
    except ValueError:
        return None  # never embed files from outside the evidence directory
    return path if path.is_file() else None


def notice_to_pdf(violation: Dict[str, Any]) -> bytes:
    """Render one violation as a printable PDF notice (details table, notice text, evidence frame).

    Args:
        violation: A violation row dict; ``notice_text`` is included when present.

    Returns:
        PDF file bytes.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Traffic Violation Notice", author="DriveGuard AI")
    story: list = [Paragraph("Traffic Violation Notice", styles["Title"]), Paragraph(f"Reference: {_pdf_safe(violation.get('id'))}", body), Spacer(1, 6 * mm)]

    details = [
        ["Plate", _pdf_safe(violation.get("plate_number") or "UNREADABLE")],
        ["Vehicle type", _pdf_safe(violation.get("vehicle_type"))],
        ["Detected speed", f"{violation.get('detected_speed')} km/h"],
        ["Speed limit", f"{violation.get('speed_limit')} km/h"],
        ["Excess", f"{violation.get('excess_speed')} km/h"],
        ["Severity", _pdf_safe(violation.get("severity"))],
        ["Location / camera", _pdf_safe(f"{violation.get('location')} ({violation.get('camera_id')})")],
        ["Time", _local_str(violation.get("timestamp"))],
        ["Status", _pdf_safe(violation.get("status"))],
    ]
    table = Table(details, colWidths=[40 * mm, 120 * mm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9FB3C8")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8EEF5")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [table, Spacer(1, 6 * mm)]

    notice = violation.get("notice_text")
    if notice:
        for para in str(notice).split("\n\n"):
            story.append(Paragraph(_para_markup(para), body))
            story.append(Spacer(1, 3 * mm))
    else:
        story.append(Paragraph("No notice text has been generated for this violation yet.", body))

    img = _evidence_image_path(violation)
    if img is not None:
        try:
            story += [Spacer(1, 4 * mm), Paragraph("Evidence frame", styles["Heading3"]), Image(str(img), width=150 * mm, height=84 * mm, kind="proportional")]
        except Exception:  # noqa: BLE001 - a corrupt image must not block the notice
            logger.warning("Could not embed evidence image %s", img)
    doc.build(story)
    return out.getvalue()


def summary_report_pdf(
    summary: Dict[str, Any], detailed: Dict[str, Any], insight_text: str, top_violations: Sequence[Dict[str, Any]],
    title: str = "DriveGuard AI - Enforcement Summary Report",
) -> bytes:
    """Render the supervisor summary: KPIs, hourly chart, top plates, recent severe cases, insights.

    Args:
        summary: Output of ``AlertSystem.get_analytics_summary``.
        detailed: Output of ``AlertSystem.get_detailed_analytics``.
        insight_text: Insight paragraph to print under "Recommendations".
        top_violations: Most serious recent violations to list.
        title: Report title.

    Returns:
        PDF file bytes.
    """
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=title, author="DriveGuard AI")
    now_local = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    story: list = [Paragraph(_pdf_safe(title), styles["Title"]), Paragraph(f"Generated {now_local}", body), Spacer(1, 5 * mm)]

    story.append(Paragraph("Executive summary", styles["Heading2"]))
    kpis = [
        ["Total violations", summary.get("total_violations", 0), "Today", summary.get("violations_today", 0)],
        ["This month", summary.get("violations_this_month", 0), "Average speed (violations)", f"{summary.get('average_speed_kmph', 0)} km/h"],
    ]
    kt = Table(kpis, colWidths=[40 * mm, 30 * mm, 55 * mm, 40 * mm])
    kt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9FB3C8")), ("FONTSIZE", (0, 0), (-1, -1), 10),
                            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8EEF5")), ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#E8EEF5"))]))
    story += [kt, Spacer(1, 5 * mm)]

    hourly = detailed.get("hourly") or [0] * 24
    if any(hourly):
        story.append(Paragraph("Violations by hour of day (local time)", styles["Heading2"]))
        d = Drawing(170 * mm, 60 * mm)
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = 10, 12, 450, 130
        chart.data = [hourly]
        chart.categoryAxis.categoryNames = [str(h) for h in range(24)]
        chart.bars[0].fillColor = colors.HexColor("#1565C0")
        chart.valueAxis.valueMin = 0
        d.add(chart)
        story += [d, Spacer(1, 3 * mm)]

    plates = summary.get("top_offending_plates") or []
    if plates:
        story.append(Paragraph("Top offending plates", styles["Heading2"]))
        pt = Table([["Plate", "Violations"]] + [[_pdf_safe(p["plate_number"]), p["violation_count"]] for p in plates], colWidths=[70 * mm, 30 * mm])
        pt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9FB3C8")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        story += [pt, Spacer(1, 5 * mm)]

    if top_violations:
        story.append(Paragraph("Most serious recent violations", styles["Heading2"]))
        rows = [["Time", "Plate", "Vehicle", "Speed / limit", "Severity"]] + [
            [_local_str(v.get("timestamp")), _pdf_safe(v.get("plate_number") or "UNREADABLE"), _pdf_safe(v.get("vehicle_type")),
             f"{v.get('detected_speed')} / {v.get('speed_limit')}", _pdf_safe(v.get("severity"))]
            for v in top_violations
        ]
        vt = Table(rows, colWidths=[42 * mm, 34 * mm, 26 * mm, 32 * mm, 26 * mm])
        vt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9FB3C8")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story += [vt, Spacer(1, 5 * mm)]

    story += [Paragraph("Recommendations", styles["Heading2"]), Paragraph(_para_markup(insight_text), body)]
    doc.build(story)
    return out.getvalue()
