"""Tests for src/report_export.py - CSV/XLSX/PDF output and CSV-injection protection."""

import io

from openpyxl import load_workbook

from src.report_export import (
    EXPORT_COLUMNS,
    notice_to_pdf,
    summary_report_pdf,
    violations_to_csv,
    violations_to_xlsx,
)

ROW = {"id": "v1", "timestamp": "2026-10-19T10:00:00+00:00", "plate_number": "MH12AB1234", "vehicle_type": "Car",
       "detected_speed": 80, "speed_limit": 60, "excess_speed": 20, "severity": "VIOLATION", "status": "PENDING",
       "camera_id": "CAM-1", "location": "NH-48", "notice_generated": 0}


def test_csv_has_header_bom_and_rows():
    data = violations_to_csv([ROW])
    assert data.startswith(b"\xef\xbb\xbf")
    lines = data.decode("utf-8-sig").splitlines()
    assert lines[0].split(",") == EXPORT_COLUMNS
    assert "MH12AB1234" in lines[1]


def test_csv_neutralises_formula_injection():
    evil = {**ROW, "plate_number": "=HYPERLINK(\"http://evil\")", "location": "+cmd|' /C calc'!A0"}
    text = violations_to_csv([evil]).decode("utf-8-sig")
    assert "'=HYPERLINK" in text and "'+cmd" in text


def test_xlsx_round_trips_and_neutralises_formulas():
    wb = load_workbook(io.BytesIO(violations_to_xlsx([{**ROW, "plate_number": "=1+1"}])))
    ws = wb["Violations"]
    assert [c.value for c in ws[1]] == EXPORT_COLUMNS
    assert ws.cell(row=2, column=EXPORT_COLUMNS.index("plate_number") + 1).value == "'=1+1"


def test_notice_pdf_is_valid_and_handles_rupee_sign_and_markup():
    pdf = notice_to_pdf({**ROW, "notice_text": "Fine \u20b91,000 <b>bold</b> & more\n\nSecond paragraph"})
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000


def test_notice_pdf_without_notice_text_still_renders():
    assert notice_to_pdf(ROW).startswith(b"%PDF")


def test_summary_pdf_renders_with_and_without_data():
    empty = summary_report_pdf({"total_violations": 0}, {}, "Not enough data", [])
    full = summary_report_pdf(
        {"total_violations": 3, "violations_today": 1, "top_offending_plates": [{"plate_number": "MH12AB1234", "violation_count": 2}]},
        {"hourly": [0] * 17 + [3] + [0] * 6}, "Peak at 5 PM.", [ROW],
    )
    assert empty.startswith(b"%PDF") and full.startswith(b"%PDF") and len(full) > len(empty)
