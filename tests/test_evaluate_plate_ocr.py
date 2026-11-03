"""Tests for scripts/evaluate_plate_ocr.py."""

import csv
import subprocess
import sys
from pathlib import Path

from scripts.evaluate_plate_ocr import evaluate, is_plate_like, levenshtein, normalize

ROOT = Path(__file__).resolve().parent.parent


def test_normalize_ignores_case_spaces_and_symbols():
    assert normalize("mh 12-ab 1234") == "MH12AB1234"
    assert normalize(None) == ""


def test_levenshtein_basics():
    assert levenshtein("KA01AB1234", "KA01AB1234") == 0
    assert levenshtein("KA01AB1234", "KA01AB1284") == 1
    assert levenshtein("", "ABC") == 3


def _rows():
    return [
        {"image": "a.jpg", "box": "0", "ocr_text": "MH12AB1234", "true_text": "MH 12 AB 1234"},   # exact
        {"image": "b.jpg", "box": "0", "ocr_text": "KA01MF5618", "true_text": "KA01MF5678"},     # 1 edit (1 vs 7)
        {"image": "c.jpg", "box": "0", "ocr_text": "XYZ", "true_text": "-"},                       # not a plate
        {"image": "d.jpg", "box": "0", "ocr_text": "DL8CAZ4521", "true_text": ""},                 # unlabeled
    ]


def test_evaluate_counts_and_rates():
    r = evaluate(_rows())
    assert (r["labeled"], r["unlabeled"], r["not_a_plate"], r["plates"]) == (3, 1, 1, 2)
    assert r["exact"] == 1 and r["exact_rate"] == 0.5
    assert abs(r["char_accuracy"] - (1.0 + 0.9) / 2) < 1e-9
    assert r["format_valid_rate"] == 1.0
    assert r["mismatches"][0]["image"] == "b.jpg" and r["mismatches"][0]["edits"] == 1


def test_cli_prints_summary_and_handles_unlabelled_csv(tmp_path):
    path = tmp_path / "r.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "box", "ocr_text", "true_text"])
        w.writeheader()
        w.writerows(_rows())
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_plate_ocr.py"), "--csv", str(path)],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "Exact-match accuracy:   1/2 = 50.0%" in out.stdout

    empty = tmp_path / "e.csv"
    empty.write_text("image,box,ocr_text,true_text\na.jpg,0,MH12AB1234,\n", encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_plate_ocr.py"), "--csv", str(empty)],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "fill the true_text column" in out.stderr


def test_is_plate_like():
    assert is_plate_like("MH 12 AB 1234") and is_plate_like("1993") and is_plate_like("TN-23-L-4547")
    assert not is_plate_like("GRILLE") and not is_plate_like("OTHER OBJECT") and not is_plate_like("4H")
    assert not is_plate_like("SHISHE PR LIKHA MOBILE NUMBER HE")


def test_descriptions_instead_of_dash_are_excluded_and_reported():
    rows = _rows() + [
        {"image": "e.jpg", "box": "1", "ocr_text": "JO", "true_text": "VEHICLE KE BACK INDICATOR"},
        {"image": "f.jpg", "box": "0", "ocr_text": "", "true_text": "GRILLE"},
    ]
    r = evaluate(rows)
    assert r["plates"] == 2 and len(r["suspicious"]) == 2
    assert {s["true_text"] for s in r["suspicious"]} == {"VEHICLE KE BACK INDICATOR", "GRILLE"}
    assert r["exact_rate"] == 0.5  # unchanged by the excluded rows
