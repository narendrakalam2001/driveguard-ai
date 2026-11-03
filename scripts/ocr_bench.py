#!/usr/bin/env python3
"""
ocr_bench.py
=============
Re-reads the plate crops you already labelled with several OCR settings and prints
one accuracy table, so you can see which setting actually helps - without labelling
anything again.

Inputs (both come from `try_plate_detector.py --ocr` + your labelling):
    plate_ocr_review.csv   with the true_text column filled in
    plate_crops/           the crop images named in the CSV's crop_file column

Variants:
    baseline               EasyOCR as in try_plate_detector.py (reads in EasyOCR's own order)
    ordered                same reads, put in reading order (top line first, then left to right)
    ordered+allowlist      ordered, and EasyOCR may only output A-Z / 0-9
    +fix                   ordered+allowlist, then snapped to an Indian plate pattern
    big+fix                like +fix but the crop is enlarged to 400 px width first

Usage:
    python scripts/ocr_bench.py
    python scripts/ocr_bench.py --csv plate_ocr_review.csv --crops-dir plate_crops

Needs EasyOCR (pip install easyocr). Rows whose true_text is blank, '-', or not
plate-like are skipped. Accuracy is on the plates only; with ~40 crops, differences of
a few points are noise.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from scripts.evaluate_plate_ocr import evaluate, is_plate_like  # noqa: E402
from src.plate_text import OCR_ALLOWLIST, correct_plate_text, order_reads, prepare_crop  # noqa: E402

ALLOWLIST = OCR_ALLOWLIST  # same character set the pipeline (PlateRecognizer) uses
VARIANTS = ["baseline", "ordered", "ordered+allowlist", "+fix", "big+fix"]


prepare = prepare_crop  # one implementation for bench and pipeline (grayscale + enlarge to min width)


def run_variant(reader, crop: np.ndarray, variant: str) -> str:
    """Return the text one variant reads from one crop."""
    width = 400 if variant == "big+fix" else 200
    image = prepare(crop, width)
    kwargs = {"allowlist": ALLOWLIST} if variant in ("ordered+allowlist", "+fix", "big+fix") else {}
    reads = reader.readtext(image, **kwargs)
    if not reads:
        return ""
    if variant == "baseline":
        text = "".join(t for _, t, _ in reads)
    else:
        text = "".join(t for t, _ in order_reads(reads))
    text = text.upper().replace(" ", "")
    return correct_plate_text(text) if variant in ("+fix", "big+fix") else text


def bench(rows: List[dict], crops_dir: Path, reader, variants: List[str]) -> Dict[str, dict]:
    """Run every variant over the labelled plate rows and score it."""
    usable = [r for r in rows if (r.get("true_text") or "").strip() not in ("", "-") and is_plate_like(r["true_text"])]
    results: Dict[str, dict] = {}
    for variant in variants:
        scored = []
        for row in usable:
            path = crops_dir / row["crop_file"]
            crop = cv2.imread(str(path)) if path.exists() else None
            if crop is None:
                continue
            scored.append({**row, "ocr_text": run_variant(reader, crop, variant)})
        results[variant] = evaluate(scored)
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare OCR settings on your labelled plate crops.")
    parser.add_argument("--csv", type=Path, default=Path("plate_ocr_review.csv"))
    parser.add_argument("--crops-dir", type=Path, default=Path("plate_crops"))
    args = parser.parse_args()
    if not args.csv.exists() or not args.crops_dir.is_dir():
        sys.exit("Need plate_ocr_review.csv (with true_text filled) and the plate_crops folder in this directory.")
    try:
        import easyocr
    except ImportError:
        sys.exit("pip install easyocr")

    with open(args.csv, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    results = bench(rows, args.crops_dir, easyocr.Reader(["en"], gpu=False), VARIANTS)

    print(f"{'variant':20} {'plates':>6} {'exact':>7} {'char acc':>9} {'valid fmt':>10}")
    for variant in VARIANTS:
        r = results[variant]
        print(f"{variant:20} {r['plates']:>6} {r['exact_rate']:>7.1%} {r['char_accuracy']:>9.1%} {r['format_valid_rate']:>10.1%}")
    print("\nSmall sample: treat gaps of a few points as noise. 'valid fmt' alone can be gamed by +fix; "
          "exact and char accuracy against your true_text are the numbers that count.")


if __name__ == "__main__":
    main()
