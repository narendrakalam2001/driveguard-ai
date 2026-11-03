#!/usr/bin/env python3
"""
evaluate_plate_ocr.py
======================
Turns the review CSV written by ``try_plate_detector.py --ocr`` into real
plate-text accuracy numbers, once you have typed the true plate for each crop
into the ``true_text`` column.

Conventions for ``true_text``:
  * the plate as you read it on the crop, e.g. MH12AB1234 (spaces/dashes/case ignored)
  * ``-``  the crop is NOT a plate (the detector fired on something else)
  * blank  not labelled yet (skipped)

Usage:
    python scripts/evaluate_plate_ocr.py --csv plate_ocr_review.csv

What the numbers mean: they cover only the plates the DETECTOR found. Plates it
missed are not in the CSV, so the detector's recall (see plate_model_card.json)
still applies on top of this.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List

INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$")


def normalize(text: str) -> str:
    """Uppercase and keep only letters and digits."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def is_plate_like(text: str) -> bool:
    """A real (or partly visible) plate has 3-11 letters/digits and at least one digit."""
    cleaned = normalize(text)
    return 3 <= len(cleaned) <= 11 and any(ch.isdigit() for ch in cleaned)


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def evaluate(rows: List[dict]) -> Dict[str, object]:
    """Compute accuracy metrics from review-CSV rows.

    Args:
        rows: Dicts with at least ``ocr_text`` and ``true_text``.

    Returns:
        Metrics dict: labeled, unlabeled, not_a_plate, plates, exact, exact_rate,
        char_accuracy, format_valid_rate and a list of mismatches.
    """
    labeled = [r for r in rows if (r.get("true_text") or "").strip()]
    not_plate = [r for r in labeled if r["true_text"].strip() == "-"]
    candidates = [r for r in labeled if r["true_text"].strip() != "-"]
    plates = [r for r in candidates if is_plate_like(r["true_text"])]
    suspicious = [r for r in candidates if not is_plate_like(r["true_text"])]

    exact = 0
    char_scores: List[float] = []
    valid = 0
    mismatches = []
    for r in plates:
        truth, pred = normalize(r["true_text"]), normalize(r.get("ocr_text", ""))
        if not truth:
            continue
        dist = levenshtein(truth, pred)
        char_scores.append(max(0.0, 1 - dist / len(truth)))
        valid += bool(INDIAN_PLATE_RE.match(pred))
        if pred == truth:
            exact += 1
        else:
            mismatches.append({"image": r.get("image"), "box": r.get("box"), "true": truth, "ocr": pred, "edits": dist})

    n = len(char_scores)
    return {
        "labeled": len(labeled),
        "unlabeled": len(rows) - len(labeled),
        "not_a_plate": len(not_plate),
        "suspicious": [{"image": r.get("image"), "box": r.get("box"), "true_text": r["true_text"].strip()} for r in suspicious],
        "plates": n,
        "exact": exact,
        "exact_rate": exact / n if n else 0.0,
        "char_accuracy": sum(char_scores) / n if n else 0.0,
        "format_valid_rate": valid / n if n else 0.0,
        "mismatches": sorted(mismatches, key=lambda m: -m["edits"]),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plate-text accuracy from the filled review CSV.")
    parser.add_argument("--csv", type=Path, default=Path("plate_ocr_review.csv"))
    parser.add_argument("--show", type=int, default=10, help="How many worst mismatches to print.")
    args = parser.parse_args()
    if not args.csv.exists():
        sys.exit(f"{args.csv} not found (create it with: python scripts/try_plate_detector.py --ocr)")

    with open(args.csv, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = evaluate(rows)
    if result["plates"] == 0 and not result["suspicious"]:
        sys.exit("No labelled plates yet - fill the true_text column first.")
    if result["plates"] == 0:
        sys.exit("Every labelled row looks like a non-plate - use '-' for those and real plate text for plates.")

    print(f"Rows: {len(rows)} | labelled: {result['labeled']} | still blank: {result['unlabeled']} "
          f"| marked '-' (not a plate): {result['not_a_plate']}")
    if result["suspicious"]:
        print(f"\nWARNING: {len(result['suspicious'])} row(s) have a true_text that does not look like a plate "
              "(a real plate has digits and is at most 11 characters).")
        print("  If the crop is not a plate, write just '-'. These rows are EXCLUDED from the accuracy below:")
        for row in result["suspicious"][:8]:
            print(f"    {row['image']} box {row['box']}: '{row['true_text']}'")
        if len(result["suspicious"]) > 8:
            print(f"    ... and {len(result['suspicious']) - 8} more")
        print()
    print(f"Plates evaluated: {result['plates']}")
    print(f"Exact-match accuracy:   {result['exact']}/{result['plates']} = {result['exact_rate']:.1%}")
    print(f"Character accuracy:     {result['char_accuracy']:.1%}")
    print(f"OCR output is a valid Indian plate format: {result['format_valid_rate']:.1%}")
    if result["mismatches"]:
        print(f"\nWorst mismatches (up to {args.show}):")
        for m in result["mismatches"][: args.show]:
            print(f"  {m['image']} box {m['box']}: true {m['true']} | ocr {m['ocr']} | {m['edits']} edit(s)")
    if result["plates"] < 100:
        print(f"\nNOTE: only {result['plates']} plates - treat these percentages as rough (+/- many points).")
    detected_only = "Covers plates the detector found only; detector recall applies on top."
    print(detected_only)


if __name__ == "__main__":
    main()
