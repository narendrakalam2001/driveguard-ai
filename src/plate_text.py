"""
plate_text.py
==============
Light-weight helpers (no heavy imports) that turn raw OCR output into an Indian
number plate string:

* :func:`order_reads` puts OCR text boxes into reading order (top line first, then
  left to right). EasyOCR returns boxes in detection order, which scrambles
  two-line plates ("2117JH05AW" instead of "JH05AW2117").
* :func:`correct_plate_text` finds the part of the text that fits an Indian plate
  layout (``AA 00 AA 0000`` and its 9/11-character variants, plus the Delhi
  ``DL 8C AZ 4521`` form), fixes look-alike characters (O/0, I/1, B/8 ...) and
  ignores extra text such as the "IND" mark.
* :func:`prepare_crop`, :func:`read_plate_text` and :func:`is_valid_plate` are the
  shared "big+fix" OCR flow used by both ``PlateRecognizer`` and ``scripts/ocr_bench.py``,
  so the benchmark measures exactly what the pipeline runs.
"""

from __future__ import annotations

import re
import statistics
from typing import List, Optional, Sequence, Tuple

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UA", "UK", "UP", "WB",
}

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# 9 / 10 / 11 character layouts: state(2 letters) district(2 digits) series(1-3 letters) number(4 digits)
LAYOUTS = ("LLDDLDDDD", "LLDDLLDDDD", "LLDDLLLDDDD")

# Delhi writes a 1-digit district followed by a letter ("DL 8C AZ 4521"). It is only tried for "DL":
# for other states it would let "JHO5AW2117" be mis-read as "JH0SAW2117".
DELHI_LAYOUT = "LLDLLLDDDD"
DELHI_STATE = "DL"

_VALID_PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")

# Other regions (for validating the OCR on non-Indian footage). UK current format: area(2 letters) age(2 digits) random(3 letters).
UK_LAYOUTS = ("LLDDLLL",)
_VALID_UK_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$")
_VALID_ANY_RE = re.compile(r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]{4,10}$")

_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


def normalize(text: str) -> str:
    """Uppercase and keep only letters and digits."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def is_valid_plate(text: str, region: str = "IN") -> bool:
    """True if ``text`` is a complete plate of the given region.

    ``IN``: real state code, 1-2 digit district, 1-3 series letters, 4 digits. ``UK``: ``AA00AAA``.
    ``ANY``: 4-10 letters/digits containing both a letter and a digit.
    """
    cleaned = normalize(text)
    if region == "UK":
        return bool(_VALID_UK_RE.match(cleaned))
    if region == "ANY":
        return bool(_VALID_ANY_RE.match(cleaned))
    return bool(_VALID_PLATE_RE.match(cleaned)) and cleaned[:2] in STATE_CODES


def order_reads(reads: Sequence[Tuple[Sequence[Sequence[float]], str, float]]) -> List[Tuple[str, float]]:
    """Sort OCR text boxes into reading order.

    Args:
        reads: EasyOCR-style ``(bbox, text, confidence)`` triples where ``bbox`` is
            four ``[x, y]`` corner points.

    Returns:
        ``(text, confidence)`` pairs: lines top to bottom, each line left to right.
    """
    items = []
    for bbox, text, conf in reads:
        xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
        items.append({"cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys), "h": max(ys) - min(ys), "text": text, "conf": conf})
    if not items:
        return []
    items.sort(key=lambda d: d["cy"])
    line_gap = 0.6 * (statistics.median(d["h"] for d in items) or 1.0)

    lines: List[dict] = []
    for item in items:
        if lines and abs(item["cy"] - lines[-1]["cy"]) < line_gap:
            lines[-1]["items"].append(item)
            lines[-1]["cy"] = statistics.mean(d["cy"] for d in lines[-1]["items"])
        else:
            lines.append({"cy": item["cy"], "items": [item]})
    ordered: List[Tuple[str, float]] = []
    for line in lines:
        ordered += [(d["text"], d["conf"]) for d in sorted(line["items"], key=lambda d: d["cx"])]
    return ordered


def _fit(window: str, layout: str, max_cost: int) -> Optional[Tuple[str, int]]:
    """Fit a window to a layout, returning (corrected text, number of changed characters)."""
    out, cost = [], 0
    for char, kind in zip(window, layout):
        if kind == "L":
            if char.isalpha():
                out.append(char)
            elif char in _TO_LETTER:
                out.append(_TO_LETTER[char])
                cost += 1
            else:
                return None
        else:
            if char.isdigit():
                out.append(char)
            elif char in _TO_DIGIT:
                out.append(_TO_DIGIT[char])
                cost += 1
            else:
                return None
        if cost > max_cost:
            return None
    return "".join(out), cost


def _correct_uk(text: str, max_cost: int) -> str:
    """Best ``AA00AAA`` window inside ``text`` (look-alike fixes by position), or the text unchanged."""
    best: Optional[Tuple[int, int, str]] = None
    for layout in UK_LAYOUTS:
        n = len(layout)
        for start in range(0, len(text) - n + 1):
            fitted = _fit(text[start : start + n], layout, max_cost)
            if fitted is None:
                continue
            candidate = (fitted[1] + (len(text) - n), start, fitted[0])
            if best is None or candidate < best:
                best = candidate
    return best[2] if best else text


def correct_plate_text(raw: str, max_cost: int = 1, region: str = "IN") -> str:
    """Return the best valid-looking Indian plate inside ``raw`` text.

    Tries every window of the raw text against the 9/10/11-character layouts, allows
    up to ``max_cost`` look-alike substitutions, and requires a real state code. Among several fits it
    prefers the one that swaps and leaves out the fewest characters. If
    nothing fits, the normalized raw text is returned unchanged.

    Args:
        raw: Raw OCR text (any case, spaces/dots allowed).
        max_cost: Maximum number of characters that may be swapped (O->0 etc.). Kept at 1 by
            default: on a hand-labelled sample, allowing 2 occasionally turned a near-miss into garbage.
    """
    text = normalize(raw)
    if region == "UK":
        return _correct_uk(text, max_cost)
    if region == "ANY":
        return text
    # Prefer the candidate that changes and leaves out the fewest characters; on a tie, the common
    # 10-character layout, then 9, then 11, then the earliest start.
    preference = {10: 0, 9: 1, 11: 2}
    best: Optional[Tuple[int, int, int, str]] = None  # (cost + unused, layout preference, start, corrected); the Delhi layout ranks with the other 10-character ones
    for layout in LAYOUTS + (DELHI_LAYOUT,):
        n = len(layout)
        for start in range(0, len(text) - n + 1):
            fitted = _fit(text[start : start + n], layout, max_cost)
            if fitted is None or fitted[0][:2] not in STATE_CODES:
                continue
            if layout == DELHI_LAYOUT and fitted[0][:2] != DELHI_STATE:
                continue
            candidate = (fitted[1] + (len(text) - n), preference[n], start, fitted[0])
            if best is None or candidate < best:
                best = candidate
    return best[3] if best else text


def prepare_crop(crop, min_width: int = 400):
    """Grayscale a plate crop and enlarge it to at least ``min_width`` pixels wide.

    Small CCTV plates read far better when enlarged; no binarisation is applied
    (adaptive thresholding was never benchmarked and tends to break EasyOCR).

    Args:
        crop: BGR (or already grayscale) ``numpy`` image of one plate.
        min_width: Minimum width in pixels; wider crops are left at their size.

    Returns:
        Grayscale image.
    """
    import cv2  # local import keeps this module light for pure-text helpers

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.shape[1] < min_width:
        scale = min_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def read_plate_text(reader, crop, min_width: int = 400, max_cost: int = 1, region: str = "IN") -> Tuple[str, float, str]:
    """Run the full "big+fix" OCR flow on one plate crop.

    Steps: enlarge -> EasyOCR restricted to A-Z/0-9 -> reading order -> snap to an Indian plate.

    Args:
        reader: An ``easyocr.Reader`` (anything with ``readtext(image, allowlist=...)``).
        crop: BGR plate crop.
        min_width: Passed to :func:`prepare_crop`.
        max_cost: Passed to :func:`correct_plate_text`.
        region: ``IN`` / ``UK`` / ``ANY`` (see :func:`is_valid_plate`).

    Returns:
        ``(text, confidence, raw_text)``: the corrected plate (or the raw text when nothing fits),
        the confidence averaged over the characters read (long fragments count more than stray
        marks like "IND"), and the uncorrected reading-order text. Empty strings and 0.0 when
        nothing was read.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return "", 0.0, ""
    reads = reader.readtext(prepare_crop(crop, min_width), allowlist=OCR_ALLOWLIST)
    ordered = order_reads(reads)
    raw = normalize("".join(text for text, _ in ordered))
    if not raw:
        return "", 0.0, ""
    weight = sum(len(normalize(text)) for text, _ in ordered) or 1
    confidence = sum(len(normalize(text)) * conf for text, conf in ordered) / weight
    return correct_plate_text(raw, max_cost=max_cost, region=region), float(confidence), raw


_DISPLAY_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$")
_DISPLAY_UK_RE = re.compile(r"^([A-Z]{2}\d{2})([A-Z]{3})$")  # UK: GX15 OGJ
_DISPLAY_DELHI_RE = re.compile(r"^(DL)(\d[A-Z])([A-Z]{2})(\d{4})$")  # Delhi: DL 8C AZ 4521


def format_plate_for_display(plate: str) -> str:
    """Put spaces in a normalised plate for display (``MH12AB1234`` -> ``MH 12 AB 1234``).

    Text that does not match an Indian plate layout is returned unchanged.
    """
    compact = plate.replace(" ", "").upper()
    match = _DISPLAY_DELHI_RE.match(compact) or _DISPLAY_RE.match(compact) or _DISPLAY_UK_RE.match(compact)
    return " ".join(match.groups()) if match else plate
