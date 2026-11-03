"""Tests for scripts/ocr_bench.py with a stub OCR reader (no EasyOCR needed)."""

import cv2
import numpy as np

from scripts.ocr_bench import ALLOWLIST, VARIANTS, bench, run_variant


def _box(x, y, w=40, h=20):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class StubReader:
    """Returns the two lines of a plate in the wrong order, and records allowlist use."""

    def __init__(self):
        self.allowlists = []

    def readtext(self, image, **kwargs):
        self.allowlists.append(kwargs.get("allowlist"))
        return [(_box(5, 60), "2117", 0.9), (_box(0, 5, 70), "JHO5AW", 0.8)]


def _crop():
    return np.full((80, 120, 3), 200, dtype=np.uint8)


def test_baseline_keeps_reader_order_and_ordered_fixes_it():
    reader = StubReader()
    assert run_variant(reader, _crop(), "baseline") == "2117JHO5AW"
    assert run_variant(reader, _crop(), "ordered") == "JHO5AW2117"


def test_fix_variant_snaps_to_plate_and_uses_allowlist():
    reader = StubReader()
    assert run_variant(reader, _crop(), "+fix") == "JH05AW2117"
    assert reader.allowlists[-1] == ALLOWLIST
    assert run_variant(reader, _crop(), "baseline") and reader.allowlists[-1] is None


def test_bench_scores_every_variant_and_skips_unusable_rows(tmp_path):
    crops = tmp_path / "crops"
    crops.mkdir()
    cv2.imwrite(str(crops / "a.jpg"), _crop())
    rows = [
        {"image": "a", "box": "0", "crop_file": "a.jpg", "true_text": "JH05AW2117"},
        {"image": "b", "box": "0", "crop_file": "a.jpg", "true_text": "-"},             # not a plate
        {"image": "c", "box": "0", "crop_file": "a.jpg", "true_text": "GRILLE"},        # not plate-like
        {"image": "d", "box": "0", "crop_file": "missing.jpg", "true_text": "MH12AB1234"},  # crop file absent
    ]
    results = bench(rows, crops, StubReader(), VARIANTS)
    assert all(results[v]["plates"] == 1 for v in VARIANTS)
    assert results["baseline"]["exact"] == 0 and results["+fix"]["exact"] == 1
