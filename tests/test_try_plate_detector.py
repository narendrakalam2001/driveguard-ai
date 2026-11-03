"""Tests for scripts/try_plate_detector.py using a stub detector / stub OCR (no ultralytics needed)."""

import csv
from pathlib import Path

import cv2
import numpy as np

from scripts.try_plate_detector import CELL_H, CELL_W, contact_sheet, dedupe, make_tile, process, write_csv


class _Arr:
    def __init__(self, a):
        self._a = np.array(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf, self._n = _Arr(xyxy), _Arr(conf), len(conf)

    def __len__(self):
        return self._n


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class StubModel:
    """Finds one plate in images whose name starts with 'hit', none otherwise."""

    def predict(self, source, conf, imgsz, verbose):
        if Path(source).name.startswith("hit"):
            return [_Result(_Boxes([[20, 30, 120, 70]], [0.91]))]
        return [_Result(_Boxes([], []))]


class StubReader:
    def readtext(self, img):
        return [(None, "MH12", 0.8), (None, "AB1234", 0.6)]


def _images(tmp_path, names):
    paths = []
    for n in names:
        p = tmp_path / n
        cv2.imwrite(str(p), np.full((150, 200, 3), 128, dtype=np.uint8))
        paths.append(p)
    return paths


def test_process_without_ocr_returns_detections_only(tmp_path):
    imgs = _images(tmp_path, ["hit1.jpg", "miss1.jpg"])
    per_image, rows = process(imgs, StubModel(), 0.25, 640)
    assert len(per_image[imgs[0]]) == 1 and per_image[imgs[1]] == []
    assert rows == []


def test_process_with_ocr_builds_review_rows_and_crops(tmp_path):
    imgs = _images(tmp_path, ["hit1.jpg", "miss1.jpg", "hit2.jpg"])
    _, rows = process(imgs, StubModel(), 0.25, 640, reader=StubReader(), crops_dir=tmp_path / "crops")
    assert [r["image"] for r in rows] == ["hit1.jpg", "hit2.jpg"]
    assert rows[0]["ocr_text"] == "MH12AB1234" and rows[0]["true_text"] == ""
    assert (tmp_path / "crops" / "hit1_box0.jpg").exists()


def test_write_csv_roundtrip(tmp_path):
    imgs = _images(tmp_path, ["hit1.jpg"])
    _, rows = process(imgs, StubModel(), 0.25, 640, reader=StubReader(), crops_dir=tmp_path / "c")
    write_csv(rows, tmp_path / "r.csv")
    with open(tmp_path / "r.csv", newline="", encoding="utf-8") as fh:
        read = list(csv.DictReader(fh))
    assert read[0]["ocr_text"] == "MH12AB1234" and "true_text" in read[0]


def test_tiles_and_sheet_dimensions(tmp_path):
    imgs = _images(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
    tiles = [make_tile(p, []) for p in imgs]
    sheet = contact_sheet(tiles, cols=2)
    assert sheet.shape == (2 * CELL_H, 2 * CELL_W, 3)  # 3 tiles padded to 2x2


class NestedStub:
    """Whole plate (0.9) + a half-plate box inside it (0.6) + a separate box elsewhere (0.5)."""

    def predict(self, source, conf, imgsz, verbose):
        return [_Result(_Boxes([[10, 10, 110, 60], [10, 10, 110, 35], [150, 100, 190, 130]], [0.9, 0.6, 0.5]))]


def test_dedupe_drops_contained_and_overlapping_boxes_keeps_separate_ones():
    dets = [(10, 10, 110, 60, 0.9), (10, 10, 110, 35, 0.6), (12, 11, 108, 58, 0.7), (150, 100, 190, 130, 0.5)]
    kept = dedupe(dets)
    assert [d[4] for d in kept] == [0.9, 0.5]


def test_process_dedupes_by_default_and_can_be_disabled(tmp_path):
    imgs = _images(tmp_path, ["hit1.jpg"])
    per_image, _ = process(imgs, NestedStub(), 0.25, 640)
    assert len(per_image[imgs[0]]) == 2
    per_image, _ = process(imgs, NestedStub(), 0.25, 640, dedupe_boxes=False)
    assert len(per_image[imgs[0]]) == 3