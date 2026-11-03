#!/usr/bin/env python3
"""
preview_yolo_labels.py
=======================
Draws the YOLO boxes on a random sample of images and saves ONE contact-sheet
picture, so you can see at a glance whether a dataset's "plate" boxes sit on
small plates inside vehicle photos (what the detector needs) or cover almost
the whole image (tight plate crops).

Read-only. Needs opencv-python(-headless).

Usage (Windows, one line):
    python scripts/preview_yolo_labels.py --images "D:/.../images" --labels "D:/.../labels" --ext .png --n 12 --out preview_png.jpg
    python scripts/preview_yolo_labels.py --images "D:/.../images" --labels "D:/.../labels" --ext .jpg --n 12 --out preview_jpg.jpg
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CELL_W, CELL_H = 480, 360


def read_boxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Read (class, xc, yc, w, h) boxes from a YOLO label file; other lines are ignored."""
    boxes = []
    for raw in label_path.read_text(errors="replace").splitlines():
        parts = raw.split()
        if len(parts) not in (5, 6):
            continue
        try:
            cls, xc, yc, w, h = int(float(parts[0])), *(float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append((cls, xc, yc, w, h))
    return boxes


def render_cell(image_path: Path, boxes: List[Tuple[int, float, float, float, float]]) -> Optional[np.ndarray]:
    """Return one CELL_W x CELL_H tile: the image (letterboxed) with its boxes and a caption."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    for _, xc, yc, w, h in boxes:
        x1, y1 = int((xc - w / 2) * width), int((yc - h / 2) * height)
        x2, y2 = int((xc + w / 2) * width), int((yc + h / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), max(2, round(max(height, width) / 200)))
    scale = min(CELL_W / width, (CELL_H - 24) / height)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    cell = np.full((CELL_H, CELL_W, 3), 30, dtype=np.uint8)
    y0, x0 = 24, (CELL_W - resized.shape[1]) // 2
    cell[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    largest = max((w * h for _, _, _, w, h in boxes), default=0.0)
    caption = f"{image_path.name[:30]} | {len(boxes)} box | biggest {largest:.0%} of image"
    cv2.putText(cell, caption, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Contact sheet of YOLO labels drawn on random images.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--ext", type=str, default="", help="Only this image type, e.g. .png or .jpg (default: all).")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("preview.jpg"))
    args = parser.parse_args()

    labels = {p.stem.lower(): p for p in args.labels.rglob("*.txt")}
    ext = args.ext.lower()
    ext = ext if not ext or ext.startswith(".") else f".{ext}"
    candidates = [
        p for p in args.images.rglob("*")
        if p.suffix.lower() in IMG_EXTS and (not ext or p.suffix.lower() == ext) and p.stem.lower() in labels
    ]
    if not candidates:
        sys.exit("No images with a matching label file found (check --images / --labels / --ext).")

    picks = random.Random(args.seed).sample(candidates, min(args.n, len(candidates)))
    cells = []
    for path in picks:
        cell = render_cell(path, read_boxes(labels[path.stem.lower()]))
        if cell is not None:
            cells.append(cell)
    while len(cells) % args.cols:
        cells.append(np.full((CELL_H, CELL_W, 3), 30, dtype=np.uint8))
    rows = [np.hstack(cells[i : i + args.cols]) for i in range(0, len(cells), args.cols)]
    cv2.imwrite(str(args.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved {args.out} ({len(picks)} images from {len(candidates)} candidates). Open it and look at the green boxes.")


if __name__ == "__main__":
    main()
