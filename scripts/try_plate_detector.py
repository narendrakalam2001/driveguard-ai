#!/usr/bin/env python3
"""
try_plate_detector.py
======================
Eyeball check for the trained plate detector. Runs models/yolov8_plate.pt on a
folder of images, draws the detections on a contact sheet, and prints a short
summary. With --ocr it also reads each detected plate with EasyOCR, saves the
crops, and writes a review CSV (image, box, confidence, predicted text and an
empty `true_text` column) that you can correct by looking at the crops - that
corrected CSV is what a real OCR-accuracy number needs.

Needs: ultralytics, opencv. For --ocr also: pip install easyocr
(first run downloads the OCR models, so it needs internet).

Usage (Windows, one line):
    python scripts/try_plate_detector.py
    python scripts/try_plate_detector.py --images "D:/some/folder" --ocr

NOTE: the --ocr here is plain EasyOCR on the raw crop. It is a starting point for
labelling, not the exact preprocessing used by src/plate_recognizer.py.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CELL_W, CELL_H = 480, 360

Detection = Tuple[float, float, float, float, float]  # x1, y1, x2, y2, confidence


def detect(model, image_path: Path, conf: float, imgsz: int) -> List[Detection]:
    """Run the detector on one image and return plain-Python boxes."""
    result = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    return [(float(a), float(b), float(c), float(d), float(p)) for (a, b, c, d), p in zip(xyxy, confs)]


def _area(d: Detection) -> float:
    return max(0.0, d[2] - d[0]) * max(0.0, d[3] - d[1])


def _overlap(a: Detection, b: Detection) -> Tuple[float, float]:
    """Return (IoU, intersection / smaller box) for two detections."""
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = _area(a) + _area(b) - inter
    smaller = min(_area(a), _area(b))
    return (inter / union if union else 0.0, inter / smaller if smaller else 0.0)


def dedupe(dets: List[Detection], iou_thresh: float = 0.5, contain_thresh: float = 0.7) -> List[Detection]:
    """Keep only the most confident box among overlapping ones.

    Two-line plates and blurry plates often get several boxes (the whole plate plus
    halves, or slightly shifted copies). A box is dropped when it overlaps a more
    confident one by IoU >= iou_thresh, or when >= contain_thresh of the smaller box
    lies inside it.
    """
    kept: List[Detection] = []
    for det in sorted(dets, key=lambda d: -d[4]):
        if all(max(_overlap(det, k)[0] / iou_thresh, _overlap(det, k)[1] / contain_thresh) < 1.0 for k in kept):
            kept.append(det)
    return kept


def ocr_crop(reader, crop: np.ndarray) -> Tuple[str, float]:
    """Read a plate crop with EasyOCR; returns (text, mean confidence)."""
    if crop.size == 0:
        return "", 0.0
    if crop.shape[1] < 200:  # small crops read better when enlarged
        scale = 200 / crop.shape[1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    reads = reader.readtext(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    if not reads:
        return "", 0.0
    text = "".join(t for _, t, _ in reads).upper().replace(" ", "")
    return text, float(np.mean([c for _, _, c in reads]))


def process(
    images: List[Path], model, conf: float, imgsz: int, reader=None, crops_dir: Optional[Path] = None,
    dedupe_boxes: bool = True,
) -> Tuple[Dict[Path, List[Detection]], List[dict]]:
    """Detect (and optionally OCR) every image.

    Returns:
        (detections per image, review-CSV rows - one per detected plate)
    """
    per_image: Dict[Path, List[Detection]] = {}
    rows: List[dict] = []
    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)
    for path in images:
        dets = detect(model, path, conf, imgsz)
        if dedupe_boxes:
            dets = dedupe(dets)
        per_image[path] = dets
        if reader is None:
            continue
        image = cv2.imread(str(path))
        for idx, (x1, y1, x2, y2, p) in enumerate(dets):
            pad = 0.05 * max(x2 - x1, y2 - y1)
            crop = image[max(0, int(y1 - pad)) : int(y2 + pad), max(0, int(x1 - pad)) : int(x2 + pad)]
            text, text_conf = ocr_crop(reader, crop)
            crop_name = f"{path.stem}_box{idx}.jpg"
            if crops_dir is not None and crop.size:
                cv2.imwrite(str(crops_dir / crop_name), crop)
            rows.append({
                "image": path.name, "box": idx, "det_conf": round(p, 3),
                "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2),
                "crop_file": crop_name, "ocr_text": text, "ocr_conf": round(text_conf, 3), "true_text": "",
            })
    return per_image, rows


def make_tile(path: Path, dets: List[Detection]) -> Optional[np.ndarray]:
    """One CELL_W x CELL_H tile: the image with its detections and a caption."""
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    thickness = max(2, round(max(height, width) / 200))
    for x1, y1, x2, y2, p in dets:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
        cv2.putText(image, f"{p:.2f}", (int(x1), max(20, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.8, width / 1200), (0, 255, 0), thickness, cv2.LINE_AA)
    scale = min(CELL_W / width, (CELL_H - 24) / height)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    tile = np.full((CELL_H, CELL_W, 3), 30, dtype=np.uint8)
    x0 = (CELL_W - resized.shape[1]) // 2
    tile[24 : 24 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.putText(tile, f"{path.name[:32]} | {len(dets)} plate(s)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def contact_sheet(tiles: List[np.ndarray], cols: int) -> np.ndarray:
    """Arrange tiles into a grid, padding the last row."""
    tiles = list(tiles)
    while len(tiles) % cols:
        tiles.append(np.full((CELL_H, CELL_W, 3), 30, dtype=np.uint8))
    return np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])


def write_csv(rows: List[dict], path: Path) -> None:
    """Write the OCR review CSV."""
    fields = ["image", "box", "det_conf", "x1", "y1", "x2", "y2", "crop_file", "ocr_text", "ocr_conf", "true_text"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Eyeball check of the trained plate detector.")
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "processed" / "plate_dataset" / "images" / "val")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolov8_plate.pt")
    parser.add_argument("--conf", type=float, default=0.4, help="Minimum detection confidence (lower = more junk boxes).")
    parser.add_argument("--no-dedupe", action="store_true", help="Keep overlapping duplicate boxes.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n", type=int, default=12, help="Images shown on the contact sheet.")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("plate_demo.jpg"))
    parser.add_argument("--ocr", action="store_true", help="Also read plates with EasyOCR and write a review CSV + crops.")
    parser.add_argument("--csv", type=Path, default=Path("plate_ocr_review.csv"))
    parser.add_argument("--crops-dir", type=Path, default=Path("plate_crops"))
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f"Weights not found: {args.weights} (train first: python scripts/train_plate_detector.py)")
    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f"No images found in {args.images}")

    from ultralytics import YOLO

    reader = None
    if args.ocr:
        try:
            import easyocr
        except ImportError:
            sys.exit("--ocr needs EasyOCR: pip install easyocr")
        reader = easyocr.Reader(["en"], gpu=False)

    model = YOLO(str(args.weights))
    per_image, rows = process(
        images, model, args.conf, args.imgsz, reader, args.crops_dir if args.ocr else None,
        dedupe_boxes=not args.no_dedupe,
    )

    shown = random.Random(args.seed).sample(images, min(args.n, len(images)))
    tiles = [t for t in (make_tile(p, per_image[p]) for p in shown) if t is not None]
    cv2.imwrite(str(args.out), contact_sheet(tiles, args.cols), [cv2.IMWRITE_JPEG_QUALITY, 90])

    counts = [len(d) for d in per_image.values()]
    confs = [det[4] for dets in per_image.values() for det in dets]
    print(f"{len(images)} images | with >=1 detection: {sum(1 for c in counts if c)} | total detections: {sum(counts)}"
          + (f" | mean confidence {np.mean(confs):.2f}" if confs else ""))
    print(f"Contact sheet saved: {args.out}")
    if rows:
        write_csv(rows, args.csv)
        print(f"Review CSV: {args.csv} ({len(rows)} plates) | crops: {args.crops_dir}")
        print("Open the CSV, look at each crop, and type the real plate into the true_text column.")


if __name__ == "__main__":
    main()