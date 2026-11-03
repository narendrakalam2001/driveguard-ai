#!/usr/bin/env python3
"""
prepare_plate_dataset.py
=========================
Builds a YOLO-format license-plate dataset (single class 0 = license_plate)
from one or more sources:

  --pair IMAGES_DIR ANNOS_DIR       Pascal-VOC XML annotations (class ``number_plate``,
                                    e.g. the DataCluster sample), matched by file name.
  --yolo-pair IMAGES_DIR LABELS_DIR Already-YOLO sources (image + .txt label with
                                    the same stem). Sub-folders are searched too.

Per image it:
  * (VOC) loads the image with OpenCV - which applies EXIF rotation - and SKIPS
    any image whose size disagrees with the XML, instead of writing wrong boxes,
  * (YOLO) keeps only label lines whose class id is in --yolo-plate-ids (default 0);
    lines that are not plain 5/6-field boxes (e.g. polygons) are ignored,
  * optionally downsizes (--max-side); labels are normalized so boxes stay valid,
  * writes a clean .jpg and a YOLO .txt label.

Splitting:
  * default: random train/val split over everything (--val-fraction),
  * --holdout-voc: train ONLY on the --yolo-pair images and validate ONLY on the
    hand-annotated --pair (VOC) images - a cleaner check when the big set has
    auto-generated labels,
  * --max-train-images N caps the training set (keeps local CPU training short).

Images are written one at a time, so memory stays low even for thousands of images.

Usage (Windows, one line):
    python scripts/prepare_plate_dataset.py ^
        --yolo-pair "D:/.../kedarsai_plates/images" "D:/.../kedarsai_plates/labels" ^
        --pair "D:/.../Indian_Number_Plates/Sample_Images" "D:/.../Annotations" ^
        --pair "D:/.../number_plate_images_ocr" "D:/.../number_plate_annos_ocr" ^
        --holdout-voc --max-train-images 800 --max-side 1024 --overwrite

Needs: opencv-python(-headless) and pyyaml.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ACCEPTED_CLASS_NAMES = {"number_plate", "license_plate", "numberplate", "plate"}
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "plate_dataset"

Box = Tuple[float, float, float, float]


def parse_voc(xml_path: Path) -> Tuple[int, int, List[Box]]:
    """Read image size and plate boxes from one Pascal-VOC XML file.

    Args:
        xml_path: Path to the annotation file.

    Returns:
        (width, height, boxes) where boxes are (xmin, ymin, xmax, ymax) in
        the XML's pixel units. Only objects whose name is an accepted plate
        class are returned.

    Raises:
        ValueError: If the XML has no usable <size>.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError("no <size> element")
    width, height = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    boxes: List[Box] = []
    for obj in root.findall("object"):
        if (obj.findtext("name") or "").strip().lower() not in ACCEPTED_CLASS_NAMES:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        boxes.append(tuple(float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")))  # type: ignore[arg-type]
    return width, height, boxes


def to_yolo_line(box: Box, width: int, height: int) -> Optional[str]:
    """Convert one pixel box into a normalized YOLO label line (class 0).

    Args:
        box: (xmin, ymin, xmax, ymax) in pixels of a width x height image.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        The label line, or None if the box is degenerate after clipping.
    """
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(0.0, min(xmin, width)), max(0.0, min(xmax, width))
    ymin, ymax = max(0.0, min(ymin, height)), max(0.0, min(ymax, height))
    if xmax - xmin < 1 or ymax - ymin < 1:
        return None
    return (
        f"0 {(xmin + xmax) / 2 / width:.6f} {(ymin + ymax) / 2 / height:.6f} "
        f"{(xmax - xmin) / width:.6f} {(ymax - ymin) / height:.6f}"
    )


@dataclass
class Item:
    """One image plus its already-converted YOLO label lines.

    Attributes:
        name: Unique output file stem.
        image: Source image path.
        lines: YOLO label lines (class 0) for this image.
        origin: "voc" or "yolo" - where the annotation came from.
    """

    name: str
    image: Path
    lines: List[str]
    origin: str


def collect_pairs(pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    """Match XML files to images by file stem across all (images, annotations) folders."""
    matched: List[Tuple[Path, Path]] = []
    for images_dir, annos_dir in pairs:
        images = {p.stem.lower(): p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        for xml_path in sorted(annos_dir.glob("*.xml")):
            image = images.get(xml_path.stem.lower())
            if image is None:
                print(f"[skip] no image for {xml_path.name}")
                continue
            matched.append((image, xml_path))
    return matched


def read_yolo_labels(label_path: Path, plate_ids: Iterable[int]) -> List[str]:
    """Read plate boxes from a YOLO label file and rewrite them as class 0.

    Args:
        label_path: A YOLO .txt label file (normalized xc yc w h).
        plate_ids: Class ids that mean "license plate" in this source.

    Returns:
        Label lines (class 0) clipped to the image; polygon/other lines are dropped.
    """
    wanted = set(plate_ids)
    lines: List[str] = []
    for raw in label_path.read_text(errors="replace").splitlines():
        parts = raw.split()
        if len(parts) not in (5, 6):  # 6th field = confidence; longer lines are polygons
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if cls not in wanted:
            continue
        x1, x2 = max(0.0, xc - w / 2), min(1.0, xc + w / 2)
        y1, y2 = max(0.0, yc - h / 2), min(1.0, yc + h / 2)
        if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
            continue
        lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return lines


def build_voc_items(pairs: Sequence[Tuple[Path, Path]]) -> Tuple[List[Item], int]:
    """Validate and convert every VOC (image, xml) pair into an :class:`Item`."""
    items: List[Item] = []
    skipped = 0
    for idx, (image_path, xml_path) in enumerate(collect_pairs(list(pairs))):
        try:
            xml_w, xml_h, boxes = parse_voc(xml_path)
        except (ET.ParseError, ValueError, TypeError) as exc:
            print(f"[skip] {xml_path.name}: unreadable annotation ({exc})")
            skipped += 1
            continue
        image = cv2.imread(str(image_path))  # applies EXIF rotation; decoded only to check the size
        if image is None:
            print(f"[skip] {image_path.name}: cannot read image")
            skipped += 1
            continue
        height, width = image.shape[:2]
        sx, sy = width / xml_w, height / xml_h
        if abs(sx - sy) / sx > 0.01:
            print(
                f"[skip] {image_path.name}: image is {width}x{height} but XML says {xml_w}x{xml_h} "
                "(orientation/size mismatch)"
            )
            skipped += 1
            continue
        lines = [
            line
            for b in boxes
            if (line := to_yolo_line((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), width, height))
        ]
        if not lines:
            print(f"[skip] {image_path.name}: no valid plate boxes")
            skipped += 1
            continue
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
        items.append(Item(f"voc{idx:04d}_{stem}", image_path, lines, "voc"))
    return items, skipped


def box_area(line: str) -> float:
    """Area of a YOLO label line as a fraction of the image (width * height)."""
    parts = line.split()
    return float(parts[3]) * float(parts[4])


def build_yolo_items(
    pairs: Sequence[Tuple[Path, Path]],
    plate_ids: Iterable[int],
    exts: Iterable[str] = (),
    max_box_area: float = 0.0,
) -> Tuple[List[Item], int]:
    """Pair YOLO-format images with their label files (searching sub-folders).

    Args:
        pairs: (images_folder, labels_folder) tuples.
        plate_ids: Class ids that mean "license plate".
        exts: If given, only images with these suffixes (e.g. {".jpg"}) are used.
        max_box_area: If > 0, images with any plate box covering more than this
            fraction of the image are skipped - those are tight plate crops, not
            scenes, and would teach the detector that "plate = the whole image".

    Returns:
        (items, number_skipped)
    """
    wanted_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    items: List[Item] = []
    skipped = 0
    too_big = 0
    idx = 0
    for images_dir, labels_dir in pairs:
        labels = {p.stem.lower(): p for p in labels_dir.rglob("*.txt")}
        for image_path in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
            if wanted_exts and image_path.suffix.lower() not in wanted_exts:
                continue  # filtered out on purpose, not counted as skipped
            label_path = labels.get(image_path.stem.lower())
            lines = read_yolo_labels(label_path, plate_ids) if label_path else []
            if not lines:
                skipped += 1  # no label file, or no plate box in it
                continue
            if max_box_area and any(box_area(line) > max_box_area for line in lines):
                skipped += 1
                too_big += 1
                continue
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
            items.append(Item(f"yolo{idx:05d}_{stem}", image_path, lines, "yolo"))
            idx += 1
    if too_big:
        print(f"[info] skipped {too_big} YOLO images with a plate box larger than {max_box_area:.0%} of the image (tight crops)")
    return items, skipped


def prepare(
    pairs: Sequence[Tuple[Path, Path]],
    out_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_side: int = 1280,
    yolo_pairs: Sequence[Tuple[Path, Path]] = (),
    holdout_voc: bool = False,
    max_train_images: int = 0,
    yolo_plate_ids: Iterable[int] = (0,),
    yolo_exts: Iterable[str] = (),
    max_box_area: float = 0.0,
) -> Dict[str, int]:
    """Convert, split and write the YOLO plate dataset.

    Args:
        pairs: VOC (images_folder, annotations_folder) tuples.
        out_dir: Output dataset root (images/ and labels/ are created inside).
        val_fraction: Validation fraction for the random split.
        seed: Seed for shuffling / sampling.
        max_side: Downscale images whose longer side exceeds this (0 = never).
        yolo_pairs: YOLO (images_folder, labels_folder) tuples.
        holdout_voc: Train only on YOLO sources, validate only on VOC sources.
        max_train_images: Cap on training images (0 = no cap).
        yolo_plate_ids: Class ids that mean "license plate" in the YOLO sources.
        yolo_exts: Only use YOLO images with these suffixes (empty = all).
        max_box_area: Skip YOLO images whose plate box exceeds this image fraction (0 = off).

    Returns:
        Summary counts: used, skipped, boxes, train, val.
    """
    voc_items, skipped_voc = build_voc_items(pairs)
    yolo_items, skipped_yolo = build_yolo_items(yolo_pairs, yolo_plate_ids, yolo_exts, max_box_area)
    skipped = skipped_voc + skipped_yolo
    items = voc_items + yolo_items
    if len(items) < 2:
        raise SystemExit(f"Only {len(items)} usable image(s); need at least 2 to make a train/val split.")

    rng = random.Random(seed)
    if holdout_voc and voc_items and yolo_items:
        train, val = list(yolo_items), list(voc_items)
    else:
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_fraction)))
        val, train = shuffled[:n_val], shuffled[n_val:]
    if max_train_images and len(train) > max_train_images:
        train = rng.sample(train, max_train_images)

    written = {"train": 0, "val": 0}
    boxes = 0
    for split, entries in (("train", train), ("val", val)):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for item in entries:
            image = cv2.imread(str(item.image))
            if image is None:
                print(f"[skip] {item.image.name}: cannot read image")
                skipped += 1
                continue
            height, width = image.shape[:2]
            if max_side and max(height, width) > max_side:
                scale = max_side / max(height, width)
                image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / "images" / split / f"{item.name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_dir / "labels" / split / f"{item.name}.txt").write_text("\n".join(item.lines) + "\n")
            written[split] += 1
            boxes += len(item.lines)

    if min(written.values()) == 0:
        raise SystemExit(f"A split ended up empty (train {written['train']}, val {written['val']}).")

    data_yaml = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val", "names": {0: "license_plate"}}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    return {
        "used": written["train"] + written["val"],
        "skipped": skipped,
        "boxes": boxes,
        "train": written["train"],
        "val": written["val"],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build a YOLO license-plate dataset from VOC and/or YOLO sources.")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("IMAGES_DIR", "ANNOS_DIR"),
                        help="Pascal-VOC source (repeatable).")
    parser.add_argument("--yolo-pair", nargs=2, action="append", default=[], metavar=("IMAGES_DIR", "LABELS_DIR"),
                        help="Already-YOLO source (repeatable).")
    parser.add_argument("--yolo-plate-ids", type=str, default="0",
                        help="Comma-separated class ids that mean license plate in the YOLO sources.")
    parser.add_argument("--yolo-exts", type=str, default="",
                        help="Only use YOLO images with these extensions, e.g. .jpg or .jpg,.jpeg (default: all).")
    parser.add_argument("--max-box-area", type=float, default=0.0,
                        help="Skip YOLO images whose plate box covers more than this fraction of the image "
                             "(e.g. 0.5 drops tight plate crops).")
    parser.add_argument("--holdout-voc", action="store_true",
                        help="Train on --yolo-pair only; validate on --pair (VOC) only.")
    parser.add_argument("--max-train-images", type=int, default=0, help="Cap the training set (0 = no cap).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-side", type=int, default=1280, help="Downscale longer side to this (0 = keep full size).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Delete --out first if it already exists.")
    args = parser.parse_args()

    if not args.pair and not args.yolo_pair:
        parser.error("give at least one --pair (VOC) or --yolo-pair (YOLO) source")
    if args.holdout_voc and not (args.pair and args.yolo_pair):
        parser.error("--holdout-voc needs both --yolo-pair (train) and --pair (validation)")
    try:
        plate_ids = [int(v) for v in args.yolo_plate_ids.split(",") if v.strip() != ""]
    except ValueError:
        parser.error("--yolo-plate-ids must be comma-separated integers, e.g. 0 or 0,2")

    pairs = [(Path(i), Path(a)) for i, a in args.pair]
    yolo_pairs = [(Path(i), Path(a)) for i, a in args.yolo_pair]
    for first, second in pairs + yolo_pairs:
        for folder in (first, second):
            if not folder.is_dir():
                sys.exit(f"Folder not found: {folder}")
    if args.out.exists():
        if not args.overwrite:
            sys.exit(f"{args.out} already exists. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(args.out)

    stats = prepare(
        pairs, args.out, args.val_fraction, args.seed, args.max_side,
        yolo_pairs=yolo_pairs, holdout_voc=args.holdout_voc,
        max_train_images=args.max_train_images, yolo_plate_ids=plate_ids,
        yolo_exts=[e for e in args.yolo_exts.split(",") if e.strip()], max_box_area=args.max_box_area,
    )
    print(
        f"\nDone: {stats['used']} images ({stats['boxes']} plate boxes), {stats['skipped']} skipped | "
        f"train {stats['train']} / val {stats['val']}\nDataset: {args.out}"
    )
    if stats["used"] < 200:
        print(
            "NOTE: this is a very small dataset. The detector will train, but validation numbers from "
            f"{stats['val']} images are unreliable and it will generalize poorly."
        )


if __name__ == "__main__":
    main()