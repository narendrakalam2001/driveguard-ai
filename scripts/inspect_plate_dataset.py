#!/usr/bin/env python3
"""
inspect_plate_dataset.py
=========================
Read-only report on the Indian number plate dataset folders: how many
images/annotation files exist, whether they pair up by filename, what the
annotation format is, which classes appear, and whether any annotation file
is actually broken (e.g. an HTML page saved instead of real data).

Standard library only - run it on your laptop. It never modifies anything.

Usage (Windows, one line):
    python scripts/inspect_plate_dataset.py ^
        --root "D:/Data_Science_Capstone_Datasets/driveguard-ai/Annotations" ^
        --root "D:/Data_Science_Capstone_Datasets/driveguard-ai/Indian_Number_Plates" ^
        --root "D:/Data_Science_Capstone_Datasets/driveguard-ai/number_plate_annos_ocr" ^
        --root "D:/Data_Science_Capstone_Datasets/driveguard-ai/number_plate_images_ocr"

Do NOT point --root at the whole driveguard-ai dataset folder: it also holds
the 280,000-file UA-DETRAC data and the scan would take forever.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANN_EXTS = {".xml", ".json", ".txt", ".csv"}


def read_head(path: Path, n: int) -> str:
    """Return the first n characters of a file, tolerant of bad encodings."""
    return path.read_bytes()[: n * 4].decode("utf-8", errors="replace")[:n]


def analyse_xml(xml_files: list[Path]) -> None:
    """Print class/box statistics for Pascal-VOC-style XML annotations."""
    names: Counter = Counter()
    child_tags: Counter = Counter()
    root_tags: Counter = Counter()
    objs_per_file: list[int] = []
    area_fracs: list[float] = []
    bad: list[tuple[Path, str]] = []

    for path in xml_files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            bad.append((path, read_head(path, 80).replace("\n", " ")))
            continue
        if root.tag.lower().endswith("html"):  # an HTML page saved with an .xml name
            bad.append((path, read_head(path, 80).replace("\n", " ")))
            continue
        root_tags[root.tag] += 1
        size = root.find("size")
        width = height = 0.0
        if size is not None:
            try:
                width, height = float(size.findtext("width") or 0), float(size.findtext("height") or 0)
            except ValueError:
                pass
        objects = root.findall("object")
        objs_per_file.append(len(objects))
        for obj in objects:
            names[(obj.findtext("name") or "").strip()] += 1
            for child in obj:
                child_tags[child.tag] += 1
            box = obj.find("bndbox")
            if box is not None and width and height:
                try:
                    w = float(box.findtext("xmax")) - float(box.findtext("xmin"))
                    h = float(box.findtext("ymax")) - float(box.findtext("ymin"))
                    area_fracs.append((w * h) / (width * height))
                except (TypeError, ValueError):
                    pass

    print(f"\n--- XML analysis ({len(xml_files)} files) ---")
    print(f"root tags: {dict(root_tags)}")
    if bad:
        print(f"UNPARSEABLE XML: {len(bad)} file(s) - first few:")
        for path, head in bad[:3]:
            print(f"   {path.name}: {head!r}")
    if not objs_per_file:
        print("No parseable XML files.")
        return
    print(f"objects per file: min {min(objs_per_file)} / mean {statistics.mean(objs_per_file):.2f} / max {max(objs_per_file)}")
    print(f"files with zero objects: {sum(1 for n in objs_per_file if n == 0)}")
    print(f"class names (<object><name>): {dict(names)}")
    print(f"tags inside <object>: {dict(child_tags)}")
    if area_fracs:
        print(
            f"box area / image area: min {min(area_fracs):.4f} / mean {statistics.mean(area_fracs):.4f} "
            f"/ max {max(area_fracs):.4f}"
        )


AREA_BUCKETS = [(0.0, 0.01, "<1%"), (0.01, 0.05, "1-5%"), (0.05, 0.20, "5-20%"), (0.20, 0.50, "20-50%"), (0.50, 1.01, ">50%")]


def analyse_yolo_txt(txt_files: list[Path], image_ext_by_stem: dict[str, str] | None = None) -> None:
    """Print class-id and box statistics if the .txt files look like YOLO labels.

    Box size is shown as buckets of image area and, when images were scanned too,
    split by image type - tight plate crops show up as boxes covering >50% of the image.
    """
    class_ids: Counter = Counter()
    field_counts: Counter = Counter()
    objs_per_file: list[int] = []
    areas: list[float] = []
    label_like_files = 0
    areas_by_ext: dict[str, list[float]] = defaultdict(list)

    for path in txt_files:
        count = 0
        file_areas: list[float] = []
        for raw in path.read_text(errors="replace").splitlines():
            parts = raw.split()
            if not parts:
                continue
            field_counts[len(parts)] += 1
            if len(parts) in (5, 6):
                try:
                    cls, _, _, w, h = (float(v) for v in parts[:5])
                except ValueError:
                    continue
                class_ids[int(cls)] += 1
                areas.append(w * h)
                file_areas.append(w * h)
                count += 1
        if count:
            label_like_files += 1
            ext = (image_ext_by_stem or {}).get(path.stem.lower(), "(no image)")
            areas_by_ext[ext].extend(file_areas)
        objs_per_file.append(count)

    if label_like_files < max(1, len(txt_files) // 2):
        return  # most .txt files are not YOLO labels (readme, lists, ...)
    print(f"\n--- YOLO label analysis ({len(txt_files)} .txt files) ---")
    print(f"class ids: {dict(sorted(class_ids.items()))}")
    print(f"fields per line (5/6 = box, 7+ = polygon): {dict(sorted(field_counts.items()))}")
    print(f"empty label files: {sum(1 for n in objs_per_file if n == 0)}")
    print(f"objects per file: min {min(objs_per_file)} / mean {statistics.mean(objs_per_file):.2f} / max {max(objs_per_file)}")
    if areas:
        print(f"box area / image area: min {min(areas):.4f} / mean {statistics.mean(areas):.4f} / max {max(areas):.4f}")
        buckets = {name: sum(1 for a in areas if lo <= a < hi) for lo, hi, name in AREA_BUCKETS}
        print(f"box size buckets (share of image area): {buckets}")
        for ext, vals in sorted(areas_by_ext.items()):
            print(
                f"  by image type {ext}: {len(vals)} boxes | mean area {statistics.mean(vals):.3f} "
                f"| median {statistics.median(vals):.3f}"
            )
        print("Tip: plates in real scenes are usually <20% of the image; boxes >50% mean tight plate crops.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Read-only report on the Indian number plate dataset.")
    parser.add_argument("--root", type=Path, action="append", required=True, help="Folder to scan (repeatable).")
    parser.add_argument("--show-chars", type=int, default=1200, help="Characters of a sample annotation to print.")
    args = parser.parse_args()

    files: list[Path] = []
    for root in args.root:
        if not root.exists():
            sys.exit(f"Folder not found: {root}")
        files += [p for p in root.rglob("*") if p.is_file() and p.name.lower() != "desktop.ini"]

    print(f"Total files scanned: {len(files)}\n\n--- Files per folder, by extension ---")
    by_dir: dict[Path, Counter] = defaultdict(Counter)
    for path in files:
        by_dir[path.parent][path.suffix.lower() or "(none)"] += 1
    for folder, counts in sorted(by_dir.items()):
        print(f"{folder}: {dict(counts)}")

    images = [p for p in files if p.suffix.lower() in IMG_EXTS]
    annotations = [p for p in files if p.suffix.lower() in ANN_EXTS]
    image_stems = {p.stem.lower() for p in images}
    ann_stems = {p.stem.lower() for p in annotations}
    print("\n--- Pairing by filename ---")
    print(f"images: {len(images)} | annotation files: {len(annotations)}")
    print(f"matched pairs: {len(image_stems & ann_stems)}")
    print(f"images without annotation: {len(image_stems - ann_stems)}")
    print(f"annotations without image: {len(ann_stems - image_stems)}")

    print("\n--- Sample annotation per format ---")
    shown: set[str] = set()
    for path in annotations:
        ext = path.suffix.lower()
        if ext in shown:
            continue
        shown.add(ext)
        print(f"\n[{ext}] {path}\n{read_head(path, args.show_chars)}")

    xml_files = [p for p in annotations if p.suffix.lower() == ".xml"]
    if xml_files:
        analyse_xml(xml_files)

    txt_files = [p for p in annotations if p.suffix.lower() == ".txt"]
    if txt_files:
        analyse_yolo_txt(txt_files, {p.stem.lower(): p.suffix.lower() for p in images})


if __name__ == "__main__":
    main()