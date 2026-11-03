#!/usr/bin/env python3
"""
make_subset.py
===============
Creates a small, evenly-sampled copy of a YOLO-formatted dataset
(images/{train,val} + labels/{train,val}) so it can be uploaded to
Kaggle as a SMALL dataset instead of attaching the full ~280,000-file one.

Pure standard library -- run it on your laptop, no pip installs needed.

Usage (Windows example):
    python scripts/make_subset.py ^
        --src "D:/Data_Science_Capstone_Datasets/driveguard-ai/content/UA-DETRAC/DETRAC_Upload" ^
        --dst "D:/detrac_subset" ^
        --train-stride 10 --val-stride 10 --zip

Keeps every Nth image (sorted by filename, so sampling stays even within
each video sequence) together with its label file. The output has the
same layout as the source, so it plugs straight into:

    python scripts/train_detector.py --yolo-dirs <subset>/DETRAC_Upload ...
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def subsample_split(src_root: Path, dst_root: Path, split: str, stride: int) -> int:
    """Copy every Nth image (and its label) of one split.

    Args:
        src_root: Source dataset root containing images/ and labels/.
        dst_root: Destination dataset root.
        split: "train" or "val".
        stride: Keep every Nth image; 1 keeps everything.

    Returns:
        Number of images copied.
    """
    src_images = src_root / "images" / split
    src_labels = src_root / "labels" / split
    dst_images = dst_root / "images" / split
    dst_labels = dst_root / "labels" / split

    if not src_images.exists():
        print(f"[skip] {src_images} does not exist")
        return 0

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in _IMG_EXTS)
    total = len(images)
    kept = 0
    for idx, img in enumerate(images):
        if idx % stride != 0:
            continue
        shutil.copy2(img, dst_images / img.name)
        label = src_labels / f"{img.stem}.txt"
        if label.exists():
            shutil.copy2(label, dst_labels / label.name)
        else:
            (dst_labels / f"{img.stem}.txt").write_text("")  # background frame
        kept += 1
        if kept % 1000 == 0:
            print(f"  {split}: copied {kept} images...")
    print(f"[{split}] kept {kept} of {total} images (stride {stride})")
    return kept


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Subsample a YOLO dataset for a small Kaggle upload.")
    parser.add_argument("--src", type=Path, required=True, help="Path to DETRAC_Upload (has images/ and labels/).")
    parser.add_argument("--dst", type=Path, required=True, help="Output folder (DETRAC_Upload is created inside).")
    parser.add_argument("--train-stride", type=int, default=10, help="Keep every Nth train image.")
    parser.add_argument("--val-stride", type=int, default=10, help="Keep every Nth val image.")
    parser.add_argument("--zip", action="store_true", help="Also create <dst>.zip (single file = easiest Kaggle upload).")
    args = parser.parse_args()

    if not (args.src / "images").exists() or not (args.src / "labels").exists():
        sys.exit(f"{args.src} must contain images/ and labels/ folders.")
    if args.train_stride < 1 or args.val_stride < 1:
        sys.exit("Strides must be >= 1.")

    out_root = args.dst / "DETRAC_Upload"
    n_train = subsample_split(args.src, out_root, "train", args.train_stride)
    n_val = subsample_split(args.src, out_root, "val", args.val_stride)

    size_mb = sum(f.stat().st_size for f in out_root.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\nDone: {n_train} train + {n_val} val images, {size_mb:.0f} MB -> {out_root}")

    if args.zip:
        archive = shutil.make_archive(str(args.dst), "zip", root_dir=args.dst)
        print(f"Zip created: {archive}  ({Path(archive).stat().st_size / (1024 * 1024):.0f} MB)")


if __name__ == "__main__":
    main()
