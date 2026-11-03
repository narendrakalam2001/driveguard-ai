#!/usr/bin/env python3
"""
find_duplicate_images.py
=========================
Tells you how many images in folder(s) B already exist in folder(s) A, even if
they were renamed, re-encoded, resized or rotated. Uses a perceptual difference
hash (dHash) on the EXIF-corrected image and also tries 90/180/270 degree
rotations.

Why: the same photos are often redistributed under different names. If you
merge two such datasets, the same image can land in both train and val, which
inflates validation metrics. Run this BEFORE merging.

Usage (Windows, one line):
    python scripts/find_duplicate_images.py ^
        --a "D:/.../Indian_Number_Plates/Sample_Images" --a "D:/.../number_plate_images_ocr" ^
        --b "D:/.../dc_hf_sample/images"

Needs: opencv-python(-headless). Read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HASH_SIZE = 8  # 8x8 differences -> 64-bit hash


def dhash(gray: np.ndarray) -> int:
    """Compute a 64-bit difference hash of a grayscale image."""
    small = cv2.resize(gray, (HASH_SIZE + 1, HASH_SIZE), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_variants(path: Path) -> Optional[List[int]]:
    """Hashes of an image at 0/90/180/270 degrees, or None if unreadable."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # applies EXIF orientation
    if gray is None:
        return None
    rotations = [
        gray,
        cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(gray, cv2.ROTATE_180),
        cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    return [dhash(r) for r in rotations]


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def list_images(folders: List[Path]) -> List[Path]:
    """All image files directly or recursively inside the given folders."""
    return sorted(p for f in folders for p in f.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def find_duplicates(a_dirs: List[Path], b_dirs: List[Path], threshold: int = 6) -> Tuple[List[Tuple[Path, Path, int]], List[Path]]:
    """Match every image in B against all images in A.

    Args:
        a_dirs: Reference folders.
        b_dirs: Folders to check.
        threshold: Max differing bits (of 64) to call two images duplicates.

    Returns:
        (duplicates as (b_path, best_a_path, distance), list of B images with no match)
    """
    a_hashes: Dict[Path, int] = {}
    for path in list_images(a_dirs):
        variants = hash_variants(path)
        if variants:
            a_hashes[path] = variants[0]

    duplicates: List[Tuple[Path, Path, int]] = []
    unique: List[Path] = []
    for path in list_images(b_dirs):
        variants = hash_variants(path)
        if not variants or not a_hashes:
            unique.append(path)
            continue
        best_path, best_dist = min(
            ((a_path, min(hamming(v, a_hash) for v in variants)) for a_path, a_hash in a_hashes.items()),
            key=lambda item: item[1],
        )
        if best_dist <= threshold:
            duplicates.append((path, best_path, best_dist))
        else:
            unique.append(path)
    return duplicates, unique


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Find images of B that already exist in A.")
    parser.add_argument("--a", type=Path, action="append", required=True, help="Reference folder (repeatable).")
    parser.add_argument("--b", type=Path, action="append", required=True, help="Folder to check (repeatable).")
    parser.add_argument("--threshold", type=int, default=6, help="Max differing bits out of 64 (default 6).")
    parser.add_argument("--verbose", action="store_true", help="List every match.")
    args = parser.parse_args()

    for folder in args.a + args.b:
        if not folder.is_dir():
            sys.exit(f"Folder not found: {folder}")

    duplicates, unique = find_duplicates(args.a, args.b, args.threshold)
    total = len(duplicates) + len(unique)
    print(f"Images checked in B: {total}")
    print(f"  already present in A (duplicates): {len(duplicates)}")
    print(f"  not found in A (new):              {len(unique)}")
    shown = duplicates if args.verbose else duplicates[:8]
    for b_path, a_path, dist in shown:
        print(f"   {b_path.name}  ~  {a_path.name}  (distance {dist}/64)")
    if not args.verbose and len(duplicates) > len(shown):
        print(f"   ... {len(duplicates) - len(shown)} more (use --verbose)")


if __name__ == "__main__":
    main()
