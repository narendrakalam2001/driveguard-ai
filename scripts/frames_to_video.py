#!/usr/bin/env python3
"""
frames_to_video.py
===================
Stitches the consecutive frames of ONE UA-DETRAC sequence (e.g. MVI_40752)
into an mp4, so you have a demo video for the dashboard / API.

Use the FULL dataset on your D: drive, not the stride-10 Kaggle subset:
the subset only keeps every 10th frame, so a video made from it would play
10x too fast and speed estimation would be wrong. The script warns if it
detects gaps between frame numbers.

Usage:
    python scripts/frames_to_video.py ^
        --images-dir "D:/Data_Science_Capstone_Datasets/driveguard-ai/content/UA-DETRAC/DETRAC_Upload/images/val" ^
        --sequence MVI_40752 --fps 25 --max-frames 750 ^
        --out sample_videos/MVI_40752.mp4
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2

_FRAME_RE = re.compile(r"_img(\d+)$")


def frame_number(path: Path) -> int:
    """Extract the numeric frame index from a name like MVI_40752_img00123.jpg."""
    match = _FRAME_RE.search(path.stem)
    return int(match.group(1)) if match else -1


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Make an mp4 from one UA-DETRAC sequence's frames.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder holding the frame images (e.g. .../images/val).")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence prefix, e.g. MVI_40752.")
    parser.add_argument("--fps", type=float, default=25.0, help="UA-DETRAC is recorded at 25 fps.")
    parser.add_argument("--max-frames", type=int, default=750, help="750 frames = 30 s at 25 fps.")
    parser.add_argument("--out", type=Path, default=None, help="Output mp4 path (default sample_videos/<sequence>.mp4).")
    args = parser.parse_args()

    files = sorted(args.images_dir.glob(f"{args.sequence}_img*.*"), key=frame_number)
    if not files:
        names = sorted({re.sub(r"_img\d+$", "", p.stem) for p in args.images_dir.glob("*_img*.*")})
        sys.exit(f"No frames for '{args.sequence}' in {args.images_dir}. Sequences found (first 15): {names[:15]}")

    files = files[: args.max_frames]
    numbers = [frame_number(p) for p in files]
    gaps = [b - a for a, b in zip(numbers, numbers[1:]) if b - a != 1]
    if gaps:
        print(
            f"WARNING: frame numbers are not consecutive ({len(gaps)} gaps, e.g. step {gaps[0]}). "
            "This looks like a subsampled folder - the video will play too fast and speed estimates will be wrong."
        )

    first = cv2.imread(str(files[0]))
    if first is None:
        sys.exit(f"Could not read {files[0]}")
    height, width = first.shape[:2]

    out_path = args.out or Path("sample_videos") / f"{args.sequence}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    if not writer.isOpened():
        sys.exit("Could not open the video writer (mp4v codec unavailable?).")

    for path in files:
        frame = cv2.imread(str(path))
        if frame is not None and frame.shape[:2] == (height, width):
            writer.write(frame)
    writer.release()
    print(f"Wrote {out_path} ({len(files)} frames, {width}x{height}, {args.fps} fps, ~{len(files) / args.fps:.0f}s)")


if __name__ == "__main__":
    main()
