"""Check the number-plate pipeline on ONE image (or one video frame), outside the traffic pipeline.

Use this to find out whether "no plates read" comes from the footage (plates too small/blurry) or from the models:
a sharp close-up of a car's plate should be read; the same car shot from far away usually cannot be.

Usage:
    python scripts/plate_check.py path/to/car.jpg
    python scripts/plate_check.py --video clip.mp4 --frame 120
    python scripts/plate_check.py car.jpg --box 100,200,420,380      # crop a vehicle out of a larger image first

Prints whether a plate was found, its pixel size, the OCR text, confidence and the verdict, and saves the plate
crop to ``plate_debug/check_plate.jpg`` so you can look at exactly what the OCR saw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

READABLE_PLATE_WIDTH_PX = 100  # rule of thumb: plates narrower than this are rarely readable


def check_image(recognizer: Any, image: np.ndarray) -> Dict[str, Any]:
    """Run plate detection then OCR on ``image`` (treated as one vehicle crop).

    Args:
        recognizer: A :class:`src.plate_recognizer.PlateRecognizer` (or a test double with the same two methods).
        image: BGR image.

    Returns:
        Dict with ``plate_box_found``; when found also ``plate_width_px``, ``plate_number`` (``PLATE_UNCLEAR`` if
        the text could not be trusted), ``confidence``, ``raw_text`` and the ``crop`` image.
    """
    crop = recognizer.detect_plate_region(image)
    if crop is None:
        return {"plate_box_found": False, "image_width_px": int(image.shape[1])}
    result = recognizer.read_plate_crop(crop)
    return {
        "plate_box_found": True,
        "image_width_px": int(image.shape[1]),
        "plate_width_px": int(crop.shape[1]),
        "plate_number": result.plate_number,
        "confidence": round(float(result.confidence), 3),
        "raw_text": getattr(result, "raw_text", ""),
        "crop": crop,
    }


def verdict(report: Dict[str, Any]) -> str:
    """One-line human explanation of a :func:`check_image` report."""
    if not report["plate_box_found"]:
        return "No plate was found. The plate may be too small, hidden or seen at an angle (or the image is not a vehicle)."
    width = report["plate_width_px"]
    if report["plate_number"] != "PLATE_UNCLEAR":
        return f"Plate read: {report['plate_number']} (confidence {report['confidence']})."
    hint = f" The plate is only {width}px wide; about {READABLE_PLATE_WIDTH_PX}px or more is needed." if width < READABLE_PLATE_WIDTH_PX else ""
    return "A plate was found but the text could not be trusted (unclear, or not a valid Indian plate)." + hint


def _load(args: argparse.Namespace) -> Optional[np.ndarray]:
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return cv2.imread(str(args.image)) if args.image else None


def _crop_box(image: np.ndarray, box: Optional[str]) -> np.ndarray:
    if not box:
        return image
    x1, y1, x2, y2 = (int(v) for v in box.split(","))
    return image[max(0, y1):y2, max(0, x1):x2]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, help="Image file (jpg/png).")
    parser.add_argument("--video", type=Path, help="Use a frame of this video instead of an image.")
    parser.add_argument("--frame", type=int, default=0, help="Frame number when --video is used.")
    parser.add_argument("--region", choices=["IN", "UK", "ANY"], help="Plate format to accept (default: PLATE_REGION from .env, normally IN).")
    parser.add_argument("--box", help="Crop 'x1,y1,x2,y2' (a vehicle) out of the image before checking.")
    args = parser.parse_args()
    if not args.image and not args.video:
        parser.error("give an image path or --video")

    image = _load(args)
    if image is None:
        print("Could not read the image/video frame.")
        return 1
    image = _crop_box(image, args.box)

    from src.plate_recognizer import PlateRecognizer

    recognizer = PlateRecognizer(region=args.region) if args.region else PlateRecognizer()
    print(f"Plate format accepted: {recognizer.region}")
    report = check_image(recognizer, image)
    print(f"Image width: {report['image_width_px']}px")
    if report["plate_box_found"]:
        print(f"Plate box found, {report['plate_width_px']}px wide")
        print(f"OCR text: {report['raw_text']!r}  ->  {report['plate_number']}  (confidence {report['confidence']})")
        if report["plate_number"] == "PLATE_UNCLEAR" and report["raw_text"] and recognizer.region == "IN":
            print("The OCR did read text. If this is not an Indian plate, try --region UK or --region ANY.")
        out = Path("plate_debug")
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / "check_plate.jpg"), report["crop"])
        print(f"Saved the plate crop to {out / 'check_plate.jpg'}")
    print(verdict(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
