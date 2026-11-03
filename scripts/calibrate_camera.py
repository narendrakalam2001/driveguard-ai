#!/usr/bin/env python3
"""
calibrate_camera.py
=====================
Interactive CLI wizard mirroring the dashboard's 4-step calibration flow:
mark 4 road-rectangle corners on a sample frame, enter lane width and
speed limit, and save the resulting :class:`CameraCalibration` to disk.

Usage:
    python scripts/calibrate_camera.py --video sample_videos/junction1.mp4 \\
        --camera-id CAM-JN-01 --speed-limit 60 --lane-width 3.5

Click 4 points on the displayed frame, in order: top-left, top-right,
bottom-right, bottom-left of a visible road rectangle. Press 'r' to
restart point selection, 'q' to cancel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2

from src.camera_calibrator import CalibrationError, CameraCalibrator
from src.config import DEFAULT_LANE_WIDTH_M, MODELS_DIR, get_logger

logger = get_logger(__name__)

_CALIBRATIONS_DIR = MODELS_DIR.parent / "data" / "calibrations"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the calibration wizard."""
    parser = argparse.ArgumentParser(description="Interactive DriveGuard AI camera calibration wizard.")
    parser.add_argument("--video", type=Path, required=True, help="Sample video or image to calibrate against.")
    parser.add_argument("--camera-id", type=str, required=True)
    parser.add_argument("--speed-limit", type=int, default=60, help="Posted speed limit in km/h.")
    parser.add_argument("--lane-width", type=float, default=DEFAULT_LANE_WIDTH_M, help="Real-world lane width in meters.")
    parser.add_argument(
        "--lane-depth", type=float, default=None,
        help="Real-world LENGTH in meters of the marked stretch. Speeds scale with it; if omitted, 4x the width is assumed (a guess).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (defaults to data/calibrations/<camera_id>.json).",
    )
    parser.add_argument(
        "--non-interactive",
        type=str,
        default=None,
        help='Skip the GUI: pass 4 points directly as "x1,y1;x2,y2;x3,y3;x4,y4".',
    )
    return parser.parse_args()


def _grab_reference_frame(video_path: Path) -> "cv2.Mat":
    """Read the first frame from a video (or the image itself) to calibrate against.

    Args:
        video_path: Path to a video file or a single image file.

    Returns:
        The BGR frame/image.

    Raises:
        RuntimeError: If no frame could be read.
    """
    if video_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        frame = cv2.imread(str(video_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {video_path}")
        return frame

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from video: {video_path}")
    return frame


def _collect_points_interactively(frame) -> list[tuple[float, float]]:
    """Open an OpenCV window and let the user click 4 calibration points.

    Args:
        frame: Reference BGR frame to display.

    Returns:
        List of 4 (x, y) pixel points in click order.

    Raises:
        RuntimeError: If the window is closed before 4 points are marked.
    """
    points: list[tuple[float, float]] = []
    display = frame.copy()
    window_name = "DriveGuard AI - Calibration (click TL, TR, BR, BL; 'r' reset; 'q' cancel)"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(display, tuple(map(int, points[-2])), (x, y), (0, 255, 0), 2)
            cv2.imshow(window_name, display)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)
    cv2.imshow(window_name, display)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if len(points) == 4:
            cv2.line(display, tuple(map(int, points[-1])), tuple(map(int, points[0])), (0, 255, 0), 2)
            cv2.imshow(window_name, display)
            cv2.waitKey(500)
            break
        if key == ord("r"):
            points.clear()
            display = frame.copy()
            cv2.imshow(window_name, display)
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled by user.")

    cv2.destroyAllWindows()
    return points


def main() -> None:
    """Entry point: grab a frame, collect 4 points, compute + save calibration."""
    args = parse_args()

    if args.non_interactive:
        points = [tuple(map(float, p.split(","))) for p in args.non_interactive.split(";")]
        if len(points) != 4:
            raise SystemExit(f"--non-interactive requires exactly 4 points, got {len(points)}")
    else:
        frame = _grab_reference_frame(args.video)
        points = _collect_points_interactively(frame)

    calibrator = CameraCalibrator()
    try:
        calibration = calibrator.calibrate(
            camera_id=args.camera_id,
            image_points=points,
            lane_width_m=args.lane_width,
            lane_depth_m=args.lane_depth,
            speed_limit_kmph=args.speed_limit,
        )
    except CalibrationError as exc:
        raise SystemExit(f"Calibration failed: {exc}")

    output_path = args.output or (_CALIBRATIONS_DIR / f"{args.camera_id}.json")
    calibration.save(output_path)

    print(f"Calibration saved: {output_path}")
    print(f"  camera_id={calibration.camera_id}")
    print(f"  lane_width_m={calibration.lane_width_m} lane_depth_m={calibration.lane_depth_m}")
    print(f"  speed_limit_kmph={calibration.speed_limit_kmph}")


if __name__ == "__main__":
    main()
