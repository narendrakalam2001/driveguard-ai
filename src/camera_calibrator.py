"""
camera_calibrator.py
=====================
Converts a camera view into a pixel-to-real-world-meters mapping via a
4-point homography, so the speed estimator can turn pixel displacement
into actual km/h.

Calibration workflow (matches the dashboard's 4-step wizard):
  1. User marks 4 points on the road forming a rectangle of known width
     (default: standard 3.5m Indian lane width) and a chosen depth.
  2. We compute a homography mapping those 4 image points to a
     real-world rectangle in meters.
  3. Any pixel coordinate can then be projected into real-world meters
     via :meth:`CameraCalibration.pixel_to_world`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import DEFAULT_LANE_WIDTH_M, get_logger

logger = get_logger(__name__)


class CalibrationError(ValueError):
    """Raised when calibration points are invalid or degenerate."""


@dataclass
class CameraCalibration:
    """Serializable calibration state for one camera/zone.

    Attributes:
        camera_id: Identifier for the camera this calibration belongs to.
        image_points: 4 [x, y] pixel points, in the order top-left,
            top-right, bottom-right, bottom-left, marking a road rectangle.
        lane_width_m: Real-world width (meters) of the marked rectangle.
        lane_depth_m: Real-world depth/length (meters) of the marked rectangle.
        homography: 3x3 homography matrix mapping image pixels -> world meters.
        speed_limit_kmph: Speed limit configured for this zone.
    """

    camera_id: str
    image_points: List[List[float]]
    lane_width_m: float
    lane_depth_m: float
    homography: List[List[float]]
    speed_limit_kmph: int

    def pixel_to_world(self, x: float, y: float) -> Tuple[float, float]:
        """Project a pixel coordinate into real-world meters.

        Args:
            x: Pixel x-coordinate.
            y: Pixel y-coordinate.

        Returns:
            (world_x_m, world_y_m) in meters, relative to the calibration
            rectangle's top-left corner.
        """
        h = np.array(self.homography, dtype=np.float64)
        point = np.array([x, y, 1.0], dtype=np.float64)
        world = h @ point
        world /= world[2]
        return float(world[0]), float(world[1])

    def save(self, path: Path) -> None:
        """Persist this calibration to a JSON file.

        Args:
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.info(f"Saved calibration for camera_id={self.camera_id} to {path}")

    @classmethod
    def load(cls, path: Path) -> "CameraCalibration":
        """Load a previously saved calibration from JSON.

        Args:
            path: Path to a calibration JSON file.

        Returns:
            A reconstructed :class:`CameraCalibration`.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        data = json.loads(path.read_text())
        return cls(**data)


class CameraCalibrator:
    """Computes a :class:`CameraCalibration` from 4 user-marked road points."""

    def calibrate(
        self,
        camera_id: str,
        image_points: List[Tuple[float, float]],
        lane_width_m: float = DEFAULT_LANE_WIDTH_M,
        lane_depth_m: Optional[float] = None,
        speed_limit_kmph: int = 60,
    ) -> CameraCalibration:
        """Compute a homography from 4 image points to a real-world rectangle.

        Args:
            camera_id: Identifier for the camera being calibrated.
            image_points: Exactly 4 (x, y) pixel points in order
                top-left, top-right, bottom-right, bottom-left, tracing
                a rectangular strip of road.
            lane_width_m: Real-world width of the marked rectangle
                (defaults to the standard 3.5m Indian lane width).
            lane_depth_m: Real-world depth of the marked rectangle. If
                omitted, defaults to 4x the width — a reasonable default
                for a visible stretch of road in typical CCTV framing.
            speed_limit_kmph: Speed limit to associate with this zone.

        Returns:
            A computed :class:`CameraCalibration`.

        Raises:
            CalibrationError: If fewer/more than 4 points are given, or
                the points are degenerate (collinear / zero area).
        """
        if len(image_points) != 4:
            raise CalibrationError(f"Calibration requires exactly 4 points, got {len(image_points)}.")

        if lane_depth_m is None:
            lane_depth_m = lane_width_m * 4

        src = np.array(image_points, dtype=np.float32)
        if self._is_degenerate(src):
            raise CalibrationError("Calibration points are collinear or degenerate; mark 4 distinct corners.")

        dst = np.array(
            [
                [0.0, 0.0],
                [lane_width_m, 0.0],
                [lane_width_m, lane_depth_m],
                [0.0, lane_depth_m],
            ],
            dtype=np.float32,
        )

        homography, _ = cv2.findHomography(src, dst, method=0)
        if homography is None:
            raise CalibrationError("Homography computation failed for the given points.")

        calibration = CameraCalibration(
            camera_id=camera_id,
            image_points=[list(p) for p in image_points],
            lane_width_m=lane_width_m,
            lane_depth_m=lane_depth_m,
            homography=homography.tolist(),
            speed_limit_kmph=speed_limit_kmph,
        )
        logger.info(
            f"Calibrated camera_id={camera_id} lane_width_m={lane_width_m} "
            f"lane_depth_m={lane_depth_m} speed_limit_kmph={speed_limit_kmph}"
        )
        return calibration

    @staticmethod
    def _is_degenerate(points: np.ndarray) -> bool:
        """Check whether 4 points are collinear or form near-zero area.

        Args:
            points: (4, 2) array of pixel coordinates.

        Returns:
            True if the points cannot form a usable quadrilateral.
        """
        # Shoelace formula for polygon area; near-zero means degenerate.
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return bool(area < 1.0)
