"""
speed_estimator.py
===================
Converts a tracked vehicle's pixel trajectory into a real-world speed
(km/h) using the homography from :mod:`camera_calibrator`.

Method: the vehicle's ground point (bottom-centre of its box, see
``TrackState.ground_point``) is projected onto the road plane; speed is the real-world
distance between the first and last point of the last N frame intervals divided by the
elapsed time. Using the net displacement over a window, rather than averaging
frame-to-frame step lengths, matters: every step carries a little detection jitter, and
summing jittery step lengths always over-estimates (a parked car "moves"). No speed is
reported until a track has ``min_points`` samples, so the first noisy frames of a new
track can never raise a violation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from src.camera_calibrator import CameraCalibration
from src.config import MPS_TO_KMPH, SPEED_MIN_POINTS, SPEED_SMOOTHING_WINDOW, get_logger
from src.tracker import TrackState

logger = get_logger(__name__)

MAX_PLAUSIBLE_KMPH = 250.0  # generous ceiling for road traffic
_SPEED_HISTORY_LEN = 300  # reported speeds kept per track


@dataclass
class SpeedEstimate:
    """Speed estimation result for one vehicle at one point in time.

    Attributes:
        track_id: The vehicle's persistent track ID.
        instant_speed_kmph: Speed from the two most recent points (noisy; diagnostics only).
        smoothed_speed_kmph: Speed over the last ``smoothing_window`` frame intervals
            (net displacement / elapsed time). This is the value used for violations.
        direction: Coarse direction of travel ("inbound"/"outbound"/"unknown").
    """

    track_id: int
    instant_speed_kmph: float
    smoothed_speed_kmph: float
    direction: str


class SpeedEstimator:
    """Computes smoothed, real-world speed for each tracked vehicle.

    Keeps a short per-track history of the speeds it reported (for the dashboard's
    "speed over time" graph), separate from the tracker's raw trajectory.
    """

    def __init__(
        self,
        calibration: CameraCalibration,
        smoothing_window: int = SPEED_SMOOTHING_WINDOW,
        min_points: int = SPEED_MIN_POINTS,
    ) -> None:
        """Initialize with a fixed camera calibration.

        Args:
            calibration: A computed :class:`CameraCalibration` for the
                camera whose frames will be fed to :meth:`estimate`.
            smoothing_window: Number of recent frame intervals the speed is measured over.
            min_points: Trajectory points a track needs before any speed is reported
                (at least 2).
        """
        self.calibration = calibration
        self.smoothing_window = max(1, smoothing_window)
        self.min_points = max(2, min_points)
        self._speed_history: Dict[int, Deque[float]] = {}

    def estimate(self, track: TrackState) -> Optional[SpeedEstimate]:
        """Estimate current speed for one tracked vehicle.

        Args:
            track: The vehicle's current :class:`TrackState`, whose
                ``centroid_history`` supplies the pixel trajectory.

        Returns:
            A :class:`SpeedEstimate`, or None while the track has fewer than
            ``min_points`` samples (warm-up) or its timestamps did not advance.
        """
        history = list(track.centroid_history)
        if len(history) < self.min_points:
            return None

        instant_speed_kmph = self._speed_between(history[-2], history[-1])
        window = history[-(self.smoothing_window + 1):]
        smoothed_speed_kmph = self._speed_between(window[0], window[-1])
        if instant_speed_kmph is None or smoothed_speed_kmph is None:
            return None  # timestamps did not advance

        buffer = self._speed_history.setdefault(track.track_id, deque(maxlen=_SPEED_HISTORY_LEN))
        buffer.append(smoothed_speed_kmph)

        direction = self._estimate_direction(history)
        track.direction = direction

        return SpeedEstimate(
            track_id=track.track_id,
            instant_speed_kmph=instant_speed_kmph,
            smoothed_speed_kmph=smoothed_speed_kmph,
            direction=direction,
        )

    def _speed_between(self, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> Optional[float]:
        """Real-world speed in km/h between two (x_px, y_px, t_s) samples, or None if t did not advance.

        The result is clamped to 0-250 km/h: anything faster is a tracker ID switch or a
        calibration glitch, not a vehicle, and must not become a SEVERE violation.
        """
        (x1, y1, t1), (x2, y2, t2) = start, end
        dt = t2 - t1
        if dt <= 0:
            return None
        wx1, wy1 = self.calibration.pixel_to_world(x1, y1)
        wx2, wy2 = self.calibration.pixel_to_world(x2, y2)
        distance_m = float(np.hypot(wx2 - wx1, wy2 - wy1))
        return float(np.clip(distance_m / dt * MPS_TO_KMPH, 0.0, MAX_PLAUSIBLE_KMPH))

    def get_speed_history(self, track_id: int) -> list:
        """Recent smoothed speeds (km/h) reported for a track, oldest first."""
        return list(self._speed_history.get(track_id, []))

    @staticmethod
    def _estimate_direction(history: list) -> str:
        """Infer coarse travel direction from the trajectory's net displacement.

        Args:
            history: List of (x, y, timestamp_s) centroid samples.

        Returns:
            "inbound" if the vehicle is moving toward increasing world-y
            (deeper into frame), "outbound" if moving away, else "unknown"
            if there isn't enough displacement to tell confidently.
        """
        if len(history) < 5:
            return "unknown"
        (x_start, y_start, _), (x_end, y_end, _) = history[0], history[-1]
        dy = y_end - y_start
        if abs(dy) < 5.0:  # pixels; not enough vertical displacement to be confident
            return "unknown"
        return "inbound" if dy > 0 else "outbound"

    def reset_track(self, track_id: int) -> None:
        """Clear stored speed history for a track (e.g. once it's been reported)."""
        self._speed_history.pop(track_id, None)

    def purge_stale(self, active_track_ids: set) -> None:
        """Drop speed history for tracks no longer active, to bound memory.

        Args:
            active_track_ids: Set of track IDs currently tracked.
        """
        stale = [tid for tid in self._speed_history if tid not in active_track_ids]
        for tid in stale:
            del self._speed_history[tid]
