"""
tracker.py
==========
Multi-object tracking engine built on DeepSORT.

Wraps ``deep-sort-realtime`` to assign persistent Track IDs to vehicles
across frames, maintain per-vehicle trajectory history, and expose the
data speed estimation and violation classification need — without those
modules having to know anything about the underlying tracker
implementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    DEEPSORT_MAX_AGE,
    DEEPSORT_MAX_COSINE_DISTANCE,
    DEEPSORT_MAX_IOU_DISTANCE,
    DEEPSORT_N_INIT,
    DEEPSORT_NN_BUDGET,
    get_logger,
)
from src.vehicle_detector import Detection

logger = get_logger(__name__)

# How many (x, y, timestamp) points to retain per vehicle for speed
# smoothing and the on-screen trajectory. Bounded so long-lived tracks
# (parked/stalled vehicles across a long clip) don't grow memory unbounded.
_TRAJECTORY_MAXLEN = 300


@dataclass
class TrackState:
    """Everything DriveGuard needs to know about one tracked vehicle.

    Attributes:
        track_id: Persistent integer ID assigned by DeepSORT.
        class_name: Most recently observed vehicle class.
        bbox: Most recent [x1, y1, x2, y2] bounding box.
        centroid_history: Deque of (x, y, timestamp_s) in arrival order, used by the
            speed estimator. (x, y) is the *ground point* (bottom-centre of the box):
            the calibration homography maps the road plane, so a point on the road is
            required - the box centre sits above the road and gives wrong distances.
        first_seen_s: Timestamp (stream-relative seconds) of first detection.
        last_seen_s: Timestamp of the most recent update.
        direction: Coarse travel direction ("inbound"/"outbound"/"unknown"),
            set by the speed estimator once enough history exists.
        confirmed: Whether DeepSORT has confirmed this as a stable track
            (vs. a tentative one that might be a false positive).
    """

    track_id: int
    class_name: str
    bbox: np.ndarray
    centroid_history: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=_TRAJECTORY_MAXLEN)
    )
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    direction: str = "unknown"
    confirmed: bool = False

    def centroid(self) -> Tuple[float, float]:
        """Return the current bounding-box centroid in pixel coordinates."""
        x1, y1, x2, y2 = self.bbox
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    def ground_point(self) -> Tuple[float, float]:
        """Return the bottom-centre of the box: where the vehicle touches the road."""
        x1, _y1, x2, y2 = self.bbox
        return float((x1 + x2) / 2), float(y2)


class VehicleTracker:
    """Persistent multi-object tracker wrapping DeepSORT.

    One instance should be created per video stream / camera session —
    track IDs are only meaningful within a single session.
    """

    def __init__(
        self,
        max_age: int = DEEPSORT_MAX_AGE,
        n_init: int = DEEPSORT_N_INIT,
        max_iou_distance: float = DEEPSORT_MAX_IOU_DISTANCE,
        max_cosine_distance: float = DEEPSORT_MAX_COSINE_DISTANCE,
        nn_budget: int = DEEPSORT_NN_BUDGET,
    ) -> None:
        """Initialize the DeepSORT tracker and internal track-state registry.

        Args:
            max_age: Frames a track survives without a matching detection
                before being deleted (handles brief occlusion).
            n_init: Consecutive detections required before a track is
                "confirmed" rather than "tentative".
            max_iou_distance: Max IoU distance for the association step.
            max_cosine_distance: Max cosine distance for the appearance
                (Re-ID) embedding matching step.
            nn_budget: Max stored appearance samples per track.
        """
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder="mobilenet",  # lightweight; swap for OSNet embedder in production
            half=True,
            bgr=True,
        )
        self._states: Dict[int, TrackState] = {}
        self._stream_time_s: float = 0.0
        logger.info(
            f"VehicleTracker initialized max_age={max_age} n_init={n_init} "
            f"max_iou_distance={max_iou_distance} max_cosine_distance={max_cosine_distance}"
        )

    def update(
        self, detections: List[Detection], frame: np.ndarray, timestamp_s: float
    ) -> List[TrackState]:
        """Advance the tracker by one frame.

        Args:
            detections: Detections for the current frame, from
                :meth:`VehicleDetector.detect`.
            frame: The current BGR frame (DeepSORT uses it to compute
                appearance embeddings for Re-ID).
            timestamp_s: Stream-relative timestamp of this frame, used to
                populate trajectory history for speed estimation.

        Returns:
            List of confirmed :class:`TrackState` objects *seen* on this frame. Tracks that
            DeepSORT is only coasting through an occlusion are kept alive internally (so the
            ID survives re-entry) but are not returned, and no predicted position is added to
            their trajectory - speed is measured from real detections only.
        """
        self._stream_time_s = timestamp_s

        ds_input = []
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            w, h = x2 - x1, y2 - y1
            # deep-sort-realtime expects ([left, top, w, h], confidence, class_name)
            ds_input.append(([float(x1), float(y1), float(w), float(h)], det.confidence, det.class_name))

        tracks = self._tracker.update_tracks(ds_input, frame=frame)

        active_states: List[TrackState] = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            track_id = int(track.track_id)
            l, t, r, b = track.to_ltrb()
            bbox = np.array([l, t, r, b], dtype=np.float32)
            class_name = track.get_det_class() or "Unknown"

            state = self._states.get(track_id)
            if state is None:
                state = TrackState(
                    track_id=track_id,
                    class_name=class_name,
                    bbox=bbox,
                    first_seen_s=timestamp_s,
                    last_seen_s=timestamp_s,
                    confirmed=True,
                )
                self._states[track_id] = state
                logger.info(f"New confirmed track_id={track_id} class={class_name}")
            else:
                state.bbox = bbox
                state.class_name = class_name
                state.last_seen_s = timestamp_s
                state.confirmed = True

            gx, gy = state.ground_point()
            state.centroid_history.append((gx, gy, timestamp_s))
            active_states.append(state)

        self._prune_stale_tracks(active_track_ids={s.track_id for s in active_states})
        return active_states

    def _prune_stale_tracks(self, active_track_ids: set) -> None:
        """Drop internal state for tracks DeepSORT has fully deleted.

        Args:
            active_track_ids: Set of track IDs DeepSORT reported this frame.
        """
        stale_ids = [
            tid
            for tid, state in self._states.items()
            if tid not in active_track_ids and (self._stream_time_s - state.last_seen_s) > 5.0
        ]
        for tid in stale_ids:
            del self._states[tid]

    def get_state(self, track_id: int) -> Optional[TrackState]:
        """Look up the current state for a specific track ID."""
        return self._states.get(track_id)

    def all_states(self) -> List[TrackState]:
        """Return every currently tracked vehicle's state (confirmed or not yet pruned)."""
        return list(self._states.values())

    def track_count(self) -> int:
        """Return the number of vehicles currently being tracked."""
        return len(self._states)
