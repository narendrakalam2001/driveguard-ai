"""
detection_service.py
======================
Orchestrates the full DriveGuard AI pipeline for one camera session:
detect -> track -> estimate speed -> classify violation -> recognize
plate -> log alert -> (optionally) generate an LLM notice.

This is the single object ``serving/api.py`` talks to — it hides the
wiring between the six Module classes so the API layer stays thin.

Logging rules (so the violations table holds events, not video frames):
  * A vehicle is logged when it first reaches a severity band (WARNING, then VIOLATION,
    then SEVERE) - at most three rows per vehicle, never one row per frame.
  * A plate read after the first row was written is attached to that vehicle's rows.
  * When a vehicle has not been seen for ``VEHICLE_LOST_AFTER_S`` (or the session ends) its
    final stats are saved to the ``vehicles`` table.
"""

from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from src.alert_system import AlertSystem
from src.camera_calibrator import CameraCalibration
from src.config import (
    MAX_ACCEL_KMPH_PER_S,
    MAX_REPORTED_SPEED_KMPH,
    SPEED_DECISION_WINDOW,
    SPEED_JUMP_SLACK_KMPH,
    SPEED_REBASE_AFTER,
    PLATE_DEBUG_CROPS,
    PLATE_DEBUG_DIR,
    PLATE_DEBUG_MAX_CROPS,
    PLATE_MIN_VEHICLE_WIDTH_PX,
    PLATE_RETRY_GROWTH,
    MIN_SPEED_SAMPLES,
    PLATE_MAX_READ_ATTEMPTS,
    PLATE_READ_INTERVAL_FRAMES,
    PLATE_UNCLEAR_LABEL,
    SEVERITY,
    VEHICLE_LOST_AFTER_S,
    get_logger,
)
from src.data_loader import VideoStream
from src.plate_recognizer import PlateRecognizer
from src.plate_text import format_plate_for_display
from src.speed_estimator import SpeedEstimator
from src.tracker import TrackState, VehicleTracker
from src.vehicle_detector import VehicleDetector
from src.violation_classifier import ViolationClassifier, ViolationRecord

logger = get_logger(__name__)

_SEVERITY_RANK = {SEVERITY.NORMAL: 0, SEVERITY.WARNING: 1, SEVERITY.VIOLATION: 2, SEVERITY.SEVERE: 3}


@dataclass
class VehicleInfo:
    """What the dashboard shows for one vehicle currently in frame.

    Attributes:
        track_id: Persistent track id.
        vehicle_type: Detector class (Car, Truck, ...).
        speed_kmph: Latest valid speed, or None while still measuring / when the reading was implausible.
        plate: Recognised plate (compact form, e.g. ``MH12AB1234``), if any.
        severity: Current severity band (``NORMAL`` until the speed is stable).
        stable: True once enough speed readings exist for this vehicle to be flagged.
    """

    track_id: int
    vehicle_type: str
    speed_kmph: Optional[float]
    plate: Optional[str]
    severity: str
    stable: bool


@dataclass
class FrameResult:
    """Everything the API/dashboard needs to render one processed frame.

    Attributes:
        frame_index: Sequential frame number within this session.
        annotated_frame: BGR frame with detection/tracking overlays drawn.
        active_tracks: Track states active this frame.
        violations_this_frame: Violation records newly logged on this frame (a vehicle
            appears here only when it first reaches a more severe band, not on every
            frame it stays over the limit).
        vehicle_count: Number of vehicles currently in frame.
    """

    frame_index: int
    annotated_frame: np.ndarray
    active_tracks: List[TrackState]
    violations_this_frame: List[ViolationRecord]
    vehicle_count: int
    vehicles: List[VehicleInfo] = field(default_factory=list)


@dataclass
class SessionSummary:
    """Aggregate stats for a completed video-processing session.

    Attributes:
        session_id: Unique identifier for this processing run.
        total_frames: Number of frames processed.
        total_vehicles: Number of unique tracked vehicles (those that got a speed reading).
        total_violations: Number of unique vehicles that reached WARNING or worse.
        average_speed_kmph: Mean of each vehicle's own mean speed (a vehicle that stays in
            view longer does not count more).
        max_speed_kmph: Highest smoothed speed observed.
        max_speed_plate: Plate number associated with the max speed, if known.
    """

    session_id: str
    total_frames: int
    total_vehicles: int
    total_violations: int
    average_speed_kmph: float
    max_speed_kmph: float
    max_speed_plate: Optional[str]
    plates_read: int = 0              # vehicles whose plate was read and accepted
    plate_attempts: int = 0           # OCR attempts made
    plate_boxes_found: int = 0        # attempts where the plate detector found a plate (read or unclear)
    vehicles_too_small_for_plate: int = 0  # vehicles that never got wide enough (see PLATE_MIN_VEHICLE_WIDTH_PX)
    plates_rejected_format: int = 0   # plate found, text read, but it is not a valid plate of the configured region
    plates_rejected_confidence: int = 0  # valid-looking text, but OCR confidence was below the threshold
    plate_region: str = ""            # plate format in force (IN / UK / ANY)
    plate_min_confidence: float = 0.0


@dataclass
class _VehicleStats:
    """Running per-vehicle bookkeeping for one session (not persisted until the vehicle is done)."""

    first_seen_iso: str
    last_seen_iso: str
    last_seen_s: float
    vehicle_type: str = "Unknown"
    speed_sum: float = 0.0
    speed_count: int = 0
    max_speed: float = 0.0
    plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    plate_attempts: int = 0
    next_plate_frame: int = 0
    last_plate_width: int = 0
    last_good_speed: Optional[float] = None   # last reading that passed the physical-plausibility check
    last_good_t: float = 0.0
    rejected_run: int = 0                     # consecutive readings ignored as glitches
    last_severity: str = SEVERITY.NORMAL
    reported_rank: int = 0
    violation_ids: List[str] = field(default_factory=list)
    plate_crop: Optional[np.ndarray] = None
    speed_trace: List[float] = field(default_factory=list)  # smoothed km/h per frame, capped (see _TRACE_MAX)

    @property
    def avg_speed(self) -> Optional[float]:
        """Mean smoothed speed, or None if no speed was ever measured."""
        return self.speed_sum / self.speed_count if self.speed_count else None


_MAX_PLATE_READ_CROPS = 300  # one small crop per vehicle whose plate was read
_TRACE_MAX = 900  # per-vehicle speed samples kept for the detail-screen graph

_SEVERITY_COLOR_BGR = {
    SEVERITY.NORMAL: (0, 200, 0),
    SEVERITY.WARNING: (0, 200, 230),
    SEVERITY.VIOLATION: (0, 0, 230),
    SEVERITY.SEVERE: (0, 0, 255),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DetectionService:
    """High-level facade running the full per-frame DriveGuard pipeline."""

    def __init__(
        self,
        calibration: CameraCalibration,
        camera_id: str,
        detector: Optional[VehicleDetector] = None,
        plate_recognizer: Optional[PlateRecognizer] = None,
        alert_system: Optional[AlertSystem] = None,
        enable_plate_recognition: bool = True,
        session_id: Optional[str] = None,
        tracker: Optional[VehicleTracker] = None,
        speed_estimator: Optional[SpeedEstimator] = None,
    ) -> None:
        """Wire up all pipeline components for one camera session.

        Args:
            calibration: A computed :class:`CameraCalibration` for this camera.
            camera_id: Identifier for this camera (used for logging/DB rows).
            detector: Pre-built :class:`VehicleDetector`; constructed lazily if None.
            plate_recognizer: Pre-built :class:`PlateRecognizer`; constructed
                lazily if None and ``enable_plate_recognition`` is True.
            alert_system: Pre-built :class:`AlertSystem`; constructed if None.
            enable_plate_recognition: Set False to skip ANPR entirely (faster,
                useful for CPU-only smoke testing).
            session_id: Identifier written to the ``vehicles`` table; a UUID if omitted.
            tracker: Pre-built :class:`VehicleTracker` (tests); a new one if None.
            speed_estimator: Pre-built :class:`SpeedEstimator` (tests); a new one if None.
        """
        self.camera_id = camera_id
        self.calibration = calibration
        self.session_id = session_id or str(uuid.uuid4())
        self.enable_plate_recognition = enable_plate_recognition

        self.detector = detector or VehicleDetector()
        self.tracker = tracker or VehicleTracker()
        self.speed_estimator = speed_estimator or SpeedEstimator(calibration=calibration)
        self.classifier = ViolationClassifier(
            speed_limit_kmph=calibration.speed_limit_kmph,
            location=camera_id,
            camera_id=camera_id,
        )
        self.plate_recognizer = plate_recognizer if not enable_plate_recognition else (
            plate_recognizer or PlateRecognizer()
        )
        self.alert_system = alert_system or AlertSystem()
        self.alert_system.register_camera(
            camera_id=camera_id,
            location_name=camera_id,
            speed_limit=calibration.speed_limit_kmph,
            calibration_matrix=calibration.homography,
        )

        self._stats: Dict[int, _VehicleStats] = {}
        self._seen_vehicles: set = set()      # every track that got a speed reading
        self._flagged_vehicles: set = set()   # tracks that reached WARNING or worse
        self._plate_counts = {"attempts": 0, "boxes_found": 0, "read": 0, "rejected_format": 0, "rejected_confidence": 0}
        self._plate_rejects_logged = 0
        self._plate_big_enough: set = set()   # tracks that were ever wide enough to try
        self._plate_debug_saved = 0
        self._plate_reads_saved = 0
        self._mean_speed_of_finished: List[float] = []
        self._frames_processed = 0
        self.fps: Optional[float] = None  # source frame rate, set from the video when known
        self.processing_fps: float = 0.0  # how fast this service is actually processing frames (EMA)
        self._last_tick: Optional[float] = None
        self._max_speed = 0.0
        self._max_speed_plate: Optional[str] = None
        self._max_speed_track_id: Optional[int] = None

        logger.info(f"DetectionService initialized for camera_id={camera_id} session_id={self.session_id}")

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp_s: float) -> FrameResult:
        """Run detection -> tracking -> speed -> classification -> logging on one frame.

        Args:
            frame: BGR frame.
            frame_index: Sequential frame number.
            timestamp_s: Stream-relative timestamp in seconds.

        Returns:
            A :class:`FrameResult` with the annotated frame and any newly logged violations.
        """
        self._frames_processed += 1
        tick = time.perf_counter()
        if self._last_tick is not None and tick > self._last_tick:
            inst = 1.0 / (tick - self._last_tick)
            self.processing_fps = inst if self.processing_fps == 0.0 else 0.9 * self.processing_fps + 0.1 * inst
        self._last_tick = tick
        detections = self.detector.detect(frame)
        active_tracks = self.tracker.update(detections, frame, timestamp_s)

        violations_this_frame: List[ViolationRecord] = []
        vehicles: List[VehicleInfo] = []
        annotated = frame.copy()
        now_iso = _now_iso()

        for track in active_tracks:
            stats = self._stats.get(track.track_id)
            if stats is None:
                stats = _VehicleStats(first_seen_iso=now_iso, last_seen_iso=now_iso, last_seen_s=timestamp_s)
                self._stats[track.track_id] = stats
            stats.last_seen_iso, stats.last_seen_s, stats.vehicle_type = now_iso, timestamp_s, track.class_name

            speed_estimate = self.speed_estimator.estimate(track)
            if speed_estimate is None:
                self._draw_box(annotated, track, speed_kmph=None, severity=SEVERITY.NORMAL, plate=stats.plate)
                vehicles.append(VehicleInfo(track.track_id, track.class_name, None, stats.plate, SEVERITY.NORMAL, False))
                continue

            raw_speed = speed_estimate.smoothed_speed_kmph
            if raw_speed >= MAX_REPORTED_SPEED_KMPH:
                # a tracker ID switch or a calibration problem, not a real speed: show the box, ignore the number
                self._draw_box(annotated, track, speed_kmph=None, severity=SEVERITY.NORMAL, plate=stats.plate)
                vehicles.append(VehicleInfo(track.track_id, track.class_name, None, stats.plate, SEVERITY.NORMAL, False))
                continue

            if not self._accept_speed(stats, raw_speed, timestamp_s):
                # a glitch (jump faster than a car can accelerate): keep showing the last good state
                shown = stats.last_good_speed
                self._draw_box(annotated, track, shown, stats.last_severity, stats.plate)
                vehicles.append(VehicleInfo(track.track_id, track.class_name, shown, stats.plate, stats.last_severity,
                                            len(stats.speed_trace) >= MIN_SPEED_SAMPLES))
                continue

            if len(stats.speed_trace) < _TRACE_MAX:
                stats.speed_trace.append(raw_speed)
            self._seen_vehicles.add(track.track_id)
            self._maybe_read_plate(track, stats, frame, frame_index)

            stable = len(stats.speed_trace) >= MIN_SPEED_SAMPLES
            if not stable:
                # still measuring: draw the reading, but do not count it or flag the vehicle yet
                self._draw_box(annotated, track, raw_speed, SEVERITY.NORMAL, stats.plate)
                vehicles.append(VehicleInfo(track.track_id, track.class_name, raw_speed, stats.plate, SEVERITY.NORMAL, False))
                continue

            speed = statistics.median(stats.speed_trace[-SPEED_DECISION_WINDOW:])  # median of the last readings: a noisy frame cannot flag a car
            stats.speed_sum += speed
            stats.speed_count += 1
            if speed > stats.max_speed:
                stats.max_speed = speed
            if speed > self._max_speed:
                self._max_speed, self._max_speed_plate, self._max_speed_track_id = speed, stats.plate, track.track_id

            record = self.classifier.classify(replace(speed_estimate, smoothed_speed_kmph=speed), track.class_name, stats.plate)
            stats.last_severity = record.severity
            self._draw_box(annotated, track, speed, record.severity, stats.plate)
            vehicles.append(VehicleInfo(track.track_id, track.class_name, speed, stats.plate, record.severity, True))

            rank = _SEVERITY_RANK.get(record.severity, 0)
            if rank > stats.reported_rank:
                stats.reported_rank = rank
                self._flagged_vehicles.add(track.track_id)
                evidence_frame = annotated.copy() if rank >= _SEVERITY_RANK[SEVERITY.VIOLATION] else None
                if stats.violation_ids:  # same vehicle, worse band: upgrade its one row instead of adding another
                    self.alert_system.escalate_violation(stats.violation_ids[0], record, evidence_frame)
                else:
                    violation_id = self.alert_system.log_violation(
                        record, plate_confidence=stats.plate_confidence, evidence_frame=evidence_frame
                    )
                    if violation_id:
                        stats.violation_ids.append(violation_id)
                        if stats.plate_crop is not None:
                            self._safe_save_plate_crop(stats.plate_crop, violation_id)
                violations_this_frame.append(record)

        self._draw_hud(annotated, len(active_tracks))
        active_ids = {t.track_id for t in active_tracks}
        self.speed_estimator.purge_stale(active_ids)
        self._flush_lost_vehicles(timestamp_s, active_ids)

        return FrameResult(
            frame_index=frame_index,
            annotated_frame=annotated,
            active_tracks=active_tracks,
            violations_this_frame=violations_this_frame,
            vehicle_count=len(active_tracks),
            vehicles=vehicles,
        )

    def _maybe_read_plate(self, track: TrackState, stats: _VehicleStats, frame: np.ndarray, frame_index: int) -> None:
        """Try to read a vehicle's plate, but only until it succeeds and not on every frame.

        OCR is the slowest step, so a vehicle is only tried once it is wide enough (``PLATE_MIN_VEHICLE_WIDTH_PX``),
        then retried at most every ``PLATE_READ_INTERVAL_FRAMES`` frames - or sooner when it has grown
        ``PLATE_RETRY_GROWTH`` times wider (it is getting closer) - and abandoned after ``PLATE_MAX_READ_ATTEMPTS``.
        A successful read is attached to any violation rows already logged for the vehicle.

        Args:
            track: The vehicle's current track state.
            stats: The vehicle's running stats (plate/attempt bookkeeping is stored here).
            frame: Current full frame, cropped to the vehicle's box.
            frame_index: Current frame number, used for the retry interval.
        """
        if (
            not self.enable_plate_recognition
            or self.plate_recognizer is None
            or stats.plate is not None
            or stats.plate_attempts >= PLATE_MAX_READ_ATTEMPTS
        ):
            return

        x1, y1, x2, y2 = track.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return
        width = x2 - x1
        if width < PLATE_MIN_VEHICLE_WIDTH_PX:
            return  # still too far away for a readable plate: do not spend an attempt on it
        self._plate_big_enough.add(track.track_id)
        # Retry on the interval, but as the vehicle approaches (grows) try again straight away: closer = sharper plate.
        if stats.plate_attempts and width < stats.last_plate_width * PLATE_RETRY_GROWTH and frame_index < stats.next_plate_frame:
            return

        stats.plate_attempts += 1
        stats.last_plate_width = width
        stats.next_plate_frame = frame_index + PLATE_READ_INTERVAL_FRAMES
        self._plate_counts["attempts"] += 1
        vehicle_crop = frame[y1:y2, x1:x2]
        try:
            result = self.plate_recognizer.read_plate(vehicle_crop)
        except Exception:  # one bad crop must not stop the video
            logger.exception(f"Plate recognition failed for track_id={track.track_id}; skipping")
            return
        if result is None:  # the plate detector found no plate on this vehicle
            self._save_plate_debug(vehicle_crop, track.track_id, frame_index, "nobox")
            return
        self._plate_counts["boxes_found"] += 1
        if result.plate_number == PLATE_UNCLEAR_LABEL:  # a plate was found but the text could not be trusted
            reason = "confidence" if result.is_valid_format else "format"
            self._plate_counts["rejected_" + reason] += 1
            raw = "".join(ch for ch in (result.raw_text or "") if ch.isalnum())[:12]
            if self._plate_rejects_logged < 15:  # enough lines to diagnose a run without flooding the log
                self._plate_rejects_logged += 1
                logger.info(
                    f"Plate rejected ({reason}) track_id={track.track_id} frame={frame_index} raw_ocr={raw!r} "
                    f"candidate={result.candidate!r} confidence={result.confidence:.2f} "
                    f"region={getattr(self.plate_recognizer, 'region', '?')} "
                    f"min_confidence={getattr(self.plate_recognizer, 'confidence_threshold', '?')}"
                )
            self._save_plate_debug(getattr(result, "crop", None), track.track_id, frame_index, f"unclear-{reason}-{raw or 'empty'}-c{int(result.confidence * 100)}")
            return
        self._plate_counts["read"] += 1

        stats.plate, stats.plate_confidence = result.plate_number, result.confidence
        stats.plate_crop = getattr(result, "crop", None)
        self._save_plate_read(stats.plate_crop, track.track_id, result.plate_number, result.confidence)
        if track.track_id == self._max_speed_track_id:
            self._max_speed_plate = result.plate_number
        for violation_id in stats.violation_ids:
            self.alert_system.update_plate(violation_id, result.plate_number, result.confidence)
            if stats.plate_crop is not None:
                self._safe_save_plate_crop(stats.plate_crop, violation_id)

    @staticmethod
    def _accept_speed(stats: "_VehicleStats", raw_speed: float, timestamp_s: float) -> bool:
        """Physical plausibility filter: reject a reading that jumps further than a car can accelerate.

        A tracker ID switch or a jittering box shows up as a sudden jump (e.g. 8 -> 120 km/h in one frame). Such
        readings are ignored; if they persist for ``SPEED_REBASE_AFTER`` readings in a row the new level is accepted.

        Returns:
            True if the reading should be used.
        """
        if stats.last_good_speed is not None and stats.rejected_run < SPEED_REBASE_AFTER:
            dt = max(timestamp_s - stats.last_good_t, 1e-3)
            if abs(raw_speed - stats.last_good_speed) > MAX_ACCEL_KMPH_PER_S * dt + SPEED_JUMP_SLACK_KMPH:
                stats.rejected_run += 1
                return False
        stats.last_good_speed, stats.last_good_t, stats.rejected_run = raw_speed, timestamp_s, 0
        return True

    def _save_plate_read(self, crop: Optional[np.ndarray], track_id: int, plate: str, confidence: float) -> None:
        """Keep the crop of every plate that was read, named ``read-track<id>-<PLATE>-c<confidence%>.jpg``, so the reads can be checked by eye."""
        if crop is None or getattr(crop, "size", 0) == 0 or self._plate_reads_saved >= _MAX_PLATE_READ_CROPS:
            return
        safe = "".join(ch for ch in plate.upper() if ch.isalnum())
        try:
            folder = PLATE_DEBUG_DIR / self.session_id
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"read-track{track_id}-{safe}-c{int(confidence * 100)}.jpg"), crop)
            self._plate_reads_saved += 1
        except Exception:
            logger.exception("Could not save the crop of a plate that was read")

    def _save_plate_debug(self, image: Optional[np.ndarray], track_id: int, frame_index: int, kind: str) -> None:
        """Keep a few failed-attempt crops (vehicle crop if no plate box, plate crop if unreadable) so failures can be inspected."""
        if not PLATE_DEBUG_CROPS or image is None or getattr(image, "size", 0) == 0 or self._plate_debug_saved >= PLATE_DEBUG_MAX_CROPS:
            return
        try:
            folder = PLATE_DEBUG_DIR / self.session_id
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"track{track_id}_frame{frame_index}_{kind}.jpg"), image)
            self._plate_debug_saved += 1
        except Exception:
            logger.exception("Could not save plate debug crop")

    def _safe_save_plate_crop(self, crop: np.ndarray, violation_id: str) -> None:
        """Save a plate crop as evidence; an I/O problem must never stop the video."""
        try:
            self.alert_system.save_plate_crop(crop, violation_id)
        except Exception:
            logger.exception(f"Could not save plate crop for violation_id={violation_id}")

    def _flush_lost_vehicles(self, timestamp_s: float, active_ids: set) -> None:
        """Save and forget vehicles that have not been seen for ``VEHICLE_LOST_AFTER_S``."""
        lost = [
            tid for tid, st in self._stats.items()
            if tid not in active_ids and timestamp_s - st.last_seen_s > VEHICLE_LOST_AFTER_S
        ]
        for tid in lost:
            self._save_vehicle(tid)

    def _save_vehicle(self, track_id: int) -> None:
        """Write one finished vehicle's stats to the DB and drop it from memory."""
        stats = self._stats.pop(track_id, None)
        if stats is None or stats.speed_count == 0:
            return  # never got a speed reading (e.g. a short false track)
        self._mean_speed_of_finished.append(stats.avg_speed or 0.0)
        for violation_id in stats.violation_ids:
            try:
                self.alert_system.save_speed_trace(violation_id, stats.speed_trace, fps=self.fps)
            except Exception:
                logger.exception(f"Could not save speed trace for violation_id={violation_id}")
        try:
            self.alert_system.save_vehicle_session(
                track_id=track_id,
                session_id=self.session_id,
                first_seen_iso=stats.first_seen_iso,
                last_seen_iso=stats.last_seen_iso,
                avg_speed=round(stats.avg_speed or 0.0, 1),
                max_speed=round(stats.max_speed, 1),
                plate_number=stats.plate,
                vehicle_type=stats.vehicle_type,
                violation_count=len(stats.violation_ids),
            )
        except Exception:
            logger.exception(f"Could not save vehicle stats for track_id={track_id}")

    def finalize(self) -> SessionSummary:
        """Save every vehicle still in memory and return the session summary.

        Call once when the video ends or the stream stops (``process_video`` does this itself).
        """
        for track_id in list(self._stats):
            self._save_vehicle(track_id)
        avg = float(np.mean(self._mean_speed_of_finished)) if self._mean_speed_of_finished else 0.0
        logger.info(
            f"Plate stats session={self.session_id}: vehicles={len(self._seen_vehicles)} attempts={self._plate_counts['attempts']} "
            f"plate_found={self._plate_counts['boxes_found']} read={self._plate_counts['read']} "
            f"too_small={len(self._seen_vehicles - self._plate_big_enough)} "
            f"rejected_format={self._plate_counts['rejected_format']} rejected_confidence={self._plate_counts['rejected_confidence']} "
            f"region={getattr(self.plate_recognizer, 'region', '?')}"
        )
        return SessionSummary(
            session_id=self.session_id,
            total_frames=self._frames_processed,
            total_vehicles=len(self._seen_vehicles),
            total_violations=len(self._flagged_vehicles),
            average_speed_kmph=round(avg, 1),
            max_speed_kmph=round(self._max_speed, 1),
            max_speed_plate=self._max_speed_plate,
            plates_read=self._plate_counts["read"],
            plate_attempts=self._plate_counts["attempts"],
            plate_boxes_found=self._plate_counts["boxes_found"],
            vehicles_too_small_for_plate=len(self._seen_vehicles - self._plate_big_enough),
            plates_rejected_format=self._plate_counts["rejected_format"],
            plates_rejected_confidence=self._plate_counts["rejected_confidence"],
            plate_region=str(getattr(self.plate_recognizer, "region", "") or ""),
            plate_min_confidence=float(getattr(self.plate_recognizer, "confidence_threshold", 0.0) or 0.0),
        )

    @staticmethod
    def _text_scale(frame: np.ndarray) -> tuple:
        """Font scale and thickness that stay readable from 360p to 4K."""
        w = frame.shape[1]
        return max(0.5, w / 1100), max(1, round(w / 700))

    @staticmethod
    def _draw_chip(frame: np.ndarray, text: str, org: tuple, bg: tuple, fg: tuple, scale: float, thick: int) -> int:
        """Draw text on a filled rectangle with its bottom-left at ``org``; returns the chip height."""
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        pad = max(2, thick + 1)
        cv2.rectangle(frame, (x, y - th - pad), (x + tw + 2 * pad, y + base + pad // 2), bg, -1)
        cv2.putText(frame, text, (x + pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)
        return th + base + pad * 2

    def _draw_hud(self, frame: np.ndarray, vehicle_count: int) -> None:
        """Draw the calibrated speed zone, a limit chip (top-left), vehicle count and FPS (bottom)."""
        h, w = frame.shape[:2]
        scale, thick = self._text_scale(frame)
        pts = np.array(self.calibration.image_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=(192, 101, 21), thickness=max(1, thick))  # police blue (BGR)
        self._draw_chip(frame, f"Speed zone {self.calibration.speed_limit_kmph} km/h", (8, int(24 * scale) + 8), (192, 101, 21), (255, 255, 255), scale, thick)
        self._draw_chip(frame, f"Vehicles: {vehicle_count}", (8, h - 10), (40, 40, 40), (255, 255, 255), scale, thick)
        fps_text = f"{self.processing_fps:.1f} FPS"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        self._draw_chip(frame, fps_text, (w - tw - 16, h - 10), (40, 40, 40), (255, 255, 255), scale, thick)

    def _draw_box(
        self,
        frame: np.ndarray,
        track: TrackState,
        speed_kmph: Optional[float],
        severity: str,
        plate: Optional[str],
    ) -> None:
        """Draw the box with a readable label: ``Car #63  54 km/h`` and, once read, the plate below it.

        Args:
            frame: Frame to draw on, mutated in place.
            track: The vehicle's current track state.
            speed_kmph: Speed to show, or None while it is being measured / was implausible.
            severity: Severity band, used to choose the colour.
            plate: Recognised plate text, if any.
        """
        x1, y1, x2, y2 = track.bbox.astype(int)
        color = _SEVERITY_COLOR_BGR.get(severity, (0, 200, 0))
        scale, thick = self._text_scale(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, thick + 1))

        first = f"{track.class_name} #{track.track_id}"
        first += f"  {speed_kmph:.0f} km/h" if speed_kmph is not None else "  measuring..."
        lines = [first] + ([format_plate_for_display(plate)] if plate else [])
        fg = (0, 0, 0) if severity in (SEVERITY.WARNING, SEVERITY.NORMAL) else (255, 255, 255)
        y = y1 - 4
        if y - len(lines) * int(28 * scale) < 0:  # no room above the box: stack the labels inside its top edge
            y = y1 + int(24 * scale)
        for line in lines:
            used = self._draw_chip(frame, line, (max(0, x1), y), color, fg, scale, thick)
            y += used if y < y1 else -used

    def process_video(
        self,
        source: str,
        session_id: Optional[str] = None,
        on_frame: Optional[Callable[[FrameResult], None]] = None,
        frame_stride: int = 1,
    ) -> SessionSummary:
        """Process an entire video file end-to-end and return a summary.

        Args:
            source: Path to a video file.
            session_id: Optional session identifier; overrides the one given at construction.
            on_frame: Optional callback run with every :class:`FrameResult` (used by the API to
                publish the live annotated preview). Exceptions in it are logged, not raised.
            frame_stride: Process every Nth frame (1 = every frame). Speeds stay correct because they use
                the real timestamps, but the work (and accuracy of fast changes) drops by N.

        Returns:
            A :class:`SessionSummary` for the completed run.
        """
        if session_id:
            self.session_id = session_id

        try:
            with VideoStream(source) as stream:
                self.fps = getattr(stream, "fps", None) or self.fps
                for packet in stream.frames():
                    if frame_stride > 1 and packet.frame_index % frame_stride:
                        continue
                    result = self.process_frame(packet.frame, packet.frame_index, packet.timestamp_s)
                    if on_frame is not None:
                        try:
                            on_frame(result)
                        except Exception:
                            logger.exception("on_frame callback failed; continuing")
        finally:
            summary = self.finalize()
        logger.info(f"Session {self.session_id} complete: {summary}")
        return summary
