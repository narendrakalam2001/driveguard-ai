"""
api.py
======
FastAPI serving layer for DriveGuard AI.

Core endpoints:
    POST /process-video     -- upload a video, process it in the background
    POST /process-stream    -- start processing a live camera/webcam stream
    GET  /violations        -- list/filter logged violations (date, type, severity, state, status)
    GET  /violation/{id}    -- fetch one violation's full detail
    GET  /analytics         -- dashboard KPI + breakdown stats
    POST /generate-report   -- draft an LLM notice for a violation
    GET  /health            -- liveness/readiness check

Dashboard support endpoints:
    GET   /sessions, /process-status/{id}, /live-frame/{id}, /live-stats
    PATCH /violation/{id}/status        PUT  /violation/{id}/notice
    POST  /violation/{id}/send-to-rto   (mock integration, audited)
    GET   /violation/{id}/evidence      GET  /violation/{id}/speed-trace   GET /violation/{id}/notice.pdf
    GET   /analytics/detailed           GET  /analytics/insights
    GET   /export/violations?format=csv|xlsx     GET /reports/summary.pdf
    GET   /cameras                      GET  /integrations

If ``DRIVEGUARD_API_KEY`` is set, every state-changing endpoint requires the header ``X-API-Key``.
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from src.alert_system import AlertSystem
from src.camera_calibrator import CameraCalibration, CameraCalibrator
from src.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    ANNOTATED_VIDEO_DIR,
    API_KEY,
    STREAM_VIDEO_FPS,
    CORS_ALLOW_ORIGINS,
    EVIDENCE_FRAMES_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    MAX_EXPORT_ROWS,
    MAX_UPLOAD_SIZE_MB,
    PLATE_CONFIDENCE_THRESHOLD,
    PLATE_DEBUG_DIR,
    PLATE_REGION,
    PROJECT_ROOT,
    get_logger,
)
from src.violation_classifier import ViolationRecord
from services.detection_service import DetectionService, SessionSummary
from src.plate_recognizer import PlateRecognizer
from src.vehicle_detector import VehicleDetector
from src.video_writer import AnnotatedVideoWriter

logger = get_logger(__name__)

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Start loading the models in the background as the API starts (set PRELOAD_MODELS=false to disable)."""
    if os.getenv("PRELOAD_MODELS", "true").lower() == "true":
        threading.Thread(target=_preload_models, daemon=True, name="preload-models").start()
    yield


app = FastAPI(
    title="DriveGuard AI API",
    description="Real-Time Traffic Violation Detection & Intelligence System",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,  # set CORS_ALLOW_ORIGINS to the dashboard origin(s) in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_alert_system = AlertSystem()
_UPLOAD_DIR = PROJECT_ROOT / "sample_videos" / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory session registry: session_id -> status dict.
# For a multi-worker production deployment this would move to Redis;
# a single-process dict is sufficient for this project's serving target.
_SESSIONS: Dict[str, dict] = {}
_SESSIONS_LOCK = threading.Lock()

# Latest annotated JPEG + counters per session, for the dashboard's live preview.
_LIVE: Dict[str, dict] = {}
_LIVE_LOCK = threading.Lock()
_LIVE_EVERY_N_FRAMES = 1       # JPEG-encode every processed frame (the pipeline itself runs at only a few FPS on CPU)
_LIVE_MAX_WIDTH = 960
_SEVERITIES = "^(WARNING|VIOLATION|SEVERE)$"
_STATUSES = "^(PENDING|RESOLVED|DISPUTED)$"
_DATE_RE = r"^\d{4}-\d{2}-\d{2}$"
_INSIGHT_CACHE: Dict[tuple, dict] = {}
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_STORED_VIDEOS = 20          # oldest processed videos are deleted beyond this

# Models are loaded once per process (loading YOLO + EasyOCR takes ~30 s) and shared by all sessions; the
# lock makes videos run one at a time because the models are not safe to call from two threads at once.
_MODELS: Dict[str, object] = {}
_MODELS_LOCK = threading.Lock()
_PIPELINE_LOCK = threading.Lock()


def _get_shared_models(enable_plate: bool):
    """Return the process-wide ``(detector, plate_recognizer_or_None)``, loading them on first use."""
    with _MODELS_LOCK:
        if "detector" not in _MODELS:
            _MODELS["detector"] = VehicleDetector()
        if enable_plate and "plate" not in _MODELS:
            _MODELS["plate"] = PlateRecognizer()
        return _MODELS["detector"], (_MODELS.get("plate") if enable_plate else None)


def _preload_models() -> None:
    """Warm the models in the background so the first uploaded video does not wait for loading."""
    try:
        _, plate = _get_shared_models(True)
        logger.info(
            f"Models preloaded; plate reading: region={getattr(plate, 'region', '?')} "
            f"min_confidence={getattr(plate, 'confidence_threshold', '?')} (set PLATE_REGION / PLATE_CONFIDENCE_THRESHOLD in .env)"
        )
    except Exception:  # noqa: BLE001 - a missing weight file must not stop the API from starting
        logger.exception("Model preload failed; models will load on the first video")



def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Dependency guarding state-changing endpoints when ``DRIVEGUARD_API_KEY`` is configured.

    Args:
        x_api_key: Value of the ``X-API-Key`` request header.

    Raises:
        HTTPException: 401 if a key is configured and the header is missing or wrong.
    """
    if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(401, "Missing or invalid X-API-Key header.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CalibrationPoint(BaseModel):
    x: float
    y: float


class ProcessVideoRequest(BaseModel):
    camera_id: str = Field(..., description="Identifier for this camera/zone.")
    speed_limit_kmph: int = Field(60, ge=5, le=140)
    lane_width_m: float = Field(3.5, gt=0)
    calibration_points: List[CalibrationPoint] = Field(
        ..., min_length=4, max_length=4, description="4 points: TL, TR, BR, BL."
    )
    enable_plate_recognition: bool = True


class ProcessStreamRequest(BaseModel):
    camera_id: str
    stream_source: str = Field(..., description="Webcam index as string (e.g. '0') or RTSP/HTTP URL.")
    speed_limit_kmph: int = Field(60, ge=5, le=140)
    lane_width_m: float = Field(3.5, gt=0)
    lane_depth_m: Optional[float] = Field(None, gt=0, description="Real length of the marked stretch (m). Strongly recommended.")
    calibration_points: List[CalibrationPoint] = Field(..., min_length=4, max_length=4)
    enable_plate_recognition: bool = True
    max_frames: Optional[int] = Field(None, description="Optional cap for demo/testing streams.")


class GenerateReportRequest(BaseModel):
    violation_id: str
    vehicle_id: Optional[str] = None
    use_llm: bool = True


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str  # QUEUED | PROCESSING | COMPLETE | FAILED
    detail: Optional[str] = None
    summary: Optional[dict] = None
    progress: Optional[float] = Field(None, description="0-1 share of frames processed (videos only)")
    video_ready: bool = Field(False, description="True once the annotated video can be played via /videos/{session_id}")
    video_playable: bool = Field(True, description="False if ffmpeg was unavailable and the video may not play in browsers")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_calibration(
    req_points: List[CalibrationPoint],
    camera_id: str,
    lane_width_m: float,
    speed_limit_kmph: int,
    lane_depth_m: Optional[float] = None,
) -> CameraCalibration:
    """Compute the homography from the 4 marked points.

    ``lane_depth_m`` is the real length of the marked stretch of road. Speed accuracy depends directly on
    it: if omitted the calibrator assumes 4x the width, which is only a guess.
    """
    calibrator = CameraCalibrator()
    points = [(p.x, p.y) for p in req_points]
    return calibrator.calibrate(
        camera_id=camera_id,
        image_points=points,
        lane_width_m=lane_width_m,
        lane_depth_m=lane_depth_m,
        speed_limit_kmph=speed_limit_kmph,
    )


def _set_session(session_id: str, **fields) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS.setdefault(session_id, {})
        _SESSIONS[session_id].update(fields)


def _publish_frame(session_id: str, result) -> None:
    """Store the latest annotated frame (JPEG) and counters for the live preview endpoints."""
    with _LIVE_LOCK:
        entry = _LIVE.setdefault(session_id, {"started": time.monotonic(), "frames": 0})
        entry["frames"] += 1
        entry["frame_index"] = result.frame_index
        entry["vehicle_count"] = result.vehicle_count
        entry["updated_at"] = time.monotonic()
        elapsed = max(entry["updated_at"] - entry["started"], 1e-6)
        entry["fps"] = round(entry["frames"] / elapsed, 1)
        counts: Dict[str, int] = {}
        for track in result.active_tracks:
            counts[track.class_name] = counts.get(track.class_name, 0) + 1
        entry["counts_by_class"] = counts
        entry["vehicles"] = [
            {"track_id": v.track_id, "vehicle_type": v.vehicle_type, "speed_kmph": None if v.speed_kmph is None else round(v.speed_kmph, 1),
             "plate": v.plate, "severity": v.severity, "stable": v.stable}
            for v in getattr(result, "vehicles", [])
        ]
        if entry["frames"] % _LIVE_EVERY_N_FRAMES:
            return
    frame = result.annotated_frame
    h, w = frame.shape[:2]
    if w > _LIVE_MAX_WIDTH:
        frame = cv2.resize(frame, (_LIVE_MAX_WIDTH, int(h * _LIVE_MAX_WIDTH / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _LIVE_LOCK:
            _LIVE[session_id]["jpeg"] = buf.tobytes()


def _save_upload_capped(upload: UploadFile, dest: Path, max_mb: int) -> float:
    """Copy an upload to disk in chunks, aborting as soon as it exceeds ``max_mb``.

    Returns:
        Final size in MB.

    Raises:
        HTTPException: 400 if the file is larger than the limit (the partial file is deleted).
    """
    limit = max_mb * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(400, f"File too large (over {max_mb}MB).")
            out.write(chunk)
    return written / (1024 * 1024)


def _video_path(session_id: str, partial: bool = False) -> Path:
    return ANNOTATED_VIDEO_DIR / (f"{session_id}.partial.mp4" if partial else f"{session_id}.mp4")


def _prune_videos() -> None:
    """Keep only the newest ``MAX_STORED_VIDEOS`` processed videos."""
    videos = sorted(ANNOTATED_VIDEO_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in videos[MAX_STORED_VIDEOS:]:
        old.unlink(missing_ok=True)


class _Recorder:
    """Per-session ``on_frame`` callback: publishes the live preview and records the annotated video."""

    def __init__(self, session_id: str, fps_fn) -> None:
        self.session_id = session_id
        self._fps_fn = fps_fn          # called on the first frame, when the source fps is known
        self._writer: Optional[AnnotatedVideoWriter] = None
        self.playable = True

    def __call__(self, result) -> None:
        _publish_frame(self.session_id, result)
        try:
            if self._writer is None:
                self._writer = AnnotatedVideoWriter(_video_path(self.session_id, partial=True), self._fps_fn())
            self._writer.write(result.annotated_frame)
        except Exception:  # noqa: BLE001 - recording problems must never stop detection
            logger.exception("Could not record annotated video frame")

    def finish(self) -> bool:
        """Close the file and publish it under its final name. Returns True if a video exists."""
        if self._writer is None:
            return False
        path = self._writer.close()
        self.playable = self._writer.browser_playable
        if path is None:
            return False
        os.replace(path, _video_path(self.session_id))
        _prune_videos()
        return True


def _summary_dict(summary: SessionSummary) -> dict:
    return {
        "total_frames": summary.total_frames,
        "total_vehicles": summary.total_vehicles,
        "total_violations": summary.total_violations,
        "average_speed_kmph": summary.average_speed_kmph,
        "max_speed_kmph": summary.max_speed_kmph,
        "max_speed_plate": summary.max_speed_plate,
        "plates_read": summary.plates_read,
        "plate_attempts": summary.plate_attempts,
        "plate_boxes_found": summary.plate_boxes_found,
        "vehicles_too_small_for_plate": summary.vehicles_too_small_for_plate,
        "plates_rejected_format": summary.plates_rejected_format,
        "plates_rejected_confidence": summary.plates_rejected_confidence,
        "plate_region": summary.plate_region,
        "plate_min_confidence": summary.plate_min_confidence,
    }


def _run_video_processing(
    session_id: str, video_path: Path, calibration: CameraCalibration, camera_id: str, enable_plate: bool, frame_stride: int = 1
) -> None:
    """Background worker: process an uploaded video end-to-end and keep the annotated result for playback."""
    _set_session(session_id, status="QUEUED", detail="Waiting for the previous video to finish")
    try:
        with _PIPELINE_LOCK:
            _set_session(session_id, status="PROCESSING", detail="Loading models and running the detection pipeline")
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            with _LIVE_LOCK:
                _LIVE.setdefault(session_id, {"started": time.monotonic(), "frames": 0})["total_frames"] = total_frames // frame_stride
            detector, plate = _get_shared_models(enable_plate)
            service = DetectionService(
                calibration=calibration, camera_id=camera_id, alert_system=_alert_system, detector=detector,
                plate_recognizer=plate, enable_plate_recognition=enable_plate, session_id=session_id,
            )
            recorder = _Recorder(session_id, lambda: (service.fps or 25.0) / frame_stride)
            summary: SessionSummary = service.process_video(
                str(video_path), session_id=session_id, on_frame=recorder, frame_stride=frame_stride
            )
            has_video = recorder.finish()
            _set_session(
                session_id, status="COMPLETE", detail="Processing finished", summary=_summary_dict(summary),
                video_ready=has_video, video_playable=recorder.playable,
            )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the client
        logger.info(f"Session {session_id} failed: {exc}")
        _set_session(session_id, status="FAILED", detail=str(exc))
    finally:
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass


def _run_stream_processing(
    session_id: str, stream_source: str, calibration: CameraCalibration, camera_id: str, enable_plate: bool, max_frames: Optional[int]
) -> None:
    """Background worker: process a live/webcam stream (optionally capped), recording the annotated video."""
    from src.data_loader import VideoStream

    _set_session(session_id, status="QUEUED", detail="Waiting for the previous job to finish")
    try:
        with _PIPELINE_LOCK:
            _set_session(session_id, status="PROCESSING", detail="Streaming from source")
            source: str | int = int(stream_source) if stream_source.isdigit() else stream_source
            detector, plate = _get_shared_models(enable_plate)
            service = DetectionService(
                calibration=calibration, camera_id=camera_id, alert_system=_alert_system, detector=detector,
                plate_recognizer=plate, enable_plate_recognition=enable_plate, session_id=session_id,
            )
            recorder = _Recorder(session_id, lambda: STREAM_VIDEO_FPS)
            frame_count = 0
            try:
                with VideoStream(source) as stream:
                    service.fps = getattr(stream, "fps", None)
                    for packet in stream.frames():
                        recorder(service.process_frame(packet.frame, packet.frame_index, packet.timestamp_s))
                        frame_count += 1
                        if max_frames and frame_count >= max_frames:
                            break
            finally:
                service.finalize()  # save the vehicles still in view when the stream stops
                has_video = recorder.finish()
            _set_session(session_id, status="COMPLETE", detail=f"Processed {frame_count} frames", video_ready=has_video, video_playable=recorder.playable)
    except Exception as exc:  # noqa: BLE001
        logger.info(f"Stream session {session_id} failed: {exc}")
        _set_session(session_id, status="FAILED", detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe.

    Returns:
        Simple status payload used by Docker/Render health checks.
    """
    return {"status": "ok", "service": "DriveGuard AI API", "version": app.version, "llm_configured": bool(GEMINI_API_KEY)}


@app.post("/process-video", response_model=SessionStatusResponse, dependencies=[Depends(require_api_key)])
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    speed_limit_kmph: int = Form(60),
    lane_width_m: float = Form(3.5),
    lane_depth_m: Optional[float] = Form(None, description="Real length (m) of the marked stretch of road"),
    frame_stride: int = Form(1, ge=1, le=6, description="Process every Nth frame (1 = every frame, slower but most accurate)"),
    calibration_points: str = Form(
        ..., description='JSON array of 4 {"x":..,"y":..} points: TL,TR,BR,BL'
    ),
    enable_plate_recognition: bool = Form(True),
) -> SessionStatusResponse:
    """Upload a traffic video for full offline processing.

    Args:
        background_tasks: FastAPI background task runner.
        file: The video file (mp4/avi/mov, max 500MB).
        camera_id: Identifier for the camera/zone this video represents.
        speed_limit_kmph: Posted speed limit for the calibrated zone.
        lane_width_m: Real-world lane width used for calibration.
        lane_depth_m: Real-world length of the marked stretch (defaults to 4x width if omitted - a guess).
        calibration_points: JSON-encoded list of 4 {x,y} pixel points.
        enable_plate_recognition: Whether to run ANPR (slower if enabled).

    Returns:
        A :class:`SessionStatusResponse` with status ``QUEUED`` — poll the
        session (or ``GET /analytics``) for completion.

    Raises:
        HTTPException: 400 if the file type/size or calibration is invalid.
    """
    import json

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_VIDEO_EXTENSIONS}")

    try:
        points_raw = json.loads(calibration_points)
        points = [CalibrationPoint(**p) for p in points_raw]
    except Exception as exc:
        raise HTTPException(400, f"Invalid calibration_points JSON: {exc}") from exc

    session_id = str(uuid.uuid4())
    dest_path = _UPLOAD_DIR / f"{session_id}{suffix}"

    _save_upload_capped(file, dest_path, MAX_UPLOAD_SIZE_MB)

    try:
        calibration = _build_calibration(points, camera_id, lane_width_m, speed_limit_kmph, lane_depth_m)
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Calibration failed: {exc}") from exc

    _set_session(session_id, status="QUEUED", detail="Video received, awaiting processing", camera_id=camera_id, kind="video")
    background_tasks.add_task(
        _run_video_processing, session_id, dest_path, calibration, camera_id, enable_plate_recognition, frame_stride
    )

    return SessionStatusResponse(session_id=session_id, status="QUEUED")


@app.post("/process-stream", response_model=SessionStatusResponse, dependencies=[Depends(require_api_key)])
async def process_stream(request: ProcessStreamRequest, background_tasks: BackgroundTasks) -> SessionStatusResponse:
    """Start processing a live webcam or RTSP/HTTP camera stream in the background.

    Args:
        request: Stream source, camera calibration, and processing options.
        background_tasks: FastAPI background task runner.

    Returns:
        A :class:`SessionStatusResponse` with status ``QUEUED``.

    Raises:
        HTTPException: 400 if calibration fails.
    """
    session_id = str(uuid.uuid4())
    try:
        calibration = _build_calibration(
            request.calibration_points, request.camera_id, request.lane_width_m, request.speed_limit_kmph, request.lane_depth_m
        )
    except Exception as exc:
        raise HTTPException(400, f"Calibration failed: {exc}") from exc

    _set_session(session_id, status="QUEUED", detail="Stream session queued", camera_id=request.camera_id, kind="stream")
    background_tasks.add_task(
        _run_stream_processing,
        session_id,
        request.stream_source,
        calibration,
        request.camera_id,
        request.enable_plate_recognition,
        request.max_frames,
    )
    return SessionStatusResponse(session_id=session_id, status="QUEUED")


@app.get("/process-status/{session_id}", response_model=SessionStatusResponse)
def process_status(session_id: str) -> SessionStatusResponse:
    """Poll the status of a queued/running video or stream processing session.

    Args:
        session_id: Session ID returned by ``/process-video`` or ``/process-stream``.

    Returns:
        Current :class:`SessionStatusResponse` for that session.

    Raises:
        HTTPException: 404 if the session is unknown.
    """
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, f"Unknown session_id: {session_id}")
    with _LIVE_LOCK:
        live = _LIVE.get(session_id) or {}
    total = live.get("total_frames") or 0
    progress = 1.0 if session.get("status") == "COMPLETE" else (
        round(min(live.get("frames", 0) / total, 1.0), 3) if total else None
    )
    return SessionStatusResponse(session_id=session_id, progress=progress, **{k: v for k, v in session.items() if k != "progress"})


@app.get("/violations")
def list_violations(
    camera_id: Optional[str] = None,
    severity: Optional[str] = Query(None, pattern=_SEVERITIES),
    status: Optional[str] = Query(None, pattern=_STATUSES),
    vehicle_type: Optional[str] = Query(None, description="Comma-separated vehicle classes, e.g. Car,Truck"),
    state_code: Optional[str] = Query(None, min_length=2, max_length=2, description="Plate state code, e.g. MH"),
    plate: Optional[str] = Query(None, max_length=15, description="Plate substring"),
    date_from: Optional[str] = Query(None, pattern=_DATE_RE, description="Local date YYYY-MM-DD, inclusive"),
    date_to: Optional[str] = Query(None, pattern=_DATE_RE, description="Local date YYYY-MM-DD, inclusive"),
    limit: int = Query(50, ge=1),
    offset: int = Query(0, ge=0),
) -> dict:
    """List logged violations, newest first, with optional filters.

    Args:
        camera_id: Restrict to one camera.
        severity: WARNING / VIOLATION / SEVERE.
        status: PENDING / RESOLVED / DISPUTED.
        vehicle_type: Comma-separated vehicle classes.
        state_code: Two-letter plate state code.
        plate: Plate substring.
        date_from: Earliest local date, inclusive.
        date_to: Latest local date, inclusive.
        limit: Page size (max 200).
        offset: Page offset.

    Returns:
        Dict with ``count`` (rows in this page), ``total`` (rows matching all filters, for
        pagination) and the ``violations`` list.
    """
    limit = min(limit, 200)
    filters = _violation_filters(camera_id, severity, status, vehicle_type, state_code, plate, date_from, date_to)
    violations = _alert_system.query_violations(limit=limit, offset=offset, **filters)
    return {"count": len(violations), "total": _alert_system.count_violations(**filters), "violations": violations}


def _violation_filters(camera_id, severity, status, vehicle_type, state_code, plate, date_from, date_to) -> dict:
    types = [t.strip() for t in vehicle_type.split(",") if t.strip()] if vehicle_type else None
    return {
        "camera_id": camera_id, "severity": severity, "status": status, "vehicle_types": types,
        "state_code": state_code, "plate_query": plate, "date_from": date_from, "date_to": date_to,
    }


@app.get("/violation/{violation_id}")
def get_violation(violation_id: str) -> dict:
    """Fetch full detail for a single violation, including its notice if generated.

    Args:
        violation_id: The violation's UUID.

    Returns:
        The violation row as a dict.

    Raises:
        HTTPException: 404 if not found.
    """
    record = _alert_system.get_violation(violation_id)
    if record is None:
        raise HTTPException(404, f"Violation not found: {violation_id}")
    return record


@app.get("/analytics")
def analytics(camera_id: Optional[str] = None) -> dict:
    """Dashboard KPI and breakdown statistics (Screen 5).

    Args:
        camera_id: Restrict stats to one camera, if given.

    Returns:
        Aggregate stats: totals, severity/vehicle-type breakdowns, top offending plates.
    """
    return _alert_system.get_analytics_summary(camera_id=camera_id)


@app.get("/live-feed")
def live_feed(limit: int = 20) -> dict:
    """Recent live alerts for the dashboard's scrolling alert panel (Screen 1).

    Args:
        limit: Max number of recent alerts to return.

    Returns:
        Dict with the ``alerts`` list, newest first.
    """
    return {"alerts": _alert_system.get_live_feed(limit=limit)}


@app.post("/generate-report", dependencies=[Depends(require_api_key)])
def generate_report(request: GenerateReportRequest) -> dict:
    """Draft (or re-draft) an official violation notice for a logged violation.

    Args:
        request: Which violation to generate a notice for, and whether to
            use the Gemini LLM path or the deterministic fallback template.

    Returns:
        Dict with the generated ``notice_text`` and its ``model_used``.

    Raises:
        HTTPException: 404 if the violation doesn't exist; 500 if both the
            LLM and fallback generation fail unexpectedly.
    """
    from datetime import datetime
    from src.report_generator import ReportGenerator

    row = _alert_system.get_violation(request.violation_id)
    if row is None:
        raise HTTPException(404, f"Violation not found: {request.violation_id}")

    record = ViolationRecord(
        violation_id=row["id"],
        track_id=-1,
        plate_number=row["plate_number"],
        vehicle_type=row["vehicle_type"],
        detected_speed_kmph=row["detected_speed"],
        speed_limit_kmph=row["speed_limit"],
        excess_speed_kmph=row["excess_speed"],
        excess_percent=round((row["excess_speed"] / row["speed_limit"]) * 100, 1) if row["speed_limit"] else 0.0,
        severity=row["severity"],
        location=row["location"],
        camera_id=row["camera_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
    )

    prior_plates = _alert_system.get_prior_plate_numbers(camera_id=row["camera_id"])
    generator = ReportGenerator()
    is_repeat = bool(record.plate_number) and prior_plates.count(record.plate_number) > 1
    vehicle_id = request.vehicle_id or f"{record.vehicle_type[:3].upper()}_{row['id'][:6].upper()}"

    try:
        if request.use_llm:
            notice = generator.generate(
                record, vehicle_id=vehicle_id, is_repeat_offender=is_repeat,
                evidence_frame_ref=Path(row["evidence_frame_path"]).name if row.get("evidence_frame_path") else None,
            )
        else:
            notice = generator.generate_fallback_notice(record, vehicle_id=vehicle_id, is_repeat_offender=is_repeat)
    except Exception as exc:  # noqa: BLE001 - LLM/RAG outage should degrade, not 500 the whole endpoint
        logger.info(f"LLM generation failed for {request.violation_id}, using fallback: {exc}")
        notice = generator.generate_fallback_notice(record, vehicle_id=vehicle_id, is_repeat_offender=is_repeat)

    _alert_system.update_notice(request.violation_id, notice.notice_text)

    return {
        "violation_id": notice.violation_id,
        "notice_text": notice.notice_text,
        "model_used": notice.model_used,
        "cited_context": notice.cited_context,
    }


# ---------------------------------------------------------------------------
# Sessions + live preview (Screen 1 / Screen 2)
# ---------------------------------------------------------------------------
@app.get("/sessions")
def list_sessions(active_only: bool = False) -> dict:
    """List known processing sessions (newest activity not guaranteed; ids are UUIDs).

    Args:
        active_only: Only sessions that are QUEUED or PROCESSING.

    Returns:
        Dict with a ``sessions`` list of ``{session_id, status, camera_id, kind, detail}``.
    """
    with _SESSIONS_LOCK:
        items = [{"session_id": sid, **data} for sid, data in _SESSIONS.items()]
    if active_only:
        items = [i for i in items if i.get("status") in ("QUEUED", "PROCESSING")]
    for item in items:
        item.pop("summary", None)
    return {"sessions": items}


@app.get("/live-frame/{session_id}")
def live_frame(session_id: str) -> Response:
    """Latest annotated frame of a running session as a JPEG.

    Args:
        session_id: Session id from ``/process-video`` or ``/process-stream``.

    Raises:
        HTTPException: 404 if the session has not produced a frame yet.
    """
    with _LIVE_LOCK:
        jpeg = (_LIVE.get(session_id) or {}).get("jpeg")
    if jpeg is None:
        raise HTTPException(404, "No frame available for this session yet.")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/live-stats")
def live_stats(session_id: Optional[str] = None) -> dict:
    """Numbers for the live-monitoring side panel.

    Args:
        session_id: Session to read frame-level numbers (vehicles in frame, FPS) from.

    Returns:
        ``vehicles_in_frame``, ``fps``, ``counts_by_class`` (None/empty without a session) plus
        ``violations_last_hour``, ``average_speed_kmph`` and ``fastest`` from the database.
    """
    stats = _alert_system.get_recent_stats(hours=1)
    with _LIVE_LOCK:
        live = dict(_LIVE.get(session_id or "", {}))
    live.pop("jpeg", None)
    return {
        "vehicles_in_frame": live.get("vehicle_count"),
        "fps": live.get("fps"),
        "counts_by_class": live.get("counts_by_class", {}),
        "vehicles": live.get("vehicles", []),
        **stats,
    }


@app.get("/cameras")
def cameras() -> dict:
    """Registered cameras with their violation counts (for the camera selector)."""
    return {"cameras": _alert_system.list_cameras()}


# ---------------------------------------------------------------------------
# Violation actions + evidence (Screen 3 / Screen 4)
# ---------------------------------------------------------------------------
class StatusUpdate(BaseModel):
    status: str = Field(..., pattern=_STATUSES)


class NoticeUpdate(BaseModel):
    notice_text: str = Field(..., min_length=1, max_length=8000)


def _get_or_404(violation_id: str) -> dict:
    row = _alert_system.get_violation(violation_id)
    if row is None:
        raise HTTPException(404, f"Violation not found: {violation_id}")
    return row


@app.patch("/violation/{violation_id}/status", dependencies=[Depends(require_api_key)])
def set_violation_status(violation_id: str, body: StatusUpdate) -> dict:
    """Mark a violation PENDING / RESOLVED / DISPUTED.

    Raises:
        HTTPException: 404 if the violation does not exist.
    """
    _get_or_404(violation_id)
    _alert_system.update_status(violation_id, body.status)
    return {"violation_id": violation_id, "status": body.status}


@app.put("/violation/{violation_id}/notice", dependencies=[Depends(require_api_key)])
def edit_notice(violation_id: str, body: NoticeUpdate) -> dict:
    """Save a manually corrected notice text for a violation.

    Raises:
        HTTPException: 404 if the violation does not exist.
    """
    _get_or_404(violation_id)
    _alert_system.update_notice_text(violation_id, body.notice_text.strip())
    return {"violation_id": violation_id, "notice_text": body.notice_text.strip()}


@app.post("/violation/{violation_id}/send-to-rto", dependencies=[Depends(require_api_key)])
def send_to_rto(violation_id: str) -> dict:
    """Send a violation to the RTO. **Mock integration**: records an audit row and returns a fake reference.

    Raises:
        HTTPException: 404 if the violation does not exist; 409 if no notice has been generated yet.
    """
    row = _get_or_404(violation_id)
    if not row.get("notice_text"):
        raise HTTPException(409, "Generate a notice before sending the violation to the RTO.")
    dispatch = _alert_system.record_rto_dispatch(violation_id, channel="RTO_PORTAL", mode="mock")
    return {**dispatch, "message": "Mock RTO portal: recorded, nothing was delivered to a real RTO."}


@app.get("/violation/{violation_id}/evidence")
def violation_evidence(violation_id: str, kind: str = Query("frame", pattern="^(frame|plate)$")) -> FileResponse:
    """Serve the evidence frame or the zoomed plate crop for a violation.

    Args:
        violation_id: The violation's UUID.
        kind: ``frame`` (annotated frame) or ``plate`` (plate crop).

    Raises:
        HTTPException: 404 if the violation or its image does not exist.
    """
    row = _get_or_404(violation_id)
    if kind == "plate":
        path = EVIDENCE_FRAMES_DIR / f"{violation_id}_plate.jpg"
    else:
        raw = row.get("evidence_frame_path")
        path = EVIDENCE_FRAMES_DIR / Path(raw).name if raw else None  # .name: DB paths may come from another host
    if path is None or not path.is_file():
        raise HTTPException(404, f"No {kind} image stored for this violation.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/violation/{violation_id}/speed-trace")
def violation_speed_trace(violation_id: str) -> dict:
    """Smoothed speed over time for the vehicle behind a violation (for the detail-screen graph).

    Raises:
        HTTPException: 404 if the violation does not exist or no trace was recorded (seeded demo data has none).
    """
    _get_or_404(violation_id)
    trace = _alert_system.load_speed_trace(violation_id)
    if trace is None:
        raise HTTPException(404, "No speed trace recorded for this violation.")
    return trace


@app.get("/violation/{violation_id}/notice.pdf")
def violation_notice_pdf(violation_id: str) -> Response:
    """Download a violation (with its notice text and evidence frame) as a PDF.

    Raises:
        HTTPException: 404 if the violation does not exist.
    """
    from src.report_export import notice_to_pdf

    row = _get_or_404(violation_id)
    return Response(
        content=notice_to_pdf(row),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="notice_{violation_id[:8]}.pdf"'},
    )


@app.get("/violation/{violation_id}/dispatches")
def violation_dispatches(violation_id: str) -> dict:
    """RTO dispatch history for a violation."""
    _get_or_404(violation_id)
    return {"dispatches": _alert_system.get_rto_dispatches(violation_id)}


# ---------------------------------------------------------------------------
# Analytics, insights, exports (Screen 5 / Screen 6)
# ---------------------------------------------------------------------------
@app.get("/analytics/detailed")
def analytics_detailed(camera_id: Optional[str] = None, months: int = Query(6, ge=1, le=24)) -> dict:
    """Chart data: hourly trend, day x hour heatmap, speed histogram, monthly trend, vehicle shares.

    Args:
        camera_id: Restrict to one camera.
        months: Months in the monthly trend.
    """
    return _alert_system.get_detailed_analytics(camera_id=camera_id, months=months)


@app.get("/analytics/insights")
def analytics_insights(camera_id: Optional[str] = None, use_llm: bool = True) -> dict:
    """Plain-language insights. Uses Gemini when configured, otherwise rule-based text built from the same stats.

    The result is cached until the violation count changes, so refreshing the dashboard does not burn LLM quota.
    """
    from src.insights import generate_insights

    summary = _alert_system.get_analytics_summary(camera_id=camera_id)
    key = (camera_id, summary["total_violations"], use_llm)
    if key not in _INSIGHT_CACHE:
        detailed = _alert_system.get_detailed_analytics(camera_id=camera_id)
        _INSIGHT_CACHE.clear()
        _INSIGHT_CACHE[key] = generate_insights(summary, detailed, use_llm=use_llm)
    return _INSIGHT_CACHE[key]


@app.get("/export/violations")
def export_violations(
    format: str = Query("csv", pattern="^(csv|xlsx)$"),
    camera_id: Optional[str] = None,
    severity: Optional[str] = Query(None, pattern=_SEVERITIES),
    status: Optional[str] = Query(None, pattern=_STATUSES),
    vehicle_type: Optional[str] = None,
    state_code: Optional[str] = Query(None, min_length=2, max_length=2),
    plate: Optional[str] = Query(None, max_length=15),
    date_from: Optional[str] = Query(None, pattern=_DATE_RE),
    date_to: Optional[str] = Query(None, pattern=_DATE_RE),
) -> Response:
    """Export every violation matching the filters (up to ``MAX_EXPORT_ROWS``) as CSV or Excel."""
    from src.report_export import violations_to_csv, violations_to_xlsx

    filters = _violation_filters(camera_id, severity, status, vehicle_type, state_code, plate, date_from, date_to)
    rows = _alert_system.query_violations(limit=MAX_EXPORT_ROWS, offset=0, **filters)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    if format == "xlsx":
        return Response(
            violations_to_xlsx(rows),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="driveguard_violations_{stamp}.xlsx"'},
        )
    return Response(
        violations_to_csv(rows),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="driveguard_violations_{stamp}.csv"'},
    )


@app.get("/reports/summary.pdf")
def summary_report(camera_id: Optional[str] = None, use_llm: bool = False) -> Response:
    """Supervisor summary report (KPIs, hourly chart, top plates, serious cases, recommendations) as a PDF.

    Args:
        camera_id: Restrict to one camera.
        use_llm: Let Gemini phrase the recommendations (default off: rule-based text, no quota use).
    """
    from src.insights import generate_insights
    from src.report_export import summary_report_pdf

    summary = _alert_system.get_analytics_summary(camera_id=camera_id)
    detailed = _alert_system.get_detailed_analytics(camera_id=camera_id)
    insight = generate_insights(summary, detailed, use_llm=use_llm)
    serious = _alert_system.query_violations(camera_id=camera_id, severity="SEVERE", limit=10)
    pdf = summary_report_pdf(summary, detailed, insight["text"] or "", serious)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="driveguard_summary_{stamp}.pdf"'})


@app.get("/integrations")
def integrations() -> dict:
    """Honest status of external integrations, for the Reports screen.

    ``mode`` is ``live`` only for things that really run; Vahan, the RTO portal and the police
    dashboard are mocks in this project.
    """
    kb_ready = (PROJECT_ROOT / "chromadb_store").exists() and any((PROJECT_ROOT / "chromadb_store").iterdir())
    return {
        "integrations": [
            {"name": "Gemini LLM", "mode": "live" if GEMINI_API_KEY else "not_configured", "detail": GEMINI_MODEL_NAME},
            {"name": "RAG knowledge base (ChromaDB)", "mode": "live" if kb_ready else "not_configured", "detail": "run scripts/ingest_knowledge_base.py" if not kb_ready else "ingested"},
            {"name": "Plate reading (ANPR)", "mode": "live", "detail": f"region {PLATE_REGION}, min confidence {PLATE_CONFIDENCE_THRESHOLD}"},
            {"name": "Vahan database", "mode": "mock", "detail": "no public API access; not implemented"},
            {"name": "RTO portal", "mode": "mock", "detail": "send-to-RTO records an audit row only"},
            {"name": "Police dashboard", "mode": "mock", "detail": "not implemented"},
        ]
    }


# ---------------------------------------------------------------------------
# Annotated video playback, live MJPEG stream, per-video vehicles
# ---------------------------------------------------------------------------
def _check_session_id(session_id: str) -> str:
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(400, "Invalid session id.")
    return session_id


@app.get("/videos")
def list_videos() -> dict:
    """Processed videos (boxes, speeds and plates drawn on), newest first. A video still being written is not listed."""
    items = []
    if ANNOTATED_VIDEO_DIR.exists():
        for path in sorted(ANNOTATED_VIDEO_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            if ".partial" in path.name:
                continue
            with _SESSIONS_LOCK:
                session = dict(_SESSIONS.get(path.stem, {}))
            items.append({
                "session_id": path.stem,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "camera_id": session.get("camera_id"),
            })
    return {"videos": items}


@app.get("/videos/{session_id}")
def get_video(session_id: str) -> FileResponse:
    """Download/stream the annotated video of a finished session (MP4, H.264).

    Raises:
        HTTPException: 400 for a malformed id, 404 if there is no finished video for it.
    """
    path = _video_path(_check_session_id(session_id))
    if not path.is_file():
        raise HTTPException(404, "No processed video for this session (still running, failed, or deleted).")
    return FileResponse(path, media_type="video/mp4", filename=f"driveguard_{session_id[:8]}.mp4")


@app.get("/sessions/{session_id}/vehicles")
def session_vehicles(session_id: str) -> dict:
    """Every vehicle that was tracked in a session: type, average/max speed, plate, violation count."""
    _check_session_id(session_id)
    return {"vehicles": _alert_system.get_session_vehicles(session_id)}


_PLATE_READ_FILE_RE = re.compile(r"^read-track(\d+)-([A-Z0-9]{3,12})-c(\d{1,3})\.jpg$")


@app.get("/sessions/{session_id}/plate-reads")
def plate_reads(session_id: str) -> dict:
    """The plates that were read in a session, with the crop each reading came from (to check them by eye).

    Returns:
        ``{"reads": [{"track_id", "plate", "confidence", "file"}, ...]}`` sorted by track id; the image of an entry is
        ``/sessions/{session_id}/plate-reads/{file}``.
    """
    folder = PLATE_DEBUG_DIR / _check_session_id(session_id)
    reads = []
    if folder.is_dir():
        for path in folder.glob("read-track*.jpg"):
            match = _PLATE_READ_FILE_RE.match(path.name)
            if match:
                reads.append({"track_id": int(match[1]), "plate": match[2], "confidence": int(match[3]) / 100, "file": path.name})
    return {"reads": sorted(reads, key=lambda r: r["track_id"])}


@app.get("/sessions/{session_id}/plate-reads/{filename}")
def plate_read_image(session_id: str, filename: str) -> FileResponse:
    """Serve one plate crop. Only names produced by the pipeline are accepted (no path tricks).

    Raises:
        HTTPException: 400 for a malformed id/name, 404 if the file does not exist.
    """
    _check_session_id(session_id)
    if not _PLATE_READ_FILE_RE.match(filename):
        raise HTTPException(400, "Invalid file name.")
    path = PLATE_DEBUG_DIR / session_id / filename
    if not path.is_file():
        raise HTTPException(404, "No such plate crop.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/live-stream/{session_id}")
async def live_stream(session_id: str) -> StreamingResponse:
    """MJPEG stream of the annotated frames of a running session; plays like a video in an ``<img>`` tag.

    The stream ends shortly after the session completes or fails (or after 2 minutes without a new frame).

    Raises:
        HTTPException: 404 for an unknown session.
    """
    _check_session_id(session_id)
    with _SESSIONS_LOCK:
        known = session_id in _SESSIONS
    with _LIVE_LOCK:
        known = known or session_id in _LIVE
    if not known:
        raise HTTPException(404, "Unknown session.")

    async def frames():
        last, idle = None, 0.0
        while idle < 120:
            with _LIVE_LOCK:
                jpeg = (_LIVE.get(session_id) or {}).get("jpeg")
            with _SESSIONS_LOCK:
                status = (_SESSIONS.get(session_id) or {}).get("status")
            if jpeg is not None and jpeg is not last:
                last, idle = jpeg, 0.0
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            elif status in ("COMPLETE", "FAILED"):
                break  # the final frame has been sent
            await asyncio.sleep(0.05)
            idle += 0.05

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})
