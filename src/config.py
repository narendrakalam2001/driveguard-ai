"""
config.py
=========
Central configuration for DriveGuard AI.

All tunable parameters, paths, model settings, speed-zone rules, and
environment-variable driven secrets live here so nothing is hardcoded
inside the pipeline modules. Import this module everywhere instead of
re-declaring constants.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured for structured JSON-style logs.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance with both a console
        and rotating file handler attached exactly once.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)
    fmt = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s",'
        '"module":"%(name)s","message":"%(message)s"}'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_DIR / "driveguard.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
PLATES_DATA_DIR: Path = DATA_DIR / "plates"
KNOWLEDGE_BASE_DIR: Path = DATA_DIR / "knowledge_base"

MODELS_DIR: Path = PROJECT_ROOT / "models"
VEHICLE_MODEL_PATH: Path = MODELS_DIR / "yolov8_vehicle.pt"
PLATE_MODEL_PATH: Path = MODELS_DIR / "yolov8_plate.pt"
REID_MODEL_PATH: Path = MODELS_DIR / "osnet_reid.pt"

VIOLATION_LOG_DIR: Path = PROJECT_ROOT / "violation_logs"
VIOLATION_DB_PATH: Path = VIOLATION_LOG_DIR / "violations.db"
EVIDENCE_FRAMES_DIR: Path = Path(os.getenv("EVIDENCE_FRAMES_DIR", str(PROJECT_ROOT / "evidence_frames")))  # point at a persistent disk in production
CHROMADB_STORE_DIR: Path = PROJECT_ROOT / "chromadb_store"

for _dir in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PLATES_DATA_DIR,
    KNOWLEDGE_BASE_DIR,
    MODELS_DIR,
    VIOLATION_LOG_DIR,
    EVIDENCE_FRAMES_DIR,
    CHROMADB_STORE_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Secrets / API keys (never hardcode — always via environment)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
KAGGLE_USERNAME: str = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY: str = os.getenv("KAGGLE_KEY", "")

# ---------------------------------------------------------------------------
# Detection model settings
# ---------------------------------------------------------------------------
VEHICLE_CLASSES: Dict[int, str] = {
    0: "Car",
    1: "Truck",
    2: "Bus",
    3: "Motorcycle",
    4: "Auto-rickshaw",
    5: "Bicycle",
    6: "Van",  # UA-DETRAC "van" (minivans); appended last so existing ids stay stable
}

YOLO_BASE_WEIGHTS: str = os.getenv("YOLO_BASE_WEIGHTS", "yolov8m.pt")
YOLO_PLATE_BASE_WEIGHTS: str = os.getenv("YOLO_PLATE_BASE_WEIGHTS", "yolov8n.pt")
DETECTION_CONF_THRESHOLD: float = float(os.getenv("DETECTION_CONF_THRESHOLD", "0.4"))
DETECTION_IOU_THRESHOLD: float = float(os.getenv("DETECTION_IOU_THRESHOLD", "0.45"))
TARGET_MAP50: float = 0.85
TARGET_FPS_GPU: int = 25
TARGET_FPS_CPU: int = 8

# ---------------------------------------------------------------------------
# Tracking (DeepSORT) settings
# ---------------------------------------------------------------------------
DEEPSORT_MAX_AGE: int = int(os.getenv("DEEPSORT_MAX_AGE", "30"))
DEEPSORT_N_INIT: int = int(os.getenv("DEEPSORT_N_INIT", "3"))
DEEPSORT_MAX_IOU_DISTANCE: float = 0.7
DEEPSORT_MAX_COSINE_DISTANCE: float = 0.3
DEEPSORT_NN_BUDGET: int = 100

# ---------------------------------------------------------------------------
# Speed estimation
# ---------------------------------------------------------------------------
DEFAULT_LANE_WIDTH_M: float = 3.5  # standard Indian lane width
SPEED_SMOOTHING_WINDOW: int = 10  # speed is measured over the last N frame intervals
SPEED_MIN_POINTS: int = int(os.getenv("SPEED_MIN_POINTS", "5"))  # no speed (so no violation) until a track has this many points
MPS_TO_KMPH: float = 3.6

# A vehicle can only be flagged after this many speed readings (about half a second of video): a 5-frame track
# is usually a tracker flicker, and its "speed" is noise amplified by perspective.
MIN_SPEED_SAMPLES: int = int(os.getenv("MIN_SPEED_SAMPLES", "12"))
# Readings at or above this are treated as measurement errors (ID switch, bad calibration): never shown as a
# speed, never counted in averages, never logged as a violation.
# A car cannot gain more than ~45 km/h per second; a bigger jump between consecutive readings is a tracker glitch
# (ID switch, box jitter) and is ignored. After SPEED_REBASE_AFTER ignored readings in a row the new level is accepted.
MAX_ACCEL_KMPH_PER_S: float = float(os.getenv("MAX_ACCEL_KMPH_PER_S", "45"))
SPEED_JUMP_SLACK_KMPH: float = 8.0
SPEED_REBASE_AFTER: int = 25
SPEED_DECISION_WINDOW: int = 9  # a vehicle is judged on the median of its last N readings
MAX_REPORTED_SPEED_KMPH: float = float(os.getenv("MAX_REPORTED_SPEED_KMPH", "200"))
ANNOTATED_VIDEO_DIR: Path = EVIDENCE_FRAMES_DIR / "annotated"  # processed videos with boxes/speeds/plates drawn on
STREAM_VIDEO_FPS: float = float(os.getenv("STREAM_VIDEO_FPS", "15"))  # fps of recordings made from a live camera

SPEED_ZONES: Dict[str, int] = {
    "School Zone": 25,
    "Residential": 40,
    "City Road": 60,
    "Highway": 100,
}

WARNING_MULTIPLIER: float = 1.1
VIOLATION_MULTIPLIER: float = 1.2
SEVERE_MULTIPLIER: float = 1.5


@dataclass(frozen=True)
class ViolationSeverity:
    """Severity band labels used consistently across modules."""

    NORMAL: str = "NORMAL"
    WARNING: str = "WARNING"
    VIOLATION: str = "VIOLATION"
    SEVERE: str = "SEVERE"


SEVERITY = ViolationSeverity()

# ---------------------------------------------------------------------------
# ANPR (plate recognition)
# ---------------------------------------------------------------------------
PLATE_OCR_LANGUAGES = ["en"]
# Minimum OCR confidence for a plate that already passed format validation. The old 0.7 rejected
# many correct reads (EasyOCR often scores a right plate at 0.3-0.6); format validation is the main
# guard now, so this only filters very shaky reads. Tune via env var.
PLATE_CONFIDENCE_THRESHOLD: float = float(os.getenv("PLATE_CONFIDENCE_THRESHOLD", "0.3"))
PLATE_DETECTION_CONF: float = float(os.getenv("PLATE_DETECTION_CONF", "0.35"))  # YOLO plate-box confidence
PLATE_OCR_MIN_WIDTH: int = 400  # plate crops are enlarged to at least this many pixels wide before OCR
PLATE_READ_INTERVAL_FRAMES: int = 10  # per vehicle, retry an unreadable plate at most every N frames
PLATE_MAX_READ_ATTEMPTS: int = 12  # then give up on that vehicle (stops OCR running on it forever)
# A plate needs pixels: below roughly 90 px of vehicle width the plate is only ~25 px wide and cannot be read, so such
# vehicles are not tried (this also stops the retry budget being burnt on far-away, tiny crops).
PLATE_MIN_VEHICLE_WIDTH_PX: int = int(os.getenv("PLATE_MIN_VEHICLE_WIDTH_PX", "90"))
PLATE_RETRY_GROWTH: float = float(os.getenv("PLATE_RETRY_GROWTH", "1.15"))  # retry early once the vehicle got this much wider
PLATE_DEBUG_CROPS: bool = os.getenv("PLATE_DEBUG_CROPS", "true").lower() == "true"  # keep a few crops to inspect failures
PLATE_DEBUG_MAX_CROPS: int = 40  # per session
PLATE_DEBUG_DIR: Path = EVIDENCE_FRAMES_DIR / "plate_debug"
PLATE_OCR_MAX_CORRECTIONS: int = 1  # look-alike character swaps (O/0, I/1 ...) allowed per plate
# Plate format the system accepts. IN = Indian plates (the product); UK / ANY exist so the OCR can be validated on
# non-Indian footage (UK: AA00 AAA; ANY: any 4-10 character text with letters and digits, no state logic).
PLATE_REGION: str = os.getenv("PLATE_REGION", "IN").strip().upper()
if PLATE_REGION not in ("IN", "UK", "ANY"):
    PLATE_REGION = "IN"
INDIAN_PLATE_REGEX: str = r"^[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{1,4}$"
PLATE_UNCLEAR_LABEL: str = "PLATE_UNCLEAR"

# ---------------------------------------------------------------------------
# LLM / RAG settings
# ---------------------------------------------------------------------------
# Gemini model names are retired every few months (1.5 -> 2.0 -> 2.5 are all gone), so nothing here is
# hardcoded in code: set GEMINI_MODEL_NAME / GEMINI_FALLBACK_MODELS in .env and check the current list at
# https://ai.google.dev/gemini-api/docs/models. If every model fails, notices fall back to a template.
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
GEMINI_FALLBACK_MODELS: List[str] = [
    m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash").split(",") if m.strip()
]
LLM_TIMEOUT_S: int = int(os.getenv("LLM_TIMEOUT_S", "30"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
RAG_COLLECTION_NAME: str = "mv_act_knowledge_base"
RAG_TOP_K: int = 4
RAG_CHUNK_SIZE: int = 800
RAG_CHUNK_OVERLAP: int = 120

# ---------------------------------------------------------------------------
# API / serving
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
MAX_UPLOAD_SIZE_MB: int = 500
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
CORS_ALLOW_ORIGINS: List[str] = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]
API_KEY: str = os.getenv("DRIVEGUARD_API_KEY", "")  # if set, mutating endpoints require header X-API-Key
MAX_EXPORT_ROWS: int = 20000

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
VEHICLE_LOST_AFTER_S: float = 2.0  # a vehicle unseen this long is finished: its stats are saved to the DB
RANDOM_SEED: int = 42
# Timestamps are stored in UTC; dashboards group/filter by local (IST, UTC+5:30) time so "peak hour" matches the clock on the road.
DISPLAY_UTC_OFFSET_MINUTES: int = int(os.getenv("DISPLAY_UTC_OFFSET_MINUTES", "330"))
FRAME_QUEUE_MAXSIZE: int = 128
