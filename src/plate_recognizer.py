"""
plate_recognizer.py
====================
Automatic Number Plate Recognition (ANPR) for Indian plates.

Two-stage pipeline:
  1. YOLOv8n (fine-tuned) crops the plate region from a vehicle's bounding box.
  2. EasyOCR reads the crop using the "big+fix" flow from ``scripts/ocr_bench.py``
     (see :mod:`src.plate_text`): enlarge -> A-Z/0-9 only -> reading order ->
     snap to an Indian plate pattern.

A read is only trusted if the corrected text is a complete, valid Indian plate
(real state code, 4-digit number) AND its confidence reaches the threshold.
Anything else is reported as ``PLATE_UNCLEAR``; the raw reading is kept in
``PlateResult.raw_text`` for review, never logged as a plate number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import (
    PLATE_CONFIDENCE_THRESHOLD,
    PLATE_DETECTION_CONF,
    PLATE_MODEL_PATH,
    PLATE_OCR_LANGUAGES,
    PLATE_REGION,
    PLATE_OCR_MAX_CORRECTIONS,
    PLATE_OCR_MIN_WIDTH,
    PLATE_UNCLEAR_LABEL,
    YOLO_PLATE_BASE_WEIGHTS,
    get_logger,
)
from src.plate_text import STATE_CODES, is_valid_plate, normalize, read_plate_text

logger = get_logger(__name__)


@dataclass
class PlateResult:
    """Structured ANPR output for one detected plate.

    Attributes:
        plate_number: Normalized plate text (e.g. "MH12AB1234"), or
            :data:`PLATE_UNCLEAR_LABEL` if the read was invalid or too uncertain.
        confidence: OCR confidence score in [0, 1].
        state_code: Two-letter state code parsed from the plate, if accepted.
        district_code: District digits parsed from the plate, if accepted.
        is_valid_format: Whether the corrected text is a valid Indian plate.
        raw_text: Uncorrected reading-order OCR text, kept even when the plate is
            rejected (for review/debugging only; never log it as a plate number).
    """

    plate_number: str
    confidence: float
    state_code: Optional[str]
    district_code: Optional[str]
    is_valid_format: bool
    raw_text: str = ""
    crop: Optional[np.ndarray] = None  # the cropped plate image that was read (kept as evidence)
    candidate: str = ""  # the corrected text when the read was rejected (for diagnosis only; never a plate number)


class PlateRecognizer:
    """Combines a YOLOv8 plate-detector with EasyOCR text recognition."""

    def __init__(
        self,
        detector_weights: Optional[Path] = None,
        confidence_threshold: float = PLATE_CONFIDENCE_THRESHOLD,
        gpu: bool = True,
        *,
        detector: Any = None,
        reader: Any = None,
        detection_conf: float = PLATE_DETECTION_CONF,
        region: str = PLATE_REGION,
    ) -> None:
        """Load the plate detector and OCR engine.

        Args:
            detector_weights: Path to fine-tuned plate-detection weights.
                Falls back to base YOLOv8n weights if not found.
            confidence_threshold: Minimum OCR confidence for a valid-format
                plate to be accepted; below this it is flagged PLATE_UNCLEAR.
            gpu: Whether EasyOCR should attempt to use a GPU.
            detector: Pre-built object with a YOLO-style ``predict`` (tests/injection).
            reader: Pre-built object with an EasyOCR-style ``readtext`` (tests/injection).
            detection_conf: Minimum YOLO confidence for a plate box.
            region: Plate format to accept: ``IN`` (default), ``UK`` or ``ANY`` (for non-Indian test footage).
        """
        self.region = region
        self.confidence_threshold = confidence_threshold
        self.detection_conf = detection_conf

        if detector is None:
            from ultralytics import YOLO

            resolved_path = Path(detector_weights) if detector_weights else PLATE_MODEL_PATH
            if not resolved_path.exists():
                logger.info(
                    f"Fine-tuned plate weights not found at {resolved_path}; "
                    f"falling back to base weights {YOLO_PLATE_BASE_WEIGHTS}."
                )
                resolved_path = YOLO_PLATE_BASE_WEIGHTS
            detector = YOLO(str(resolved_path))
            logger.info(f"PlateRecognizer loaded detector={resolved_path}")
        if reader is None:
            import easyocr

            reader = easyocr.Reader(PLATE_OCR_LANGUAGES, gpu=gpu)

        self._plate_detector = detector
        self._reader = reader

    def detect_plate_region(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """Locate and crop the license-plate sub-region within a vehicle crop.

        Args:
            vehicle_crop: BGR image cropped to a single vehicle's bounding box.

        Returns:
            The cropped plate image, or None if no plate was detected.
        """
        if vehicle_crop.size == 0:
            return None

        results = self._plate_detector.predict(source=vehicle_crop, conf=self.detection_conf, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        # Take the highest-confidence plate box.
        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(vehicle_crop.shape[1], x2)
        y2 = min(vehicle_crop.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return vehicle_crop[y1:y2, x1:x2]

    def read_plate(self, vehicle_crop: np.ndarray) -> Optional[PlateResult]:
        """Full ANPR pipeline: detect plate region -> OCR -> validate.

        Args:
            vehicle_crop: BGR image cropped to a single vehicle's bounding box.

        Returns:
            A :class:`PlateResult`, or None if no plate region was found
            at all (as opposed to being found but unreadable, which
            returns a result flagged PLATE_UNCLEAR).
        """
        plate_crop = self.detect_plate_region(vehicle_crop)
        if plate_crop is None:
            return None
        result = self.read_plate_crop(plate_crop)
        result.crop = plate_crop
        return result

    def read_plate_crop(self, plate_crop: np.ndarray) -> PlateResult:
        """OCR an already-cropped plate image and decide whether to trust the reading.

        Args:
            plate_crop: BGR image of just the plate.

        Returns:
            A :class:`PlateResult`. ``plate_number`` is the plate only when the corrected
            text is a valid plate of the configured region (Indian by default) with confidence >= ``confidence_threshold``;
            otherwise it is :data:`PLATE_UNCLEAR_LABEL`.
        """
        try:
            text, confidence, raw_text = read_plate_text(
                self._reader, plate_crop, min_width=PLATE_OCR_MIN_WIDTH, max_cost=PLATE_OCR_MAX_CORRECTIONS, region=self.region
            )
        except Exception:  # OCR failure must never kill the video pipeline
            logger.exception("Plate OCR failed; treating plate as unclear")
            return PlateResult(PLATE_UNCLEAR_LABEL, 0.0, None, None, False)

        if not text:
            return PlateResult(PLATE_UNCLEAR_LABEL, 0.0, None, None, False)

        is_valid = is_valid_plate(text, self.region)
        if not is_valid or confidence < self.confidence_threshold:
            return PlateResult(PLATE_UNCLEAR_LABEL, confidence, None, None, is_valid, raw_text=raw_text, candidate=text)

        state_code, district_code = self._parse_state_district(text) if self.region == "IN" else (None, None)
        return PlateResult(text, confidence, state_code, district_code, True, raw_text=raw_text)

    @staticmethod
    def _normalize_plate_text(raw_text: str) -> str:
        """Clean OCR output into a canonical uppercase, no-space plate string.

        Args:
            raw_text: Concatenated raw OCR text.

        Returns:
            Normalized string, e.g. "MH12AB1234".
        """
        return normalize(raw_text)

    @staticmethod
    def _parse_state_district(plate_number: str) -> tuple[Optional[str], Optional[str]]:
        """Extract state and district codes from a validated plate string.

        Args:
            plate_number: Normalized plate string matching the Indian format.

        Returns:
            (state_code, district_code) tuple; either may be None if parsing fails.
        """
        state_code = plate_number[:2]
        if state_code not in STATE_CODES:
            return None, None
        district_match = re.match(r"^[A-Z]{2}(\d{1,2})", plate_number)
        district_code = district_match.group(1) if district_match else None
        return state_code, district_code
