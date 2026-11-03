"""
vehicle_detector.py
====================
Vehicle detection engine wrapping Ultralytics YOLOv8.

Responsible only for detection (bounding boxes + class + confidence).
Tracking, speed, and plate logic live in their own modules to keep
this class single-responsibility and easy to unit test / swap models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.config import (
    DETECTION_CONF_THRESHOLD,
    DETECTION_IOU_THRESHOLD,
    VEHICLE_CLASSES,
    VEHICLE_MODEL_PATH,
    YOLO_BASE_WEIGHTS,
    get_logger,
)

logger = get_logger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before a model is loaded."""


@dataclass
class Detection:
    """A single detected vehicle in one frame.

    Attributes:
        xyxy: [x1, y1, x2, y2] pixel bounding box in original image space.
        confidence: Model confidence score in [0, 1].
        class_id: Integer class id from :data:`VEHICLE_CLASSES`.
        class_name: Human-readable class name (Car, Truck, ...).
    """

    xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """YOLOv8-based multi-class vehicle detector.

    Wraps Ultralytics' YOLO API. Supports loading either the fine-tuned
    production weights (``models/yolov8_vehicle.pt``) or falling back to
    base COCO weights for a quick smoke test when no fine-tuned model
    exists yet.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        conf_threshold: float = DETECTION_CONF_THRESHOLD,
        iou_threshold: float = DETECTION_IOU_THRESHOLD,
        device: str = "auto",
    ) -> None:
        """Initialize and load the detection model.

        Args:
            weights_path: Path to .pt weights. Defaults to the fine-tuned
                production weights path; falls back to base COCO weights
                with a warning if that file does not exist yet.
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: NMS IoU threshold.
            device: "cpu", "cuda", "cuda:0", or "auto" to let Ultralytics decide.
        """
        from ultralytics import YOLO  # local import: heavy dependency, only needed here

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        resolved_path = Path(weights_path) if weights_path else VEHICLE_MODEL_PATH
        if not resolved_path.exists():
            logger.info(
                f"Fine-tuned weights not found at {resolved_path}; "
                f"falling back to base weights {YOLO_BASE_WEIGHTS}. "
                "Run scripts/train_detector.py to produce production weights."
            )
            resolved_path = YOLO_BASE_WEIGHTS

        self._model = YOLO(str(resolved_path))
        self._is_finetuned = resolved_path != YOLO_BASE_WEIGHTS
        logger.info(f"VehicleDetector loaded weights={resolved_path} finetuned={self._is_finetuned}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Args:
            frame: BGR image array (H, W, 3), as read by OpenCV.

        Returns:
            List of :class:`Detection` objects, one per detected vehicle,
            filtered to the vehicle classes of interest.

        Raises:
            ModelNotLoadedError: If called before the model finished loading.
        """
        if self._model is None:
            raise ModelNotLoadedError("YOLO model is not loaded.")

        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=None if self.device == "auto" else self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        names = result.names  # model-specific class id -> name mapping
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name_raw = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            mapped_name = self._map_class_name(cls_name_raw)
            if mapped_name is None:
                continue  # not a vehicle class we care about (e.g. pedestrian)

            xyxy = box.xyxy.cpu().numpy().reshape(-1)
            conf = float(box.conf.item())
            detections.append(
                Detection(xyxy=xyxy, confidence=conf, class_id=cls_id, class_name=mapped_name)
            )

        return detections

    @staticmethod
    def _map_class_name(raw_name: str) -> Optional[str]:
        """Map raw model class names (COCO or fine-tuned) onto DriveGuard's taxonomy.

        Args:
            raw_name: Class name as reported by the underlying model.

        Returns:
            The mapped DriveGuard class name, or None if the class should
            be ignored (e.g. "person", "traffic light" from base COCO weights).
        """
        coco_map = {
            "car": "Car",
            "truck": "Truck",
            "bus": "Bus",
            "motorcycle": "Motorcycle",
            "bicycle": "Bicycle",
        }
        lowered = raw_name.lower()
        if lowered in coco_map:
            return coco_map[lowered]
        if raw_name in VEHICLE_CLASSES.values():
            return raw_name
        return None

    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection over a batch of frames for throughput efficiency.

        Args:
            frames: List of BGR frames.

        Returns:
            List of detection lists, one per input frame (same order).
        """
        results = self._model.predict(
            source=frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=None if self.device == "auto" else self.device,
            verbose=False,
        )
        batched: List[List[Detection]] = []
        for result in results:
            frame_dets: List[Detection] = []
            if result.boxes is not None:
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name_raw = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    mapped_name = self._map_class_name(cls_name_raw)
                    if mapped_name is None:
                        continue
                    xyxy = box.xyxy.cpu().numpy().reshape(-1)
                    conf = float(box.conf.item())
                    frame_dets.append(
                        Detection(xyxy=xyxy, confidence=conf, class_id=cls_id, class_name=mapped_name)
                    )
            batched.append(frame_dets)
        return batched
