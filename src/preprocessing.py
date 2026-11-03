"""
preprocessing.py
=================
Frame-level preprocessing: resizing, normalization, and light
augmentation used both at training time (via Albumentations) and at
inference time (fast OpenCV-only path, no augmentation).
"""

from __future__ import annotations

from typing import Tuple

import albumentations as A
import cv2
import numpy as np

from src.config import get_logger

logger = get_logger(__name__)


def resize_letterbox(
    image: np.ndarray, target_size: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize an image to target_size while preserving aspect ratio via padding.

    This mirrors YOLO's standard letterbox resize so detections map back
    to original coordinates correctly.

    Args:
        image: Source BGR image.
        target_size: (width, height) the model expects.
        color: Padding color.

    Returns:
        Tuple of (letterboxed_image, scale_ratio, (pad_x, pad_y)).
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return padded, scale, (pad_x, pad_y)


def undo_letterbox_coords(
    boxes_xyxy: np.ndarray, scale: float, pad: Tuple[int, int]
) -> np.ndarray:
    """Map letterboxed-image box coordinates back to the original image space.

    Args:
        boxes_xyxy: Array of shape (N, 4) in [x1, y1, x2, y2] format,
            in letterboxed-image pixel coordinates.
        scale: Scale ratio returned by :func:`resize_letterbox`.
        pad: (pad_x, pad_y) returned by :func:`resize_letterbox`.

    Returns:
        Array of shape (N, 4) in original-image pixel coordinates.
    """
    pad_x, pad_y = pad
    boxes = boxes_xyxy.copy().astype(np.float32)
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale
    return boxes


def normalize_frame(image: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] float32, RGB channel order.

    Args:
        image: Source BGR uint8 image.

    Returns:
        Float32 RGB image in range [0, 1].
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32)) / 255.0


def denoise_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
    """Preprocessing pipeline specifically for ANPR: grayscale + threshold + denoise.

    Args:
        plate_crop: Cropped BGR image of a license plate region.

    Returns:
        A cleaned, binarized grayscale image ready for OCR.
    """
    if plate_crop.size == 0:
        return plate_crop

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=15
    )
    return thresh


def build_training_augmentations(image_size: int = 640) -> A.Compose:
    """Build the Albumentations training-time augmentation pipeline.

    Chosen augmentations are deliberately conservative — traffic footage
    is already highly varied (lighting, weather, occlusion), so we avoid
    aggressive geometric distortion that could break bounding-box alignment.

    Args:
        image_size: Square target size for training crops.

    Returns:
        A configured ``albumentations.Compose`` pipeline with
        ``bbox_params`` set for Pascal-VOC style boxes.
    """
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.RandomRain(p=0.1),
            A.RandomFog(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3),
    )


def apply_augmentation(
    augmenter: A.Compose,
    image: np.ndarray,
    boxes: np.ndarray,
    class_labels: list,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Apply a built Albumentations pipeline to one image + its boxes.

    Args:
        augmenter: Pipeline from :func:`build_training_augmentations`.
        image: BGR image.
        boxes: Array of shape (N, 4) in Pascal-VOC [x1, y1, x2, y2] format.
        class_labels: List of length N with class names/ids for each box.

    Returns:
        (augmented_image, augmented_boxes, augmented_class_labels)
    """
    if len(boxes) == 0:
        result = augmenter(image=image, bboxes=[], class_labels=[])
        return result["image"], np.zeros((0, 4)), []

    result = augmenter(image=image, bboxes=boxes.tolist(), class_labels=class_labels)
    aug_boxes = np.array(result["bboxes"]) if result["bboxes"] else np.zeros((0, 4))
    return result["image"], aug_boxes, result["class_labels"]
