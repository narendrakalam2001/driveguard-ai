"""Tests for src/preprocessing.py — letterbox resize/coord mapping and plate denoising."""

import numpy as np
import pytest

from src.preprocessing import denoise_plate_crop, normalize_frame, resize_letterbox, undo_letterbox_coords


def test_resize_letterbox_output_shape():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out, scale, pad = resize_letterbox(img, target_size=(640, 640))
    assert out.shape == (640, 640, 3)


def test_resize_letterbox_preserves_aspect_ratio_via_scale():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out, scale, pad = resize_letterbox(img, target_size=(640, 640))
    assert scale == pytest.approx(640 / 640, rel=0.01)  # width-limited: scale = 1.0


def test_undo_letterbox_coords_roundtrip():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out, scale, pad = resize_letterbox(img, target_size=(640, 640))
    # A box drawn directly on the letterboxed image...
    boxes = np.array([[0, 80, 640, 560]], dtype=np.float32)  # accounts for vertical padding
    original = undo_letterbox_coords(boxes, scale, pad)
    # ...should map back close to the original image's full extent.
    assert original[0, 0] == pytest.approx(0, abs=2)
    assert original[0, 2] == pytest.approx(640, abs=2)


def test_normalize_frame_range_and_dtype():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    normalized = normalize_frame(img)
    assert normalized.dtype == np.float32
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0


def test_denoise_plate_crop_handles_empty_input():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    result = denoise_plate_crop(empty)
    assert result.size == 0


def test_denoise_plate_crop_returns_grayscale():
    crop = np.random.randint(0, 255, (40, 120, 3), dtype=np.uint8)
    result = denoise_plate_crop(crop)
    assert result.ndim == 2  # grayscale, not BGR
