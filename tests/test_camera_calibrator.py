"""Tests for src/camera_calibrator.py — homography correctness and error handling."""

import pytest

from src.camera_calibrator import CalibrationError, CameraCalibration, CameraCalibrator


def test_calibration_maps_top_left_to_origin(square_calibration):
    wx, wy = square_calibration.pixel_to_world(100, 100)
    assert wx == pytest.approx(0.0, abs=1e-6)
    assert wy == pytest.approx(0.0, abs=1e-6)


def test_calibration_maps_bottom_right_to_full_extent(square_calibration):
    wx, wy = square_calibration.pixel_to_world(300, 400)
    assert wx == pytest.approx(3.5, abs=1e-6)
    assert wy == pytest.approx(14.0, abs=1e-6)  # 4x lane width default depth


def test_calibration_midpoint_is_proportional(square_calibration):
    wx, wy = square_calibration.pixel_to_world(200, 250)
    assert wx == pytest.approx(1.75, abs=0.05)
    assert wy == pytest.approx(7.0, abs=0.05)


def test_calibrate_rejects_wrong_point_count():
    calibrator = CameraCalibrator()
    with pytest.raises(CalibrationError):
        calibrator.calibrate("CAM-1", [(0, 0), (10, 0), (10, 10)])  # only 3 points


def test_calibrate_rejects_collinear_points():
    calibrator = CameraCalibrator()
    with pytest.raises(CalibrationError):
        calibrator.calibrate("CAM-1", [(0, 0), (10, 0), (20, 0), (30, 0)])


def test_calibration_custom_lane_depth():
    calibrator = CameraCalibrator()
    cal = calibrator.calibrate(
        "CAM-2", [(0, 0), (350, 0), (350, 1000), (0, 1000)], lane_width_m=3.5, lane_depth_m=50.0
    )
    wx, wy = cal.pixel_to_world(350, 1000)
    assert wx == pytest.approx(3.5, abs=1e-3)
    assert wy == pytest.approx(50.0, abs=1e-3)


def test_save_and_load_roundtrip(tmp_path, square_calibration):
    path = tmp_path / "calibration.json"
    square_calibration.save(path)
    reloaded = CameraCalibration.load(path)
    assert reloaded.camera_id == square_calibration.camera_id
    assert reloaded.speed_limit_kmph == square_calibration.speed_limit_kmph
    wx, wy = reloaded.pixel_to_world(300, 400)
    assert wx == pytest.approx(3.5, abs=1e-6)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CameraCalibration.load(tmp_path / "does_not_exist.json")
