"""Tests for src/speed_estimator.py — trajectory-to-speed math and smoothing."""

from collections import deque

import pytest

from src.speed_estimator import SpeedEstimator
from src.tracker import TrackState
import numpy as np


def _make_track(track_id: int, points):
    """Build a TrackState with a pre-populated centroid history."""
    track = TrackState(track_id=track_id, class_name="Car", bbox=np.array([0, 0, 10, 10], dtype=np.float32))
    for x, y, t in points:
        track.centroid_history.append((x, y, t))
    return track


def test_estimate_returns_none_with_single_point(square_calibration):
    estimator = SpeedEstimator(square_calibration, min_points=2)
    track = _make_track(1, [(100, 100, 0.0)])
    assert estimator.estimate(track) is None


def test_estimate_computes_plausible_speed(square_calibration):
    """A vehicle moving 3.5m in 1 second along calibrated depth => 3.5 m/s => 12.6 km/h."""
    estimator = SpeedEstimator(square_calibration, min_points=2)
    # world_y moves from 0 to 3.5m over 1 second (since depth scale = 3.5/(400-100)*300... use calibration directly)
    # Easier: move from (100,100) to (100, 100 + (300/14)*3.5) over 1s -> world dy = 3.5m
    track = _make_track(2, [(100, 100, 0.0), (100, 100 + (300 / 14) * 3.5, 1.0)])
    estimate = estimator.estimate(track)
    assert estimate is not None
    assert estimate.instant_speed_kmph == pytest.approx(3.5 * 3.6, rel=0.05)


def test_estimate_zero_dt_returns_none(square_calibration):
    estimator = SpeedEstimator(square_calibration, min_points=2)
    track = _make_track(3, [(100, 100, 1.0), (110, 110, 1.0)])  # same timestamp
    assert estimator.estimate(track) is None


def test_speed_is_clamped_to_plausible_ceiling(square_calibration):
    estimator = SpeedEstimator(square_calibration, min_points=2)
    # Enormous, physically impossible jump in a tiny time window.
    track = _make_track(4, [(100, 100, 0.0), (100, 100000, 0.001)])
    estimate = estimator.estimate(track)
    assert estimate.instant_speed_kmph <= 250.0


def test_smoothed_speed_is_measured_over_the_window_not_the_last_step(square_calibration):
    """One big jump among steady steps must not dominate the reported speed."""
    estimator = SpeedEstimator(square_calibration, smoothing_window=4, min_points=2)
    px_per_m = 300 / 14
    track = _make_track(5, [(100, 100, 0.0)])
    y = 100.0
    for i, step_m in enumerate([1.0, 1.0, 1.0, 5.0], start=1):  # 1 m/s steps, then a 5 m glitch
        y += step_m * px_per_m
        track.centroid_history.append((100, y, float(i)))
    est = estimator.estimate(track)
    assert est.instant_speed_kmph == pytest.approx(5.0 * 3.6, rel=0.02)
    assert est.smoothed_speed_kmph == pytest.approx(8.0 / 4 * 3.6, rel=0.02)  # 8 m over 4 s


def test_jitter_on_a_stationary_vehicle_reads_as_zero_not_as_motion(square_calibration):
    """Summing jittery step lengths would report a speed for a parked car; net displacement does not."""
    estimator = SpeedEstimator(square_calibration, smoothing_window=10, min_points=2)
    points = [(100 + (2 if i % 2 else -2), 200 + (2 if i % 2 else -2), i * 0.04) for i in range(11)]
    est = estimator.estimate(_make_track(9, points))
    assert est.smoothed_speed_kmph < 1.0
    assert est.instant_speed_kmph > 5.0  # the single step on its own looks like motion


def test_no_speed_until_enough_points_have_been_seen(square_calibration):
    """The first noisy frames of a new track must never be able to raise a violation."""
    estimator = SpeedEstimator(square_calibration)  # default warm-up
    assert estimator.min_points >= 2
    track = _make_track(10, [(100, 100, 0.0)])
    for i in range(1, estimator.min_points - 1):
        track.centroid_history.append((100, 100 + i * 50, i * 0.04))
        assert estimator.estimate(track) is None
    track.centroid_history.append((100, 100 + 50 * estimator.min_points, estimator.min_points * 0.04))
    assert estimator.estimate(track) is not None


def test_speed_history_is_recorded_for_the_dashboard_graph(square_calibration):
    estimator = SpeedEstimator(square_calibration, min_points=2)
    track = _make_track(11, [(100, 100, 0.0), (100, 150, 1.0)])
    estimator.estimate(track)
    track.centroid_history.append((100, 200, 2.0))
    estimator.estimate(track)
    assert len(estimator.get_speed_history(11)) == 2
    assert estimator.get_speed_history(999) == []


def test_ground_point_is_bottom_centre_of_the_box():
    track = TrackState(track_id=12, class_name="Truck", bbox=np.array([100, 50, 200, 150], dtype=np.float32))
    assert track.ground_point() == (150.0, 150.0)
    assert track.centroid() == (150.0, 100.0)


def test_purge_stale_removes_untracked_history(square_calibration):
    estimator = SpeedEstimator(square_calibration, min_points=2)
    track = _make_track(6, [(100, 100, 0.0), (100, 150, 1.0)])
    estimator.estimate(track)
    assert 6 in estimator._speed_history
    estimator.purge_stale(active_track_ids=set())
    assert 6 not in estimator._speed_history


def test_direction_unknown_with_little_displacement(square_calibration):
    estimator = SpeedEstimator(square_calibration, min_points=2)
    track = _make_track(7, [(100, 100, float(i)) for i in range(6)])  # no y movement
    estimator.estimate(track)
    assert track.direction == "unknown"


def test_direction_inbound_with_positive_dy(square_calibration):
    estimator = SpeedEstimator(square_calibration, min_points=2)
    points = [(100, 100 + i * 10, float(i)) for i in range(6)]  # y increasing
    track = _make_track(8, points)
    estimator.estimate(track)
    assert track.direction == "inbound"
