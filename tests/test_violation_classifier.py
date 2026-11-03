"""Tests for src/violation_classifier.py — severity thresholds and record construction."""

import pytest

from src.config import SEVERITY
from src.speed_estimator import SpeedEstimate
from src.violation_classifier import ViolationClassifier


@pytest.fixture
def classifier():
    return ViolationClassifier(speed_limit_kmph=60, location="Test Rd", camera_id="CAM-1")


def _speed(v):
    return SpeedEstimate(track_id=1, instant_speed_kmph=v, smoothed_speed_kmph=v, direction="inbound")


def test_normal_speed_below_warning_threshold(classifier):
    assert classifier.classify_severity(60) == SEVERITY.NORMAL
    assert classifier.classify_severity(65) == SEVERITY.NORMAL  # exactly at 1.1x boundary is not > 


def test_warning_band(classifier):
    assert classifier.classify_severity(67) == SEVERITY.WARNING  # 60*1.1=66, so 67 > 66


def test_violation_band(classifier):
    assert classifier.classify_severity(73) == SEVERITY.VIOLATION  # 60*1.2=72


def test_severe_band(classifier):
    assert classifier.classify_severity(91) == SEVERITY.SEVERE  # 60*1.5=90


def test_classify_builds_full_record(classifier):
    record = classifier.classify(_speed(90), vehicle_type="Truck", plate_number="MH12AB1234")
    assert record.severity == SEVERITY.VIOLATION  # 90 is not > 90? 90>90 false -> VIOLATION band (>72,<=90)
    assert record.plate_number == "MH12AB1234"
    assert record.vehicle_type == "Truck"
    assert record.speed_limit_kmph == 60
    assert record.excess_speed_kmph == pytest.approx(30.0)
    assert record.excess_percent == pytest.approx(50.0)
    assert record.camera_id == "CAM-1"
    assert record.location == "Test Rd"


def test_classify_normal_speed_has_zero_excess(classifier):
    record = classifier.classify(_speed(55), vehicle_type="Car")
    assert record.severity == SEVERITY.NORMAL
    assert record.excess_speed_kmph == 0.0


def test_is_repeat_offender_true_when_plate_seen_before(classifier):
    assert classifier.is_repeat_offender("MH12AB1234", ["MH12AB1234", "KA01XY5678"]) is True


def test_is_repeat_offender_false_when_plate_not_seen(classifier):
    assert classifier.is_repeat_offender("DL8CAZ4521", ["MH12AB1234"]) is False


def test_violation_id_is_unique_across_calls(classifier):
    r1 = classifier.classify(_speed(90), vehicle_type="Car")
    r2 = classifier.classify(_speed(90), vehicle_type="Car")
    assert r1.violation_id != r2.violation_id
