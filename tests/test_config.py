"""Tests for src/config.py — thresholds, zones, and severity constants."""

from src.config import (
    DEFAULT_LANE_WIDTH_M,
    SEVERE_MULTIPLIER,
    SEVERITY,
    SPEED_ZONES,
    VEHICLE_CLASSES,
    VIOLATION_MULTIPLIER,
    WARNING_MULTIPLIER,
    get_logger,
)


def test_speed_zones_match_spec():
    assert SPEED_ZONES == {
        "School Zone": 25,
        "Residential": 40,
        "City Road": 60,
        "Highway": 100,
    }


def test_violation_multiplier_ordering():
    """Warning < violation < severe thresholds must be strictly increasing."""
    assert WARNING_MULTIPLIER < VIOLATION_MULTIPLIER < SEVERE_MULTIPLIER


def test_default_lane_width_is_standard_indian_lane():
    assert DEFAULT_LANE_WIDTH_M == 3.5


def test_vehicle_classes_cover_seven_categories():
    assert len(VEHICLE_CLASSES) == 7
    assert set(VEHICLE_CLASSES.values()) == {
        "Car",
        "Truck",
        "Bus",
        "Motorcycle",
        "Auto-rickshaw",
        "Bicycle",
        "Van",
    }


def test_severity_labels_are_distinct():
    labels = {SEVERITY.NORMAL, SEVERITY.WARNING, SEVERITY.VIOLATION, SEVERITY.SEVERE}
    assert len(labels) == 4


def test_get_logger_is_idempotent():
    """Calling get_logger twice for the same name must not duplicate handlers."""
    logger_a = get_logger("test.duplicate")
    logger_b = get_logger("test.duplicate")
    assert logger_a is logger_b
    assert len(logger_a.handlers) == 2  # console + file, not 4
