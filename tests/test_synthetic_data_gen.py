"""Tests for src/synthetic_data_gen.py — synthetic incident generation."""

import re
from datetime import datetime, timezone

from src.config import INDIAN_PLATE_REGEX
from src.synthetic_data_gen import (
    _generate_plate_number,
    _severity_for_excess,
    generate_incident,
    generate_synthetic_dataset,
)


def test_generated_plate_matches_indian_format():
    for _ in range(50):
        plate = _generate_plate_number()
        assert re.match(INDIAN_PLATE_REGEX, plate), f"Invalid plate generated: {plate}"


def test_severity_for_excess_bands():
    assert _severity_for_excess(5) == "NORMAL"
    assert _severity_for_excess(15) == "WARNING"
    assert _severity_for_excess(30) == "VIOLATION"
    assert _severity_for_excess(60) == "SEVERE"


def test_generate_incident_has_required_fields():
    incident = generate_incident(datetime.now(timezone.utc))
    required_fields = {
        "vehicle_id", "plate_number", "vehicle_type", "detected_speed", "speed_limit",
        "violation_type", "severity", "excess_percent", "location", "timestamp",
        "evidence_frame", "weather", "road_condition", "camera_id",
    }
    assert required_fields.issubset(incident.keys())
    assert incident["violation_type"] == "SPEEDING"


def test_generate_synthetic_dataset_count_and_reproducibility():
    batch1 = generate_synthetic_dataset(n=25, seed=7)
    batch2 = generate_synthetic_dataset(n=25, seed=7)
    assert len(batch1) == 25
    # Same seed should reproduce the same plate numbers (order-sensitive).
    assert [i["plate_number"] for i in batch1] == [i["plate_number"] for i in batch2]


def test_generate_synthetic_dataset_speed_always_nonnegative():
    batch = generate_synthetic_dataset(n=50, seed=1)
    assert all(incident["detected_speed"] >= 0 for incident in batch)
