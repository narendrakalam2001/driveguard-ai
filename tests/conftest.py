"""Shared pytest configuration: repo-root path setup and common fixtures."""

import os
import sys
from pathlib import Path

# Tests must not depend on the developer's .env (python-dotenv never overrides variables that already exist).
# Force the code defaults before any project module is imported.
os.environ["DRIVEGUARD_API_KEY"] = ""      # API auth off
os.environ["DRIVEGUARD_DEMO"] = "0"        # tests exercise the full API mode; demo-mode tests switch it on themselves
os.environ["GEMINI_API_KEY"] = ""          # never call the real LLM from tests
os.environ["PLATE_CONFIDENCE_THRESHOLD"] = "0.3"
os.environ["PLATE_REGION"] = "IN"          # the product default; tests must not depend on a developer's .env
os.environ["MAX_ACCEL_KMPH_PER_S"] = "45"
os.environ["DISPLAY_UTC_OFFSET_MINUTES"] = "330"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture
def tmp_db_path(tmp_path):
    """A temporary SQLite DB path for AlertSystem tests, isolated per test."""
    return tmp_path / "test_violations.db"


@pytest.fixture
def square_calibration():
    """A simple, exact 3.5m x 14m calibration for deterministic math checks."""
    from src.camera_calibrator import CameraCalibrator

    calibrator = CameraCalibrator()
    return calibrator.calibrate(
        camera_id="CAM-TEST",
        image_points=[(100, 100), (300, 100), (300, 400), (100, 400)],
        lane_width_m=3.5,
        speed_limit_kmph=60,
    )


@pytest.fixture
def make_record():
    """Factory for ViolationRecord objects with controllable speed, plate, camera and timestamp."""
    import uuid
    from datetime import datetime, timezone

    from src.violation_classifier import ViolationRecord

    def _make(speed=80.0, limit=60, plate="MH12AB1234", vehicle_type="Car", camera_id="CAM-1", ts=None,
              severity=None, location="Test Junction"):
        excess = round(speed - limit, 1)
        if severity is None:
            severity = "SEVERE" if speed > limit * 1.5 else "VIOLATION" if speed > limit * 1.2 else "WARNING"
        return ViolationRecord(
            violation_id=str(uuid.uuid4()), track_id=1, plate_number=plate, vehicle_type=vehicle_type,
            detected_speed_kmph=speed, speed_limit_kmph=limit, excess_speed_kmph=excess,
            excess_percent=round(excess / limit * 100, 1), severity=severity, location=location,
            camera_id=camera_id, timestamp=ts or datetime.now(timezone.utc),
        )

    return _make


class FakeGenai:
    """Test double for ``google.genai.Client``: ``models.generate_content`` replays scripted outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # each item: str (reply text) or Exception (raised)
        self.calls = []
        outer = self

        class _Models:
            def generate_content(self, model, contents, config=None):
                outer.calls.append(model)
                item = outer.outcomes.pop(0)
                if isinstance(item, Exception):
                    raise item

                class _R:
                    text = item

                return _R()

        self.models = _Models()


@pytest.fixture
def fake_genai():
    return FakeGenai
