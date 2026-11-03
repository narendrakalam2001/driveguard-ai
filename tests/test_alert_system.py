"""Tests for src/alert_system.py — SQLite persistence, live feed, and analytics."""

import pytest

from src.alert_system import AlertSystem
from src.config import SEVERITY
from src.speed_estimator import SpeedEstimate
from src.violation_classifier import ViolationClassifier


@pytest.fixture
def alert_system(tmp_db_path):
    system = AlertSystem(db_path=tmp_db_path)
    system.register_camera("CAM-1", "Test Junction", speed_limit=60)
    return system


@pytest.fixture
def classifier():
    return ViolationClassifier(speed_limit_kmph=60, location="Test Junction", camera_id="CAM-1")


def _log(alert_system, classifier, speed_kmph, plate="MH12AB1234", vehicle_type="Truck"):
    se = SpeedEstimate(track_id=1, instant_speed_kmph=speed_kmph, smoothed_speed_kmph=speed_kmph, direction="inbound")
    record = classifier.classify(se, vehicle_type=vehicle_type, plate_number=plate)
    return record, alert_system.log_violation(record)


def test_schema_applies_and_camera_registers(alert_system):
    with alert_system._connect() as conn:
        row = conn.execute("SELECT * FROM cameras WHERE camera_id='CAM-1'").fetchone()
    assert row is not None
    assert row["speed_limit"] == 60


def test_normal_speed_not_persisted_by_default(alert_system, classifier):
    record, vid = _log(alert_system, classifier, speed_kmph=55)
    assert record.severity == SEVERITY.NORMAL
    assert vid is None


def test_warning_and_above_is_persisted(alert_system, classifier):
    record, vid = _log(alert_system, classifier, speed_kmph=90)
    assert vid is not None
    fetched = alert_system.get_violation(vid)
    assert fetched is not None
    assert fetched["plate_number"] == "MH12AB1234"


def test_get_violation_returns_none_for_unknown_id(alert_system):
    assert alert_system.get_violation("nonexistent") is None


def test_update_notice_persists_text(alert_system, classifier):
    _, vid = _log(alert_system, classifier, speed_kmph=90)
    ok = alert_system.update_notice(vid, "TEST NOTICE TEXT")
    assert ok is True
    fetched = alert_system.get_violation(vid)
    assert fetched["notice_text"] == "TEST NOTICE TEXT"
    assert fetched["notice_generated"] == 1


def test_update_notice_returns_false_for_unknown_id(alert_system):
    assert alert_system.update_notice("nonexistent", "text") is False


def test_update_status_changes_row(alert_system, classifier):
    _, vid = _log(alert_system, classifier, speed_kmph=90)
    assert alert_system.update_status(vid, "RESOLVED") is True
    assert alert_system.get_violation(vid)["status"] == "RESOLVED"


def test_live_feed_newest_first(alert_system, classifier):
    _, vid1 = _log(alert_system, classifier, speed_kmph=68)
    _, vid2 = _log(alert_system, classifier, speed_kmph=95)
    feed = alert_system.get_live_feed()
    assert feed[0]["violation_id"] == vid2
    assert feed[1]["violation_id"] == vid1


def test_query_violations_filters_by_severity(alert_system, classifier):
    _log(alert_system, classifier, speed_kmph=68)  # WARNING
    _log(alert_system, classifier, speed_kmph=95)  # SEVERE
    severe_only = alert_system.query_violations(severity="SEVERE")
    assert len(severe_only) == 1
    assert severe_only[0]["severity"] == "SEVERE"


def test_query_violations_filters_by_camera(alert_system, classifier):
    alert_system.register_camera("CAM-2", "Other Rd", speed_limit=40)
    other_classifier = ViolationClassifier(speed_limit_kmph=40, location="Other Rd", camera_id="CAM-2")
    _log(alert_system, classifier, speed_kmph=90)
    _log(alert_system, other_classifier, speed_kmph=60)
    cam1_only = alert_system.query_violations(camera_id="CAM-1")
    assert all(v["camera_id"] == "CAM-1" for v in cam1_only)


def test_get_prior_plate_numbers_includes_duplicates(alert_system, classifier):
    _log(alert_system, classifier, speed_kmph=90, plate="MH12AB1234")
    _log(alert_system, classifier, speed_kmph=95, plate="MH12AB1234")
    plates = alert_system.get_prior_plate_numbers()
    assert plates.count("MH12AB1234") == 2


def test_analytics_summary_counts_and_breakdowns(alert_system, classifier):
    _log(alert_system, classifier, speed_kmph=68, vehicle_type="Truck")  # WARNING
    _log(alert_system, classifier, speed_kmph=95, vehicle_type="Car")  # SEVERE
    summary = alert_system.get_analytics_summary()
    assert summary["total_violations"] == 2
    assert summary["severity_breakdown"].get("WARNING") == 1
    assert summary["severity_breakdown"].get("SEVERE") == 1
    assert summary["vehicle_type_breakdown"].get("Truck") == 1
    assert summary["vehicle_type_breakdown"].get("Car") == 1


def test_upsert_vehicle_session_creates_and_updates(alert_system):
    alert_system.upsert_vehicle_session(
        track_id=1,
        session_id="sess-1",
        speed_kmph=60.0,
        vehicle_type="Car",
        plate_number="MH12AB1234",
        first_seen_iso="2026-01-01T00:00:00",
        last_seen_iso="2026-01-01T00:00:01",
        is_violation=False,
    )
    alert_system.upsert_vehicle_session(
        track_id=1,
        session_id="sess-1",
        speed_kmph=80.0,
        vehicle_type="Car",
        plate_number="MH12AB1234",
        first_seen_iso="2026-01-01T00:00:00",
        last_seen_iso="2026-01-01T00:00:02",
        is_violation=True,
    )
    with alert_system._connect() as conn:
        row = conn.execute("SELECT * FROM vehicles WHERE track_id=1 AND session_id='sess-1'").fetchone()
    assert row["max_speed"] == 80.0
    assert row["violation_count"] == 1
