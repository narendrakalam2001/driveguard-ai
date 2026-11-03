"""Tests for serving/api.py — endpoint contracts via FastAPI's TestClient.

Uses the module-level AlertSystem instance directly to seed data, mirroring
how the endpoints themselves read/write it.
"""

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to an isolated, temporary violations DB per test.

    The module-level ``_alert_system`` in serving.api is swapped for a
    fresh instance pointed at a per-test tmp_path DB, since AlertSystem's
    default db_path is bound at import time (not patchable via
    monkeypatching src.config after the fact).
    """
    import serving.api as api_module
    from src.alert_system import AlertSystem

    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(api_module, "_UPLOAD_DIR", tmp_path / "uploads")  # keep test uploads out of the project
    api_module._alert_system = AlertSystem(db_path=tmp_path / "violations.db")
    api_module._SESSIONS.clear()
    return TestClient(api_module.app), api_module


def test_health_check(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_violations_empty_by_default(client):
    test_client, _ = client
    response = test_client.get("/violations")
    assert response.status_code == 200
    assert response.json() == {"count": 0, "total": 0, "violations": []}


def test_violation_not_found_returns_404(client):
    test_client, _ = client
    response = test_client.get("/violation/does-not-exist")
    assert response.status_code == 404


def test_process_video_rejects_bad_file_extension(client):
    test_client, _ = client
    response = test_client.post(
        "/process-video",
        files={"file": ("clip.txt", b"not a video", "text/plain")},
        data={
            "camera_id": "CAM-1",
            "speed_limit_kmph": 60,
            "lane_width_m": 3.5,
            "calibration_points": json.dumps(
                [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}]
            ),
        },
    )
    assert response.status_code == 400


def test_process_video_rejects_degenerate_calibration(client):
    test_client, _ = client
    response = test_client.post(
        "/process-video",
        files={"file": ("clip.mp4", b"fake bytes", "video/mp4")},
        data={
            "camera_id": "CAM-1",
            "speed_limit_kmph": 60,
            "lane_width_m": 3.5,
            "calibration_points": json.dumps(
                [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 20, "y": 0}, {"x": 30, "y": 0}]
            ),
        },
    )
    assert response.status_code == 400


def test_process_stream_requires_four_calibration_points(client):
    test_client, _ = client
    response = test_client.post(
        "/process-stream",
        json={"camera_id": "CAM-1", "stream_source": "0"},  # missing calibration_points
    )
    assert response.status_code == 422


def test_generate_report_fallback_for_unknown_violation(client):
    test_client, _ = client
    response = test_client.post(
        "/generate-report", json={"violation_id": "does-not-exist", "use_llm": False}
    )
    assert response.status_code == 404


def test_full_violation_lifecycle_via_api(client):
    """Seed a violation directly through AlertSystem, then read/report it via the API."""
    test_client, api_module = client
    from src.violation_classifier import ViolationClassifier
    from src.speed_estimator import SpeedEstimate

    api_module._alert_system.register_camera("CAM-9", "Test Rd", 60)
    clf = ViolationClassifier(speed_limit_kmph=60, location="Test Rd", camera_id="CAM-9")
    se = SpeedEstimate(track_id=1, instant_speed_kmph=95, smoothed_speed_kmph=95, direction="inbound")
    record = clf.classify(se, vehicle_type="Car", plate_number="KA01MF5678")
    vid = api_module._alert_system.log_violation(record)

    get_response = test_client.get(f"/violation/{vid}")
    assert get_response.status_code == 200
    assert get_response.json()["severity"] == "SEVERE"

    report_response = test_client.post("/generate-report", json={"violation_id": vid, "use_llm": False})
    assert report_response.status_code == 200
    assert "KA01MF5678" in report_response.json()["notice_text"]

    analytics_response = test_client.get("/analytics")
    assert analytics_response.json()["total_violations"] == 1
