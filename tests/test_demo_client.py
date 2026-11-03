"""Lite (no-API) dashboard: DemoClient endpoints, demo-mode switch, and every screen on bundled demo data."""

import pytest
from streamlit.testing.v1 import AppTest

from frontend.api_client import ApiError
from frontend.demo_client import DEMO_DB, DemoClient, is_demo_mode

SCREENS = ["1. Live monitoring", "2. Upload and analysis", "3. Violations log", "4. Violation detail", "5. Analytics", "6. Reports and export"]


@pytest.fixture(scope="module")
def client() -> DemoClient:
    return DemoClient()


@pytest.fixture
def demo_env(monkeypatch):
    """Switch the dashboard into demo mode for AppTest runs and reset Streamlit's cached client."""
    import streamlit as st

    monkeypatch.setenv("DRIVEGUARD_DEMO", "1")
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def test_bundled_demo_database_exists_and_is_small():
    assert DEMO_DB.is_file()
    assert DEMO_DB.stat().st_size < 5 * 1024 * 1024


@pytest.mark.parametrize("flag,url,expected", [
    ("1", "http://localhost:8000", True), ("true", "", True), ("0", "", False), ("off", "", False),
    ("", "", True), ("", "http://localhost:8000", False), ("", "   ", True),
])
def test_is_demo_mode_rules(monkeypatch, flag, url, expected):
    monkeypatch.setenv("DRIVEGUARD_DEMO", flag)
    monkeypatch.setenv("DRIVEGUARD_API_URL", url)
    assert is_demo_mode() is expected


def test_health_and_cameras(client):
    assert client.json("GET", "/health")["status"] == "ok"
    assert len(client.json("GET", "/cameras")["cameras"]) == 8


def test_violation_list_filters_and_paging(client):
    everything = client.json("GET", "/violations", {"limit": 5})
    assert everything["total"] == 1800 and everything["count"] == 5
    severe = client.json("GET", "/violations", {"severity": "SEVERE", "limit": 200})
    assert 0 < severe["total"] < 1800 and all(v["severity"] == "SEVERE" for v in severe["violations"])
    trucks = client.json("GET", "/violations", {"vehicle_type": "Truck,Bus", "limit": 50})
    assert {v["vehicle_type"] for v in trucks["violations"]} <= {"Truck", "Bus"}
    page2 = client.json("GET", "/violations", {"limit": 5, "offset": 5})["violations"]
    assert {v["id"] for v in page2}.isdisjoint({v["id"] for v in everything["violations"]})


def test_newest_violation_is_not_in_the_future(client):
    from datetime import datetime, timezone

    newest = client.json("GET", "/violations", {"limit": 1})["violations"][0]["timestamp"]
    assert datetime.fromisoformat(newest) <= datetime.now(timezone.utc)


def test_unknown_violation_and_route_raise_api_error(client):
    with pytest.raises(ApiError):
        client.json("GET", "/violation/does-not-exist")
    with pytest.raises(ApiError):
        client.json("GET", "/nope")


def test_status_update_roundtrip_and_validation(client):
    vid = client.json("GET", "/violations", {"status": "PENDING", "limit": 1})["violations"][0]["id"]
    client.json("PATCH", f"/violation/{vid}/status", json={"status": "RESOLVED"})
    assert client.json("GET", f"/violation/{vid}")["status"] == "RESOLVED"
    with pytest.raises(ApiError):
        client.json("PATCH", f"/violation/{vid}/status", json={"status": "BOGUS"})


def test_notice_template_edit_pdf_and_rto_flow(client):
    rows = client.json("GET", "/violations", {"severity": "VIOLATION", "limit": 200})["violations"]
    vid = next(r["id"] for r in rows if not r["notice_text"])
    with pytest.raises(ApiError, match="409"):  # no notice yet -> cannot send to the RTO
        client.json("POST", f"/violation/{vid}/send-to-rto")
    res = client.json("POST", "/generate-report", json={"violation_id": vid, "use_llm": True})
    assert "TRAFFIC VIOLATION NOTICE" in res["notice_text"] and "template" in res["model_used"]
    client.json("PUT", f"/violation/{vid}/notice", json={"notice_text": "edited by officer"})
    assert client.json("GET", f"/violation/{vid}")["notice_text"] == "edited by officer"
    assert client.bytes(f"/violation/{vid}/notice.pdf").startswith(b"%PDF")
    assert "Mock" in client.json("POST", f"/violation/{vid}/send-to-rto")["message"]
    assert len(client.json("GET", f"/violation/{vid}/dispatches")["dispatches"]) == 1


def test_evidence_and_speed_trace_are_absent_not_faked(client):
    vid = client.json("GET", "/violations", {"limit": 1})["violations"][0]["id"]
    with pytest.raises(ApiError):
        client.bytes(f"/violation/{vid}/evidence", {"kind": "frame"})
    with pytest.raises(ApiError):
        client.json("GET", f"/violation/{vid}/speed-trace")


def test_exports_and_summary_pdf(client):
    csv_bytes = client.bytes("/export/violations", {"format": "csv", "severity": "SEVERE"})
    assert b"SEVERE" in csv_bytes and b"WARNING" not in csv_bytes
    assert client.bytes("/export/violations", {"format": "xlsx"})[:2] == b"PK"  # xlsx is a zip
    assert client.bytes("/reports/summary.pdf").startswith(b"%PDF")


def test_analytics_and_insights(client):
    s, d = client.json("GET", "/analytics"), client.json("GET", "/analytics/detailed")
    assert s["total_violations"] == 1800 and len(d["hourly"]) == 24 and len(d["heatmap"]) == 7
    peak = d["peak_window"]
    assert 17 <= peak["start_hour"] <= 19  # synthetic evening rush hour survives the whole-day time shift
    assert client.json("GET", "/analytics/insights")["source"] != "llm"


def test_video_processing_is_disabled_with_a_clear_message(client):
    with pytest.raises(ApiError, match="lite demo"):
        client.json("POST", "/process-video")


@pytest.mark.parametrize("screen", SCREENS)
def test_every_screen_renders_in_demo_mode(demo_env, screen):
    at = AppTest.from_file("frontend/app.py", default_timeout=120)
    at.session_state["nav"] = screen
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("Lite demo" in i.value for i in at.info)
