"""Tests for the dashboard-support endpoints in serving/api.py."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from src.llm_client import LLMUnavailableError


@pytest.fixture
def api(tmp_path, monkeypatch):
    import serving.api as api_module
    import src.alert_system as alert_module
    from src.alert_system import AlertSystem

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    monkeypatch.setattr(alert_module, "EVIDENCE_FRAMES_DIR", evidence)
    monkeypatch.setattr(api_module, "EVIDENCE_FRAMES_DIR", evidence)
    monkeypatch.setattr(api_module, "API_KEY", "")
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(api_module, "_UPLOAD_DIR", uploads)  # never write test uploads into the project folder
    system = AlertSystem(db_path=tmp_path / "v.db")
    system.register_camera("CAM-1", "Test Junction", 60)
    api_module._alert_system = system
    api_module._SESSIONS.clear()
    api_module._LIVE.clear()
    api_module._INSIGHT_CACHE.clear()
    return SimpleNamespace(client=TestClient(api_module.app), mod=api_module, system=system, evidence=evidence)


@pytest.fixture
def seeded(api, make_record):
    ids = [
        api.system.log_violation(make_record(speed=80, plate="MH12AB1234", vehicle_type="Truck")),
        api.system.log_violation(make_record(speed=100, plate="KA01MF5678", vehicle_type="Car")),
        api.system.log_violation(make_record(speed=70, plate="MH14XY0001", vehicle_type="Bus")),
    ]
    return api, ids


def test_violations_filters_and_total_for_pagination(seeded):
    api, _ = seeded
    r = api.client.get("/violations", params={"state_code": "MH", "limit": 1}).json()
    assert r["count"] == 1 and r["total"] == 2
    assert api.client.get("/violations", params={"vehicle_type": "Car"}).json()["total"] == 1
    assert api.client.get("/violations", params={"severity": "SEVERE"}).json()["total"] == 1


def test_violations_rejects_bad_filter_values(api):
    assert api.client.get("/violations", params={"severity": "bogus"}).status_code == 422
    assert api.client.get("/violations", params={"date_from": "20-10-2026"}).status_code == 422
    assert api.client.get("/violations", params={"state_code": "MAH"}).status_code == 422


def test_status_update_and_validation(seeded):
    api, ids = seeded
    assert api.client.patch(f"/violation/{ids[0]}/status", json={"status": "RESOLVED"}).json()["status"] == "RESOLVED"
    assert api.client.get(f"/violation/{ids[0]}").json()["status"] == "RESOLVED"
    assert api.client.patch(f"/violation/{ids[0]}/status", json={"status": "DELETED"}).status_code == 422
    assert api.client.patch("/violation/nope/status", json={"status": "RESOLVED"}).status_code == 404


def test_edit_notice_persists_manual_text(seeded):
    api, ids = seeded
    r = api.client.put(f"/violation/{ids[0]}/notice", json={"notice_text": "  Corrected notice  "})
    assert r.json()["notice_text"] == "Corrected notice"
    row = api.client.get(f"/violation/{ids[0]}").json()
    assert row["notice_text"] == "Corrected notice" and row["notice_generated"] == 1
    assert api.client.put(f"/violation/{ids[0]}/notice", json={"notice_text": ""}).status_code == 422


def test_send_to_rto_requires_notice_then_records_mock_dispatch(seeded):
    api, ids = seeded
    assert api.client.post(f"/violation/{ids[0]}/send-to-rto").status_code == 409
    api.system.update_notice(ids[0], "notice")
    r = api.client.post(f"/violation/{ids[0]}/send-to-rto").json()
    assert r["mode"] == "mock" and "nothing was delivered" in r["message"]
    assert len(api.client.get(f"/violation/{ids[0]}/dispatches").json()["dispatches"]) == 1
    assert api.client.post("/violation/nope/send-to-rto").status_code == 404


def test_evidence_frame_and_plate_served_and_missing_is_404(seeded):
    api, ids = seeded
    assert api.client.get(f"/violation/{ids[0]}/evidence").status_code == 404
    api.system.save_evidence_frame(np.zeros((20, 20, 3), dtype=np.uint8), ids[0])
    api.system.save_plate_crop(np.zeros((8, 24, 3), dtype=np.uint8), ids[0])
    with api.system._connect() as conn:
        conn.execute("UPDATE violations SET evidence_frame_path=? WHERE id=?", (str(api.evidence / f"{ids[0]}.jpg"), ids[0]))
    assert api.client.get(f"/violation/{ids[0]}/evidence").headers["content-type"] == "image/jpeg"
    assert api.client.get(f"/violation/{ids[0]}/evidence", params={"kind": "plate"}).status_code == 200
    assert api.client.get(f"/violation/{ids[0]}/evidence", params={"kind": "../x"}).status_code == 422


def test_evidence_path_traversal_in_db_cannot_escape_directory(seeded):
    api, ids = seeded
    with api.system._connect() as conn:
        conn.execute("UPDATE violations SET evidence_frame_path='../../../../etc/passwd' WHERE id=?", (ids[0],))
    assert api.client.get(f"/violation/{ids[0]}/evidence").status_code == 404


def test_speed_trace_endpoint(seeded):
    api, ids = seeded
    assert api.client.get(f"/violation/{ids[0]}/speed-trace").status_code == 404
    api.system.save_speed_trace(ids[0], [55, 62.5], fps=25)
    assert api.client.get(f"/violation/{ids[0]}/speed-trace").json()["speeds_kmph"] == [55.0, 62.5]


def test_notice_pdf_download(seeded):
    api, ids = seeded
    r = api.client.get(f"/violation/{ids[0]}/notice.pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert "attachment" in r.headers["content-disposition"]
    assert api.client.get("/violation/nope/notice.pdf").status_code == 404


def test_analytics_detailed_and_rule_based_insights(seeded):
    api, _ = seeded
    d = api.client.get("/analytics/detailed").json()
    assert sum(d["hourly"]) == 3 and len(d["heatmap"]) == 7
    ins = api.client.get("/analytics/insights", params={"use_llm": False}).json()
    assert ins["source"] == "rules" and "Not enough" in ins["text"]  # only 3 violations (< 5)


def test_export_csv_and_xlsx_respect_filters(seeded):
    api, _ = seeded
    csv_resp = api.client.get("/export/violations", params={"state_code": "MH"})
    assert csv_resp.headers["content-type"].startswith("text/csv")
    assert len(csv_resp.content.decode("utf-8-sig").strip().splitlines()) == 3  # header + 2 MH rows
    x = api.client.get("/export/violations", params={"format": "xlsx", "vehicle_type": "Car"})
    assert load_workbook(io.BytesIO(x.content))["Violations"].max_row == 2
    assert api.client.get("/export/violations", params={"format": "exe"}).status_code == 422


def test_summary_report_pdf(seeded):
    api, _ = seeded
    r = api.client.get("/reports/summary.pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")


def test_cameras_and_integrations_are_honest_about_mocks(seeded):
    api, _ = seeded
    assert api.client.get("/cameras").json()["cameras"][0]["violation_count"] == 3
    modes = {i["name"]: i["mode"] for i in api.client.get("/integrations").json()["integrations"]}
    assert modes["Vahan database"] == "mock" and modes["RTO portal"] == "mock"


def test_live_frame_and_stats_after_publish(api):
    assert api.client.get("/live-frame/s1").status_code == 404
    frame = np.full((100, 200, 3), 128, dtype=np.uint8)
    tracks = [SimpleNamespace(class_name="Car"), SimpleNamespace(class_name="Car"), SimpleNamespace(class_name="Truck")]
    result = SimpleNamespace(frame_index=2, vehicle_count=3, active_tracks=tracks, annotated_frame=frame)
    for _ in range(2):  # every 2nd frame is encoded
        api.mod._publish_frame("s1", result)
    r = api.client.get("/live-frame/s1")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    stats = api.client.get("/live-stats", params={"session_id": "s1"}).json()
    assert stats["vehicles_in_frame"] == 3 and stats["counts_by_class"] == {"Car": 2, "Truck": 1}
    assert api.client.get("/live-stats").json()["vehicles_in_frame"] is None


def test_sessions_listing_filters_active(api):
    api.mod._set_session("a", status="PROCESSING", camera_id="C1", kind="video")
    api.mod._set_session("b", status="COMPLETE", camera_id="C1", kind="video", summary={"x": 1})
    assert len(api.client.get("/sessions").json()["sessions"]) == 2
    active = api.client.get("/sessions", params={"active_only": True}).json()["sessions"]
    assert [s["session_id"] for s in active] == ["a"]


def test_api_key_protects_mutations_but_not_reads(seeded, monkeypatch):
    api, ids = seeded
    monkeypatch.setattr(api.mod, "API_KEY", "s3cret")
    assert api.client.get("/violations").status_code == 200
    assert api.client.patch(f"/violation/{ids[0]}/status", json={"status": "RESOLVED"}).status_code == 401
    bad = api.client.patch(f"/violation/{ids[0]}/status", json={"status": "RESOLVED"}, headers={"X-API-Key": "nope"})
    assert bad.status_code == 401
    ok = api.client.patch(f"/violation/{ids[0]}/status", json={"status": "RESOLVED"}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


def test_upload_larger_than_limit_is_rejected_and_not_kept(api, monkeypatch):
    monkeypatch.setattr(api.mod, "MAX_UPLOAD_SIZE_MB", 0)
    uploads_before = set(api.mod._UPLOAD_DIR.glob("*"))
    pts = [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 100}, {"x": 0, "y": 100}]
    r = api.client.post(
        "/process-video", files={"file": ("c.mp4", b"x" * 2048, "video/mp4")},
        data={"camera_id": "C", "calibration_points": json.dumps(pts)},
    )
    assert r.status_code == 400 and "too large" in r.json()["detail"]
    assert set(api.mod._UPLOAD_DIR.glob("*")) == uploads_before


def test_generate_report_degrades_to_template_when_llm_is_down(seeded, monkeypatch):
    api, ids = seeded
    import src.report_generator as rg

    def _down(self, *a, **k):
        raise LLMUnavailableError("all models failed")

    monkeypatch.setattr(rg.ReportGenerator, "generate", _down)
    r = api.client.post("/generate-report", json={"violation_id": ids[0], "use_llm": True})
    assert r.status_code == 200 and r.json()["model_used"] == "fallback_template"
    assert api.client.get(f"/violation/{ids[0]}").json()["notice_generated"] == 1


# ---------------------------------------------------------------------------
# Calibration depth, progress, state breakdown, HUD
# ---------------------------------------------------------------------------
_PTS = [{"x": 100, "y": 100}, {"x": 300, "y": 100}, {"x": 300, "y": 400}, {"x": 100, "y": 400}]


def test_lane_depth_is_passed_into_calibration_for_uploads(api, monkeypatch):
    captured = {}
    monkeypatch.setattr(api.mod, "_run_video_processing", lambda sid, path, calib, cam, plate, stride=1: captured.update(calib=calib, stride=stride))
    r = api.client.post(
        "/process-video", files={"file": ("c.mp4", b"x" * 64, "video/mp4")},
        data={"camera_id": "C", "lane_width_m": 3.5, "lane_depth_m": 20, "calibration_points": json.dumps(_PTS)},
    )
    assert r.status_code == 200
    assert captured["calib"].lane_depth_m == 20 and captured["calib"].lane_width_m == 3.5


def test_missing_lane_depth_falls_back_to_four_times_width(api):
    calib = api.mod._build_calibration([api.mod.CalibrationPoint(**p) for p in _PTS], "C", 3.5, 60)
    assert calib.lane_depth_m == 14.0


def test_stream_request_accepts_and_validates_lane_depth(api, monkeypatch):
    captured = {}
    monkeypatch.setattr(api.mod, "_run_stream_processing", lambda sid, src, calib, cam, plate, mf: captured.update(calib=calib))
    body = {"camera_id": "C", "stream_source": "0", "calibration_points": _PTS, "lane_depth_m": 25}
    assert api.client.post("/process-stream", json=body).status_code == 200
    assert captured["calib"].lane_depth_m == 25
    assert api.client.post("/process-stream", json={**body, "lane_depth_m": -5}).status_code == 422


def test_process_status_reports_progress(api):
    api.mod._set_session("p1", status="PROCESSING", detail="x")
    with api.mod._LIVE_LOCK:
        api.mod._LIVE["p1"] = {"frames": 25, "total_frames": 100}
    assert api.client.get("/process-status/p1").json()["progress"] == 0.25
    api.mod._set_session("p1", status="COMPLETE")
    assert api.client.get("/process-status/p1").json()["progress"] == 1.0
    api.mod._set_session("p2", status="QUEUED")
    assert api.client.get("/process-status/p2").json()["progress"] is None


def test_state_breakdown_groups_by_plate_prefix(seeded):
    api, _ = seeded
    assert api.client.get("/analytics/detailed").json()["state_breakdown"] == {"MH": 2, "KA": 1}


def test_unreadable_plates_are_not_counted_in_state_breakdown(api, make_record):
    api.system.log_violation(make_record(plate=None))
    assert api.client.get("/analytics/detailed").json()["state_breakdown"] == {}


# ---------------------------------------------------------------------------
# Annotated video playback, live MJPEG stream, per-video vehicles, shared models
# ---------------------------------------------------------------------------
@pytest.fixture
def videos(api, tmp_path, monkeypatch):
    d = tmp_path / "annotated"
    d.mkdir()
    monkeypatch.setattr(api.mod, "ANNOTATED_VIDEO_DIR", d)
    return d


def test_videos_list_and_download_only_finished_files(api, videos):
    (videos / "abc123.mp4").write_bytes(b"x" * 2048)
    (videos / "def456.partial.mp4").write_bytes(b"y" * 10)  # still being written: must not be offered
    api.mod._set_session("abc123", camera_id="CAM-1", status="COMPLETE")
    listed = api.client.get("/videos").json()["videos"]
    assert [v["session_id"] for v in listed] == ["abc123"] and listed[0]["camera_id"] == "CAM-1"
    r = api.client.get("/videos/abc123")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and len(r.content) == 2048
    assert api.client.get("/videos/def456").status_code == 404


def test_video_endpoint_rejects_path_traversal_ids(api, videos):
    assert api.client.get("/videos/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)
    assert api.client.get("/videos/bad.id").status_code == 400


def test_recorder_publishes_frames_and_produces_a_playable_video(api, videos):
    pytest.importorskip("imageio_ffmpeg")
    import cv2

    tracks = [SimpleNamespace(class_name="Car")]
    rec = api.mod._Recorder("sess1", lambda: 10.0)
    for i in range(12):
        frame = np.full((90, 160, 3), i * 10, dtype=np.uint8)
        rec(SimpleNamespace(frame_index=i, vehicle_count=1, active_tracks=tracks, annotated_frame=frame, vehicles=[]))
    assert rec.finish() is True and rec.playable is True
    path = videos / "sess1.mp4"
    assert path.is_file() and not (videos / "sess1.partial.mp4").exists()
    cap = cv2.VideoCapture(str(path))
    assert sum(1 for _ in iter(lambda: cap.read()[0], False)) == 12
    assert api.client.get("/live-frame/sess1").status_code == 200  # the live preview was fed by the same frames


def test_recorder_without_frames_leaves_no_file(api, videos):
    assert api.mod._Recorder("empty", lambda: 10.0).finish() is False
    assert list(videos.iterdir()) == []


def test_old_videos_are_pruned(api, videos, monkeypatch):
    import os
    import time

    monkeypatch.setattr(api.mod, "MAX_STORED_VIDEOS", 2)
    now = time.time()
    for i in range(4):
        p = videos / f"v{i}.mp4"
        p.write_bytes(b"x")
        os.utime(p, (now + i, now + i))
    api.mod._prune_videos()
    assert sorted(p.name for p in videos.iterdir()) == ["v2.mp4", "v3.mp4"]


def test_live_stream_sends_multipart_jpeg_frames_then_ends(api):
    frame = np.full((60, 80, 3), 90, dtype=np.uint8)
    result = SimpleNamespace(frame_index=0, vehicle_count=0, active_tracks=[], annotated_frame=frame, vehicles=[])
    api.mod._set_session("live1", status="PROCESSING")
    api.mod._publish_frame("live1", result)
    api.mod._set_session("live1", status="COMPLETE")  # lets the stream finish after the frame
    r = api.client.get("/live-stream/live1")
    assert r.status_code == 200 and r.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"--frame" in r.content and b"\xff\xd8" in r.content
    assert api.client.get("/live-stream/unknown").status_code == 404


def test_live_stats_lists_visible_vehicles_with_speed_and_plate(api):
    v = SimpleNamespace(track_id=3, vehicle_type="Truck", speed_kmph=71.234, plate="MH12AB1234", severity="VIOLATION", stable=True)
    result = SimpleNamespace(frame_index=0, vehicle_count=1, active_tracks=[SimpleNamespace(class_name="Truck")],
                             annotated_frame=np.zeros((20, 20, 3), dtype=np.uint8), vehicles=[v])
    api.mod._publish_frame("s9", result)
    got = api.client.get("/live-stats", params={"session_id": "s9"}).json()["vehicles"]
    assert got == [{"track_id": 3, "vehicle_type": "Truck", "speed_kmph": 71.2, "plate": "MH12AB1234", "severity": "VIOLATION", "stable": True}]


def test_session_vehicles_endpoint_orders_fastest_first(api):
    api.system.save_vehicle_session(1, "sx", "2026-10-19T10:00:00+00:00", "2026-10-19T10:00:05+00:00", 40, 55, None, "Car", 0)
    api.system.save_vehicle_session(2, "sx", "2026-10-19T10:00:00+00:00", "2026-10-19T10:00:05+00:00", 60, 88, "MH12AB1234", "Truck", 1)
    api.system.save_vehicle_session(3, "other", "2026-10-19T10:00:00+00:00", "2026-10-19T10:00:05+00:00", 10, 12, None, "Bus", 0)
    rows = api.client.get("/sessions/sx/vehicles").json()["vehicles"]
    assert [r["track_id"] for r in rows] == [2, 1] and rows[0]["plate_number"] == "MH12AB1234"


def test_models_are_loaded_once_and_shared(api, monkeypatch):
    built = []
    monkeypatch.setattr(api.mod, "VehicleDetector", lambda: built.append("det") or object())
    monkeypatch.setattr(api.mod, "PlateRecognizer", lambda: built.append("plate") or object())
    api.mod._MODELS.clear()
    d1, p1 = api.mod._get_shared_models(True)
    d2, p2 = api.mod._get_shared_models(True)
    assert d1 is d2 and p1 is p2 and built == ["det", "plate"]
    assert api.mod._get_shared_models(False)[1] is None
    api.mod._MODELS.clear()


def test_frame_stride_is_forwarded_and_validated(api, monkeypatch):
    captured = {}
    monkeypatch.setattr(api.mod, "_run_video_processing", lambda sid, path, calib, cam, plate, stride=1: captured.update(stride=stride))
    data = {"camera_id": "C", "lane_depth_m": 14, "calibration_points": json.dumps(_PTS), "frame_stride": 3}
    assert api.client.post("/process-video", files={"file": ("c.mp4", b"x" * 64, "video/mp4")}, data=data).status_code == 200
    assert captured["stride"] == 3
    assert api.client.post("/process-video", files={"file": ("c.mp4", b"x" * 64, "video/mp4")}, data={**data, "frame_stride": 99}).status_code == 422


def test_session_summary_dict_carries_the_plate_statistics(api):
    from services.detection_service import SessionSummary

    d = api.mod._summary_dict(SessionSummary("s", 10, 5, 2, 50.0, 80.0, None, plates_read=1, plate_attempts=4,
                                             plate_boxes_found=2, vehicles_too_small_for_plate=3))
    assert (d["plates_read"], d["plate_attempts"], d["plate_boxes_found"], d["vehicles_too_small_for_plate"]) == (1, 4, 2, 3)


def test_integrations_show_the_plate_settings_in_force(api):
    items = {i["name"]: i for i in api.client.get("/integrations").json()["integrations"]}
    assert "region IN" in items["Plate reading (ANPR)"]["detail"] and "min confidence 0.3" in items["Plate reading (ANPR)"]["detail"]


@pytest.fixture
def plate_dir(api, tmp_path, monkeypatch):
    d = tmp_path / "plate_debug"
    monkeypatch.setattr(api.mod, "PLATE_DEBUG_DIR", d)
    (d / "sess1").mkdir(parents=True)
    for name in ("read-track117-BG65USJ-c62.jpg", "read-track9-CE61LYL-c41.jpg", "track5_frame80_unclear-format-X-c10.jpg"):
        (d / "sess1" / name).write_bytes(b"\xff\xd8jpg")
    return d


def test_plate_reads_list_only_real_reads_sorted_by_track(api, plate_dir):
    reads = api.client.get("/sessions/sess1/plate-reads").json()["reads"]
    assert reads == [
        {"track_id": 9, "plate": "CE61LYL", "confidence": 0.41, "file": "read-track9-CE61LYL-c41.jpg"},
        {"track_id": 117, "plate": "BG65USJ", "confidence": 0.62, "file": "read-track117-BG65USJ-c62.jpg"},
    ]
    assert api.client.get("/sessions/none/plate-reads").json() == {"reads": []}


def test_plate_read_image_is_served_and_names_are_validated(api, plate_dir):
    r = api.client.get("/sessions/sess1/plate-reads/read-track117-BG65USJ-c62.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert api.client.get("/sessions/sess1/plate-reads/track5_frame80_unclear-format-X-c10.jpg").status_code == 400  # debug files are not served
    assert api.client.get("/sessions/sess1/plate-reads/..%2F..%2Fsecret.jpg").status_code in (400, 404)
    assert api.client.get("/sessions/sess1/plate-reads/read-track1-ZZ99ZZZ-c50.jpg").status_code == 404
