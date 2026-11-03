"""Dashboard tests: pure helpers, the API client, and every Streamlit screen run against the real FastAPI app."""

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from frontend.api_client import ApiClient, ApiError, normalize_base_url
from frontend.helpers import (
    default_points, draw_calibration, excess_percent, first_frame, fmt_local, format_plate,
    is_convex_quad, plate_state, whatsapp_link,
)


# ------------------------------- helpers -----------------------------------
def test_format_plate_variants():
    assert format_plate("MH12AB1234") == "MH 12 AB 1234"
    assert format_plate("DL8CAZ4521") == "DL 8C AZ 4521"
    assert format_plate(None) == "UNREADABLE" and format_plate("odd") == "odd"


def test_plate_state_and_excess_percent():
    assert plate_state("KA01MF5678") == "Karnataka" and plate_state("ZZ01AB1234") == "Unknown" and plate_state(None) == "Unknown"
    assert excess_percent(87, 60) == 45.0 and excess_percent(None, 60) is None


def test_fmt_local_converts_utc_to_ist():
    assert fmt_local("2026-10-19T20:00:00+00:00") == "20 Oct 2026 01:30:00"
    assert fmt_local(None) == "-" and fmt_local("garbage") == "garbage"


def test_default_points_form_valid_quad_and_bad_order_is_rejected():
    assert is_convex_quad(default_points(1280, 720))
    assert not is_convex_quad([(0, 0), (10, 10), (10, 0), (0, 10)])  # bow-tie
    assert not is_convex_quad([(0, 0), (1, 1), (2, 2)])


def test_draw_calibration_does_not_mutate_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_calibration(frame, default_points(100, 100))
    assert out.any() and not frame.any()


def test_first_frame_reads_video_and_rejects_garbage(tmp_path):
    import cv2

    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(3):
        w.write(np.full((48, 64, 3), 90, dtype=np.uint8))
    w.release()
    assert first_frame(p.read_bytes()).shape == (48, 64, 3)
    assert first_frame(b"not a video") is None


def test_whatsapp_link_is_url_encoded_and_truncated():
    link = whatsapp_link({"id": "abcdef123", "plate_number": "MH12AB1234", "vehicle_type": "Car", "detected_speed": 80,
                          "speed_limit": 60, "severity": "VIOLATION", "timestamp": "2026-10-19T10:00:00+00:00"}, "x" * 5000)
    assert link.startswith("https://wa.me/?text=") and "MH%2012%20AB%201234" in link and len(link) < 2600


# ------------------------------- client ------------------------------------
def test_normalize_base_url():
    assert normalize_base_url("api.onrender.com:10000") == "http://api.onrender.com:10000"
    assert normalize_base_url("https://x.io/") == "https://x.io" and normalize_base_url("") == "http://localhost:8000"


def test_client_wraps_network_and_http_errors(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "request", boom)
    with pytest.raises(ApiError, match="Cannot reach"):
        ApiClient("http://x").json("GET", "/health")

    class R:
        status_code, text = 404, "nope"
        def json(self): return {"detail": "Violation not found"}

    monkeypatch.setattr(requests, "request", lambda *a, **k: R())
    with pytest.raises(ApiError, match="Violation not found"):
        ApiClient("http://x").json("GET", "/violation/1")


def test_client_sends_api_key_and_drops_empty_params(monkeypatch):
    seen = {}

    class R:
        status_code = 200
        def json(self): return {}

    def fake(method, url, headers, timeout, params=None, **kw):
        seen.update(headers=headers, params=params)
        return R()

    monkeypatch.setattr(requests, "request", fake)
    ApiClient("http://x", api_key="k").json("GET", "/v", {"a": 1, "b": None, "c": "", "d": []})
    assert seen["headers"]["X-API-Key"] == "k" and seen["params"] == {"a": 1}


# ------------------------------- screens -----------------------------------
@pytest.fixture
def live_app(tmp_path, monkeypatch, make_record):
    """The real FastAPI app on a temp DB, with ``requests`` routed into it, plus seeded violations."""
    import serving.api as api_module
    import src.alert_system as alert_module
    from src.alert_system import AlertSystem

    ev = tmp_path / "ev"
    ev.mkdir()
    monkeypatch.setattr(alert_module, "EVIDENCE_FRAMES_DIR", ev)
    monkeypatch.setattr(api_module, "EVIDENCE_FRAMES_DIR", ev)
    monkeypatch.setattr(api_module, "API_KEY", "")
    system = AlertSystem(db_path=tmp_path / "d.db")
    system.register_camera("CAM-1", "Test Junction", 60)
    for i in range(6):
        vid = system.log_violation(make_record(speed=75 + i * 7, plate=f"MH12AB{1000 + i}", vehicle_type=["Car", "Truck"][i % 2]),
                                   plate_confidence=0.82, evidence_frame=np.zeros((40, 60, 3), dtype=np.uint8))
    system.save_speed_trace(vid, [40, 55, 70, 90], fps=25)
    system.update_notice(vid, "TRAFFIC VIOLATION NOTICE\n\nSection 183 applies.")
    api_module._alert_system = system
    api_module._SESSIONS.clear(); api_module._LIVE.clear(); api_module._INSIGHT_CACHE.clear()
    tc = TestClient(api_module.app)

    def routed(method, url, headers=None, timeout=None, params=None, **kw):
        path = "/" + url.split("//", 1)[1].split("/", 1)[1]  # strip scheme + host
        return tc.request(method, path, params=params, **{k: v for k, v in kw.items() if k in ("json", "data", "files")})

    monkeypatch.setattr(requests, "request", routed)
    return system, vid


@pytest.mark.parametrize("screen", [
    "1. Live monitoring", "2. Upload and analysis", "3. Violations log", "4. Violation detail", "5. Analytics", "6. Reports and export",
])
def test_every_screen_renders_without_exception(live_app, screen):
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = screen
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_log_screen_lists_seeded_violations(live_app):
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "3. Violations log"
    at.run()
    assert any("6 violations" in n.label for n in at.number_input)
    assert len(at.dataframe[0].value) == 6


def test_detail_screen_shows_selected_violation_and_notice(live_app):
    system, vid = live_app
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "4. Violation detail"
    at.session_state["selected_violation_id"] = vid
    at.run()
    assert not at.exception
    assert any("Section 183" in t.value for t in at.text_area)
    assert any("Maharashtra" in m.value for m in at.markdown)


def test_analytics_screen_has_charts_and_rule_based_insight(live_app):
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "5. Analytics"
    at.run()
    assert not at.exception
    assert any("Generated directly from the statistics" in c.value for c in at.caption)


def test_offline_api_shows_clear_error_not_traceback(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "request", boom)
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)


# ---------------------------------------------------------------------------
# Click-to-place calibration corners
# ---------------------------------------------------------------------------
def test_scale_click_maps_shrunk_display_back_to_frame_pixels_and_clamps():
    from frontend.helpers import scale_click

    assert scale_click({"x": 450, "y": 100, "width": 900, "height": 450}, (1800, 900)) == (900, 200)
    assert scale_click({"x": 5000, "y": -20, "width": 900, "height": 450}, (900, 450)) == (899, 0)


def test_draw_calibration_scales_markers_and_marks_active_corner():
    from frontend.helpers import draw_calibration

    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pts = [(500, 400), (900, 400), (1000, 800), (400, 800)]
    plain, active = draw_calibration(big, pts), draw_calibration(big, pts, active=2)
    assert (active != plain).any()  # the amber ring only appears when a corner is active


def _wizard_at_step_two(monkeypatch, clicks):
    """Run the wizard on step 2 with a fake click component that returns the scripted clicks."""
    import streamlit_image_coordinates as sic

    queue = list(clicks)
    monkeypatch.setattr(sic, "streamlit_image_coordinates", lambda *a, **k: queue[0] if queue else None)
    frame = np.zeros((450, 900, 3), dtype=np.uint8)
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "2. Upload and analysis"
    pts = default_points(900, 450)
    at.session_state["wiz"] = {"step": 1, "name": "v.mp4", "data": b"", "size": (900, 450), "frame": frame, "points": pts,
                               "active": 0, "last_click": None}
    for i, (x, y) in enumerate(pts):
        at.session_state[f"px{i}"], at.session_state[f"py{i}"] = x, y
    return at


def test_clicking_the_image_moves_the_active_corner_and_advances(monkeypatch, live_app):
    at = _wizard_at_step_two(monkeypatch, [{"x": 300, "y": 120, "width": 900, "height": 450, "unix_time": 1}])
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert (at.session_state["px0"], at.session_state["py0"]) == (300, 120)  # corner 1 followed the click
    assert at.session_state["wiz"]["active"] == 1  # next click will place corner 2
    # re-running with the same (stale) click value must not move another corner
    at.run()
    assert (at.session_state["px1"], at.session_state["py1"]) == default_points(900, 450)[1]
    assert at.session_state["wiz"]["active"] == 1


def test_four_clicks_place_all_corners_in_order(monkeypatch, live_app):
    clicks = [{"x": 250, "y": 200, "width": 900, "height": 450, "unix_time": 1}]
    at = _wizard_at_step_two(monkeypatch, clicks)
    at.run()
    import streamlit_image_coordinates as sic

    targets = [(650, 200), (850, 420), (80, 420)]
    for n, (x, y) in enumerate(targets, start=2):
        sic.streamlit_image_coordinates = lambda *a, _c={"x": x, "y": y, "width": 900, "height": 450, "unix_time": n}, **k: _c
        at.run()
    got = [(at.session_state[f"px{i}"], at.session_state[f"py{i}"]) for i in range(4)]
    assert got == [(250, 200), (650, 200), (850, 420), (80, 420)]
    assert at.session_state["wiz"]["active"] == 0  # wrapped around, ready to adjust corner 1 again


# ---------------------------------------------------------------------------
# Live stream, processed-video playback, vehicles tables, processing mode
# ---------------------------------------------------------------------------
def _seed_session(api_module, tmp_path, monkeypatch, sid, status, video=False, vehicles=True):
    import types

    videos = tmp_path / "annotated"
    videos.mkdir(exist_ok=True)
    monkeypatch.setattr(api_module, "ANNOTATED_VIDEO_DIR", videos)
    api_module._set_session(sid, status=status, camera_id="CAM-UP", kind="video", video_ready=video, video_playable=True,
                            summary={"total_vehicles": 4, "total_violations": 1, "average_speed_kmph": 42.5, "max_speed_kmph": 77.0,
                                     "max_speed_plate": "MH12AB1234"} if status == "COMPLETE" else None)
    if video:
        (videos / f"{sid}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 64)
    if vehicles:
        from src.alert_system import AlertSystem  # noqa: F401  (system already attached to the module by the fixture)

        api_module._alert_system.save_vehicle_session(1, sid, "2026-10-19T10:00:00+00:00", "2026-10-19T10:00:05+00:00", 40.0, 77.0, "MH12AB1234", "Car", 1)
    live = types.SimpleNamespace(track_id=7, vehicle_type="Truck", speed_kmph=51.2, plate="KA01MF5678", severity="NORMAL", stable=True)
    api_module._publish_frame(sid, types.SimpleNamespace(frame_index=0, vehicle_count=1, active_tracks=[types.SimpleNamespace(class_name="Truck")],
                                                          annotated_frame=np.zeros((20, 20, 3), dtype=np.uint8), vehicles=[live]))


def _all_markdown(at):
    return " ".join(m.value for m in at.markdown)


def test_live_screen_streams_the_running_video_and_lists_vehicles_with_plates(live_app, tmp_path, monkeypatch):
    import serving.api as api_module

    _seed_session(api_module, tmp_path, monkeypatch, "run0001", "PROCESSING")
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "1. Live monitoring"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "/live-stream/run0001" in _all_markdown(at)        # smooth MJPEG video, not a 1-second slideshow
    frame_tables = [d.value for d in at.dataframe]
    assert any("KA 01 MF 5678" in df.to_string() and "Truck #7" in df.to_string() for df in frame_tables)  # type + id + plate + speed


def test_live_screen_plays_the_finished_annotated_video(live_app, tmp_path, monkeypatch):
    import serving.api as api_module

    _seed_session(api_module, tmp_path, monkeypatch, "done0002", "COMPLETE", video=True)
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "1. Live monitoring"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("video")) == 1                          # the whole processed video is playable after processing
    assert "/live-stream/" not in _all_markdown(at)
    assert any("MH 12 AB 1234" in d.value.to_string() for d in at.dataframe)  # per-vehicle summary of the whole video


def test_upload_results_show_video_summary_and_vehicle_table(live_app, tmp_path, monkeypatch):
    import serving.api as api_module

    _seed_session(api_module, tmp_path, monkeypatch, "done0003", "COMPLETE", video=True)
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "2. Upload and analysis"
    at.session_state["wiz"] = {"step": 3, "name": "cars.avi", "camera_id": "CAM-UP", "lane_width": 3.5, "lane_depth": 14.0,
                               "limit": 60, "anpr": True, "stride": 2, "points": [(1, 1)] * 4, "session_id": "done0003"}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("video")) == 1
    assert any("Processed video" in s.value for s in at.subheader)
    assert any("every 2 frame" in t.value.to_string() for t in at.table)
    assert any("MH 12 AB 1234" in d.value.to_string() for d in at.dataframe)


def test_upload_while_processing_shows_live_stream_and_progress(live_app, tmp_path, monkeypatch):
    import serving.api as api_module

    _seed_session(api_module, tmp_path, monkeypatch, "run0004", "PROCESSING", vehicles=False)
    with api_module._LIVE_LOCK:
        api_module._LIVE["run0004"].update(frames=25, total_frames=100)
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "2. Upload and analysis"
    at.session_state["wiz"] = {"step": 3, "name": "c.avi", "camera_id": "C", "lane_width": 3.5, "lane_depth": 14.0, "limit": 60,
                               "anpr": True, "points": [(1, 1)] * 4, "session_id": "run0004"}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "/live-stream/run0004" in _all_markdown(at)
    assert len(at.get("progress")) == 1


def test_processing_mode_selector_offers_frame_skipping(live_app):
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "2. Upload and analysis"
    at.session_state["wiz"] = {"step": 2, "name": "c.avi", "points": [(1, 1)] * 4, "size": (640, 480)}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    mode = next(sb for sb in at.selectbox if sb.label == "Processing mode")
    mode.select("Fast: every 3rd frame").run()
    assert at.session_state["wiz"]["stride"] == 3


def test_failed_session_shows_the_error_message(live_app, tmp_path, monkeypatch):
    import serving.api as api_module

    api_module._set_session("bad0005", status="FAILED", detail="No module named 'x'")
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "2. Upload and analysis"
    at.session_state["wiz"] = {"step": 3, "name": "c.avi", "camera_id": "C", "lane_width": 3.5, "lane_depth": 14.0, "limit": 60,
                               "anpr": True, "points": [(1, 1)] * 4, "session_id": "bad0005"}
    at.run()
    assert any("No module named 'x'" in e.value for e in at.error)


# ---------------------------------------------------------------------------
# Plate statistics on the results page + the plate_check script
# ---------------------------------------------------------------------------
def _results_page(live_app, sid, summary):
    import serving.api as api_module

    api_module._set_session(sid, status="COMPLETE", camera_id="C", kind="video", video_ready=False, summary=summary)
    at = AppTest.from_file("frontend/app.py", default_timeout=60)
    at.session_state["nav"] = "2. Upload and analysis"
    at.session_state["wiz"] = {"step": 3, "name": "c.avi", "camera_id": "C", "lane_width": 3.5, "lane_depth": 14.0, "limit": 60,
                               "anpr": True, "points": [(1, 1)] * 4, "session_id": sid}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


_BASE = {"total_vehicles": 55, "total_violations": 31, "average_speed_kmph": 66.4, "max_speed_kmph": 90.0, "max_speed_plate": None}


def test_results_explain_why_no_plates_were_read(live_app):
    at = _results_page(live_app, "np0001", {**_BASE, "plates_read": 0, "plate_attempts": 9, "plate_boxes_found": 0, "vehicles_too_small_for_plate": 40})
    assert any("0 read of 55 vehicles" in c.value and "40 vehicles too small" in c.value for c in at.caption)
    assert any("No number plate could be read" in w.value and "plate_check.py" in w.value for w in at.warning)


def test_results_show_plate_counts_without_a_warning_when_plates_were_read(live_app):
    at = _results_page(live_app, "np0002", {**_BASE, "plates_read": 12, "plate_attempts": 30, "plate_boxes_found": 20, "vehicles_too_small_for_plate": 10})
    assert any("12 read of 55 vehicles" in c.value for c in at.caption)
    assert not any("No number plate could be read" in w.value for w in at.warning)


class _FakeRecognizer:
    def __init__(self, crop, text, conf):
        self.crop, self.text, self.conf = crop, text, conf

    def detect_plate_region(self, image):
        return self.crop

    def read_plate_crop(self, crop):
        from src.plate_recognizer import PlateResult

        return PlateResult(self.text, self.conf, None, None, False, "RAW TEXT")


def test_plate_check_reports_found_unclear_and_missing_plates():
    from scripts.plate_check import check_image, verdict

    img = np.zeros((200, 300, 3), dtype=np.uint8)
    ok = check_image(_FakeRecognizer(np.zeros((30, 120, 3), dtype=np.uint8), "MH12AB1234", 0.9), img)
    assert ok["plate_box_found"] and "Plate read: MH12AB1234" in verdict(ok)
    small = check_image(_FakeRecognizer(np.zeros((10, 40, 3), dtype=np.uint8), "PLATE_UNCLEAR", 0.1), img)
    assert "only 40px wide" in verdict(small)
    none = check_image(_FakeRecognizer(None, "", 0), img)
    assert none == {"plate_box_found": False, "image_width_px": 300} and "No plate was found" in verdict(none)


def test_frontend_formats_uk_plates_too():
    assert format_plate("GX15OGJ") == "GX15 OGJ" and format_plate("MH12AB1234") == "MH 12 AB 1234"


def test_results_diagnose_plates_found_but_rejected_by_format(live_app):
    at = _results_page(live_app, "np0003", {**_BASE, "plates_read": 0, "plate_attempts": 1393, "plate_boxes_found": 586, "vehicles_too_small_for_plate": 30})
    warnings = " ".join(w.value for w in at.warning)
    assert "found plates 586 times" in warnings and "PLATE_REGION=UK" in warnings
    assert "too small or far away" not in warnings   # the right explanation, not the resolution one


def test_results_blame_the_format_check_and_name_the_active_region(live_app):
    s = {**_BASE, "plates_read": 0, "plate_attempts": 90, "plate_boxes_found": 60, "vehicles_too_small_for_plate": 3,
         "plates_rejected_format": 55, "plates_rejected_confidence": 5, "plate_region": "IN", "plate_min_confidence": 0.3}
    at = _results_page(live_app, "np0004", s)
    assert any("55 failed the format check" in c.value and "format IN, min confidence 0.3" in c.value for c in at.caption)
    w = " ".join(x.value for x in at.warning)
    assert "failed the **IN** plate format check" in w and "restart the API" in w and "file names show what the OCR read" in w


def test_results_blame_low_confidence_when_that_is_the_main_reason(live_app):
    s = {**_BASE, "plates_read": 0, "plate_attempts": 90, "plate_boxes_found": 60, "vehicles_too_small_for_plate": 3,
         "plates_rejected_format": 4, "plates_rejected_confidence": 56, "plate_region": "UK", "plate_min_confidence": 0.5}
    at = _results_page(live_app, "np0005", s)
    w = " ".join(x.value for x in at.warning)
    assert "confidence below 0.5" in w and "PLATE_CONFIDENCE_THRESHOLD" in w and "failed the **" not in w


def test_results_show_each_read_plate_next_to_its_crop(live_app, tmp_path, monkeypatch):
    import cv2
    import serving.api as api_module

    d = tmp_path / "plate_debug" / "np0009"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "read-track117-BG65USJ-c62.jpg"), np.full((20, 60, 3), 180, dtype=np.uint8))
    monkeypatch.setattr(api_module, "PLATE_DEBUG_DIR", tmp_path / "plate_debug")
    at = _results_page(live_app, "np0009", {**_BASE, "plates_read": 1, "plate_attempts": 9, "plate_boxes_found": 5, "vehicles_too_small_for_plate": 2})
    assert any("Check the 1 plate(s) that were read" in e.label for e in at.expander)
    images = at.get("imgs")
    assert len(images) == 1
    assert "Track #117 - BG65 USJ (confidence 0.62)" in images[0].proto.imgs[0].caption
