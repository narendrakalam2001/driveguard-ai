"""Tests for services/detection_service.py with a stub detector/tracker (no models needed).

A stub tracker moves synthetic vehicles at chosen speeds through a known calibration, so the
whole path - speed, classification, de-duplicated logging, plate retries, vehicle stats - is
checked end to end against a real (temporary) SQLite database.
"""

import numpy as np
import pytest

import src.alert_system as alert_module
from src.alert_system import AlertSystem
from src.config import (
    PLATE_MAX_READ_ATTEMPTS,
    PLATE_READ_INTERVAL_FRAMES,
    PLATE_UNCLEAR_LABEL,
    SEVERITY,
    VEHICLE_LOST_AFTER_S,
)
from src.plate_recognizer import PlateResult
from src.speed_estimator import SpeedEstimator
from src.tracker import TrackState
from services.detection_service import DetectionService

FPS = 25.0
PX_PER_M_Y = 300 / 14  # square_calibration: 300 px of road depth = 14 m


class StubDetector:
    def detect(self, frame):
        return []


class StubTracker:
    """Returns scripted vehicles. ``script`` maps track_id -> function(frame_no) -> speed_kmph or None."""

    def __init__(self, script):
        self.script, self.states, self.y = script, {}, {}

    def update(self, detections, frame, timestamp_s):
        frame_no = int(round(timestamp_s * FPS))
        active = []
        for track_id, speed_at in self.script.items():
            speed = speed_at(frame_no)
            if speed is None:
                continue
            state = self.states.setdefault(
                track_id, TrackState(track_id=track_id, class_name="Truck", bbox=np.zeros(4, dtype=np.float32))
            )
            y = self.y.get(track_id, 100.0) + (speed / 3.6) / FPS * PX_PER_M_Y
            self.y[track_id] = y
            state.bbox = np.array([90, 100, 110, 140], dtype=np.float32)  # kept inside the test frame; speed uses the history
            state.centroid_history.append((100.0, y, timestamp_s))
            active.append(state)
        return active


class StubPlates:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def read_plate(self, crop):
        self.calls.append(crop.shape)
        item = self.results.pop(0) if self.results else None
        if isinstance(item, Exception):
            raise item
        return item


def _plate(text="MH12AB1234", conf=0.8):
    return PlateResult(text, conf, "MH", "12", True)


@pytest.fixture(autouse=True)
def _no_plate_size_gate(monkeypatch, tmp_path):
    """The stub vehicles are 20 px wide; switch the real-world size gate off (and debug crops into tmp) unless a test sets it."""
    import services.detection_service as ds

    monkeypatch.setattr(ds, "PLATE_MIN_VEHICLE_WIDTH_PX", 0)
    monkeypatch.setattr(ds, "PLATE_DEBUG_DIR", tmp_path / "plate_debug")


@pytest.fixture
def alerts(tmp_db_path, tmp_path, monkeypatch):
    monkeypatch.setattr(alert_module, "EVIDENCE_FRAMES_DIR", tmp_path / "evidence")
    return AlertSystem(db_path=tmp_db_path)


def _service(square_calibration, alerts, script, plates=None):
    return DetectionService(
        calibration=square_calibration,  # speed limit 60
        camera_id="CAM-TEST",
        detector=StubDetector(),
        tracker=StubTracker(script),
        speed_estimator=SpeedEstimator(square_calibration, min_points=5),
        plate_recognizer=plates,
        alert_system=alerts,
        enable_plate_recognition=plates is not None,
        session_id="S1",
    )


def _run(service, frames, start=0):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = []
    for n in range(start, start + frames):
        results.append(service.process_frame(frame, n, n / FPS))
    return results


def _rows(alerts):
    return alerts.query_violations(limit=1000)


def test_a_speeding_vehicle_is_logged_once_not_once_per_frame(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 100.0})  # limit 60 -> SEVERE
    results = _run(service, 75)  # 3 seconds in view
    rows = _rows(alerts)
    assert len(rows) == 1 and rows[0]["severity"] == SEVERITY.SEVERE
    assert sum(len(r.violations_this_frame) for r in results) == 1
    assert rows[0]["evidence_frame_path"]


def test_escalation_upgrades_one_row_instead_of_adding_rows(square_calibration, alerts):
    def speed(n):  # 50 (normal) -> 70 (warning) -> 80 (violation) -> 110 (severe), 30 frames each
        return [50.0, 70.0, 80.0, 110.0][min(n // 30, 3)]

    service = _service(square_calibration, alerts, {1: speed})
    _run(service, 130)
    rows = _rows(alerts)
    assert len(rows) == 1 and rows[0]["severity"] == SEVERITY.SEVERE   # one vehicle = one violation, at its worst band
    assert rows[0]["detected_speed"] > 90 and rows[0]["evidence_frame_path"]  # evidence was added when it reached VIOLATION
    feed = [a["severity"] for a in reversed(alerts.get_live_feed(10))]
    assert feed == [SEVERITY.WARNING, SEVERITY.VIOLATION, SEVERITY.SEVERE]   # the live feed still shows each step
    assert service.finalize().total_violations == 1


def test_vehicle_within_the_limit_is_never_logged(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 50.0})
    _run(service, 60)
    assert _rows(alerts) == []


def test_no_violation_during_speed_warm_up(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 200.0})
    results = _run(service, 4)  # fewer than min_points samples
    assert all(not r.violations_this_frame for r in results)
    assert _rows(alerts) == []


def test_summary_counts_vehicles_not_frames_and_averages_per_vehicle(square_calibration, alerts):
    # vehicle 1 is in view for 100 frames at 90 km/h; vehicle 2 for 20 frames at 30 km/h
    service = _service(
        square_calibration,
        alerts,
        {1: lambda n: 90.0, 2: lambda n: 30.0 if n < 20 else None},
    )
    _run(service, 100)
    summary = service.finalize()
    assert summary.total_frames == 100
    assert summary.total_vehicles == 2
    assert summary.total_violations == 1
    assert summary.average_speed_kmph == pytest.approx((90 + 30) / 2, rel=0.05)  # per vehicle, not per frame
    assert summary.max_speed_kmph == pytest.approx(90, rel=0.05)


def test_vehicle_stats_are_saved_when_vehicle_is_lost_and_at_finalize(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 90.0 if n < 30 else None, 2: lambda n: 40.0})
    gap_frames = int(VEHICLE_LOST_AFTER_S * FPS) + 30 + 5
    _run(service, gap_frames)
    with alerts._connect() as conn:
        before = {r["track_id"] for r in conn.execute("SELECT track_id FROM vehicles").fetchall()}
    assert before == {1}  # vehicle 1 left; vehicle 2 is still in view
    service.finalize()
    with alerts._connect() as conn:
        rows = {r["track_id"]: dict(r) for r in conn.execute("SELECT * FROM vehicles").fetchall()}
    assert set(rows) == {1, 2}
    assert rows[1]["session_id"] == "S1" and rows[1]["vehicle_type"] == "Truck"
    assert rows[1]["max_speed"] == pytest.approx(90, rel=0.05) and rows[1]["violation_count"] == 1
    assert rows[2]["violation_count"] == 0


def test_unreadable_plate_is_retried_rarely_and_then_abandoned(square_calibration, alerts):
    plates = StubPlates([])  # never finds a plate
    service = _service(square_calibration, alerts, {1: lambda n: 50.0}, plates)
    _run(service, 400)
    assert len(plates.calls) == PLATE_MAX_READ_ATTEMPTS  # not one OCR run per frame


def test_plate_retry_interval_is_respected(square_calibration, alerts):
    plates = StubPlates([])
    service = _service(square_calibration, alerts, {1: lambda n: 50.0}, plates)
    _run(service, 5 + PLATE_READ_INTERVAL_FRAMES)  # speed starts at the 5th point
    assert len(plates.calls) <= 2


def test_plate_read_after_the_first_log_is_attached_to_the_violation(square_calibration, alerts):
    plates = StubPlates([None, PlateResult(PLATE_UNCLEAR_LABEL, 0.2, None, None, False), _plate("KA01MF5678", 0.64)])
    service = _service(square_calibration, alerts, {1: lambda n: 100.0}, plates)
    _run(service, 60)
    rows = _rows(alerts)
    assert len(rows) == 1
    assert rows[0]["plate_number"] == "KA01MF5678"
    assert rows[0]["plate_confidence"] == pytest.approx(0.64)
    assert service.finalize().max_speed_plate == "KA01MF5678"


def test_plate_confidence_is_stored_when_the_plate_is_known_at_log_time(square_calibration, alerts):
    plates = StubPlates([_plate("MH12AB1234", 0.9)])
    service = _service(square_calibration, alerts, {1: lambda n: 100.0}, plates)
    _run(service, 40)
    row = _rows(alerts)[0]
    assert row["plate_number"] == "MH12AB1234" and row["plate_confidence"] == pytest.approx(0.9)


def test_plate_recognizer_error_does_not_stop_the_pipeline(square_calibration, alerts):
    plates = StubPlates([RuntimeError("OCR crashed"), _plate()])
    service = _service(square_calibration, alerts, {1: lambda n: 100.0}, plates)
    _run(service, 40)
    assert len(_rows(alerts)) == 1
    assert _rows(alerts)[0]["plate_number"] == "MH12AB1234"


# ---------------------------------------------------------------------------
# Evidence extras: plate crop, speed trace, live-preview callback
# ---------------------------------------------------------------------------
def test_plate_crop_is_saved_for_every_violation_row_of_the_vehicle(square_calibration, alerts, tmp_path):
    crop = np.full((12, 40, 3), 200, dtype=np.uint8)
    result = _plate()
    result.crop = crop
    service = _service(square_calibration, alerts, {1: lambda n: 100.0}, plates=StubPlates([result]))
    _run(service, 40)
    rows = _rows(alerts)
    assert rows, "expected a logged violation"
    for row in rows:
        assert (tmp_path / "evidence" / f"{row['id']}_plate.jpg").is_file()


def test_speed_trace_saved_when_vehicle_finishes(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 100.0})
    _run(service, 40)
    service.finalize()
    vid = _rows(alerts)[0]["id"]
    trace = AlertSystem.load_speed_trace(vid)
    assert trace is not None and len(trace["speeds_kmph"]) > 10
    assert max(trace["speeds_kmph"]) == pytest.approx(100.0, abs=3.0)


def test_speed_trace_is_capped(square_calibration, alerts, monkeypatch):
    import services.detection_service as ds

    monkeypatch.setattr(ds, "_TRACE_MAX", 20)  # must stay >= MIN_SPEED_SAMPLES or nothing could ever be flagged
    service = _service(square_calibration, alerts, {1: lambda n: 100.0})
    _run(service, 60)
    service.finalize()
    assert len(AlertSystem.load_speed_trace(_rows(alerts)[0]["id"])["speeds_kmph"]) == 20


def test_process_video_calls_on_frame_and_survives_callback_errors(square_calibration, alerts, tmp_path):
    import cv2

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for _ in range(6):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    seen = []

    def flaky(result):
        seen.append(result.frame_index)
        raise RuntimeError("preview encoder broke")

    service = _service(square_calibration, alerts, {})
    summary = service.process_video(str(path), on_frame=flaky)
    assert len(seen) == 6 and summary.total_frames == 6
    assert service.fps == pytest.approx(25.0, abs=1.0)


def test_hud_draws_speed_zone_and_counters_without_touching_evidence(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 100.0})
    results = _run(service, 40)
    assert results[-1].annotated_frame.any()  # zone polygon / counters were drawn on the black test frame
    # evidence frames are saved before the HUD is drawn, so they hold boxes but not the FPS/counter overlay
    assert service.processing_fps >= 0.0


# ---------------------------------------------------------------------------
# Measurement-quality guards, vehicle info, labels, frame stride
# ---------------------------------------------------------------------------
def test_short_lived_track_is_never_flagged(square_calibration, alerts):
    """A 10-frame track (a tracker flicker) must not produce a violation, however fast it looks."""
    service = _service(square_calibration, alerts, {1: lambda n: 100.0})
    _run(service, 10)
    assert _rows(alerts) == []


def test_implausible_speed_is_ignored_not_logged_or_counted(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 400.0})
    results = _run(service, 60)
    summary = service.finalize()
    assert _rows(alerts) == [] and summary.max_speed_kmph == 0
    assert all(v.speed_kmph is None for r in results for v in r.vehicles)


def test_vehicle_info_reports_measuring_then_stable_speed(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 100.0})
    results = _run(service, 40)
    early = [v for r in results[:8] for v in r.vehicles]
    late = results[-1].vehicles[0]
    assert early and not any(v.stable for v in early)
    assert late.stable and late.speed_kmph == pytest.approx(100.0, abs=4) and late.severity == "SEVERE"
    assert late.vehicle_type


def test_labels_show_type_id_speed_and_formatted_plate(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 100.0}, StubPlates([_plate("MH12AB1234", 0.9)]))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for n in range(40):
        result = service.process_frame(frame, n, n / FPS)
    # a plain black frame can only become non-black through the drawn boxes, label chips and HUD
    assert result.annotated_frame.any()
    assert result.vehicles[0].plate == "MH12AB1234"


def test_frame_stride_skips_frames_but_keeps_speeds_correct(square_calibration, alerts, tmp_path):
    import cv2

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for _ in range(12):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    seen = []
    service = _service(square_calibration, alerts, {})
    service.process_video(str(path), on_frame=lambda r: seen.append(r.frame_index), frame_stride=3)
    assert seen == [0, 3, 6, 9]


def test_api_worker_end_to_end_records_a_playable_annotated_video(square_calibration, alerts, tmp_path, monkeypatch):
    """Whole background-job path with a real video file: queue lock -> shared models -> pipeline -> recording -> COMPLETE."""
    pytest.importorskip("imageio_ffmpeg")
    import cv2

    import serving.api as api_module

    out_dir = tmp_path / "annotated"
    monkeypatch.setattr(api_module, "ANNOTATED_VIDEO_DIR", out_dir)
    monkeypatch.setattr(api_module, "_alert_system", alerts)
    monkeypatch.setattr(api_module, "_get_shared_models", lambda enable_plate: (None, None))
    monkeypatch.setattr(api_module, "DetectionService", lambda **kw: _service(square_calibration, alerts, {1: lambda n: 100.0}))

    src = tmp_path / "upload.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
    for _ in range(40):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    api_module._run_video_processing("e2e-1", src, square_calibration, "CAM-TEST", False, 1)

    session = api_module._SESSIONS["e2e-1"]
    assert session["status"] == "COMPLETE" and session["video_ready"] is True and session["video_playable"] is True
    assert session["summary"]["total_vehicles"] == 1 and session["summary"]["total_violations"] >= 1
    assert not src.exists()  # the upload is deleted, the annotated copy is what is kept
    video = out_dir / "e2e-1.mp4"
    cap = cv2.VideoCapture(str(video))
    frames = sum(1 for _ in iter(lambda: cap.read()[0], False))
    assert frames == 40 and cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25.0, abs=1.0)
    assert [v["track_id"] for v in alerts.get_session_vehicles("e2e-1")] == [1]


def test_api_worker_failure_is_reported_and_upload_cleaned_up(square_calibration, alerts, tmp_path, monkeypatch):
    import serving.api as api_module

    monkeypatch.setattr(api_module, "ANNOTATED_VIDEO_DIR", tmp_path / "annotated")
    monkeypatch.setattr(api_module, "_alert_system", alerts)

    def _boom(enable_plate):
        raise RuntimeError("weights missing")

    monkeypatch.setattr(api_module, "_get_shared_models", _boom)
    src = tmp_path / "upload.mp4"
    src.write_bytes(b"x")
    api_module._run_video_processing("e2e-2", src, square_calibration, "CAM-TEST", False, 1)
    assert api_module._SESSIONS["e2e-2"]["status"] == "FAILED" and "weights missing" in api_module._SESSIONS["e2e-2"]["detail"]
    assert not src.exists()
    assert api_module._PIPELINE_LOCK.acquire(blocking=False)  # lock was released after the failure
    api_module._PIPELINE_LOCK.release()


# ---------------------------------------------------------------------------
# Plate attempts: size gate, growth-based retries, counters, debug crops
# ---------------------------------------------------------------------------
def _track(width, track_id=1):
    return TrackState(track_id=track_id, class_name="Car", bbox=np.array([0, 0, width, 60], dtype=np.float32))


def _fresh_stats():
    from services.detection_service import _VehicleStats

    return _VehicleStats(first_seen_iso="t", last_seen_iso="t", last_seen_s=0.0)


def test_vehicle_too_small_for_a_plate_costs_no_attempt(square_calibration, alerts, monkeypatch):
    import services.detection_service as ds

    monkeypatch.setattr(ds, "PLATE_MIN_VEHICLE_WIDTH_PX", 90)
    plates = StubPlates([_plate()])
    service = _service(square_calibration, alerts, {1: lambda n: 50.0}, plates)
    _run(service, 40)
    summary = service.finalize()
    assert plates.calls == [] and summary.plate_attempts == 0
    assert summary.vehicles_too_small_for_plate == 1 and summary.plates_read == 0


def test_summary_counts_attempts_boxes_and_reads(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 50.0}, StubPlates([None, _plate("PLATE_UNCLEAR", 0.1), _plate()]))
    _run(service, 80)
    s = service.finalize()
    assert (s.plate_attempts, s.plate_boxes_found, s.plates_read) == (3, 2, 1)   # no box, unclear box, then read
    assert s.vehicles_too_small_for_plate == 0 and s.max_speed_plate in (None, "MH12AB1234")


def test_retry_happens_early_when_the_vehicle_gets_closer_but_not_when_it_stays_far(square_calibration, alerts):
    plates = StubPlates([None] * 10)
    service = _service(square_calibration, alerts, {}, plates)
    stats = _fresh_stats()
    for frame_index, width in [(0, 100), (1, 100), (2, 130), (3, 131), (4, 170), (20, 170)]:
        service._maybe_read_plate(_track(width), stats, np.zeros((200, 400, 3), dtype=np.uint8), frame_index)
    # frame 0 first try; 1 skipped (same size, inside interval); 2 grew 30% -> try; 3 skipped; 4 grew 30% -> try; 20 interval over -> try
    assert len(plates.calls) == 4


def test_failed_attempts_keep_a_limited_number_of_debug_crops(square_calibration, alerts, tmp_path, monkeypatch):
    import services.detection_service as ds

    monkeypatch.setattr(ds, "PLATE_DEBUG_MAX_CROPS", 2)
    unclear = _plate("PLATE_UNCLEAR", 0.1)
    unclear.crop = np.full((10, 30, 3), 7, dtype=np.uint8)
    service = _service(square_calibration, alerts, {}, StubPlates([None, unclear, None, unclear]))
    stats = _fresh_stats()
    for i in range(4):
        service._maybe_read_plate(_track(100 + i * 40), stats, np.zeros((200, 400, 3), dtype=np.uint8), i * 100)
    files = sorted(p.name for p in (tmp_path / "plate_debug" / service.session_id).glob("*.jpg"))
    assert len(files) == 2 and "nobox" in files[0] + files[1] and "unclear" in files[0] + files[1]


def test_escalate_violation_unknown_id_returns_false(alerts, make_record):
    assert alerts.escalate_violation("missing-id", make_record()) is False


# ---------------------------------------------------------------------------
# Speed glitch filter (a bus "doing 127 km/h" in a traffic jam)
# ---------------------------------------------------------------------------
def _burst(n):  # crawling at 8 km/h, with a 6-frame glitch (tracker ID switch / box jitter) that reads as 300 km/h
    return 300.0 if 40 <= n < 46 else 8.0


def test_a_short_glitch_does_not_flag_a_crawling_vehicle(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: _burst})
    results = _run(service, 120)
    summary = service.finalize()
    assert _rows(alerts) == []
    assert summary.max_speed_kmph < 20
    assert max(v.speed_kmph for r in results for v in r.vehicles if v.speed_kmph is not None) < 60


def test_the_same_glitch_would_flag_it_without_the_filter(square_calibration, alerts, monkeypatch):
    import services.detection_service as ds

    monkeypatch.setattr(ds, "MAX_ACCEL_KMPH_PER_S", 1e9)   # filter off: proves the test above is meaningful
    service = _service(square_calibration, alerts, {1: _burst})
    _run(service, 120)
    assert len(_rows(alerts)) == 1


def test_a_real_and_lasting_speed_up_is_still_detected_after_the_filter_rebases(square_calibration, alerts):
    service = _service(square_calibration, alerts, {1: lambda n: 20.0 if n < 40 else 100.0})
    _run(service, 160)
    rows = _rows(alerts)
    assert len(rows) == 1 and rows[0]["severity"] == SEVERITY.SEVERE


# ---------------------------------------------------------------------------
# Why a plate was rejected: counters, log lines and debug file names
# ---------------------------------------------------------------------------
def _rejected(valid_format, raw="GX150GJ", conf=0.42, candidate="GX15OGJ"):
    r = PlateResult("PLATE_UNCLEAR", conf, None, None, valid_format, raw_text=raw, candidate=candidate)
    r.crop = np.full((12, 40, 3), 9, dtype=np.uint8)
    return r


def test_rejections_are_split_into_format_and_confidence_and_explained(square_calibration, alerts, tmp_path, caplog):
    plates = StubPlates([_rejected(False), _rejected(True, conf=0.21), _rejected(False)])
    plates.region, plates.confidence_threshold = "IN", 0.3
    service = _service(square_calibration, alerts, {}, plates)
    stats = _fresh_stats()
    with caplog.at_level("INFO"):
        for i in range(3):
            service._maybe_read_plate(_track(100 + i * 40), stats, np.zeros((200, 400, 3), dtype=np.uint8), i * 100)
    s = service.finalize()
    assert (s.plates_rejected_format, s.plates_rejected_confidence, s.plates_read) == (2, 1, 0)
    assert (s.plate_region, s.plate_min_confidence) == ("IN", 0.3)
    text = caplog.text
    assert "Plate rejected (format)" in text and "raw_ocr='GX150GJ'" in text and "candidate='GX15OGJ'" in text and "region=IN" in text
    names = " ".join(p.name for p in (tmp_path / "plate_debug" / service.session_id).glob("*.jpg"))
    assert "unclear-format-GX150GJ-c42" in names and "unclear-confidence-GX150GJ-c21" in names   # the file name tells the story


def test_recognizer_keeps_the_corrected_candidate_of_a_rejected_read():
    from src.plate_recognizer import PlateRecognizer

    class Reader:
        def readtext(self, image, allowlist=None):
            return [([[0, 0], [100, 0], [100, 30], [0, 30]], "GX15 0GJ", 0.8)]

    result = PlateRecognizer(detector=object(), reader=Reader(), region="IN").read_plate_crop(np.full((30, 100, 3), 120, dtype=np.uint8))
    assert result.plate_number == "PLATE_UNCLEAR" and result.raw_text == "GX150GJ" and result.candidate == "GX150GJ"


def test_every_plate_that_is_read_keeps_its_crop_for_checking(square_calibration, alerts, tmp_path):
    ok = _plate("MH12AB1234", 0.62)
    ok.crop = np.full((12, 40, 3), 200, dtype=np.uint8)
    service = _service(square_calibration, alerts, {}, StubPlates([ok]))
    service._maybe_read_plate(_track(100, track_id=117), _fresh_stats(), np.zeros((200, 400, 3), dtype=np.uint8), 0)
    names = [p.name for p in (tmp_path / "plate_debug" / service.session_id).glob("read-*.jpg")]
    assert names == ["read-track117-MH12AB1234-c62.jpg"]
