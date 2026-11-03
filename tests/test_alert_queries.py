"""Tests for the dashboard-facing queries in src/alert_system.py: filters, local-time analytics, evidence files."""

from datetime import datetime, timezone

import pytest

import src.alert_system as alert_module
from src.alert_system import AlertSystem


@pytest.fixture
def system(tmp_db_path):
    s = AlertSystem(db_path=tmp_db_path)
    s.register_camera("CAM-1", "Test Junction", 60)
    s.register_camera("CAM-2", "Second Junction", 60)
    return s


def _ts(day: int, hour: int, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, 0, 0, tzinfo=timezone.utc)


def test_filters_by_vehicle_type_state_plate_and_status(system, make_record):
    system.log_violation(make_record(plate="MH12AB1234", vehicle_type="Truck"))
    system.log_violation(make_record(plate="KA01MF5678", vehicle_type="Car"))
    vid = system.log_violation(make_record(plate="MH14XY0001", vehicle_type="Bus"))
    system.update_status(vid, "RESOLVED")

    assert len(system.query_violations(vehicle_types=["Truck", "Bus"])) == 2
    assert len(system.query_violations(state_code="mh")) == 2
    assert len(system.query_violations(plate_query="mf 56")) == 1  # spaces and case ignored
    assert [r["id"] for r in system.query_violations(status="RESOLVED")] == [vid]


def test_count_matches_query_and_ignores_pagination(system, make_record):
    for _ in range(5):
        system.log_violation(make_record())
    assert system.count_violations() == 5
    assert len(system.query_violations(limit=2)) == 2
    assert system.count_violations(severity="SEVERE") == len(system.query_violations(severity="SEVERE"))


def test_date_filters_use_local_ist_day_not_utc(system, make_record):
    # 20:00 UTC on 19 Oct is 01:30 IST on 20 Oct -> belongs to local day 20 Oct.
    system.log_violation(make_record(ts=_ts(19, 20)))
    assert system.count_violations(date_from="2026-10-20", date_to="2026-10-20") == 1
    assert system.count_violations(date_from="2026-10-19", date_to="2026-10-19") == 0


def test_filter_values_are_bound_not_interpolated(system, make_record):
    system.log_violation(make_record())
    assert system.query_violations(plate_query="'; DROP TABLE violations; --") == []
    assert system.count_violations() == 1  # table still there


def test_hourly_and_heatmap_use_local_time(system, make_record):
    system.log_violation(make_record(ts=_ts(19, 20)))  # Mon 20:00 UTC = Tue 01:30 IST
    d = system.get_detailed_analytics()
    assert d["hourly"][1] == 1 and sum(d["hourly"]) == 1
    assert d["heatmap"][1][1] == 1  # row 1 = Tuesday, hour 1


def test_heatmap_monday_is_row_zero(system, make_record):
    system.log_violation(make_record(ts=_ts(19, 6)))  # Mon 06:00 UTC = Mon 11:30 IST
    assert system.get_detailed_analytics()["heatmap"][0][11] == 1


def test_speed_histogram_and_monthly_trend(system, make_record):
    system.log_violation(make_record(speed=71, ts=_ts(5, 8, month=9)))
    system.log_violation(make_record(speed=78, ts=_ts(6, 8, month=10)))
    system.log_violation(make_record(speed=92, ts=_ts(7, 8, month=10)))
    d = system.get_detailed_analytics()
    bins = {b["from_kmph"]: b["count"] for b in d["speed_histogram"]}
    assert bins == {70: 2, 90: 1}
    assert d["monthly_trend"] == [{"month": "2026-09", "count": 1}, {"month": "2026-10", "count": 2}]


def test_peak_window_and_empty_analytics(system, make_record):
    assert system.get_detailed_analytics()["peak_window"] is None
    for h in (12, 12, 12):  # 12:00 UTC -> 17:30 IST -> hour 17
        system.log_violation(make_record(ts=_ts(19, h)))
    assert system.get_detailed_analytics()["peak_window"]["start_hour"] in (16, 17)


def test_vehicle_share_compares_severe_to_traffic(system, make_record):
    for _ in range(3):
        system.log_violation(make_record(speed=100, vehicle_type="Truck"))  # SEVERE
    system.save_vehicle_session(1, "s", "2026-10-19T10:00:00+00:00", "2026-10-19T10:01:00+00:00", 50, 90, None, "Truck", 1)
    for i in range(2, 10):
        system.save_vehicle_session(i, "s", "2026-10-19T10:00:00+00:00", "2026-10-19T10:01:00+00:00", 50, 60, None, "Car", 0)
    share = {r["vehicle_type"]: r for r in system.get_detailed_analytics()["vehicle_share"]}
    assert share["Truck"]["severe_share_pct"] == 100.0
    assert share["Truck"]["traffic_share_pct"] == pytest.approx(11.1, abs=0.1)


def test_list_cameras_counts_violations(system, make_record):
    system.log_violation(make_record(camera_id="CAM-1"))
    cams = {c["camera_id"]: c for c in system.list_cameras()}
    assert cams["CAM-1"]["violation_count"] == 1 and cams["CAM-2"]["violation_count"] == 0
    assert "calibration_matrix" not in cams["CAM-1"]


def test_rto_dispatch_is_audited_and_marked_mock(system, make_record):
    vid = system.log_violation(make_record())
    row = system.record_rto_dispatch(vid)
    assert row["mode"] == "mock" and row["reference"].startswith("MOCK-")
    assert len(system.get_rto_dispatches(vid)) == 1
    with pytest.raises(KeyError):
        system.record_rto_dispatch("missing-id")


def test_speed_trace_and_plate_crop_round_trip(system, tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setattr(alert_module, "EVIDENCE_FRAMES_DIR", tmp_path)
    system.save_speed_trace("abc", [50.04, 61.0], fps=25.0)
    assert AlertSystem.load_speed_trace("abc") == {"speeds_kmph": [50.0, 61.0], "fps": 25.0}
    assert AlertSystem.load_speed_trace("nope") is None
    assert system.save_plate_crop(np.zeros((10, 30, 3), dtype=np.uint8), "abc").name == "abc_plate.jpg"
    assert system.save_plate_crop(np.zeros((0, 0, 3), dtype=np.uint8), "abc") is None


def test_recent_stats_counts_only_last_hour(system, make_record):
    system.log_violation(make_record())  # now
    system.log_violation(make_record(ts=_ts(1, 1, month=1)))  # long ago
    now_iso = datetime.now(timezone.utc).isoformat()
    system.save_vehicle_session(1, "s", now_iso, now_iso, 55.0, 91.0, "MH12AB1234", "Car", 1)
    stats = system.get_recent_stats(hours=1)
    assert stats["violations_last_hour"] == 1
    assert stats["fastest"] == {"speed_kmph": 91.0, "plate_number": "MH12AB1234"}
    assert stats["average_speed_kmph"] == 55.0
