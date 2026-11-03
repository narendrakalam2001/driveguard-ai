#!/usr/bin/env python3
"""Build the bundled demo database used by the lite (no-API) Streamlit deployment.

The lite dashboard cannot run the AI pipeline on Streamlit Community Cloud (1 GB RAM), so it reads a
small SQLite file of SYNTHETIC violations instead. This script builds that file deterministically:
8 named cameras, about 1,800 violations spread over 6 months, with rush-hour peaks, Tuesday/Friday
peaks and trucks over-represented among SEVERE cases, so every analytics chart has a realistic shape.

Usage:
    python scripts/build_demo_db.py                       # writes data/demo/demo_violations.db
    python scripts/build_demo_db.py --count 2500 --seed 7

All plates and incidents are randomly generated. No real vehicle or person is represented.
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alert_system import AlertSystem
from src.config import PROJECT_ROOT, SEVERITY, SPEED_ZONES, get_logger
from src.report_generator import ReportGenerator
from src.violation_classifier import ViolationRecord

logger = get_logger("build_demo_db")
IST = timezone(timedelta(hours=5, minutes=30))
DEFAULT_OUT = PROJECT_ROOT / "data" / "demo" / "demo_violations.db"

CAMERAS = [  # (camera_id, location, zone type, share of traffic)
    ("CAM-PUN-01", "NH-48, Pune-Mumbai Highway, KM marker 42", "Highway", 16),
    ("CAM-BLR-02", "MG Road Junction, Bengaluru", "City Road", 17),
    ("CAM-DEL-03", "Ring Road, Sector 18, Delhi", "City Road", 17),
    ("CAM-HYD-04", "Outer Ring Road, Hyderabad", "Highway", 12),
    ("CAM-CHE-05", "Anna Salai, Chennai", "City Road", 13),
    ("CAM-LDH-06", "Model Town Residential Zone, Ludhiana", "Residential", 9),
    ("CAM-MUM-07", "St. Xavier's School Zone, Mumbai", "School Zone", 8),
    ("CAM-AMB-08", "NH-44, Ambala Bypass", "Highway", 8),
]
STATES = ["MH", "KA", "DL", "TN", "UP", "GJ", "RJ", "WB", "TS", "KL", "PB", "HR", "MP", "AP"]
STATE_WEIGHTS = [26, 14, 12, 9, 8, 6, 5, 3, 5, 3, 3, 2, 2, 2]
VEHICLES = ["Car", "Truck", "Bus", "Motorcycle", "Auto-rickshaw", "Bicycle", "Van"]
VEHICLE_W_ALL = [38, 12, 6, 30, 8, 1, 5]
VEHICLE_W_SEVERE = [26, 34, 8, 20, 6, 0, 6]  # trucks are over-represented among the worst offenders
HOUR_W = [1, 1, 1, 1, 2, 3, 5, 9, 14, 11, 7, 6, 6, 6, 6, 7, 9, 13, 16, 13, 8, 5, 3, 2]  # IST rush hours
DAY_W = [10, 13, 10, 10, 13, 9, 6]  # Mon..Sun, Tuesday and Friday peaks
BANDS = {SEVERITY.WARNING: (10.0, 19.9), SEVERITY.VIOLATION: (20.0, 49.9), SEVERITY.SEVERE: (50.0, 85.0)}


def make_plate(rng: random.Random) -> str:
    """Return a random, format-valid Indian plate such as ``MH12AB1234``."""
    state = rng.choices(STATES, STATE_WEIGHTS)[0]
    series = "".join(rng.choices("ABCDEFGHJKLMNPRSTUVWXYZ", k=rng.choice([1, 2])))
    return f"{state}{rng.randint(1, 60):02d}{series}{rng.randint(1, 9999):04d}"


def build(count: int, seed: int, out: Path, months: int = 6) -> int:
    """Create the demo database and return the number of violations written.

    Args:
        count: Violations to generate.
        seed: Random seed (same seed gives the same database content).
        out: SQLite file to create (replaced if it exists).
        months: How many months back the data spreads.
    """
    rng = random.Random(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    system = AlertSystem(db_path=out)
    for cam_id, loc, zone, _ in CAMERAS:
        system.register_camera(cam_id, loc, SPEED_ZONES[zone])

    now = datetime.now(IST).replace(second=0, microsecond=0)
    ids: list[tuple[str, ViolationRecord]] = []
    weights = [c[3] for c in CAMERAS]
    for _ in range(count):
        cam_id, loc, zone, _w = rng.choices(CAMERAS, weights)[0]
        limit = SPEED_ZONES[zone]
        severity = rng.choices([SEVERITY.WARNING, SEVERITY.VIOLATION, SEVERITY.SEVERE], [44, 39, 17])[0]
        lo, hi = BANDS[severity]
        excess_pct = round(rng.uniform(lo, hi), 1)
        speed = round(limit * (1 + excess_pct / 100), 1)
        vtype = rng.choices(VEHICLES, VEHICLE_W_SEVERE if severity == SEVERITY.SEVERE else VEHICLE_W_ALL)[0]
        while True:  # pick a day (weekday-weighted) and an hour (rush-hour weighted) in the last N months
            day = (now - timedelta(days=rng.randint(0, months * 30 - 1))).replace(hour=0, minute=0)
            if rng.random() < DAY_W[day.weekday()] / max(DAY_W):
                break
        when = day.replace(hour=rng.choices(range(24), HOUR_W)[0], minute=rng.randint(0, 59))
        if when > now:
            when -= timedelta(days=1)
        record = ViolationRecord(
            violation_id=str(uuid.uuid4()), track_id=-1, plate_number=make_plate(rng), vehicle_type=vtype,
            detected_speed_kmph=speed, speed_limit_kmph=limit, excess_speed_kmph=round(speed - limit, 1),
            excess_percent=excess_pct, severity=severity, location=loc, camera_id=cam_id,
            timestamp=when.astimezone(timezone.utc),
        )
        if system.log_violation(record, plate_confidence=round(rng.uniform(0.62, 0.98), 2)):
            ids.append((record.violation_id, record))

    # Realistic case-work state: some resolved/disputed, and example notices on the worst cases.
    generator = ReportGenerator()
    for vid, record in ids:
        roll = rng.random()
        if roll < 0.24:
            system.update_status(vid, "RESOLVED")
        elif roll < 0.29:
            system.update_status(vid, "DISPUTED")
    severe = sorted((r for r in ids if r[1].severity == SEVERITY.SEVERE), key=lambda r: r[1].timestamp, reverse=True)[:30]
    for vid, record in severe:
        notice = generator.generate_fallback_notice(record, vehicle_id=f"{record.vehicle_type[:3].upper()}_{vid[:6].upper()}")
        system.update_notice(vid, notice.notice_text)
    logger.info("Built %s with %d violations, %d cameras, %d notices", out, len(ids), len(CAMERAS), len(severe))
    return len(ids)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build the bundled demo database (synthetic data).")
    parser.add_argument("--count", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(f"Wrote {build(args.count, args.seed, args.out)} synthetic violations to {args.out}")


if __name__ == "__main__":
    main()
