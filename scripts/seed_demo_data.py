#!/usr/bin/env python3
"""
seed_demo_data.py
==================
Loads synthetic violation incidents (Faker-based, see
``src/synthetic_data_gen.py``) into the SQLite violations database so the
dashboard's Violations Log, Analytics and notice-generation screens have
data to show without processing a real video first.

Usage:
    python scripts/seed_demo_data.py --count 1000

These rows are SYNTHETIC. Do not present dashboard numbers from a seeded
database as real detections. To start clean, stop the API and delete
``violation_logs/violations.db``.
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.alert_system import AlertSystem
from src.config import VIOLATION_DB_PATH, get_logger
from src.synthetic_data_gen import generate_synthetic_dataset
from src.violation_classifier import ViolationRecord

logger = get_logger(__name__)


def seed(count: int, seed_value: int, db_path: Path) -> int:
    """Generate synthetic incidents and persist the WARNING+ ones.

    Args:
        count: Number of synthetic incidents to generate.
        seed_value: Random seed for reproducibility.
        db_path: SQLite database file to write into.

    Returns:
        Number of violation rows actually persisted (NORMAL-severity
        incidents are skipped, same rule as the live pipeline).
    """
    alert_system = AlertSystem(db_path=db_path)
    incidents = generate_synthetic_dataset(n=count, seed=seed_value)
    rng = random.Random(seed_value)

    registered: set[str] = set()
    persisted = 0
    for incident in incidents:
        camera_id = incident["camera_id"]
        if camera_id not in registered:
            alert_system.register_camera(camera_id, incident["location"], incident["speed_limit"])
            registered.add(camera_id)

        timestamp = datetime.strptime(incident["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        record = ViolationRecord(
            violation_id=str(uuid.uuid4()),
            track_id=-1,
            plate_number=incident["plate_number"],
            vehicle_type=incident["vehicle_type"],
            detected_speed_kmph=incident["detected_speed"],
            speed_limit_kmph=incident["speed_limit"],
            excess_speed_kmph=round(max(0.0, incident["detected_speed"] - incident["speed_limit"]), 1),
            excess_percent=incident["excess_percent"],
            severity=incident["severity"],
            location=incident["location"],
            camera_id=camera_id,
            timestamp=timestamp,
        )
        if alert_system.log_violation(record, plate_confidence=round(rng.uniform(0.7, 0.98), 2)):
            persisted += 1

    logger.info(f"Seeded {persisted} synthetic violations into {db_path}")
    return persisted


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Seed the DriveGuard DB with synthetic violations.")
    parser.add_argument("--count", type=int, default=1000, help="Incidents to generate (NORMAL ones are skipped).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=Path, default=VIOLATION_DB_PATH)
    args = parser.parse_args()

    persisted = seed(args.count, args.seed, args.db)
    print(f"Seeded {persisted} synthetic violations into {args.db}")
    print("Refresh the dashboard (Violations Log / Analytics). These rows are synthetic demo data.")


if __name__ == "__main__":
    main()
