"""
synthetic_data_gen.py
=======================
Generates synthetic traffic-violation incidents for testing the LLM
report-generation pipeline (Module 5) and the dashboard end-to-end,
without needing a real video processed first.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from faker import Faker

from src.config import PROCESSED_DATA_DIR, SEVERITY, SPEED_ZONES, VEHICLE_CLASSES, get_logger

logger = get_logger(__name__)
fake = Faker("en_IN")

_INDIAN_STATE_CODES = [
    "MH", "KA", "DL", "TN", "UP", "GJ", "RJ", "WB", "TS", "KL", "PB", "HR", "MP", "AP", "OD",
]

_LOCATIONS = [
    ("NH-48, Pune-Mumbai Highway, KM marker 42", "Highway"),
    ("MG Road Junction, Bengaluru", "City Road"),
    ("Ring Road, Sector 18, Delhi", "City Road"),
    ("Outer Ring Road, Hyderabad", "Highway"),
    ("Anna Salai, Chennai", "City Road"),
    ("Model Town Residential Zone, Ludhiana", "Residential"),
    ("St. Xavier's School Zone, Mumbai", "School Zone"),
    ("NH-44, Ambala Bypass", "Highway"),
]

_WEATHER_OPTIONS = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Foggy", "Hazy"]
_ROAD_CONDITIONS = ["Dry", "Wet", "Waterlogged"]


def _generate_plate_number() -> str:
    """Generate a plausible, regex-valid synthetic Indian plate number."""
    state = random.choice(_INDIAN_STATE_CODES)
    district = f"{random.randint(1, 99):02d}"
    series = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=random.choice([1, 2])))
    number = f"{random.randint(1, 9999):04d}"
    return f"{state}{district}{series}{number}"


def _severity_for_excess(excess_percent: float) -> str:
    """Map an excess-speed percentage onto a severity label, mirroring violation_classifier."""
    if excess_percent >= 50:
        return SEVERITY.SEVERE
    if excess_percent >= 20:
        return SEVERITY.VIOLATION
    if excess_percent >= 10:
        return SEVERITY.WARNING
    return SEVERITY.NORMAL


def generate_incident(reference_time: datetime) -> dict:
    """Generate one synthetic violation incident.

    Args:
        reference_time: Base timestamp; the incident's timestamp is a
            random offset before this.

    Returns:
        A dict matching the LLM report-generator's expected input schema
        (see ``src/report_generator.py``, built on Day 3).
    """
    location, zone_type = random.choice(_LOCATIONS)
    speed_limit = SPEED_ZONES[zone_type]

    # Skew toward realistic mild-violation-heavy distribution rather than uniform.
    excess_percent = max(0.0, random.gauss(mu=18, sigma=15))
    detected_speed = round(speed_limit * (1 + excess_percent / 100), 1)
    severity = _severity_for_excess(excess_percent)

    vehicle_type = random.choices(
        list(VEHICLE_CLASSES.values()), weights=[35, 15, 8, 30, 10, 2, 6], k=1
    )[0]

    incident_time = reference_time - timedelta(
        days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59)
    )

    return {
        "vehicle_id": f"{vehicle_type[:3].upper()}_{random.randint(1, 999):03d}",
        "plate_number": _generate_plate_number(),
        "vehicle_type": vehicle_type,
        "detected_speed": detected_speed,
        "speed_limit": speed_limit,
        "violation_type": "SPEEDING",
        "severity": severity,
        "excess_percent": round(excess_percent, 1),
        "location": location,
        "timestamp": incident_time.strftime("%Y-%m-%d %H:%M:%S"),
        "evidence_frame": f"frame_{random.randint(1000, 9999)}.jpg",
        "weather": random.choice(_WEATHER_OPTIONS),
        "road_condition": random.choice(_ROAD_CONDITIONS),
        "camera_id": f"CAM-{fake.city_suffix().upper()[:3]}-{random.randint(1, 20):02d}",
    }


def generate_synthetic_dataset(n: int = 1000, seed: int = 42) -> List[dict]:
    """Generate a full batch of synthetic incidents.

    Args:
        n: Number of incidents to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of incident dicts.
    """
    random.seed(seed)
    Faker.seed(seed)
    reference_time = datetime.now(timezone.utc)
    incidents = [generate_incident(reference_time) for _ in range(n)]
    logger.info(f"Generated {len(incidents)} synthetic violation incidents")
    return incidents


def main() -> None:
    """CLI entry point: generate incidents and write them to a JSON file."""
    parser = argparse.ArgumentParser(description="Generate synthetic DriveGuard AI violation incidents.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "synthetic_violations.json",
    )
    args = parser.parse_args()

    incidents = generate_synthetic_dataset(n=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(incidents, indent=2))
    logger.info(f"Wrote {len(incidents)} synthetic incidents to {args.output}")

    severity_counts: dict[str, int] = {}
    for incident in incidents:
        severity_counts[incident["severity"]] = severity_counts.get(incident["severity"], 0) + 1
    print(f"Generated {len(incidents)} incidents -> {args.output}")
    print(f"Severity breakdown: {severity_counts}")


if __name__ == "__main__":
    main()
