"""
violation_classifier.py
=========================
Applies speed-zone rules to a vehicle's estimated speed and classifies
it into NORMAL / WARNING / VIOLATION / SEVERE, per the thresholds in
``src/config.py``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.config import (
    SEVERE_MULTIPLIER,
    SEVERITY,
    VIOLATION_MULTIPLIER,
    WARNING_MULTIPLIER,
    get_logger,
)
from src.speed_estimator import SpeedEstimate

logger = get_logger(__name__)


@dataclass
class ViolationRecord:
    """A single, fully-classified violation event ready for logging/reporting.

    Attributes:
        violation_id: Unique identifier for this event.
        track_id: The vehicle's tracker ID.
        plate_number: Recognized plate text, or None if not yet read.
        vehicle_type: Vehicle class (Car, Truck, ...).
        detected_speed_kmph: Smoothed speed at time of classification.
        speed_limit_kmph: Applicable speed limit for the zone.
        excess_speed_kmph: How far over the limit, in km/h.
        excess_percent: How far over the limit, as a percentage.
        severity: One of NORMAL / WARNING / VIOLATION / SEVERE.
        location: Human-readable location string.
        camera_id: Identifier of the camera that captured this event.
        timestamp: UTC timestamp of the violation.
    """

    violation_id: str
    track_id: int
    plate_number: Optional[str]
    vehicle_type: str
    detected_speed_kmph: float
    speed_limit_kmph: int
    excess_speed_kmph: float
    excess_percent: float
    severity: str
    location: str
    camera_id: str
    timestamp: datetime


class ViolationClassifier:
    """Classifies vehicle speeds against a configured zone's limit."""

    def __init__(
        self,
        speed_limit_kmph: int,
        location: str,
        camera_id: str,
        warning_multiplier: float = WARNING_MULTIPLIER,
        violation_multiplier: float = VIOLATION_MULTIPLIER,
        severe_multiplier: float = SEVERE_MULTIPLIER,
    ) -> None:
        """Configure the classifier for one camera/zone.

        Args:
            speed_limit_kmph: Posted speed limit for this zone.
            location: Human-readable location description, used in reports.
            camera_id: Identifier of the camera producing events.
            warning_multiplier: Fraction over limit that triggers WARNING (default 1.1x).
            violation_multiplier: Fraction over limit that triggers VIOLATION (default 1.2x).
            severe_multiplier: Fraction over limit that triggers SEVERE (default 1.5x).
        """
        self.speed_limit_kmph = speed_limit_kmph
        self.location = location
        self.camera_id = camera_id
        self.warning_multiplier = warning_multiplier
        self.violation_multiplier = violation_multiplier
        self.severe_multiplier = severe_multiplier

    def classify_severity(self, speed_kmph: float) -> str:
        """Map a raw speed onto a severity band.

        Args:
            speed_kmph: Smoothed detected speed.

        Returns:
            One of :class:`src.config.ViolationSeverity`'s labels.
        """
        if speed_kmph > self.speed_limit_kmph * self.severe_multiplier:
            return SEVERITY.SEVERE
        if speed_kmph > self.speed_limit_kmph * self.violation_multiplier:
            return SEVERITY.VIOLATION
        if speed_kmph > self.speed_limit_kmph * self.warning_multiplier:
            return SEVERITY.WARNING
        return SEVERITY.NORMAL

    def classify(
        self,
        speed_estimate: SpeedEstimate,
        vehicle_type: str,
        plate_number: Optional[str] = None,
    ) -> ViolationRecord:
        """Build a full :class:`ViolationRecord` from a speed estimate.

        Always returns a record, even for NORMAL speeds — callers (e.g.
        the alert system) decide whether NORMAL records are worth
        persisting or are only used for live dashboard stats.

        Args:
            speed_estimate: Output of :meth:`SpeedEstimator.estimate`.
            vehicle_type: Vehicle class for this track.
            plate_number: Recognized plate text, if available yet.

        Returns:
            A populated :class:`ViolationRecord`.
        """
        severity = self.classify_severity(speed_estimate.smoothed_speed_kmph)
        excess_speed = max(0.0, speed_estimate.smoothed_speed_kmph - self.speed_limit_kmph)
        excess_percent = (excess_speed / self.speed_limit_kmph) * 100 if self.speed_limit_kmph else 0.0

        record = ViolationRecord(
            violation_id=str(uuid.uuid4()),
            track_id=speed_estimate.track_id,
            plate_number=plate_number,
            vehicle_type=vehicle_type,
            detected_speed_kmph=round(speed_estimate.smoothed_speed_kmph, 1),
            speed_limit_kmph=self.speed_limit_kmph,
            excess_speed_kmph=round(excess_speed, 1),
            excess_percent=round(excess_percent, 1),
            severity=severity,
            location=self.location,
            camera_id=self.camera_id,
            timestamp=datetime.now(timezone.utc),
        )

        if severity != SEVERITY.NORMAL:
            # debug, not info: this runs for every vehicle on every frame. The service logs the
            # event itself when a vehicle first reaches a band.
            logger.debug(
                f"{severity} track_id={record.track_id} plate={plate_number} "
                f"speed={record.detected_speed_kmph} limit={self.speed_limit_kmph} "
                f"excess={record.excess_percent}%"
            )
        return record

    def is_repeat_offender(self, plate_number: str, prior_plate_numbers: list[str]) -> bool:
        """Check whether a plate already has a violation on record.

        Args:
            plate_number: Plate to check.
            prior_plate_numbers: Plate numbers from violation records logged *before*
                this one (e.g. queried from the violations DB).

        Returns:
            True if the plate appears at least once in ``prior_plate_numbers``.
            (If the current violation is already in that list, test ``count > 1`` instead.)
        """
        return prior_plate_numbers.count(plate_number) > 0
