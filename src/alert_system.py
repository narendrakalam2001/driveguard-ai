"""
alert_system.py
================
Real-time alert generation and persistence.

Responsible for:
  - Applying the SQLite schema on first run.
  - Persisting classified violations (and upserting per-vehicle session stats).
  - Saving evidence frames to disk alongside the DB record.
  - Publishing a bounded in-memory live-alert feed for the dashboard's
    "Live Alert Feed" panel (Screen 1) without requiring a separate broker.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import (
    DISPLAY_UTC_OFFSET_MINUTES,
    EVIDENCE_FRAMES_DIR,
    SEVERITY,
    VIOLATION_DB_PATH,
    VIOLATION_LOG_DIR,
    get_logger,
)
from src.violation_classifier import ViolationRecord

logger = get_logger(__name__)

_SCHEMA_PATH = VIOLATION_LOG_DIR / "schema.sql"
_LIVE_FEED_MAXLEN = 200


class AlertSystem:
    """Persists violations to SQLite and maintains a live in-memory alert feed.

    Thread-safe: designed to be called from the frame-processing loop
    while a FastAPI/Streamlit thread concurrently reads the live feed.
    """

    def __init__(self, db_path: Path = VIOLATION_DB_PATH) -> None:
        """Initialize the alert system and ensure the database schema exists.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._live_feed: Deque[dict] = deque(maxlen=_LIVE_FEED_MAXLEN)
        self._apply_schema()

    def _apply_schema(self) -> None:
        """Apply ``violation_logs/schema.sql`` if the DB hasn't been initialized yet."""
        if not _SCHEMA_PATH.exists():
            logger.info(f"Schema file not found at {_SCHEMA_PATH}; skipping auto-apply.")
            return
        schema_sql = _SCHEMA_PATH.read_text()
        with self._connect() as conn:
            conn.executescript(schema_sql)
        logger.info(f"Applied schema to database at {self.db_path}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context-managed SQLite connection with row factory set for dict-like access."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def register_camera(
        self, camera_id: str, location_name: str, speed_limit: int, calibration_matrix: Optional[list] = None
    ) -> None:
        """Insert or update a camera's metadata.

        Args:
            camera_id: Unique camera identifier.
            location_name: Human-readable location string.
            speed_limit: Posted speed limit for this camera's zone.
            calibration_matrix: The camera's 3x3 homography, if calibrated.
        """
        from datetime import datetime, timezone

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cameras (camera_id, location_name, speed_limit, calibration_matrix, status, last_active)
                VALUES (?, ?, ?, ?, 'ACTIVE', ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    location_name=excluded.location_name,
                    speed_limit=excluded.speed_limit,
                    calibration_matrix=excluded.calibration_matrix,
                    status='ACTIVE',
                    last_active=excluded.last_active
                """,
                (
                    camera_id,
                    location_name,
                    speed_limit,
                    json.dumps(calibration_matrix) if calibration_matrix else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        logger.info(f"Registered camera_id={camera_id} location={location_name}")

    def save_evidence_frame(self, frame: np.ndarray, violation_id: str) -> Path:
        """Save the evidence frame for a violation to disk.

        Args:
            frame: BGR frame at the moment of violation, ideally with the
                bounding box / speed overlay already drawn.
            violation_id: The violation's UUID, used as the filename.

        Returns:
            Path to the saved JPEG file.
        """
        EVIDENCE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EVIDENCE_FRAMES_DIR / f"{violation_id}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return out_path

    def escalate_violation(
        self, violation_id: str, record: ViolationRecord, evidence_frame: Optional[np.ndarray] = None
    ) -> bool:
        """Upgrade an existing violation row when the same vehicle reaches a worse severity band.

        A vehicle that goes WARNING -> VIOLATION -> SEVERE stays **one** row (one notice, one count) instead
        of three; the row keeps its id and takes the new speed, severity and (if given) a new evidence frame.

        Args:
            violation_id: The existing row to upgrade.
            record: The newly classified record for the same vehicle.
            evidence_frame: Frame to store as the new evidence, if any.

        Returns:
            True if a row was updated.
        """
        evidence_path = self.save_evidence_frame(evidence_frame, violation_id) if evidence_frame is not None else None
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """UPDATE violations SET detected_speed = ?, speed_limit = ?, excess_speed = ?, severity = ?,
                          evidence_frame_path = COALESCE(?, evidence_frame_path) WHERE id = ?""",
                (record.detected_speed_kmph, record.speed_limit_kmph, record.excess_speed_kmph, record.severity,
                 str(evidence_path) if evidence_path else None, violation_id),
            )
        if cur.rowcount:
            payload = {**asdict(record), "violation_id": violation_id, "timestamp": record.timestamp.isoformat()}
            self._live_feed.appendleft(payload)
            logger.info(f"Escalated violation_id={violation_id} to severity={record.severity}")
        return bool(cur.rowcount)

    def save_plate_crop(self, plate_crop: np.ndarray, violation_id: str) -> Optional[Path]:
        """Save the zoomed plate image for a violation (``<id>_plate.jpg`` next to the evidence frame).

        Args:
            plate_crop: BGR crop of just the number plate.
            violation_id: The violation's UUID.

        Returns:
            Path of the saved JPEG, or None if the crop was empty.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None
        EVIDENCE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EVIDENCE_FRAMES_DIR / f"{violation_id}_plate.jpg"
        cv2.imwrite(str(out_path), plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return out_path

    def save_speed_trace(self, violation_id: str, speeds_kmph: List[float], fps: Optional[float] = None) -> Path:
        """Save the vehicle's smoothed speed over time for the detail screen's speed graph.

        Args:
            violation_id: The violation's UUID.
            speeds_kmph: Smoothed speed per processed frame, oldest first.
            fps: Frame rate of the source, if known (lets the UI label the x-axis in seconds).

        Returns:
            Path of the JSON file (``<id>_speed.json``).
        """
        EVIDENCE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EVIDENCE_FRAMES_DIR / f"{violation_id}_speed.json"
        out_path.write_text(json.dumps({"speeds_kmph": [round(float(s), 1) for s in speeds_kmph], "fps": fps}))
        return out_path

    @staticmethod
    def load_speed_trace(violation_id: str) -> Optional[dict]:
        """Load a saved speed trace, or None if none was recorded (e.g. seeded demo data)."""
        path = EVIDENCE_FRAMES_DIR / f"{violation_id}_speed.json"
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def log_violation(
        self,
        record: ViolationRecord,
        plate_confidence: Optional[float] = None,
        evidence_frame: Optional[np.ndarray] = None,
        min_severity_to_persist: str = SEVERITY.WARNING,
    ) -> Optional[str]:
        """Persist a violation record and push it onto the live alert feed.

        NORMAL-severity records are intentionally not persisted by default
        (they'd flood the DB with every non-violating vehicle) — only used
        transiently for live dashboard stats by the caller.

        Args:
            record: A classified :class:`ViolationRecord`.
            plate_confidence: OCR confidence for the plate read, if available.
            evidence_frame: Frame to save as evidence; skipped if None.
            min_severity_to_persist: Minimum severity band that gets written
                to the database (default: WARNING and above).

        Returns:
            The violation_id if persisted, else None.
        """
        severity_rank = {SEVERITY.NORMAL: 0, SEVERITY.WARNING: 1, SEVERITY.VIOLATION: 2, SEVERITY.SEVERE: 3}
        if severity_rank.get(record.severity, 0) < severity_rank.get(min_severity_to_persist, 1):
            return None

        evidence_path: Optional[Path] = None
        if evidence_frame is not None:
            evidence_path = self.save_evidence_frame(evidence_frame, record.violation_id)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO violations (
                    id, timestamp, plate_number, plate_confidence, vehicle_type,
                    detected_speed, speed_limit, excess_speed, severity, location,
                    camera_id, evidence_frame_path, notice_generated, notice_text,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 'PENDING', ?)
                """,
                (
                    record.violation_id,
                    record.timestamp.isoformat(),
                    record.plate_number,
                    plate_confidence,
                    record.vehicle_type,
                    record.detected_speed_kmph,
                    record.speed_limit_kmph,
                    record.excess_speed_kmph,
                    record.severity,
                    record.location,
                    record.camera_id,
                    str(evidence_path) if evidence_path else None,
                    record.timestamp.isoformat(),
                ),
            )

        alert_payload = {**asdict(record), "timestamp": record.timestamp.isoformat()}
        self._live_feed.appendleft(alert_payload)
        logger.info(f"Logged violation_id={record.violation_id} severity={record.severity}")
        return record.violation_id

    def upsert_vehicle_session(
        self,
        track_id: int,
        session_id: str,
        speed_kmph: float,
        vehicle_type: str,
        plate_number: Optional[str],
        first_seen_iso: str,
        last_seen_iso: str,
        is_violation: bool,
    ) -> None:
        """Upsert running per-vehicle statistics for the current session.

        Args:
            track_id: Vehicle's tracker ID.
            session_id: Identifier for the current processing session/video.
            speed_kmph: Latest smoothed speed sample.
            vehicle_type: Vehicle class.
            plate_number: Recognized plate, if available.
            first_seen_iso: ISO timestamp of first detection.
            last_seen_iso: ISO timestamp of latest detection.
            is_violation: Whether this update corresponds to a violation event.
        """
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT avg_speed, max_speed, violation_count FROM vehicles WHERE track_id=? AND session_id=?",
                (track_id, session_id),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO vehicles (
                        track_id, session_id, first_seen, last_seen, avg_speed,
                        max_speed, plate_number, vehicle_type, violation_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        track_id,
                        session_id,
                        first_seen_iso,
                        last_seen_iso,
                        speed_kmph,
                        speed_kmph,
                        plate_number,
                        vehicle_type,
                        1 if is_violation else 0,
                    ),
                )
            else:
                new_max = max(existing["max_speed"], speed_kmph)
                new_avg = (existing["avg_speed"] + speed_kmph) / 2  # simple running blend
                new_violation_count = existing["violation_count"] + (1 if is_violation else 0)
                conn.execute(
                    """
                    UPDATE vehicles
                    SET last_seen=?, avg_speed=?, max_speed=?, plate_number=COALESCE(?, plate_number),
                        vehicle_type=?, violation_count=?
                    WHERE track_id=? AND session_id=?
                    """,
                    (
                        last_seen_iso,
                        new_avg,
                        new_max,
                        plate_number,
                        vehicle_type,
                        new_violation_count,
                        track_id,
                        session_id,
                    ),
                )

    def get_live_feed(self, limit: int = 20) -> List[dict]:
        """Return the most recent live alerts (newest first), for dashboard polling.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of alert dicts, newest first.
        """
        return list(self._live_feed)[:limit]

    @staticmethod
    def _local(expr: str = "timestamp") -> str:
        """SQL expression converting a stored UTC timestamp to local (display) time."""
        return f"datetime({expr}, '{DISPLAY_UTC_OFFSET_MINUTES:+d} minutes')"

    def _build_where(
        self,
        camera_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        vehicle_types: Optional[Sequence[str]] = None,
        state_code: Optional[str] = None,
        plate_query: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Tuple[str, list]:
        """Build a parameterised WHERE clause shared by listing, counting and export.

        Args:
            camera_id: Exact camera id.
            severity: WARNING / VIOLATION / SEVERE.
            status: PENDING / RESOLVED / DISPUTED.
            vehicle_types: Any-of list of vehicle classes.
            state_code: Two-letter plate prefix, e.g. ``MH``.
            plate_query: Substring of the plate (spaces ignored).
            date_from: Local date ``YYYY-MM-DD``, inclusive.
            date_to: Local date ``YYYY-MM-DD``, inclusive.

        Returns:
            ``(where_sql, params)`` - values are always bound, never interpolated.
        """
        clauses: List[str] = []
        params: list = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if vehicle_types:
            clauses.append(f"vehicle_type IN ({','.join('?' * len(vehicle_types))})")
            params.extend(vehicle_types)
        if state_code:
            clauses.append("REPLACE(plate_number, ' ', '') LIKE ?")
            params.append(f"{state_code.strip().upper()}%")
        if plate_query:
            clauses.append("REPLACE(plate_number, ' ', '') LIKE ?")
            params.append(f"%{plate_query.replace(' ', '').upper()}%")
        if date_from:
            clauses.append(f"date({self._local()}) >= date(?)")
            params.append(date_from)
        if date_to:
            clauses.append(f"date({self._local()}) <= date(?)")
            params.append(date_to)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def query_violations(
        self,
        camera_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        vehicle_types: Optional[Sequence[str]] = None,
        state_code: Optional[str] = None,
        plate_query: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[dict]:
        """Query persisted violations with optional filters, newest first.

        Args:
            camera_id: Restrict to a single camera, if given.
            severity: Restrict to a single severity band, if given.
            limit: Page size.
            offset: Page offset.
            status: Restrict to one resolution status.
            vehicle_types: Restrict to any of these vehicle classes.
            state_code: Restrict to plates starting with this state code.
            plate_query: Substring match on the plate.
            date_from: Earliest local date (``YYYY-MM-DD``), inclusive.
            date_to: Latest local date (``YYYY-MM-DD``), inclusive.

        Returns:
            List of violation rows as dicts.
        """
        where, params = self._build_where(
            camera_id, severity, status, vehicle_types, state_code, plate_query, date_from, date_to
        )
        query = f"SELECT * FROM violations{where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        with self._connect() as conn:
            rows = conn.execute(query, [*params, limit, offset]).fetchall()
        return [dict(row) for row in rows]

    def count_violations(self, **filters) -> int:
        """Count violations matching the same filters as :meth:`query_violations` (for pagination).

        Args:
            **filters: Any of the filter keyword arguments of :meth:`query_violations`
                (``limit``/``offset`` are not accepted).

        Returns:
            Number of matching rows.
        """
        where, params = self._build_where(**filters)
        with self._connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) AS c FROM violations{where}", params).fetchone()["c"])

    def list_cameras(self) -> List[dict]:
        """Return all registered cameras with their violation counts.

        Returns:
            List of camera dicts (calibration matrix omitted - it is large and not for display).
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.camera_id, c.location_name, c.speed_limit, c.status, c.last_active,
                          COUNT(v.id) AS violation_count
                   FROM cameras c LEFT JOIN violations v ON v.camera_id = c.camera_id
                   GROUP BY c.camera_id ORDER BY c.camera_id"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_detailed_analytics(self, camera_id: Optional[str] = None, months: int = 6) -> dict:
        """Time-series and distribution data behind the analytics charts (Screen 5).

        All time grouping uses local time (``DISPLAY_UTC_OFFSET_MINUTES``), not UTC.

        Args:
            camera_id: Restrict to one camera, if given.
            months: How many trailing calendar months to return in the monthly trend.

        Returns:
            Dict with ``hourly`` (24 counts), ``heatmap`` (7 rows Mon..Sun x 24 hours),
            ``speed_histogram`` (10 km/h bins), ``monthly_trend``, ``peak_window`` (busiest two
            consecutive hours), ``state_breakdown`` (violations per plate state code), ``most_common_severity``, and ``vehicle_share`` comparing each vehicle
            type's share of SEVERE violations with its share of all tracked vehicles.
        """
        where, params = self._build_where(camera_id=camera_id)
        local = self._local()
        with self._connect() as conn:
            hourly = [0] * 24
            for row in conn.execute(
                f"SELECT CAST(strftime('%H', {local}) AS INTEGER) AS h, COUNT(*) AS c FROM violations{where} GROUP BY h", params
            ):
                hourly[row["h"]] = row["c"]

            heatmap = [[0] * 24 for _ in range(7)]  # row 0 = Monday
            for row in conn.execute(
                f"""SELECT (CAST(strftime('%w', {local}) AS INTEGER) + 6) % 7 AS d,
                           CAST(strftime('%H', {local}) AS INTEGER) AS h, COUNT(*) AS c
                    FROM violations{where} GROUP BY d, h""",
                params,
            ):
                heatmap[row["d"]][row["h"]] = row["c"]

            bins: dict = {}
            for row in conn.execute(
                f"SELECT CAST(detected_speed / 10 AS INTEGER) * 10 AS b, COUNT(*) AS c FROM violations{where} GROUP BY b ORDER BY b",
                params,
            ):
                bins[int(row["b"])] = row["c"]
            speed_histogram = [{"from_kmph": b, "to_kmph": b + 10, "count": c} for b, c in sorted(bins.items())]

            monthly = [
                {"month": r["m"], "count": r["c"]}
                for r in conn.execute(
                    f"""SELECT strftime('%Y-%m', {local}) AS m, COUNT(*) AS c FROM violations{where}
                        GROUP BY m ORDER BY m DESC LIMIT ?""",
                    [*params, months],
                )
            ][::-1]

            sev_where = (where + " AND" if where else " WHERE") + " severity = 'SEVERE'"
            severe_by_type = {
                r["vehicle_type"]: r["c"]
                for r in conn.execute(
                    f"SELECT vehicle_type, COUNT(*) AS c FROM violations{sev_where} GROUP BY vehicle_type", params
                )
            }
            fleet = {
                r["vehicle_type"]: r["c"]
                for r in conn.execute(
                    "SELECT vehicle_type, COUNT(*) AS c FROM vehicles WHERE vehicle_type IS NOT NULL GROUP BY vehicle_type"
                )
            }
            state_breakdown = {
                r["st"]: r["c"]
                for r in conn.execute(
                    f"""SELECT SUBSTR(REPLACE(plate_number, ' ', ''), 1, 2) AS st, COUNT(*) AS c FROM violations
                        {(where + " AND") if where else "WHERE"} plate_number IS NOT NULL
                        GROUP BY st ORDER BY c DESC""",
                    params,
                )
            }
            sev_counts = {
                r["severity"]: r["c"]
                for r in conn.execute(f"SELECT severity, COUNT(*) AS c FROM violations{where} GROUP BY severity", params)
            }

        peak_start = max(range(24), key=lambda h: hourly[h] + hourly[(h + 1) % 24]) if any(hourly) else None
        severe_total, fleet_total = sum(severe_by_type.values()), sum(fleet.values())
        vehicle_share = [
            {
                "vehicle_type": vt,
                "severe_share_pct": round(100 * severe_by_type.get(vt, 0) / severe_total, 1) if severe_total else 0.0,
                "traffic_share_pct": round(100 * fleet.get(vt, 0) / fleet_total, 1) if fleet_total else None,
            }
            for vt in sorted(set(severe_by_type) | set(fleet))
        ]
        return {
            "hourly": hourly,
            "heatmap": heatmap,
            "speed_histogram": speed_histogram,
            "monthly_trend": monthly,
            "peak_window": None if peak_start is None else {"start_hour": peak_start, "end_hour": (peak_start + 2) % 24},
            "most_common_severity": max(sev_counts, key=sev_counts.get) if sev_counts else None,
            "severity_counts": sev_counts,
            "state_breakdown": state_breakdown,
            "vehicle_share": vehicle_share,
            "utc_offset_minutes": DISPLAY_UTC_OFFSET_MINUTES,
        }

    def get_violation(self, violation_id: str) -> Optional[dict]:
        """Fetch a single violation record by its ID.

        Args:
            violation_id: The violation's UUID.

        Returns:
            The violation row as a dict, or None if not found.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM violations WHERE id = ?", (violation_id,)).fetchone()
        return dict(row) if row else None

    def update_plate(self, violation_id: str, plate_number: str, plate_confidence: Optional[float] = None) -> bool:
        """Attach a plate that was read *after* the violation was first logged.

        Args:
            violation_id: The violation's UUID.
            plate_number: Plate text that was read.
            plate_confidence: OCR confidence of that read.

        Returns:
            True if a row was updated.
        """
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE violations SET plate_number = ?, plate_confidence = ? WHERE id = ?",
                (plate_number, plate_confidence, violation_id),
            )
        for alert in self._live_feed:
            if alert.get("violation_id") == violation_id:
                alert["plate_number"] = plate_number
        return cursor.rowcount > 0

    def save_vehicle_session(
        self,
        track_id: int,
        session_id: str,
        first_seen_iso: str,
        last_seen_iso: str,
        avg_speed: Optional[float],
        max_speed: Optional[float],
        plate_number: Optional[str],
        vehicle_type: str,
        violation_count: int,
    ) -> None:
        """Write one finished vehicle's final stats (the ``vehicles`` table), replacing any earlier row.

        Args:
            track_id: Vehicle's tracker ID.
            session_id: Processing session/video identifier.
            first_seen_iso: ISO timestamp of first detection.
            last_seen_iso: ISO timestamp of last detection.
            avg_speed: Mean smoothed speed over the time it was measured, km/h.
            max_speed: Highest smoothed speed, km/h.
            plate_number: Plate that was read, if any.
            vehicle_type: Vehicle class.
            violation_count: Number of violation records logged for this vehicle.
        """
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO vehicles (
                    track_id, session_id, first_seen, last_seen, avg_speed,
                    max_speed, plate_number, vehicle_type, violation_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (track_id, session_id, first_seen_iso, last_seen_iso, avg_speed, max_speed,
                 plate_number, vehicle_type, violation_count),
            )

    def update_notice(self, violation_id: str, notice_text: str) -> bool:
        """Persist a generated LLM notice back onto its violation row.

        Args:
            violation_id: The violation's UUID.
            notice_text: The generated notice text.

        Returns:
            True if a row was updated, False if the violation_id didn't exist.
        """
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE violations SET notice_generated = 1, notice_text = ? WHERE id = ?",
                (notice_text, violation_id),
            )
        return cursor.rowcount > 0

    def update_status(self, violation_id: str, status: str) -> bool:
        """Update a violation's resolution status.

        Args:
            violation_id: The violation's UUID.
            status: One of PENDING / RESOLVED / DISPUTED.

        Returns:
            True if a row was updated, False if the violation_id didn't exist.
        """
        with self._lock, self._connect() as conn:
            cursor = conn.execute("UPDATE violations SET status = ? WHERE id = ?", (status, violation_id))
        return cursor.rowcount > 0

    def get_analytics_summary(self, camera_id: Optional[str] = None) -> dict:
        """Aggregate KPI stats for the analytics dashboard (Screen 5).

        Args:
            camera_id: Restrict to a single camera, if given.

        Returns:
            Dict with total counts, severity/type/vehicle breakdowns, and
            the top-10 most-violating plate numbers.
        """
        where = "WHERE camera_id = ?" if camera_id else ""
        params: tuple = (camera_id,) if camera_id else ()

        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) AS c FROM violations {where}", params).fetchone()["c"]

            today_where = where + (" AND" if where else "WHERE") + " date(" + self._local() + ") = date('now', '" + f"{DISPLAY_UTC_OFFSET_MINUTES:+d} minutes" + "')"
            today = conn.execute(f"SELECT COUNT(*) AS c FROM violations {today_where}", params).fetchone()["c"]

            month_where = where + (" AND" if where else "WHERE") + " strftime('%Y-%m', " + self._local() + ") = strftime('%Y-%m', 'now', '" + f"{DISPLAY_UTC_OFFSET_MINUTES:+d} minutes" + "')"
            this_month = conn.execute(f"SELECT COUNT(*) AS c FROM violations {month_where}", params).fetchone()["c"]

            severity_rows = conn.execute(
                f"SELECT severity, COUNT(*) AS c FROM violations {where} GROUP BY severity", params
            ).fetchall()
            severity_breakdown = {row["severity"]: row["c"] for row in severity_rows}

            vehicle_rows = conn.execute(
                f"SELECT vehicle_type, COUNT(*) AS c FROM violations {where} GROUP BY vehicle_type", params
            ).fetchall()
            vehicle_breakdown = {row["vehicle_type"]: row["c"] for row in vehicle_rows}

            top_plates_rows = conn.execute(
                f"""SELECT plate_number, COUNT(*) AS c FROM violations
                    {where}{' AND' if where else 'WHERE'} plate_number IS NOT NULL
                    GROUP BY plate_number ORDER BY c DESC LIMIT 10""",
                params,
            ).fetchall()
            top_plates = [{"plate_number": row["plate_number"], "violation_count": row["c"]} for row in top_plates_rows]

            avg_speed_row = conn.execute(f"SELECT AVG(detected_speed) AS avg_s FROM violations {where}", params).fetchone()
            avg_speed = round(avg_speed_row["avg_s"], 1) if avg_speed_row["avg_s"] is not None else 0.0

        return {
            "total_violations": total,
            "violations_today": today,
            "violations_this_month": this_month,
            "severity_breakdown": severity_breakdown,
            "vehicle_type_breakdown": vehicle_breakdown,
            "top_offending_plates": top_plates,
            "average_speed_kmph": avg_speed,
        }

    def record_rto_dispatch(self, violation_id: str, channel: str = "RTO_PORTAL", mode: str = "mock") -> dict:
        """Record that a violation was sent to the RTO and return the audit row.

        The RTO portal integration is a mock: this writes the audit trail and a synthetic reference
        number only. ``mode`` becomes ``"live"`` only when a real integration is wired in.

        Args:
            violation_id: The violation's UUID (must exist).
            channel: Destination channel name.
            mode: ``"mock"`` or ``"live"``.

        Returns:
            The dispatch row as a dict.

        Raises:
            KeyError: If the violation does not exist.
        """
        import uuid
        from datetime import datetime, timezone

        if self.get_violation(violation_id) is None:
            raise KeyError(violation_id)
        row = {
            "id": str(uuid.uuid4()),
            "violation_id": violation_id,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "channel": channel,
            "reference": f"{'MOCK' if mode == 'mock' else 'RTO'}-{uuid.uuid4().hex[:10].upper()}",
            "mode": mode,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO rto_dispatches (id, violation_id, sent_at, channel, reference, mode) VALUES (?,?,?,?,?,?)",
                tuple(row.values()),
            )
        return row

    def get_rto_dispatches(self, violation_id: str) -> List[dict]:
        """Return the RTO dispatch history of one violation, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM rto_dispatches WHERE violation_id = ? ORDER BY sent_at DESC", (violation_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_notice_text(self, violation_id: str, notice_text: str) -> bool:
        """Overwrite the notice with a manually edited version (same storage as a generated notice)."""
        return self.update_notice(violation_id, notice_text)

    def get_session_vehicles(self, session_id: str) -> List[dict]:
        """All vehicles tracked in one processing session, fastest first (vehicles without a speed last).

        Args:
            session_id: Session id the vehicles were saved under.

        Returns:
            List of ``vehicles`` rows as dicts.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM vehicles WHERE session_id = ? ORDER BY max_speed IS NULL, max_speed DESC, track_id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_stats(self, hours: int = 1) -> dict:
        """Headline numbers for the live monitoring panel.

        Args:
            hours: Look-back window.

        Returns:
            Dict with ``violations_last_hour`` (in the window), ``average_speed_kmph`` over vehicles
            seen in the window, and ``fastest`` (``{"speed_kmph", "plate_number"}`` or None).
        """
        cutoff = f"-{int(hours)} hours"
        with self._connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS c FROM violations WHERE datetime(timestamp) >= datetime('now', ?)", (cutoff,)
            ).fetchone()["c"]
            avg = conn.execute(
                "SELECT AVG(avg_speed) AS a FROM vehicles WHERE datetime(last_seen) >= datetime('now', ?)", (cutoff,)
            ).fetchone()["a"]
            fast = conn.execute(
                """SELECT max_speed, plate_number FROM vehicles WHERE max_speed IS NOT NULL
                   AND datetime(last_seen) >= datetime('now', ?) ORDER BY max_speed DESC LIMIT 1""",
                (cutoff,),
            ).fetchone()
        return {
            "violations_last_hour": n,
            "average_speed_kmph": round(avg, 1) if avg is not None else None,
            "fastest": {"speed_kmph": fast["max_speed"], "plate_number": fast["plate_number"]} if fast else None,
        }

    def get_prior_plate_numbers(self, camera_id: Optional[str] = None) -> List[str]:
        """Fetch all previously logged plate numbers, for repeat-offender checks.

        Args:
            camera_id: Restrict to a single camera, if given.

        Returns:
            List of plate number strings (may contain duplicates, which is
            the point — callers count occurrences).
        """
        query = "SELECT plate_number FROM violations WHERE plate_number IS NOT NULL"
        params: list = []
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [row["plate_number"] for row in rows]
