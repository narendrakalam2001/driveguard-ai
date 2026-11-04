"""
demo_client.py
==============
Drop-in replacement for :class:`frontend.api_client.ApiClient` that answers the dashboard's requests
from a bundled SQLite file of SYNTHETIC violations instead of a running FastAPI server.

It powers the *lite* deployment (Streamlit Community Cloud, no API, no torch). The AI pipeline (YOLO,
tracking, plate OCR) is not available here; everything that only reads the database does work:
violations log, detail view, status changes, notices (fixed template), analytics, insights,
CSV/Excel export and PDF reports. It reuses the real backend modules (``AlertSystem``, ``report_export``,
``insights``, ``ReportGenerator``), none of which import torch.

AI notices and insights are OFF by default. Add ``GEMINI_API_KEY`` to the app's secrets to switch them on
(still grounded by the exact fine/section lookup and the notice validator, and capped per app run so a
public demo cannot burn the whole quota). Without a key, or when the cap is reached, the fixed template is used.

Changes made in the demo (status, edited notices) live in a temp copy of the database, are shared by all
visitors of the running app, and are discarded when the app restarts.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from frontend.api_client import ApiError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DB = PROJECT_ROOT / "data" / "demo" / "demo_violations.db"
MAX_EXPORT_ROWS = 5000
LATEST_VIOLATION_AGE = timedelta(minutes=9)  # the newest demo violation ends up at most this recent (never in the future)

DEMO_DISABLED_MSG = (
    "Video processing is switched off in the lite demo (Streamlit Cloud cannot run the AI models). "
    "Run the full stack locally with `python scripts/run_api.py` to process videos."
)


DEMO_LLM_LIMIT = int(os.getenv("DRIVEGUARD_DEMO_LLM_LIMIT", "60"))  # max Gemini calls per app run (public-demo guard)
_llm_calls = 0


def llm_enabled() -> bool:
    """True when a Gemini key is configured (Streamlit secrets or environment) for the lite demo."""
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def _take_llm_call() -> bool:
    """Reserve one Gemini call; False once the per-run cap is used up."""
    global _llm_calls
    if not llm_enabled() or _llm_calls >= DEMO_LLM_LIMIT:
        return False
    _llm_calls += 1
    return True


def is_demo_mode() -> bool:
    """Return True when the dashboard should use bundled demo data instead of a live API.

    Demo mode is on when ``DRIVEGUARD_DEMO`` is truthy, or when ``DRIVEGUARD_API_URL`` is not set at all
    (the situation on a fresh Streamlit Cloud deployment).
    """
    flag = os.getenv("DRIVEGUARD_DEMO", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    return not os.getenv("DRIVEGUARD_API_URL", "").strip()


def _prepare_database(source: Path) -> Path:
    """Copy the bundled DB to a writable temp file and shift its timestamps by whole days so the data looks current.

    Args:
        source: Bundled demo database.

    Returns:
        Path of the working copy.

    Raises:
        ApiError: If the bundled database is missing.
    """
    if not source.is_file():
        raise ApiError(f"Demo database not found at {source}. Run `python scripts/build_demo_db.py`.")
    target = Path(tempfile.gettempdir()) / f"driveguard_demo_{os.getpid()}.db"
    shutil.copyfile(source, target)
    conn = sqlite3.connect(target)
    try:
        newest = conn.execute("SELECT MAX(timestamp) FROM violations").fetchone()[0]
        if newest:
            # shift by WHOLE DAYS only, so hour-of-day and weekday patterns (rush hours) stay intact
            gap = (datetime.now(timezone.utc) - LATEST_VIOLATION_AGE) - datetime.fromisoformat(newest)
            delta = timedelta(days=gap // timedelta(days=1))
            rows = conn.execute("SELECT id, timestamp, created_at FROM violations").fetchall()
            conn.executemany(
                "UPDATE violations SET timestamp = ?, created_at = ? WHERE id = ?",
                [((datetime.fromisoformat(t) + delta).isoformat(), (datetime.fromisoformat(c) + delta).isoformat(), i) for i, t, c in rows],
            )
        conn.execute("UPDATE cameras SET last_active = ?", (datetime.now(timezone.utc).isoformat(),))
        conn.commit()
    finally:
        conn.close()
    return target


class DemoClient:
    """Serves the subset of the DriveGuard API that the dashboard uses, from local demo data."""

    is_demo = True
    api_key = ""

    def __init__(self, source_db: Path = DEMO_DB) -> None:
        """Create the client and its temp database.

        Args:
            source_db: Bundled demo database to copy.
        """
        from src.alert_system import AlertSystem  # light import: no torch (see module docstring)

        self.base_url = "demo://bundled-data"
        self._system = AlertSystem(db_path=_prepare_database(source_db))
        self._routes: List[tuple[str, re.Pattern[str], Callable[..., Any]]] = []
        self._register_routes()

    # ------------------------------------------------------------------ public API (same as ApiClient)
    def json(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """Handle a JSON endpoint.

        Args:
            method: HTTP method.
            path: Endpoint path starting with ``/``.
            params: Query parameters (``None`` and empty values are ignored).
            **kwargs: ``json=`` body is honoured; other ``requests`` options are ignored.

        Raises:
            ApiError: For unknown routes, missing records, or features the demo does not have.
        """
        clean = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
        return self._dispatch(method.upper(), path, clean, kwargs.get("json") or {})

    def bytes(self, path: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> bytes:
        """Handle a binary endpoint (PDF / CSV / Excel / images); same arguments as :meth:`json`."""
        clean = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
        result = self._dispatch("GET", path, clean, {})
        if not isinstance(result, (bytes, bytearray)):
            raise ApiError(f"GET {path} did not return a file.")
        return bytes(result)

    # ------------------------------------------------------------------ routing
    def _route(self, method: str, pattern: str, handler: Callable[..., Any]) -> None:
        self._routes.append((method, re.compile(f"^{pattern}$"), handler))

    def _dispatch(self, method: str, path: str, params: Dict[str, Any], body: Dict[str, Any]) -> Any:
        for m, regex, handler in self._routes:
            match = regex.match(path)
            if m == method and match:
                return handler(*match.groups(), params=params, body=body)
        raise ApiError(f"{method} {path} is not available in the lite demo.")

    def _register_routes(self) -> None:
        r = self._route
        r("GET", r"/health", self._health)
        r("GET", r"/cameras", lambda **_: {"cameras": self._system.list_cameras()})
        r("GET", r"/sessions", lambda **_: {"sessions": []})
        r("GET", r"/videos", lambda **_: {"videos": []})
        r("GET", r"/sessions/([\w-]+)/vehicles", lambda *_a, **_k: {"vehicles": []})
        r("GET", r"/live-feed", lambda **_: {"alerts": []})  # empty: the UI then falls back to the database
        r("GET", r"/live-stats", self._live_stats)
        r("GET", r"/violations", self._list_violations)
        r("GET", r"/violation/([\w-]+)", self._get_violation)
        r("GET", r"/violation/([\w-]+)/evidence", self._no_evidence)
        r("GET", r"/violation/([\w-]+)/speed-trace", lambda *_a, **_k: self._not_found("No speed trace in the demo data."))
        r("GET", r"/violation/([\w-]+)/dispatches", self._dispatches)
        r("GET", r"/violation/([\w-]+)/notice\.pdf", self._notice_pdf)
        r("PATCH", r"/violation/([\w-]+)/status", self._set_status)
        r("PUT", r"/violation/([\w-]+)/notice", self._edit_notice)
        r("POST", r"/violation/([\w-]+)/send-to-rto", self._send_to_rto)
        r("POST", r"/generate-report", self._generate_report)
        r("GET", r"/analytics", lambda **k: self._system.get_analytics_summary(camera_id=k["params"].get("camera_id")))
        r("GET", r"/analytics/detailed", lambda **k: self._system.get_detailed_analytics(camera_id=k["params"].get("camera_id")))
        r("GET", r"/analytics/insights", self._insights)
        r("GET", r"/export/violations", self._export)
        r("GET", r"/reports/summary\.pdf", self._summary_pdf)
        r("GET", r"/integrations", self._integrations)
        r("POST", r"/process-video", self._disabled)
        r("POST", r"/process-stream", self._disabled)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _not_found(message: str) -> Any:
        raise ApiError(f"404: {message}")

    @staticmethod
    def _disabled(**_: Any) -> Any:
        raise ApiError(DEMO_DISABLED_MSG)

    def _row(self, violation_id: str) -> dict:
        row = self._system.get_violation(violation_id)
        if row is None:
            raise ApiError(f"404: Violation not found: {violation_id}")
        return row

    @staticmethod
    def _filters(p: Dict[str, Any]) -> dict:
        types = [t.strip() for t in str(p["vehicle_type"]).split(",") if t.strip()] if p.get("vehicle_type") else None
        return {
            "camera_id": p.get("camera_id"), "severity": p.get("severity"), "status": p.get("status"), "vehicle_types": types,
            "state_code": p.get("state_code"), "plate_query": p.get("plate"), "date_from": p.get("date_from"), "date_to": p.get("date_to"),
        }

    # ------------------------------------------------------------------ handlers
    def _health(self, **_: Any) -> dict:
        return {"status": "ok", "service": "DriveGuard AI (lite demo)", "version": "lite", "llm_configured": llm_enabled(), "mode": "demo"}

    def _live_stats(self, **_: Any) -> dict:
        return {"vehicles_in_frame": None, "fps": None, "counts_by_class": {}, "vehicles": [], **self._system.get_recent_stats(hours=1)}

    def _list_violations(self, params: Dict[str, Any], **_: Any) -> dict:
        limit, offset = min(int(params.get("limit", 50)), 200), int(params.get("offset", 0))
        filters = self._filters(params)
        rows = self._system.query_violations(limit=limit, offset=offset, **filters)
        return {"count": len(rows), "total": self._system.count_violations(**filters), "violations": rows}

    def _get_violation(self, vid: str, **_: Any) -> dict:
        return self._row(vid)

    def _no_evidence(self, vid: str, **_: Any) -> Any:
        self._row(vid)
        return self._not_found("The demo data has no evidence images.")

    def _dispatches(self, vid: str, **_: Any) -> dict:
        self._row(vid)
        return {"dispatches": self._system.get_rto_dispatches(vid)}

    def _notice_pdf(self, vid: str, **_: Any) -> bytes:
        from src.report_export import notice_to_pdf

        return notice_to_pdf(self._row(vid))

    def _set_status(self, vid: str, body: Dict[str, Any], **_: Any) -> dict:
        self._row(vid)
        status = str(body.get("status", ""))
        if status not in {"PENDING", "RESOLVED", "DISPUTED"}:
            raise ApiError("422: status must be PENDING, RESOLVED or DISPUTED.")
        self._system.update_status(vid, status)
        return {"violation_id": vid, "status": status}

    def _edit_notice(self, vid: str, body: Dict[str, Any], **_: Any) -> dict:
        self._row(vid)
        text = str(body.get("notice_text", "")).strip()
        if not text:
            raise ApiError("422: notice_text must not be empty.")
        self._system.update_notice_text(vid, text)
        return {"violation_id": vid, "notice_text": text}

    def _send_to_rto(self, vid: str, **_: Any) -> dict:
        if not self._row(vid).get("notice_text"):
            raise ApiError("409: Generate a notice before sending the violation to the RTO.")
        dispatch = self._system.record_rto_dispatch(vid, channel="RTO_PORTAL", mode="mock")
        return {**dispatch, "message": "Mock RTO portal: recorded, nothing was delivered to a real RTO."}

    def _generate_report(self, body: Dict[str, Any], **_: Any) -> dict:
        """Write a notice: Gemini when a key is set and the cap allows it, otherwise the fixed template."""
        from src.report_generator import ReportGenerator
        from src.violation_classifier import ViolationRecord

        row = self._row(str(body.get("violation_id", "")))
        limit = row["speed_limit"]
        record = ViolationRecord(
            violation_id=row["id"], track_id=-1, plate_number=row["plate_number"], vehicle_type=row["vehicle_type"],
            detected_speed_kmph=row["detected_speed"], speed_limit_kmph=limit, excess_speed_kmph=row["excess_speed"],
            excess_percent=round(row["excess_speed"] / limit * 100, 1) if limit else 0.0, severity=row["severity"],
            location=row["location"], camera_id=row["camera_id"], timestamp=datetime.fromisoformat(row["timestamp"]),
        )
        repeat = bool(record.plate_number) and self._system.get_prior_plate_numbers(camera_id=row["camera_id"]).count(record.plate_number) > 1
        vehicle_id = f"{record.vehicle_type[:3].upper()}_{row['id'][:6].upper()}"
        generator = ReportGenerator()
        notice, label = None, "fixed template (lite demo, no LLM)"
        if body.get("use_llm", True) and _take_llm_call():
            try:
                notice = generator.generate(record, vehicle_id=vehicle_id, is_repeat_offender=repeat)
                label = notice.model_used
            except Exception:  # noqa: BLE001 - quota, network or validation trouble: degrade to the template
                notice = None
        if notice is None:
            notice = generator.generate_fallback_notice(record, vehicle_id=vehicle_id, is_repeat_offender=repeat)
        self._system.update_notice(row["id"], notice.notice_text)
        return {"violation_id": row["id"], "notice_text": notice.notice_text, "model_used": label, "cited_context": notice.cited_context}

    def _insights(self, params: Dict[str, Any], **_: Any) -> dict:
        from src.insights import generate_insights

        cam = params.get("camera_id")
        want_llm = str(params.get("use_llm", "true")).lower() != "false" and _take_llm_call()
        return generate_insights(self._system.get_analytics_summary(camera_id=cam), self._system.get_detailed_analytics(camera_id=cam), use_llm=want_llm)

    def _export(self, params: Dict[str, Any], **_: Any) -> bytes:
        from src.report_export import violations_to_csv, violations_to_xlsx

        rows = self._system.query_violations(limit=MAX_EXPORT_ROWS, offset=0, **self._filters(params))
        return violations_to_xlsx(rows) if params.get("format") == "xlsx" else violations_to_csv(rows)

    def _summary_pdf(self, params: Dict[str, Any], **_: Any) -> bytes:
        from src.insights import generate_insights
        from src.report_export import summary_report_pdf

        cam = params.get("camera_id")
        summary, detailed = self._system.get_analytics_summary(camera_id=cam), self._system.get_detailed_analytics(camera_id=cam)
        insight = generate_insights(summary, detailed, use_llm=False)
        return summary_report_pdf(summary, detailed, insight["text"] or "", self._system.query_violations(camera_id=cam, severity="SEVERE", limit=10))

    def _integrations(self, **_: Any) -> dict:
        from src.config import GEMINI_MODEL_NAME

        return {"integrations": [
            {"name": "Demo data", "mode": "live", "detail": "1,800 synthetic violations, 8 cameras, 6 months"},
            {"name": "AI pipeline (YOLOv8 + DeepSORT + ANPR)", "mode": "not_configured", "detail": "off in the lite demo; run the full stack locally"},
            {"name": "Gemini LLM", "mode": "live" if llm_enabled() else "not_configured",
             "detail": f"{GEMINI_MODEL_NAME}, max {DEMO_LLM_LIMIT} calls per app run" if llm_enabled() else "no GEMINI_API_KEY in the app secrets; notices use the fixed template"},
            {"name": "RAG knowledge base (ChromaDB)", "mode": "not_configured", "detail": "vector store is not built in the lite demo; exact fine and section lookup is used"},
            {"name": "Vahan database", "mode": "mock", "detail": "no public API access; not implemented"},
            {"name": "RTO portal", "mode": "mock", "detail": "send-to-RTO records an audit row only"},
            {"name": "Police dashboard", "mode": "mock", "detail": "not implemented"},
        ]}
