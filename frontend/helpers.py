"""
helpers.py
==========
Pure helper functions for the dashboard (no Streamlit import, so they are unit-tested directly).
"""

from __future__ import annotations

import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import quote

import cv2
import numpy as np

from frontend.states import STATE_NAMES

IST = timezone(timedelta(hours=5, minutes=30))
SEVERITY_EMOJI = {"WARNING": "🟡", "VIOLATION": "🟠", "SEVERE": "🔴", "NORMAL": "🟢"}
SPEED_ZONES = {"School Zone": 25, "Residential": 40, "City Road": 60, "Highway": 100}
CORNER_NAMES = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
Point = Tuple[float, float]


def ist_now() -> datetime:
    """Current time in IST."""
    return datetime.now(IST)


def fmt_local(iso: Optional[str], fmt: str = "%d %b %Y %H:%M:%S") -> str:
    """Format a stored UTC ISO timestamp as IST text.

    Args:
        iso: ISO-8601 string (naive values are treated as UTC), or None.
        fmt: ``strftime`` format.

    Returns:
        Formatted local time, ``"-"`` for empty input, or the raw string if it cannot be parsed.
    """
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime(fmt)


def plate_state(plate: Optional[str]) -> str:
    """State name for a plate such as ``MH12AB1234`` (``"Unknown"`` if unreadable or unrecognised)."""
    code = (plate or "").replace(" ", "")[:2].upper()
    return STATE_NAMES.get(code, "Unknown")


_PLATE_RE = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$")
_UK_RE = re.compile(r"^([A-Z]{2}\d{2})([A-Z]{3})$")  # UK test footage: GX15 OGJ
_DELHI_RE = re.compile(r"^(DL)(\d[A-Z])([A-Z]{2})(\d{4})$")  # Delhi: DL 8C AZ 4521


def format_plate(plate: Optional[str]) -> str:
    """Show a plate with spaces (``MH12AB1234`` -> ``MH 12 AB 1234``); ``UNREADABLE`` when empty.

    Text that does not fit an Indian plate layout is returned unchanged.
    """
    if not plate:
        return "UNREADABLE"
    p = plate.replace(" ", "").upper()
    m = _DELHI_RE.match(p) or _PLATE_RE.match(p) or _UK_RE.match(p)
    return " ".join(m.groups()) if m else plate


def severity_label(severity: Optional[str]) -> str:
    """Emoji + name for a severity band."""
    return f"{SEVERITY_EMOJI.get(severity or '', '⚪')} {severity or 'UNKNOWN'}"


def excess_percent(detected: Optional[float], limit: Optional[float]) -> Optional[float]:
    """How far over the limit (percent), or None if the numbers are missing."""
    if not detected or not limit:
        return None
    return round((detected - limit) / limit * 100, 1)


def default_points(width: int, height: int) -> List[Point]:
    """A starting trapezoid (TL, TR, BR, BL) over the lower-middle of the frame for the wizard to adjust."""
    return [
        (round(width * 0.38), round(height * 0.55)),
        (round(width * 0.62), round(height * 0.55)),
        (round(width * 0.78), round(height * 0.90)),
        (round(width * 0.22), round(height * 0.90)),
    ]


def is_convex_quad(points: Sequence[Point]) -> bool:
    """True if 4 points (in order) form a convex, non-self-intersecting quadrilateral."""
    if len(points) != 4:
        return False
    signs = []
    for i in range(4):
        (x1, y1), (x2, y2), (x3, y3) = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        signs.append(cross)
    return all(c > 0 for c in signs) or all(c < 0 for c in signs)


def draw_calibration(frame: np.ndarray, points: Sequence[Point], active: Optional[int] = None) -> np.ndarray:
    """Return a copy of ``frame`` (BGR) with the calibration quad and numbered corners drawn on it.

    Marker size scales with the frame width so corners stay visible on HD video once the image is shrunk
    to the dashboard width.

    Args:
        frame: BGR frame.
        points: Four (x, y) corners in frame pixels.
        active: Index (0-3) of the corner that the next click will move; it gets an amber ring.
    """
    out = frame.copy()
    w = out.shape[1]
    radius, thick, font = max(6, round(w / 110)), max(2, round(w / 400)), max(0.6, w / 1100)
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, (192, 101, 21), thick)
    for idx, (x, y) in enumerate(points, start=1):
        centre = (int(x), int(y))
        if active is not None and idx - 1 == active:
            cv2.circle(out, centre, radius + max(4, radius // 2), (23, 127, 245), thick)  # amber ring (BGR)
        cv2.circle(out, centre, radius, (47, 47, 211), -1)
        cv2.putText(out, str(idx), (centre[0] + radius + 2, centre[1] - radius - 2), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick)
    return out


def scale_click(click: dict, frame_size: Tuple[int, int]) -> Tuple[int, int]:
    """Convert a click on the *displayed* (possibly shrunk) image into frame pixel coordinates.

    Args:
        click: Dict from ``streamlit_image_coordinates`` with ``x``, ``y`` and the displayed ``width``/``height``.
        frame_size: ``(width, height)`` of the original video frame.

    Returns:
        ``(x, y)`` in frame pixels, clamped to the frame.
    """
    fw, fh = frame_size
    x = click["x"] * fw / max(click.get("width") or fw, 1)
    y = click["y"] * fh / max(click.get("height") or fh, 1)
    return int(min(max(round(x), 0), fw - 1)), int(min(max(round(y), 0), fh - 1))


def first_frame(video_bytes: bytes, suffix: str = ".mp4") -> Optional[np.ndarray]:
    """Decode the first frame of an uploaded video (None if the file cannot be read)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        path = Path(tmp.name)
    try:
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    finally:
        path.unlink(missing_ok=True)


def whatsapp_link(violation: dict, notice_text: Optional[str] = None, max_chars: int = 1200) -> str:
    """Build a ``wa.me`` share link with a short summary (and the start of the notice) pre-filled."""
    head = (
        f"Traffic violation {violation.get('id', '')[:8]}: {format_plate(violation.get('plate_number'))} "
        f"({violation.get('vehicle_type')}) at {violation.get('detected_speed')} km/h in a "
        f"{violation.get('speed_limit')} km/h zone, {violation.get('severity')}, {fmt_local(violation.get('timestamp'))}."
    )
    body = f"{head}\n\n{notice_text}" if notice_text else head
    return "https://wa.me/?text=" + quote(body[:max_chars])
