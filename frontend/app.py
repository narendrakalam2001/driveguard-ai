"""
app.py
======
DriveGuard AI - Traffic Control Dashboard (Streamlit).

A dark, control-room style UI over the FastAPI backend (``serving/api.py``). Six screens:
  1 Live monitoring      2 Upload and analysis    3 Violations log
  4 Violation detail     5 Analytics              6 Reports and export

Run:  streamlit run frontend/app.py      (API URL from DRIVEGUARD_API_URL, default http://localhost:8000)
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # same .env as the API, so DRIVEGUARD_API_KEY / DRIVEGUARD_API_URL match (no-op where python-dotenv or .env is absent)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:  # click-to-place calibration; the wizard falls back to number boxes only if it is missing
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # pragma: no cover
    streamlit_image_coordinates = None

from frontend.api_client import ApiClient, ApiError
from frontend.demo_client import DEMO_DISABLED_MSG, DemoClient, is_demo_mode
from frontend.helpers import (
    CORNER_NAMES, SPEED_ZONES, default_points, draw_calibration, excess_percent, first_frame,
    fmt_local, format_plate, ist_now, is_convex_quad, plate_state, scale_click, severity_label, whatsapp_link,
)
from frontend.states import STATE_NAMES

DEMO = is_demo_mode()  # lite deployment: bundled synthetic data, no API server
DOCS = Path(__file__).resolve().parent.parent / "docs"
BG, PANEL, BLUE, RED, AMBER, GREEN, TEXT = "#0D1B2A", "#13273D", "#1565C0", "#D32F2F", "#F57F17", "#2E7D32", "#FFFFFF"
SEV_COLOR = {"WARNING": AMBER, "VIOLATION": "#EF6C00", "SEVERE": RED, "NORMAL": GREEN}
VEHICLE_TYPES = ["Car", "Truck", "Bus", "Motorcycle", "Auto-rickshaw", "Bicycle", "Van"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="DriveGuard AI - Traffic Control", page_icon="🚦", layout="wide")
st.markdown(
    f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
    html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', sans-serif; }}
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    section[data-testid="stSidebar"] {{ background-color: #0A1420; }}
    .dg-kpi {{ background: {PANEL}; border-radius: 6px; padding: 14px 18px; border-left: 4px solid {BLUE}; }}
    .dg-kpi.alert {{ border-left-color: {RED}; }} .dg-kpi.warn {{ border-left-color: {AMBER}; }} .dg-kpi.ok {{ border-left-color: {GREEN}; }}
    .dg-kpi h3 {{ margin: 0; font-size: .82rem; color: #9FB3C8; font-weight: 500; }}
    .dg-kpi p {{ margin: 4px 0 0; font: 500 1.7rem 'IBM Plex Mono', monospace; color: {TEXT}; }}
    .dg-row {{ background: {PANEL}; border-radius: 4px; padding: 7px 10px; margin-bottom: 5px; font-size: .84rem; border-left: 3px solid {BLUE}; }}
    .dg-row.SEVERE {{ border-left-color: {RED}; }} .dg-row.VIOLATION {{ border-left-color: #EF6C00; }} .dg-row.WARNING {{ border-left-color: {AMBER}; }}
    .dg-clock {{ font: 500 1.05rem 'IBM Plex Mono', monospace; color: #9FB3C8; }}
    .dg-on {{ color: #66BB6A; font-weight: 600; }} .dg-off {{ color: #EF5350; font-weight: 600; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client() -> ApiClient | DemoClient:
    """One client per server process: bundled demo data in lite mode, otherwise the FastAPI backend."""
    if DEMO:
        return DemoClient()
    return ApiClient(os.getenv("DRIVEGUARD_API_URL", "http://localhost:8000"), os.getenv("DRIVEGUARD_API_KEY", ""))


def api(method: str, path: str, params: dict | None = None, quiet: bool = False, **kw):
    """Call the API; on failure show a warning (unless ``quiet``) and return None."""
    try:
        return get_client().json(method, path, params, **kw)
    except ApiError as exc:
        if not quiet:
            st.warning(str(exc))
        return None


def api_bytes(path: str, params: dict | None = None):
    """GET raw bytes; None (with a warning) on failure."""
    try:
        return get_client().bytes(path, params, timeout=60)
    except ApiError as exc:
        st.warning(str(exc))
        return None


def kpi(label: str, value, tone: str = "") -> None:
    st.markdown(f'<div class="dg-kpi {tone}"><h3>{label}</h3><p>{value}</p></div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=TEXT, height=height, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(gridcolor="#1E3550")
    fig.update_yaxes(gridcolor="#1E3550")
    return fig


def goto(screen: str, violation_id: str | None = None) -> None:
    """Button callback: jump to a screen (optionally selecting a violation)."""
    st.session_state["nav"] = screen
    if violation_id:
        st.session_state["selected_violation_id"] = violation_id


def cam_filter() -> str | None:
    """Camera chosen in the top bar (None = all cameras)."""
    cam = st.session_state.get("camera", "All cameras")
    return None if cam == "All cameras" else cam


# ---------------------------------------------------------------------------
# Screen 1 - Live monitoring
# ---------------------------------------------------------------------------
def public_api_url() -> str:
    """API address as seen by the *browser* (used for the live video stream; set DRIVEGUARD_PUBLIC_API_URL when deployed)."""
    return (os.getenv("DRIVEGUARD_PUBLIC_API_URL") or get_client().base_url).rstrip("/")


def live_stream_view(session_id: str) -> None:
    """Show the running detection as live video (an MJPEG stream the browser plays by itself, no polling)."""
    st.markdown(
        f'<img src="{public_api_url()}/live-stream/{session_id}" alt="Live detection" style="width:100%;border-radius:6px;background:#000">',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=900, show_spinner="Loading video...")
def _fetch_video(base_url: str, api_key: str, session_id: str) -> bytes:
    return ApiClient(base_url, api_key).bytes(f"/videos/{session_id}", timeout=120)


def processed_video_player(session_id: str) -> None:
    """Play the finished, annotated video (boxes, type, speed and plate drawn on every frame) with a download button."""
    try:
        client = get_client()
        data = _fetch_video(client.base_url, client.api_key, session_id)
    except ApiError as exc:
        st.info(f"No processed video is stored for this session ({exc}).")
        return
    st.video(data, format="video/mp4")
    st.download_button("Download processed video", data, f"driveguard_{session_id[:8]}.mp4", "video/mp4", key=f"dlv_{session_id}")


@st.cache_data(ttl=900, show_spinner=False)
def _plate_crop(base_url: str, api_key: str, session_id: str, filename: str) -> bytes:
    return ApiClient(base_url, api_key).bytes(f"/sessions/{session_id}/plate-reads/{filename}")


def plate_reads_gallery(session_id: str) -> None:
    """Show each plate that was read next to the crop it came from, so wrong readings are easy to spot."""
    reads = (api("GET", f"/sessions/{session_id}/plate-reads", quiet=True) or {}).get("reads", [])
    if not reads:
        return
    with st.expander(f"Check the {len(reads)} plate(s) that were read", expanded=True):
        st.caption("Compare the text with the picture: a wrong reading here would be a wrong person on a notice.")
        client = get_client()
        cols = st.columns(4)
        for i, r in enumerate(reads[:24]):
            try:
                img = _plate_crop(client.base_url, client.api_key, session_id, r["file"])
            except ApiError:
                continue
            with cols[i % 4]:
                st.image(img, caption=f"Track #{r['track_id']} - {format_plate(r['plate'])} (confidence {r['confidence']:.2f})", use_container_width=True)


def vehicles_table(rows: list, live: bool) -> None:
    """Table of vehicles: type, speed and plate (live frame list, or the whole video's summary)."""
    if not rows:
        st.caption("No vehicles yet.")
        return
    if live:
        df = pd.DataFrame([{
            "Vehicle": f"{r['vehicle_type']} #{r['track_id']}",
            "Speed (km/h)": r["speed_kmph"],
            "Plate": format_plate(r["plate"]) if r["plate"] else "-",
            "Status": severity_label(r["severity"]) if r["stable"] else "measuring...",
        } for r in rows])
    else:
        df = pd.DataFrame([{
            "Vehicle": f"{r['vehicle_type'] or '?'} #{r['track_id']}",
            "Average (km/h)": None if r["avg_speed"] is None else round(r["avg_speed"], 1),
            "Max (km/h)": None if r["max_speed"] is None else round(r["max_speed"], 1),
            "Plate": format_plate(r["plate_number"]) if r["plate_number"] else "-",
            "Violations": r["violation_count"],
        } for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True)


def live_side_panel(session_id: str | None, was_active: bool) -> None:
    """Right-hand panel of the live screen: KPIs, vehicles in frame (or in the whole video), latest alerts."""
    if was_active and session_id:  # the video finished while we were watching: redraw the page with the finished video
        status = api("GET", f"/process-status/{session_id}", quiet=True)
        if status and status["status"] not in ("QUEUED", "PROCESSING"):
            st.rerun()
    stats = api("GET", "/live-stats", {"session_id": session_id}, quiet=True) or {}
    c1, c2 = st.columns(2)
    with c1:
        kpi("Vehicles in frame", stats.get("vehicles_in_frame") if stats.get("vehicles_in_frame") is not None else "-")
    with c2:
        kpi("Violations, last hour", stats.get("violations_last_hour", 0), "alert" if stats.get("violations_last_hour") else "")
    c3, c4 = st.columns(2)
    avg, fast = stats.get("average_speed_kmph"), stats.get("fastest")
    with c3:
        kpi("Average speed", f"{avg} km/h" if avg is not None else "-")
    with c4:
        kpi("Fastest", f"{fast['speed_kmph']:.0f} km/h" if fast else "-", "warn" if fast else "")
    if fast and fast.get("plate_number"):
        st.caption(f"Fastest plate: {format_plate(fast['plate_number'])}")
    if stats.get("fps") and was_active:
        st.caption(f"Processing speed: {stats['fps']} FPS")

    if was_active:
        st.markdown("**Vehicles in frame now**")
        vehicles_table(stats.get("vehicles", []), live=True)
    elif session_id:
        st.markdown("**Vehicles in this video**")
        vehicles_table((api("GET", f"/sessions/{session_id}/vehicles", quiet=True) or {}).get("vehicles", []), live=False)

    st.markdown("**Latest alerts**")
    feed = (api("GET", "/live-feed", {"limit": 8}, quiet=True) or {}).get("alerts", [])
    rows = [{"severity": a.get("severity"), "ts": a.get("timestamp"), "plate": a.get("plate_number"),
             "speed": a.get("detected_speed_kmph"), "limit": a.get("speed_limit_kmph")} for a in feed]
    if not rows:  # the in-memory feed is empty after an API restart: fall back to the database
        for v in (api("GET", "/violations", {"limit": 8, "camera_id": cam_filter()}, quiet=True) or {}).get("violations", []):
            rows.append({"severity": v["severity"], "ts": v["timestamp"], "plate": v["plate_number"],
                         "speed": v["detected_speed"], "limit": v["speed_limit"]})
    if not rows:
        st.caption("No alerts yet.")
    for r in rows:
        st.markdown(
            f'<div class="dg-row {r["severity"]}">{severity_label(r["severity"]).split()[0]} {fmt_local(r["ts"], "%H:%M:%S")} - '
            f'{format_plate(r["plate"])} - {r["speed"]:.0f} km/h (limit {r["limit"]}) - {r["severity"]}</div>',
            unsafe_allow_html=True,
        )


def demo_shot(name: str, caption: str) -> None:
    """Show a recorded screenshot from docs/screenshots (lite mode replaces the live AI screens with these)."""
    path = DOCS / "screenshots" / name
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)


def screen_live_demo() -> None:
    """Live monitoring in the lite demo: a recording of the real screen plus live numbers from the demo database."""
    left, right = st.columns([2, 1])
    with left:
        st.caption("Recorded from the full system. Live detection needs the AI backend, which the lite demo does not run.")
        gif = DOCS / "gifs" / "User_Portal.gif"
        if gif.is_file():
            st.image(str(gif), use_container_width=True)
        else:
            demo_shot("live_monitoring.png", "Live monitoring: boxes, speeds and plates drawn on the video")
    with right:
        live_side_panel(None, False)


def screen_live() -> None:
    if DEMO:
        return screen_live_demo()
    sessions = (api("GET", "/sessions") or {}).get("sessions", [])
    saved = {v["session_id"]: v for v in (api("GET", "/videos", quiet=True) or {}).get("videos", [])}
    entries: dict = {}
    for s in sorted(sessions, key=lambda x: x.get("status") not in ("QUEUED", "PROCESSING")):
        label = f"{s.get('camera_id', '?')} - {s['status'].lower()} ({s['session_id'][:8]})"
        entries[label] = (s["session_id"], s["status"], s["session_id"] in saved or bool(s.get("video_ready")))
    for sid, v in saved.items():  # videos that survived an API restart
        if sid not in {e[0] for e in entries.values()}:
            entries[f"{v.get('camera_id') or 'saved video'} - saved ({sid[:8]})"] = (sid, "COMPLETE", True)

    choice = st.selectbox("Session to watch", list(entries) or ["(none)"], key="live_session")
    sid, status, has_video = entries.get(choice, (None, None, False))
    active = status in ("QUEUED", "PROCESSING")
    left, right = st.columns([2, 1])
    with left:
        if active:
            st.caption("Live detection (plays as fast as the models can process; the full-speed video appears when it finishes)" if status == "PROCESSING" else "Waiting for the previous video to finish...")
            live_stream_view(sid)
        elif has_video:
            processed_video_player(sid)
        else:
            st.info("No video or stream is being processed. Start one from 'Upload and analysis' or POST /process-stream.")
    with right:
        st.fragment(run_every=1.0 if active else None)(lambda: live_side_panel(sid, active))()


# ---------------------------------------------------------------------------
# Screen 2 - Upload and analysis (4-step wizard)
# ---------------------------------------------------------------------------
PROCESSING_MODES = {"Accurate: every frame (slowest)": 1, "Balanced: every 2nd frame": 2, "Fast: every 3rd frame": 3}
STEPS = ["Upload video", "Mark the road", "Lane and speed limit", "Confirm and run"]


def set_points(points) -> None:
    """Store the 4 corners in the number-box widget state (single source of truth for the wizard)."""
    for i, (x, y) in enumerate(points):
        st.session_state[f"px{i}"], st.session_state[f"py{i}"] = int(x), int(y)


def current_points(w: dict) -> list:
    """The 4 corners as currently held by the number boxes (falls back to the wizard's stored points)."""
    return [(st.session_state.get(f"px{i}", w["points"][i][0]), st.session_state.get(f"py{i}", w["points"][i][1])) for i in range(4)]


def wizard_state() -> dict:
    return st.session_state.setdefault("wiz", {"step": 0, "points": None, "session_id": None})


def progress_panel(session_id: str) -> None:
    """Progress bar for a running video; redraws the whole page with the results when it finishes."""
    data = api("GET", f"/process-status/{session_id}", quiet=True)
    if not data:
        st.warning("Could not read the processing status.")
        return
    if data["status"] in ("QUEUED", "PROCESSING"):
        progress = data.get("progress")
        label = "Waiting for the previous video to finish..." if data["status"] == "QUEUED" else "Detecting vehicles, tracking and measuring speed..."
        st.progress(progress or 0.02, text=f"{label} {int((progress or 0) * 100)}%")
    else:
        st.rerun()


def results_panel(session_id: str, data: dict | None) -> None:
    """Summary, the full processed video (with detections drawn on) and the per-vehicle table."""
    if not data:
        st.warning("Could not read the processing status.")
        return
    if data["status"] == "FAILED":
        st.error(f"Processing failed: {data.get('detail')}")
        return
    st.progress(1.0, text="Processing complete")
    s = data.get("summary") or {}
    total, viol = s.get("total_vehicles", 0), s.get("total_violations", 0)
    c = st.columns(4)
    with c[0]:
        kpi("Total vehicles", f"{total:,}")
    with c[1]:
        kpi("Violations", f"{viol:,} ({viol / total * 100:.1f}%)" if total else "0", "alert" if viol else "ok")
    with c[2]:
        kpi("Average speed", f"{s.get('average_speed_kmph', 0)} km/h")
    with c[3]:
        kpi("Max speed", f"{s.get('max_speed_kmph', 0)} km/h")
    if s.get("max_speed_plate"):
        st.caption(f"Fastest vehicle: {format_plate(s['max_speed_plate'])}")
    if "plates_read" in s and total:
        st.caption(
            f"Number plates: {s['plates_read']} read of {total} vehicles - {s['plate_attempts']} attempts, plate found {s['plate_boxes_found']} times "
            f"({s.get('plates_rejected_format', 0)} failed the format check, {s.get('plates_rejected_confidence', 0)} had low OCR confidence), "
            f"{s['vehicles_too_small_for_plate']} vehicles too small or far away to try. "
            f"Settings in force: format {s.get('plate_region') or '?'}, min confidence {s.get('plate_min_confidence', '?')}."
        )
        fmt, low = s.get("plates_rejected_format", 0), s.get("plates_rejected_confidence", 0)
        if s["plates_read"] == 0 and s["plate_boxes_found"] > 0:
            region = s.get("plate_region") or "IN"
            if fmt >= low:
                st.warning(
                    f"The plate detector found plates {s['plate_boxes_found']} times, but {fmt} readings failed the **{region}** plate format check, "
                    "so nothing was logged. Open `evidence_frames/plate_debug/<session id>/`: the file names show what the OCR read. "
                    "If the footage is not Indian, set `PLATE_REGION=UK` (or `ANY`) in `.env` and restart the API (the setting is read at start-up). "
                    "If it is Indian footage, the OCR text in those file names shows what it misreads."
                )
            else:
                st.warning(
                    f"Plates were found and the text looked valid, but {low} readings had OCR confidence below {s.get('plate_min_confidence', '?')}. "
                    "Lower `PLATE_CONFIDENCE_THRESHOLD` in `.env` (for example 0.2) and restart the API. The crops are in "
                    "`evidence_frames/plate_debug/<session id>/` with the confidence in the file name."
                )
        elif s["plates_read"] == 0:
            st.warning(
                "No number plate could be read in this video. A plate needs roughly 100 px of width to be readable; in footage like this, "
                "vehicles are usually too small or far away, or seen from an angle. Try a closer, higher-resolution camera view. "
                "To test the models on one close-up photo run `python scripts/plate_check.py <image>`."
            )
    if data.get("video_ready"):
        st.subheader("Processed video")
        if not data.get("video_playable", True):
            st.warning("The video was saved in a format browsers may not play. Run `pip install imageio-ffmpeg` and process it again.")
        processed_video_player(session_id)
    st.subheader("Vehicles in this video")
    vehicles_table((api("GET", f"/sessions/{session_id}/vehicles", quiet=True) or {}).get("vehicles", []), live=False)
    plate_reads_gallery(session_id)
    st.button("Open violations log", on_click=goto, args=("3. Violations log",))


def screen_upload_demo() -> None:
    """Upload screen in the lite demo: explain the 4-step flow with real screenshots instead of running it."""
    st.info(DEMO_DISABLED_MSG)
    st.markdown("**How video analysis works in the full system**: upload a clip, mark the road, set the lane width and speed limit, "
                "then the models detect, track and measure every vehicle and log the violations you see in the other screens.")
    shots = [("upload_&_analysis.png", "1. Upload the video"), ("mark_the_road.png", "2. Mark the road (4 corners)"),
             ("lane_&_speed_limit.png", "3. Lane width and speed limit"), ("processing_complete.png", "4. Results and violations")]
    for row in (shots[:2], shots[2:]):
        for col, (name, cap) in zip(st.columns(2), row):
            with col:
                demo_shot(name, cap)
    st.code("git clone https://github.com/narendrakalam2001/driveguard-ai\ncd driveguard-ai && pip install -r requirements.txt\n"
            "python scripts/run_api.py        # terminal 1\npython scripts/run_dashboard.py  # terminal 2", language="bash")


def screen_upload() -> None:
    if DEMO:
        return screen_upload_demo()
    w = wizard_state()
    st.caption("  ›  ".join(f"**{n}**" if i == w["step"] else n for i, n in enumerate(STEPS)))
    step = w["step"]

    if step == 0:
        up = st.file_uploader("Drop a traffic video here (MP4, AVI or MOV, up to 500 MB)", type=["mp4", "avi", "mov"])
        if up is not None:
            frame = first_frame(up.getvalue(), Path(up.name).suffix or ".mp4")
            if frame is None:
                st.error("This video could not be decoded. Try re-encoding it as H.264 MP4.")
            else:
                h, wd = frame.shape[:2]
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"{up.name} - {wd}x{h}", use_container_width=True)
                if st.button("Next: mark the road", type="primary"):
                    w.update(step=1, name=up.name, data=up.getvalue(), size=(wd, h), frame=frame, points=default_points(wd, h),
                             active=0, last_click=None, epoch=w.get("epoch", 0) + 1)
                    set_points(w["points"])
                    st.rerun()

    elif step == 1:
        wd, h = w["size"]
        pts = current_points(w)
        active = w.setdefault("active", 0)
        st.markdown("Mark **4 corners of a rectangular stretch of road** whose real width and length you know "
                    "(lane markings work well). **Click on the image** to place each corner.")
        # no key on purpose: when a click auto-advances ``active`` the index changes and Streamlit rebuilds the radio at the new value
        picked = st.radio("Corner to place next", [f"{i + 1} {n}" for i, n in enumerate(CORNER_NAMES)], index=active, horizontal=True,
                          help="The next corner is selected automatically after every click. Pick another one here to redo it.")
        active = w["active"] = int(picked.split()[0]) - 1
        disp_w = min(wd, 900)
        shown = cv2.resize(draw_calibration(w["frame"], pts, active), (disp_w, round(h * disp_w / wd)))
        shown = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        if streamlit_image_coordinates is not None:
            click = streamlit_image_coordinates(shown, width=disp_w, key=f"calib_click_{w.get('epoch', 0)}")  # new key per visit: no stale click
            stamp = (click or {}).get("unix_time", (click or {}).get("x"))
            if click and stamp != w.get("last_click"):
                w["last_click"] = stamp
                x, y = scale_click(click, (wd, h))
                set_points([(x, y) if i == active else p for i, p in enumerate(pts)])
                w["active"] = (active + 1) % 4
                st.rerun()
        else:
            st.image(shown, use_container_width=True)
            st.info("Install `streamlit-image-coordinates` to place corners by clicking. You can still use the number boxes below.")

        c = st.columns(4)
        new_pts = []
        for i, name in enumerate(CORNER_NAMES):
            with c[i]:
                x = st.number_input(f"{i + 1} {name} x", 0, wd, int(pts[i][0]), key=f"px{i}")
                y = st.number_input(f"{i + 1} {name} y", 0, h, int(pts[i][1]), key=f"py{i}")
                new_pts.append((float(x), float(y)))
        w["points"] = new_pts
        ok = is_convex_quad(new_pts)
        if not ok:
            st.error("The corners cross over or are in the wrong order. Use top-left, top-right, bottom-right, bottom-left.")
        b1, b2, b3 = st.columns([1, 1, 2])
        b1.button("Back", on_click=lambda: w.update(step=0))
        b2.button("Reset corners", on_click=lambda: (set_points(default_points(wd, h)), w.update(active=0, last_click=None, epoch=w.get("epoch", 0) + 1)))
        if b3.button("Next: lane and speed limit", type="primary", disabled=not ok):
            w["step"] = 2
            st.rerun()

    elif step == 2:
        c1, c2 = st.columns(2)
        with c1:
            w["lane_width"] = st.number_input("Width of the marked stretch (m)", 1.0, 20.0, w.get("lane_width", 3.5), 0.1, help="A standard Indian lane is 3.5 m.")
            w["lane_depth"] = st.number_input("Length of the marked stretch (m)", 2.0, 200.0, w.get("lane_depth", 14.0), 0.5,
                                              help="Measure it (dashed lane markings are usually 3 m with 6 m gaps).")
            st.warning("Every speed scales with this length: a wrong value gives wrong speeds by the same percentage.")
        with c2:
            zone = st.selectbox("Zone type", [*SPEED_ZONES, "Custom"], index=2)
            default_limit = SPEED_ZONES.get(zone, 60)
            w["limit"] = st.number_input("Speed limit (km/h)", 5, 140, default_limit if zone != "Custom" else w.get("limit", 60), key=f"lim_{zone}")
            w["camera_id"] = st.text_input("Camera / zone ID", w.get("camera_id", "CAM-UPLOAD-01"))
            w["anpr"] = st.checkbox("Read number plates (slower)", w.get("anpr", True))
            modes = list(PROCESSING_MODES)
            current = next((i for i, v in enumerate(PROCESSING_MODES.values()) if v == w.get("stride", 1)), 0)
            w["stride"] = PROCESSING_MODES[st.selectbox("Processing mode", modes, index=current,
                                                         help="CPU processing is slow. Skipping frames makes it faster; speeds stay correct because they use real timestamps.")]
        b1, b2 = st.columns(2)
        b1.button("Back", on_click=lambda: w.update(step=1, last_click=None, epoch=w.get("epoch", 0) + 1))
        if b2.button("Next: confirm", type="primary"):
            w["step"] = 3
            st.rerun()

    elif step == 3:
        st.table(pd.DataFrame({
            "Setting": ["Video", "Camera", "Marked stretch", "Speed limit", "Plate reading", "Processing"],
            "Value": [w["name"], w["camera_id"], f"{w['lane_width']} m wide x {w['lane_depth']} m long", f"{w['limit']} km/h",
                      "On" if w["anpr"] else "Off", f"every {w.get('stride', 1)} frame(s)"],
        }))
        if not w.get("session_id"):
            b1, b2 = st.columns(2)
            b1.button("Back", on_click=lambda: w.update(step=2))
            if b2.button("Confirm and process video", type="primary"):
                import json

                pts = [{"x": x, "y": y} for x, y in w["points"]]
                res = api("POST", "/process-video", files={"file": (w["name"], w["data"], "video/mp4")}, timeout=300, data={
                    "camera_id": w["camera_id"], "speed_limit_kmph": w["limit"], "lane_width_m": w["lane_width"],
                    "lane_depth_m": w["lane_depth"], "calibration_points": json.dumps(pts), "enable_plate_recognition": w["anpr"],
                    "frame_stride": w.get("stride", 1)})
                if res:
                    w["session_id"] = res["session_id"]
                    st.rerun()
        else:
            sid = w["session_id"]
            data = api("GET", f"/process-status/{sid}", quiet=True)
            if data and data["status"] in ("QUEUED", "PROCESSING"):
                live_stream_view(sid)  # watch the detection happen while it is processed
                st.fragment(run_every=1.5)(lambda: progress_panel(sid))()
            else:
                results_panel(sid, data)
            st.button("Analyse another video", on_click=lambda: st.session_state.pop("wiz", None))


# ---------------------------------------------------------------------------
# Screen 3 - Violations log
# ---------------------------------------------------------------------------
PAGE_SIZE = 50


def filter_params(prefix: str) -> dict:
    """Render the filter bar and return API query parameters."""
    c = st.columns([2, 1, 1, 2, 1, 1])
    with c[0]:
        rng = st.date_input("Date range (IST)", value=(), key=f"{prefix}_dates")
    with c[1]:
        sev = st.selectbox("Severity", ["All", "WARNING", "VIOLATION", "SEVERE"], key=f"{prefix}_sev")
    with c[2]:
        status = st.selectbox("Status", ["All", "PENDING", "RESOLVED", "DISPUTED"], key=f"{prefix}_status")
    with c[3]:
        vt = st.multiselect("Vehicle type", VEHICLE_TYPES, key=f"{prefix}_vt")
    with c[4]:
        state = st.selectbox("State", ["All", *sorted(STATE_NAMES)], key=f"{prefix}_state")
    with c[5]:
        plate = st.text_input("Plate contains", key=f"{prefix}_plate")
    params = {"camera_id": cam_filter(), "severity": None if sev == "All" else sev, "status": None if status == "All" else status,
              "vehicle_type": ",".join(vt), "state_code": None if state == "All" else state, "plate": plate.replace(" ", "") or None}
    if isinstance(rng, (tuple, list)) and rng:
        params["date_from"] = rng[0].isoformat()
        params["date_to"] = (rng[1] if len(rng) > 1 else rng[0]).isoformat()
    return params


def screen_log() -> None:
    params = filter_params("log")
    st.caption("Violation type: speeding (red-light, wrong-lane and overloading detection are on the roadmap).")
    first = api("GET", "/violations", {**params, "limit": 1}) or {"total": 0}
    total = first.get("total", 0)
    pages = max(1, -(-total // PAGE_SIZE))
    page = st.number_input(f"Page (of {pages}) - {total:,} violations", 1, pages, 1)
    rows = (api("GET", "/violations", {**params, "limit": PAGE_SIZE, "offset": (page - 1) * PAGE_SIZE}) or {}).get("violations", [])
    if not rows:
        st.info("No violations match these filters.")
        return
    df = pd.DataFrame([{
        "ID": r["id"][:8], "Time (IST)": fmt_local(r["timestamp"]), "Plate": format_plate(r["plate_number"]), "Vehicle": r["vehicle_type"],
        "Speed": r["detected_speed"], "Limit": r["speed_limit"], "Severity": severity_label(r["severity"]), "Status": r["status"],
        "Notice": "yes" if r["notice_generated"] else "no"} for r in rows])
    event = st.dataframe(df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
    sel = event.selection.rows if event and event.selection else []
    if sel:
        row = rows[sel[0]]
        st.markdown(f"Selected **{format_plate(row['plate_number'])}** - {row['detected_speed']:.0f} km/h - {row['severity']}")
        a = st.columns(4)
        a[0].button("View evidence", on_click=goto, args=("4. Violation detail", row["id"]), use_container_width=True)
        if a[1].button("Generate notice", use_container_width=True):
            if api("POST", "/generate-report", json={"violation_id": row["id"], "use_llm": True}, timeout=90):
                st.toast("Notice generated"); st.rerun()
        if a[2].button("Mark resolved", use_container_width=True):
            if api("PATCH", f"/violation/{row['id']}/status", json={"status": "RESOLVED"}):
                st.toast("Marked resolved"); st.rerun()
        if a[3].button("Send to RTO", use_container_width=True):
            res = api("POST", f"/violation/{row['id']}/send-to-rto")
            if res:
                st.toast(res["message"])
    else:
        st.caption("Select a row to view evidence, generate a notice, resolve it or send it to the RTO.")
    export_block("log", params)


def export_block(prefix: str, params: dict) -> None:
    st.markdown("**Export the filtered violations**")
    if st.button("Prepare CSV and Excel files", key=f"{prefix}_prep"):
        st.session_state[f"{prefix}_csv"] = api_bytes("/export/violations", {**params, "format": "csv"})
        st.session_state[f"{prefix}_xlsx"] = api_bytes("/export/violations", {**params, "format": "xlsx"})
    c = st.columns(2)
    if st.session_state.get(f"{prefix}_csv"):
        c[0].download_button("Download CSV", st.session_state[f"{prefix}_csv"], "driveguard_violations.csv", "text/csv", key=f"{prefix}_dl_csv")
    if st.session_state.get(f"{prefix}_xlsx"):
        c[1].download_button("Download Excel", st.session_state[f"{prefix}_xlsx"], "driveguard_violations.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{prefix}_dl_xlsx")


# ---------------------------------------------------------------------------
# Screen 4 - Violation detail
# ---------------------------------------------------------------------------
def screen_detail() -> None:
    recent = (api("GET", "/violations", {"limit": 25, "camera_id": cam_filter()}, quiet=True) or {}).get("violations", [])
    current = st.session_state.get("selected_violation_id", "")
    options = [r["id"] for r in recent]
    if current and current not in options:
        options.insert(0, current)
    if not options:
        st.info("No violations yet. Process a video first.")
        return
    vid = st.selectbox("Violation", options, index=options.index(current) if current in options else 0,
                       format_func=lambda i: next((f"{i[:8]} - {format_plate(r['plate_number'])} - {r['detected_speed']:.0f} km/h" for r in recent if r["id"] == i), i))
    v = api("GET", f"/violation/{vid}")
    if not v:
        return
    over = excess_percent(v["detected_speed"], v["speed_limit"])
    left, right = st.columns(2)
    with left:
        st.subheader("Evidence")
        for kind, cap in (("frame", "Frame at the moment of violation"), ("plate", "Number plate (zoomed)")):
            try:
                st.image(get_client().bytes(f"/violation/{vid}/evidence", {"kind": kind}), caption=cap, use_container_width=True)
            except ApiError:
                st.caption(f"No {'evidence frame' if kind == 'frame' else 'plate crop'} stored (only WARNING-level or demo records lack them).")
    with right:
        st.subheader("Vehicle")
        st.markdown(f"- **Plate:** {format_plate(v['plate_number'])}\n- **State:** {plate_state(v['plate_number'])}\n- **Type:** {v['vehicle_type']}\n"
                    f"- **Plate read confidence:** {v['plate_confidence'] * 100:.1f}%" if v.get("plate_confidence") else
                    f"- **Plate:** {format_plate(v['plate_number'])}\n- **State:** {plate_state(v['plate_number'])}\n- **Type:** {v['vehicle_type']}\n- **Plate read confidence:** not recorded")
        st.subheader("Violation")
        st.markdown(f"- **Speed:** {v['detected_speed']:.1f} km/h in a {v['speed_limit']} km/h zone\n- **Excess:** {v['excess_speed']:.1f} km/h ({over}% over)\n"
                    f"- **Severity:** {severity_label(v['severity'])}\n- **Location / camera:** {v['location']} ({v['camera_id']})\n"
                    f"- **Time:** {fmt_local(v['timestamp'])} IST\n- **Status:** {v['status']}")
        s = st.columns(3)
        for col, (label, status) in zip(s, [("Mark resolved", "RESOLVED"), ("Mark disputed", "DISPUTED"), ("Reopen", "PENDING")]):
            if col.button(label, key=f"st_{status}", use_container_width=True, disabled=v["status"] == status):
                if api("PATCH", f"/violation/{vid}/status", json={"status": status}):
                    st.rerun()
    st.divider()
    speed_graph(vid, v)
    st.divider()
    notice_editor(vid, v)


def speed_graph(vid: str, v: dict) -> None:
    st.subheader("Speed over time")
    trace = api("GET", f"/violation/{vid}/speed-trace", quiet=True)
    if not trace:
        st.caption("No speed trace was recorded for this violation (demo data, or recorded before traces existed).")
        return
    fps = trace.get("fps") or 25
    ys = trace["speeds_kmph"]
    fig = go.Figure(go.Scatter(x=[i / fps for i in range(len(ys))], y=ys, mode="lines", name="Speed", line=dict(color=BLUE, width=3)))
    lim = v["speed_limit"]
    for mult, name, color in ((1.0, "Limit", GREEN), (1.2, "Violation (+20%)", "#EF6C00"), (1.5, "Severe (+50%)", RED)):
        fig.add_hline(y=lim * mult, line_dash="dash", line_color=color, annotation_text=name, annotation_position="top left")
    fig.update_layout(xaxis_title="Seconds in view", yaxis_title="km/h")
    st.plotly_chart(style(fig, 300), use_container_width=True)


def notice_editor(vid: str, v: dict) -> None:
    st.subheader("Notice")
    text = st.text_area("Notice text (editable)", value=v.get("notice_text") or "", height=260, key=f"notice_{vid}",
                        placeholder="No notice yet. Generate one below.")
    c = st.columns(5)
    if c[0].button("Generate with AI", use_container_width=True):
        res = api("POST", "/generate-report", json={"violation_id": vid, "use_llm": True}, timeout=90)
        if res:
            st.session_state.pop(f"notice_{vid}", None)
            st.toast(f"Written by {res['model_used']}"); st.rerun()
    if c[1].button("Use fixed template", use_container_width=True):
        if api("POST", "/generate-report", json={"violation_id": vid, "use_llm": False}):
            st.session_state.pop(f"notice_{vid}", None); st.rerun()
    if c[2].button("Save my edits", use_container_width=True, disabled=not text.strip()):
        if api("PUT", f"/violation/{vid}/notice", json={"notice_text": text}):
            st.toast("Notice saved")
    pdf = api_bytes(f"/violation/{vid}/notice.pdf")
    if pdf:
        c[3].download_button("Download PDF", pdf, f"notice_{vid[:8]}.pdf", "application/pdf", use_container_width=True)
    c[4].link_button("Share on WhatsApp", whatsapp_link(v, text or None), use_container_width=True)
    if st.button("Send to RTO (mock)", disabled=not v.get("notice_text")):
        res = api("POST", f"/violation/{vid}/send-to-rto")
        if res:
            st.success(f"{res['message']} Reference {res['reference']}.")
    for d in (api("GET", f"/violation/{vid}/dispatches", quiet=True) or {}).get("dispatches", []):
        st.caption(f"Sent {fmt_local(d['sent_at'])} via {d['channel']} ({d['mode']}), reference {d['reference']}")


# ---------------------------------------------------------------------------
# Screen 5 - Analytics
# ---------------------------------------------------------------------------
def screen_analytics() -> None:
    cam = cam_filter()
    s = api("GET", "/analytics", {"camera_id": cam})
    d = api("GET", "/analytics/detailed", {"camera_id": cam})
    if not s or not d:
        return
    peak = d.get("peak_window")
    k = st.columns(4)
    with k[0]: kpi("Violations today", f"{s['violations_today']:,}", "alert" if s["violations_today"] else "")
    with k[1]: kpi("This month", f"{s['violations_this_month']:,}")
    with k[2]: kpi("Most common", d["most_common_severity"] or "-")
    with k[3]: kpi("Peak hours", f"{peak['start_hour']:02d}:00 - {peak['end_hour']:02d}:00" if peak else "-", "warn" if peak else "")
    if not s["total_violations"]:
        st.info("No violations logged yet.")
        return

    st.subheader("Insights")
    c1, c2 = st.columns([4, 1])
    with c2:
        plain = st.checkbox("Without AI", help="Write the insights from the statistics only (no LLM call).")
    ins = api("GET", "/analytics/insights", {"camera_id": cam, "use_llm": "false" if plain else "true"}, timeout=60)
    if ins:
        c1.info(ins["text"])
        c1.caption(f"Written by {ins['model']}" if ins["source"] == "llm" else "Generated directly from the statistics.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Violations by hour (IST)**")
        st.plotly_chart(style(px.line(x=list(range(24)), y=d["hourly"], markers=True, labels={"x": "Hour", "y": "Violations"}).update_traces(line_color=BLUE)), use_container_width=True)
    with b:
        st.markdown("**Severity**")
        sev = s["severity_breakdown"]
        st.plotly_chart(style(go.Figure(go.Pie(labels=list(sev), values=list(sev.values()), hole=.55, marker=dict(colors=[SEV_COLOR.get(x, "#888") for x in sev])))), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Vehicle types**")
        vt = s["vehicle_type_breakdown"]
        st.plotly_chart(style(px.bar(x=list(vt), y=list(vt.values()), labels={"x": "", "y": "Violations"}).update_traces(marker_color=BLUE)), use_container_width=True)
    with b:
        st.markdown("**Speed distribution (km/h)**")
        h = d["speed_histogram"]
        st.plotly_chart(style(px.bar(x=[f"{r['from_kmph']}-{r['to_kmph']}" for r in h], y=[r["count"] for r in h], labels={"x": "Speed", "y": "Violations"}).update_traces(marker_color=AMBER)), use_container_width=True)
    st.markdown("**Day and hour heatmap**")
    st.plotly_chart(style(px.imshow(d["heatmap"], x=[f"{i:02d}" for i in range(24)], y=DAYS, aspect="auto", color_continuous_scale="Blues", labels={"x": "Hour", "y": "", "color": "Violations"}), 300), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Monthly trend**")
        m = d["monthly_trend"]
        st.plotly_chart(style(px.bar(x=[r["month"] for r in m], y=[r["count"] for r in m], labels={"x": "", "y": "Violations"}).update_traces(marker_color=BLUE)), use_container_width=True)
    with b:
        st.markdown("**By state (plate prefix)**")
        sb = d["state_breakdown"]
        if sb:
            st.plotly_chart(style(px.bar(x=[STATE_NAMES.get(k, k) for k in sb], y=list(sb.values()), labels={"x": "", "y": "Violations"}).update_traces(marker_color=GREEN)), use_container_width=True)
        else:
            st.caption("No readable plates yet.")
    st.markdown("**Top 10 plates**")
    if s["top_offending_plates"]:
        st.dataframe(pd.DataFrame(s["top_offending_plates"]).assign(plate_number=lambda x: x.plate_number.map(format_plate)), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Screen 6 - Reports and export
# ---------------------------------------------------------------------------
def screen_reports() -> None:
    st.subheader("Summary report (PDF)")
    st.caption("Executive summary, hourly chart, top plates, most serious cases and recommendations.")
    ai = st.checkbox("Let AI word the recommendations (uses LLM quota)")
    if st.button("Build summary report", type="primary"):
        st.session_state["summary_pdf"] = api_bytes("/reports/summary.pdf", {"camera_id": cam_filter(), "use_llm": "true" if ai else "false"})
    if st.session_state.get("summary_pdf"):
        st.download_button("Download summary report", st.session_state["summary_pdf"], f"driveguard_summary_{date.today():%Y%m%d}.pdf", "application/pdf")
    st.divider()
    st.subheader("Export violation data")
    export_block("rep", filter_params("rep"))
    st.divider()
    st.subheader("Scheduled reports")
    st.markdown("The dashboard does not run background jobs. Schedule the report endpoint with cron or a GitHub Actions workflow:")
    st.code("# daily 07:00 IST (01:30 UTC)\n30 1 * * *  curl -s -o /reports/daily.pdf \"$DRIVEGUARD_API_URL/reports/summary.pdf\"", language="bash")
    st.divider()
    st.subheader("Integrations")
    for i in (api("GET", "/integrations") or {}).get("integrations", []):
        icon = {"live": "✅", "mock": "⚠️", "not_configured": "⛔"}.get(i["mode"], "⚪")
        st.markdown(f"{icon} **{i['name']}** - {i['mode'].replace('_', ' ')} ({i['detail']})")


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------
SCREENS = {
    "1. Live monitoring": screen_live, "2. Upload and analysis": screen_upload, "3. Violations log": screen_log,
    "4. Violation detail": screen_detail, "5. Analytics": screen_analytics, "6. Reports and export": screen_reports,
}


def top_bar() -> None:
    cams = [c["camera_id"] for c in (api("GET", "/cameras", quiet=True) or {}).get("cameras", [])]
    health = api("GET", "/health", quiet=True)
    c = st.columns([3, 2, 2, 1])
    c[0].markdown("## 🚦 DriveGuard AI")
    c[1].selectbox("Camera", ["All cameras", *cams], key="camera", label_visibility="collapsed")
    c[2].markdown(f'<div class="dg-clock">{ist_now():%d %b %Y  %H:%M:%S} IST</div>', unsafe_allow_html=True)
    c[3].markdown('<span class="dg-on">🟢 Active</span>' if health else '<span class="dg-off">🔴 Offline</span>', unsafe_allow_html=True)
    if not health:
        st.error(f"The API is not reachable at {get_client().base_url}. Start it with `python scripts/run_api.py`.")


def main() -> None:
    st.sidebar.title("Navigate")
    st.sidebar.radio("Screen", list(SCREENS), key="nav", label_visibility="collapsed")
    top_bar()
    if DEMO:
        st.info("Lite demo: this deployment shows **synthetic data** (no real vehicles or people) and runs without the AI backend. "
                "Violations, notices, analytics, exports and reports are live; video processing is not. Changes you make reset when the app restarts.")
    SCREENS[st.session_state["nav"]]()


if __name__ == "__main__":
    main()
