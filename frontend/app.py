"""
VortexVision Streamlit frontend.
Live feed viewer, anomaly event log, NL query, and system monitoring.
"""
import base64
import logging
import os
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import streamlit as st

logger = logging.getLogger(__name__)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
# WS_BASE: in production the browser connects to the public hostname, not the internal service name.
# VORTEX_PUBLIC_HOST allows overriding for production deployments.
_public_host = os.getenv("VORTEX_PUBLIC_HOST", "")
if _public_host:
    WS_BASE = f"ws://{_public_host}" if not _public_host.startswith("https") else f"wss://{_public_host.replace('https://', '')}"
else:
    WS_BASE = API_BASE.replace("http://", "ws://").replace("https://", "wss://")

st.set_page_config(
    page_title="VortexVision",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "frame_queue" not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=5)
if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = None


def _auth_headers() -> dict:
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {"X-API-Key": "vortex-dev-key-change-me"}


def _api_get(path: str, params: dict = None, silent: bool = False) -> Optional[dict]:
    try:
        r = requests.get(f"{API_BASE}{path}", headers=_auth_headers(), params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        if not silent:
            st.warning("API not reachable. Is the stack running? (`make up`)")
        return None
    except Exception as e:
        if not silent:
            st.error(f"API error: {e}")
        return None


def _api_post(path: str, payload: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}{path}", headers=_auth_headers(), json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.warning("API not reachable. Is the stack running? (`make up`)")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _start_ws_stream(stream_id: str, frame_q: queue.Queue):
    """Background thread: connects to WebSocket with exponential backoff reconnection."""
    import websocket

    ws_url = f"{WS_BASE}/ws/stream/{stream_id}"
    backoff = 1.0
    max_backoff = 30.0
    connected = threading.Event()

    def on_open(ws):
        nonlocal backoff
        backoff = 1.0
        connected.set()
        logger.debug("WS connected: %s", ws_url)

    def on_message(ws, message):
        buf = np.frombuffer(message, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                frame_q.put_nowait(frame_rgb)
            except queue.Full:
                try:
                    frame_q.get_nowait()
                    frame_q.put_nowait(frame_rgb)
                except Exception:
                    pass

    def on_error(ws, error):
        logger.debug("WS error on %s: %s", stream_id, error)

    def on_close(ws, close_status_code, close_msg):
        connected.clear()
        logger.debug("WS closed: %s", stream_id)

    while True:
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.debug("WS exception: %s", e)

        # Exponential backoff before reconnecting
        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)


# ── Login sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://placehold.co/200x60/1a1a2e/00d4ff?text=VortexVision", use_column_width=True)
    st.divider()

    # Auth
    with st.expander("🔐 Login", expanded=st.session_state.token is None):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="vortex-admin-pass")
        if st.button("Login"):
            resp = requests.post(f"{API_BASE}/auth/token",
                                 json={"username": username, "password": password}, timeout=5)
            if resp.ok:
                st.session_state.token = resp.json()["access_token"]
                st.success("Logged in")
                st.rerun()
            else:
                st.error("Invalid credentials")

    if st.session_state.token:
        st.success("✅ Authenticated")
        if st.button("Logout"):
            st.session_state.token = None
            st.rerun()

    st.divider()
    st.header("📡 Stream Management")
    stream_id_input = st.text_input("Stream ID", value="cam-01")
    source_url = st.text_input("Source URL", value="rtsp://example.com/stream1")
    fps_limit = st.slider("FPS Limit", 1, 30, 15)

    col_add, col_stop = st.columns(2)
    with col_add:
        if st.button("▶ Start", use_container_width=True):
            result = _api_post("/streams/", {
                "stream_id": stream_id_input,
                "source": source_url,
                "fps_limit": fps_limit,
            })
            if result:
                st.success(f"Started")

    with col_stop:
        if st.button("⏹ Stop", use_container_width=True):
            try:
                r = requests.delete(f"{API_BASE}/streams/{stream_id_input}",
                                    headers=_auth_headers(), timeout=5)
                if r.ok:
                    st.success("Stopped")
                else:
                    st.error(r.json().get("detail", "Error"))
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.header("Active Streams")
    streams_data = _api_get("/streams/", silent=True)
    if streams_data:
        for s in streams_data:
            icon = "🟢" if s["active"] else "🔴"
            st.write(f"{icon} `{s['stream_id']}`")
            st.caption(s["source"][:50] + "..." if len(s["source"]) > 50 else s["source"])
    else:
        st.caption("No streams or API offline")

    st.divider()
    # System health
    health = _api_get("/health/deep", silent=True)
    if health:
        st.header("System Health")
        status_icon = "✅" if health.get("status") == "healthy" else "⚠️"
        st.write(f"{status_icon} {health.get('status', 'unknown').upper()}")
        for comp, ok in health.get("components", {}).items():
            st.write(f"{'✅' if ok else '❌'} {comp.replace('_', ' ').title()}")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🎯 VortexVision — Real-Time Video Analytics")

tab_live, tab_events, tab_query, tab_monitoring = st.tabs([
    "📹 Live Feed", "🚨 Events", "🔍 NL Query", "📊 Monitoring"
])

# ── Tab 1: Live Feed ──────────────────────────────────────────────────────────
with tab_live:
    col_stream_sel, col_refresh = st.columns([3, 1])
    with col_stream_sel:
        watch_stream = st.text_input("Watch Stream ID", value="cam-01", key="watch_stream")
    with col_refresh:
        auto_refresh = st.checkbox("Auto-refresh", value=True)

    col_feed, col_stats = st.columns([2, 1])

    with col_feed:
        st.subheader("Annotated Live Feed")
        frame_placeholder = st.empty()

        # Start WebSocket thread if not running
        if watch_stream and (st.session_state.ws_thread is None or
                             not st.session_state.ws_thread.is_alive()):
            t = threading.Thread(
                target=_start_ws_stream,
                args=(watch_stream, st.session_state.frame_queue),
                daemon=True,
            )
            t.start()
            st.session_state.ws_thread = t

        # Display latest frame
        try:
            frame = st.session_state.frame_queue.get_nowait()
            frame_placeholder.image(frame, use_column_width=True, caption=f"Stream: {watch_stream}")
        except queue.Empty:
            frame_placeholder.info(f"Waiting for frames from stream '{watch_stream}'...")

    with col_stats:
        st.subheader("Detection Stats")
        stats = _api_get("/events/stats")
        if stats:
            st.metric("Total Anomalies", stats.get("total", 0))
            for atype, count in stats.get("by_type", {}).items():
                st.metric(atype.replace("_", " ").title(), count)

        st.subheader("Recent Alerts")
        recent = _api_get("/events/", params={"limit": 5})
        if recent:
            for ev in recent:
                ts = time.strftime("%H:%M:%S", time.localtime(ev.get("timestamp", 0)))
                atype = ev.get("anomaly_type", "unknown")
                conf = ev.get("confidence", 0)
                st.warning(f"[{ts}] {atype} ({conf:.0%}) on `{ev.get('stream_id')}`")

    if auto_refresh:
        time.sleep(0.1)
        st.rerun()

# ── Tab 2: Events ─────────────────────────────────────────────────────────────
with tab_events:
    st.subheader("Anomaly Event Log")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_stream = st.text_input("Filter by Stream", key="ev_stream")
    with col_f2:
        filter_type = st.selectbox("Anomaly Type", ["", "fight", "crowd_rush", "accident",
                                                     "weapon", "trespassing", "unknown"])
    with col_f3:
        ev_limit = st.slider("Max results", 10, 500, 100)

    params = {"limit": ev_limit}
    if filter_stream:
        params["stream_id"] = filter_stream
    if filter_type:
        params["anomaly_type"] = filter_type

    events = _api_get("/events/", params=params)
    if events:
        import pandas as pd
        df = pd.DataFrame(events)
        if "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        display_cols = ["time", "stream_id", "anomaly_type", "confidence",
                        "autoencoder_score", "transformer_score", "clip_path"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, height=400)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "events.csv", "text/csv")
    else:
        st.info("No events found.")

# ── Tab 3: NL Query ───────────────────────────────────────────────────────────
with tab_query:
    st.subheader("Natural Language Video Query")
    st.caption("Ask questions like: 'Show me all red cars speeding in the last 5 minutes' "
               "or 'Were there any fights near the entrance today?'")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        question = st.text_area("Your question", height=80, placeholder="Describe what you're looking for...")
    with col_q2:
        time_window = st.number_input("Time window (min)", min_value=1, max_value=1440, value=5)
        query_stream = st.text_input("Stream (optional)", key="q_stream")

    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying VLM... this may take a few seconds"):
                payload = {
                    "question": question,
                    "time_window_seconds": time_window * 60,
                }
                if query_stream:
                    payload["stream_id"] = query_stream
                result = _api_post("/query/", payload)
                if result:
                    st.success(f"Found {result['clips_found']} relevant clips "
                               f"({result['processing_ms']:.0f}ms)")
                    st.markdown("### Answer")
                    st.write(result["answer"])

    st.divider()
    st.subheader("Query History")
    history = _api_get("/query/history", params={"limit": 10})
    if history:
        for item in history:
            with st.expander(f"Q: {item['question'][:80]}..."):
                st.write(f"**Answer:** {item['answer']}")
                st.caption(f"Clips: {item['clips_found']} | Latency: {item['processing_ms']:.0f}ms")

# ── Tab 4: Monitoring ─────────────────────────────────────────────────────────
with tab_monitoring:
    st.subheader("System Monitoring")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("**Grafana Dashboard**")
        st.link_button("Open Grafana", "http://localhost:3000", use_container_width=True)
    with col_m2:
        st.markdown("**MLflow Experiments**")
        st.link_button("Open MLflow", "http://localhost:5000", use_container_width=True)
    with col_m3:
        st.markdown("**Jaeger Tracing**")
        st.link_button("Open Jaeger", "http://localhost:16686", use_container_width=True)

    st.divider()
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("**Prometheus Metrics**")
        st.link_button("Open Prometheus", "http://localhost:9090", use_container_width=True)
    with col_p2:
        st.markdown("**Kafka UI**")
        st.link_button("Open Kafka UI", "http://localhost:8080", use_container_width=True)

    st.divider()
    st.subheader("API Health")
    health_data = _api_get("/health/deep")
    if health_data:
        import pandas as pd
        comp_df = pd.DataFrame([
            {"Component": k.replace("_", " ").title(), "Status": "✅ OK" if v else "❌ Down"}
            for k, v in health_data.get("components", {}).items()
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
