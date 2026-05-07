"""
Shared utilities: API client, auth helpers, demo data generator.
"""
import os
import queue
import random
import threading
import time
from typing import Optional

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
_public_host = os.getenv("VORTEX_PUBLIC_HOST", "")
if _public_host:
    WS_BASE = (
        f"wss://{_public_host.replace('https://', '')}"
        if _public_host.startswith("https")
        else f"ws://{_public_host}"
    )
else:
    WS_BASE = API_BASE.replace("http://", "ws://").replace("https://", "wss://")


# ── Auth ──────────────────────────────────────────────────────────────────────

def auth_headers() -> dict:
    if st.session_state.get("token"):
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {"X-API-Key": "vortex-dev-key-change-me"}


def login(username: str, password: str) -> bool:
    try:
        r = requests.post(
            f"{API_BASE}/auth/token",
            json={"username": username, "password": password},
            timeout=5,
        )
        if r.ok:
            st.session_state.token = r.json()["access_token"]
            st.session_state.username = username
            return True
        return False
    except Exception:
        return False


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None, silent: bool = False) -> Optional[dict]:
    if st.session_state.get("demo_mode"):
        return None  # caller should use demo data
    try:
        r = requests.get(
            f"{API_BASE}{path}",
            headers=auth_headers(),
            params=params,
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        if not silent:
            st.toast("⚠️ API offline — enable Demo mode in sidebar", icon="⚠️")
        return None
    except Exception as e:
        if not silent:
            st.toast(f"API error: {e}", icon="❌")
        return None


def api_post(path: str, payload: dict, timeout: int = 30) -> Optional[dict]:
    try:
        r = requests.post(
            f"{API_BASE}{path}",
            headers=auth_headers(),
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.toast("⚠️ API offline — enable Demo mode in sidebar", icon="⚠️")
        return None
    except Exception as e:
        st.toast(f"API error: {e}", icon="❌")
        return None


def api_delete(path: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}{path}", headers=auth_headers(), timeout=5)
        return r.ok
    except Exception:
        return False


# ── WebSocket frame consumer ──────────────────────────────────────────────────

def start_ws_stream(stream_id: str, frame_q: queue.Queue):
    """Background thread: WebSocket → frame queue with exponential backoff."""
    import logging

    import cv2
    import numpy as np
    import websocket

    logger = logging.getLogger(__name__)
    ws_url = f"{WS_BASE}/ws/stream/{stream_id}"
    backoff = 1.0

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

    while True:
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=lambda ws, e: logger.debug("WS error: %s", e),
                on_close=lambda ws, c, m: None,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.debug("WS exception: %s", e)
        time.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


# ── Demo data generators ──────────────────────────────────────────────────────

ANOMALY_TYPES = ["fight", "crowd_rush", "accident", "weapon", "trespassing", "unknown"]
STREAM_IDS = ["cam-01", "cam-02", "cam-03", "entrance", "parking-lot"]


def demo_stats() -> dict:
    return {
        "total": random.randint(12, 80),
        "by_type": {
            "fight": random.randint(2, 15),
            "crowd_rush": random.randint(1, 8),
            "accident": random.randint(0, 5),
            "weapon": random.randint(0, 3),
            "trespassing": random.randint(1, 10),
            "unknown": random.randint(3, 20),
        },
    }


def demo_events(limit: int = 20) -> list:
    now = time.time()
    events = []
    for i in range(min(limit, 20)):
        atype = random.choice(ANOMALY_TYPES)
        events.append(
            {
                "id": i + 1,
                "stream_id": random.choice(STREAM_IDS),
                "timestamp": now - random.randint(0, 3600),
                "frame_id": random.randint(100, 9999),
                "anomaly_type": atype,
                "confidence": round(random.uniform(0.55, 0.99), 3),
                "autoencoder_score": round(random.uniform(0.04, 0.25), 4),
                "transformer_score": round(random.uniform(0.6, 0.98), 4),
                "clip_path": f"/data/clips/{atype}_{int(now - i * 180)}.mp4",
                "description": None,
            }
        )
    return sorted(events, key=lambda e: e["timestamp"], reverse=True)


def demo_streams() -> list:
    return [
        {"stream_id": "cam-01", "source": "rtsp://192.168.1.10/stream1", "active": True},
        {"stream_id": "cam-02", "source": "rtsp://192.168.1.11/stream1", "active": True},
        {"stream_id": "cam-03", "source": "https://youtube.com/watch?v=demo", "active": False},
    ]


def demo_health() -> dict:
    return {
        "status": "degraded",
        "components": {
            "detector": False,
            "anomaly_scorer": False,
            "behavioral_detector": False,
            "vlm_engine": False,
            "clip_buffer": True,
            "database": True,
            "tracing": False,
        },
        "load_errors": {"detector": "Demo mode — no GPU"},
        "version": "1.0.0",
    }


def demo_frame():
    """Generate a synthetic annotated frame for demo mode."""
    import cv2
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (20, 20, 30)

    # Fake bounding boxes
    boxes = [
        ((80, 100, 200, 380), "person #1", (0, 200, 100)),
        ((300, 120, 420, 370), "person #2", (0, 180, 80)),
        ((450, 200, 580, 350), "car #3", (0, 100, 255)),
    ]
    for (x1, y1, x2, y2), label, color in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Timestamp overlay
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"DEMO  {ts}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "cam-01", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
