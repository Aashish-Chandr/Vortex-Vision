"""
Live Streams page — add/remove streams, view annotated live feed.
"""
import queue
import threading
import time

import streamlit as st

from frontend.utils import (
    api_delete,
    api_get,
    api_post,
    demo_frame,
    demo_streams,
    start_ws_stream,
)


def render():
    st.title("📹 Live Streams")

    demo = st.session_state.demo_mode

    # ── Add stream form ───────────────────────────────────────────────────────
    with st.expander("➕ Add New Stream", expanded=False):
        with st.form("add_stream_form"):
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                new_id = st.text_input("Stream ID", placeholder="cam-entrance")
            with col2:
                new_src = st.text_input(
                    "Source URL",
                    placeholder="rtsp://192.168.1.10/stream  or  https://youtube.com/watch?v=...",
                )
            with col3:
                new_fps = st.number_input("FPS", min_value=1, max_value=30, value=15)

            submitted = st.form_submit_button("▶ Start Stream", type="primary", use_container_width=True)
            if submitted:
                if not new_id or not new_src:
                    st.error("Stream ID and Source URL are required.")
                elif demo:
                    st.success(f"[Demo] Stream '{new_id}' would be started.")
                else:
                    result = api_post(
                        "/streams/",
                        {"stream_id": new_id, "source": new_src, "fps_limit": new_fps},
                    )
                    if result:
                        st.success(f"Stream '{new_id}' started.")
                        st.rerun()

    st.divider()

    # ── Stream list ───────────────────────────────────────────────────────────
    streams = demo_streams() if demo else (api_get("/streams/", silent=True) or [])

    if not streams:
        st.info("No streams configured. Add one above or enable Demo mode.")
        return

    # ── Live viewer ───────────────────────────────────────────────────────────
    st.subheader("Live Feed Viewer")

    stream_ids = [s["stream_id"] for s in streams]
    watch_id = st.selectbox("Select stream to watch", stream_ids, key="watch_select")

    col_feed, col_info = st.columns([3, 1])

    with col_feed:
        frame_placeholder = st.empty()

        if demo:
            # Show synthetic demo frame
            frame = demo_frame()
            frame_placeholder.image(
                frame,
                caption=f"[DEMO] {watch_id} — synthetic annotated feed",
                use_column_width=True,
            )
        else:
            # Real WebSocket stream
            ws_key = f"ws_thread_{watch_id}"
            fq_key = f"frame_q_{watch_id}"

            if fq_key not in st.session_state:
                st.session_state[fq_key] = queue.Queue(maxsize=5)

            frame_q = st.session_state[fq_key]
            ws_thread = st.session_state.get(ws_key)

            if ws_thread is None or not ws_thread.is_alive():
                t = threading.Thread(
                    target=start_ws_stream,
                    args=(watch_id, frame_q),
                    daemon=True,
                )
                t.start()
                st.session_state[ws_key] = t

            try:
                frame = frame_q.get_nowait()
                frame_placeholder.image(
                    frame,
                    caption=f"Stream: {watch_id}",
                    use_column_width=True,
                )
            except queue.Empty:
                frame_placeholder.markdown(
                    """
                    <div style="background:#161b22; border:1px dashed #30363d;
                    border-radius:8px; padding:60px; text-align:center; color:#8b949e;">
                        <p style="font-size:2rem; margin:0;">📷</p>
                        <p>Waiting for frames from <b>{}</b>…</p>
                        <p style="font-size:0.8rem;">Make sure the stream is active and the detector is running.</p>
                    </div>
                    """.format(watch_id),
                    unsafe_allow_html=True,
                )

    with col_info:
        # Stream details
        selected = next((s for s in streams if s["stream_id"] == watch_id), None)
        if selected:
            st.markdown("**Stream Details**")
            status = "🟢 Active" if selected.get("active") else "🔴 Stopped"
            st.markdown(f"**Status:** {status}")
            st.markdown(f"**ID:** `{selected['stream_id']}`")
            src = selected.get("source", "")
            st.markdown(f"**Source:**")
            st.code(src, language=None)

            if st.button("⏹ Stop Stream", use_container_width=True, type="secondary"):
                if demo:
                    st.info("[Demo] Stream would be stopped.")
                else:
                    if api_delete(f"/streams/{watch_id}"):
                        st.success("Stream stopped.")
                        st.rerun()
                    else:
                        st.error("Failed to stop stream.")

    st.divider()

    # ── All streams table ─────────────────────────────────────────────────────
    st.subheader("All Streams")
    cols = st.columns([1, 3, 1, 1])
    cols[0].markdown("**ID**")
    cols[1].markdown("**Source**")
    cols[2].markdown("**Status**")
    cols[3].markdown("**Action**")

    for s in streams:
        c1, c2, c3, c4 = st.columns([1, 3, 1, 1])
        c1.code(s["stream_id"], language=None)
        src = s.get("source", "")
        c2.caption(src[:60] + "…" if len(src) > 60 else src)
        c3.markdown("🟢 Active" if s.get("active") else "🔴 Stopped")
        if c4.button("Stop", key=f"stop_{s['stream_id']}", use_container_width=True):
            if not demo:
                api_delete(f"/streams/{s['stream_id']}")
                st.rerun()

    # Auto-refresh
    time.sleep(2)
    st.rerun()
