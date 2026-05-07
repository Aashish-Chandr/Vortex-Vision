"""
Dashboard page — overview metrics, live alert feed, system status.
"""
import time

import pandas as pd
import streamlit as st

from frontend.utils import (
    api_get,
    demo_events,
    demo_health,
    demo_stats,
    demo_streams,
    login,
)

ALERT_COLORS = {
    "fight": "#ff4b4b",
    "weapon": "#ff0000",
    "accident": "#ffa500",
    "crowd_rush": "#ffff00",
    "trespassing": "#ff69b4",
    "unknown": "#888888",
}

ALERT_ICONS = {
    "fight": "🥊",
    "weapon": "🔫",
    "accident": "💥",
    "crowd_rush": "🏃",
    "trespassing": "🚫",
    "unknown": "❓",
}


def render():
    # ── Login gate ────────────────────────────────────────────────────────────
    if not st.session_state.token and not st.session_state.demo_mode:
        st.title("🎯 VortexVision")
        st.info("Sign in using the sidebar, or enable **Demo mode** to explore without a backend.")
        with st.form("login_form"):
            st.subheader("Sign in")
            col1, col2 = st.columns(2)
            with col1:
                uname = st.text_input("Username", value="admin")
            with col2:
                pwd = st.text_input("Password", type="password", value="vortex-admin-pass")
            if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                if login(uname, pwd):
                    st.success("Signed in!")
                    st.rerun()
                else:
                    st.error("Invalid credentials or API offline.")
        return

    st.title("🎯 Dashboard")
    st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

    # ── Fetch data ────────────────────────────────────────────────────────────
    demo = st.session_state.demo_mode
    stats = demo_stats() if demo else (api_get("/events/stats", silent=True) or demo_stats())
    health = demo_health() if demo else (api_get("/health/deep", silent=True) or demo_health())
    streams = demo_streams() if demo else (api_get("/streams/", silent=True) or [])
    recent_events = demo_events(10) if demo else (api_get("/events/", params={"limit": 10}, silent=True) or [])

    # ── KPI row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    total = stats.get("total", 0)
    by_type = stats.get("by_type", {})
    active_streams = sum(1 for s in streams if s.get("active"))

    col1.metric("Total Anomalies", total, delta=f"+{by_type.get('unknown', 0)} unknown")
    col2.metric("Active Streams", active_streams, delta=f"{len(streams)} total")
    col3.metric("Fights Detected", by_type.get("fight", 0))
    col4.metric("Weapons Detected", by_type.get("weapon", 0))
    col5.metric("Accidents", by_type.get("accident", 0))

    st.divider()

    # ── Main columns ──────────────────────────────────────────────────────────
    left, right = st.columns([2, 1])

    with left:
        # Anomaly breakdown chart
        st.subheader("Anomaly Breakdown")
        if by_type:
            chart_df = pd.DataFrame(
                [{"Type": k.replace("_", " ").title(), "Count": v} for k, v in by_type.items() if v > 0]
            ).sort_values("Count", ascending=False)
            st.bar_chart(chart_df.set_index("Type"), color="#00d4ff", height=250)
        else:
            st.info("No anomaly data yet.")

        # Recent events table
        st.subheader("Recent Events")
        if recent_events:
            df = pd.DataFrame(recent_events)
            df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
            df["conf"] = df["confidence"].apply(lambda x: f"{x:.0%}")
            display = df[["time", "stream_id", "anomaly_type", "conf"]].rename(
                columns={
                    "time": "Time",
                    "stream_id": "Stream",
                    "anomaly_type": "Type",
                    "conf": "Confidence",
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True, height=220)
        else:
            st.info("No events yet.")

    with right:
        # Live alert feed
        st.subheader("🔴 Live Alerts")
        alert_placeholder = st.empty()

        with alert_placeholder.container():
            if recent_events:
                for ev in recent_events[:8]:
                    atype = ev.get("anomaly_type", "unknown")
                    icon = ALERT_ICONS.get(atype, "❓")
                    color = ALERT_COLORS.get(atype, "#888")
                    ts = time.strftime("%H:%M:%S", time.localtime(ev.get("timestamp", 0)))
                    conf = ev.get("confidence", 0)
                    sid = ev.get("stream_id", "?")
                    st.markdown(
                        f'<div style="background:{color}18; border-left:3px solid {color}; '
                        f'padding:6px 10px; border-radius:4px; margin:3px 0; font-size:0.82rem;">'
                        f"{icon} <b>{atype.replace('_',' ').title()}</b> "
                        f"<span style='color:#8b949e'>{ts} · {sid} · {conf:.0%}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No alerts yet.")

        # System health
        st.subheader("System Health")
        status = health.get("status", "unknown")
        status_color = "#00ff88" if status == "healthy" else "#ffa500" if status == "degraded" else "#ff4b4b"
        st.markdown(
            f'<p style="color:{status_color}; font-weight:600; font-size:1rem;">● {status.upper()}</p>',
            unsafe_allow_html=True,
        )
        components = health.get("components", {})
        for comp, ok in components.items():
            icon = "✅" if ok else "❌"
            label = comp.replace("_", " ").title()
            st.markdown(f"{icon} {label}", unsafe_allow_html=False)

        # Stream status
        st.subheader("Streams")
        if streams:
            for s in streams:
                dot = "🟢" if s.get("active") else "🔴"
                src = s.get("source", "")
                src_short = src[:35] + "…" if len(src) > 35 else src
                st.markdown(f"{dot} **{s['stream_id']}**")
                st.caption(src_short)
        else:
            st.caption("No streams configured.")

    # Auto-refresh every 3 seconds
    time.sleep(3)
    st.rerun()
