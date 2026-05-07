"""
Events page — anomaly event log with filtering, charts, and CSV export.
"""
import time

import pandas as pd
import streamlit as st

from frontend.utils import api_get, demo_events

ALERT_ICONS = {
    "fight": "🥊",
    "weapon": "🔫",
    "accident": "💥",
    "crowd_rush": "🏃",
    "trespassing": "🚫",
    "unknown": "❓",
}

ALERT_COLORS = {
    "fight": "#ff4b4b",
    "weapon": "#ff0000",
    "accident": "#ffa500",
    "crowd_rush": "#ffff00",
    "trespassing": "#ff69b4",
    "unknown": "#888888",
}


def render():
    st.title("🚨 Anomaly Events")

    demo = st.session_state.demo_mode

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            filter_stream = st.text_input("Filter by Stream ID", placeholder="cam-01")
        with col2:
            filter_type = st.selectbox(
                "Anomaly Type",
                ["All", "fight", "crowd_rush", "accident", "weapon", "trespassing", "unknown"],
            )
        with col3:
            ev_limit = st.selectbox("Show", [25, 50, 100, 250, 500], index=1)
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            refresh = st.button("🔄 Refresh", use_container_width=True)

    # ── Fetch ─────────────────────────────────────────────────────────────────
    params = {"limit": ev_limit}
    if filter_stream:
        params["stream_id"] = filter_stream
    if filter_type != "All":
        params["anomaly_type"] = filter_type

    if demo:
        events = demo_events(ev_limit)
        if filter_type != "All":
            events = [e for e in events if e["anomaly_type"] == filter_type]
        if filter_stream:
            events = [e for e in events if filter_stream in e["stream_id"]]
    else:
        events = api_get("/events/", params=params) or []

    if not events:
        st.info("No events found. Try adjusting filters or enable Demo mode.")
        return

    df = pd.DataFrame(events)
    df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    df["conf_pct"] = df["confidence"].apply(lambda x: f"{x:.1%}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", len(df))
    col2.metric("Unique Streams", df["stream_id"].nunique())
    col3.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
    col4.metric(
        "Most Common",
        df["anomaly_type"].value_counts().index[0].replace("_", " ").title()
        if len(df) > 0
        else "—",
    )

    st.divider()

    # ── Charts row ────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Events by Type")
        type_counts = df["anomaly_type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        type_counts["Type"] = type_counts["Type"].str.replace("_", " ").str.title()
        st.bar_chart(type_counts.set_index("Type"), color="#ff4b4b", height=220)

    with chart_col2:
        st.subheader("Events by Stream")
        stream_counts = df["stream_id"].value_counts().reset_index()
        stream_counts.columns = ["Stream", "Count"]
        st.bar_chart(stream_counts.set_index("Stream"), color="#00d4ff", height=220)

    st.divider()

    # ── Event table ───────────────────────────────────────────────────────────
    st.subheader("Event Log")

    display_cols = {
        "time": "Time",
        "stream_id": "Stream",
        "anomaly_type": "Type",
        "conf_pct": "Confidence",
        "autoencoder_score": "AE Score",
        "transformer_score": "TF Score",
    }
    available = {k: v for k, v in display_cols.items() if k in df.columns}
    display_df = df[list(available.keys())].rename(columns=available)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Type": st.column_config.TextColumn(width="medium"),
            "Confidence": st.column_config.TextColumn(width="small"),
            "AE Score": st.column_config.NumberColumn(format="%.4f"),
            "TF Score": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    # ── Export ────────────────────────────────────────────────────────────────
    col_dl1, col_dl2 = st.columns([1, 4])
    with col_dl1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Export CSV",
            csv,
            f"vortexvision_events_{int(time.time())}.csv",
            "text/csv",
            use_container_width=True,
        )

    # ── Event detail cards ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Event Details")
    st.caption("Click an event above to see details, or browse below.")

    for _, row in df.head(5).iterrows():
        atype = row.get("anomaly_type", "unknown")
        icon = ALERT_ICONS.get(atype, "❓")
        color = ALERT_COLORS.get(atype, "#888")
        with st.expander(
            f"{icon} {atype.replace('_', ' ').title()} — {row['stream_id']} @ {row['time']}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{row['confidence']:.1%}")
            c2.metric("AE Score", f"{row.get('autoencoder_score', 0):.4f}")
            c3.metric("TF Score", f"{row.get('transformer_score', 0):.4f}")
            if row.get("clip_path"):
                st.markdown(f"**Clip:** `{row['clip_path']}`")
            if row.get("description"):
                st.markdown(f"**VLM Description:** {row['description']}")
