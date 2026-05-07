"""
NL Query page — natural language search over video content.
"""
import time

import streamlit as st

from frontend.utils import api_get, api_post

EXAMPLE_QUERIES = [
    "Show me all red cars speeding in the last 5 minutes",
    "Were there any fights near the entrance today?",
    "Find all instances of people running in the parking lot",
    "Show me any suspicious activity in the last hour",
    "Were there any weapons detected this morning?",
    "Find crowd rushes in the last 30 minutes",
]


def render():
    st.title("🔍 Natural Language Query")
    st.caption(
        "Ask questions about your video streams in plain English. "
        "Powered by Qwen2.5-VL multimodal AI."
    )

    demo = st.session_state.demo_mode

    # ── Query form ────────────────────────────────────────────────────────────
    with st.container():
        col_q, col_opts = st.columns([3, 1])

        with col_q:
            question = st.text_area(
                "Your question",
                height=100,
                placeholder="e.g. Show me all red cars speeding in the last 5 minutes",
                key="nl_question",
            )

        with col_opts:
            time_window = st.number_input(
                "Time window (minutes)",
                min_value=1,
                max_value=1440,
                value=5,
                help="How far back to search",
            )
            stream_filter = st.text_input(
                "Stream ID (optional)",
                placeholder="cam-01",
                help="Leave blank to search all streams",
            )

        # Example queries
        st.markdown("**Quick examples:**")
        ex_cols = st.columns(3)
        for i, ex in enumerate(EXAMPLE_QUERIES):
            if ex_cols[i % 3].button(
                ex[:45] + "…" if len(ex) > 45 else ex,
                key=f"ex_{i}",
                use_container_width=True,
            ):
                st.session_state.nl_question = ex
                st.rerun()

        search_clicked = st.button(
            "🔍 Search",
            type="primary",
            use_container_width=True,
            disabled=not question.strip(),
        )

    # ── Execute query ─────────────────────────────────────────────────────────
    if search_clicked and question.strip():
        if demo:
            # Simulate a VLM response
            with st.spinner("Querying VLM… (demo mode)"):
                time.sleep(1.5)
            st.success("Found 3 relevant clips (demo — 1247ms)")
            st.markdown("### Answer")
            st.markdown(
                f"> **Demo response for:** *{question}*\n\n"
                "I can see several vehicles moving through the frame. "
                "One vehicle in the left lane appears to be moving significantly faster "
                "than the surrounding traffic at timestamps 14:23:05 and 14:31:42. "
                "The vehicle appears to be a red sedan. "
                "No other anomalies were detected in the requested time window."
            )
            with st.expander("📎 Matched clips (demo)"):
                for i in range(3):
                    st.markdown(f"- `/data/clips/clip_{int(time.time()) - i * 300}.mp4`")
        else:
            if not st.session_state.token:
                st.warning("You must be signed in to use NL Query.")
                return

            payload = {
                "question": question,
                "time_window_seconds": time_window * 60,
            }
            if stream_filter:
                payload["stream_id"] = stream_filter

            with st.spinner("Querying VLM… this may take a few seconds"):
                result = api_post("/query/", payload, timeout=60)

            if result:
                st.success(
                    f"Found **{result['clips_found']}** relevant clips "
                    f"({result['processing_ms']:.0f}ms)"
                )
                st.markdown("### Answer")
                st.markdown(f"> {result['answer']}")
            else:
                st.error(
                    "Query failed. Make sure the VLM service is running "
                    "(check `docker-compose logs vlm`)."
                )

    st.divider()

    # ── Query history ─────────────────────────────────────────────────────────
    st.subheader("Query History")

    if demo:
        history = [
            {
                "question": "Show me all red cars in the last 5 minutes",
                "answer": "Two red vehicles were detected at 14:23 and 14:31.",
                "clips_found": 2,
                "processing_ms": 1340,
            },
            {
                "question": "Any fights near the entrance?",
                "answer": "No fights detected in the requested time window.",
                "clips_found": 0,
                "processing_ms": 890,
            },
        ]
    else:
        history = api_get("/query/history", params={"limit": 10}) or []

    if history:
        for item in history:
            q_short = item["question"][:70] + "…" if len(item["question"]) > 70 else item["question"]
            with st.expander(f"❓ {q_short}"):
                st.markdown(f"**Answer:** {item.get('answer', '—')}")
                col1, col2 = st.columns(2)
                col1.metric("Clips found", item.get("clips_found", 0))
                col2.metric("Latency", f"{item.get('processing_ms', 0):.0f}ms")
    else:
        st.info("No query history yet. Run a search above.")
