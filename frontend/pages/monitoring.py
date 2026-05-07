"""
Monitoring page — system health, component status, links to observability tools.
"""
import time

import streamlit as st

from frontend.utils import api_get, demo_health


def render():
    st.title("📊 Monitoring")

    demo = st.session_state.demo_mode
    health = demo_health() if demo else (api_get("/health/deep", silent=True) or demo_health())

    # ── Overall status banner ─────────────────────────────────────────────────
    status = health.get("status", "unknown")
    if status == "healthy":
        st.success("✅ All systems operational")
    elif status == "degraded":
        st.warning("⚠️ System degraded — some components offline")
    else:
        st.error("❌ System unhealthy")

    st.caption(f"Version: {health.get('version', '—')} · Checked at {time.strftime('%H:%M:%S')}")

    st.divider()

    # ── Component health grid ─────────────────────────────────────────────────
    st.subheader("Component Status")
    components = health.get("components", {})
    load_errors = health.get("load_errors", {})

    comp_cols = st.columns(4)
    for i, (comp, ok) in enumerate(components.items()):
        with comp_cols[i % 4]:
            label = comp.replace("_", " ").title()
            if ok:
                st.markdown(
                    f'<div style="background:#00ff8818; border:1px solid #00ff8844; '
                    f'border-radius:8px; padding:12px; text-align:center; margin:4px 0;">'
                    f'<p style="margin:0; font-size:1.5rem;">✅</p>'
                    f'<p style="margin:0; font-size:0.85rem; color:#00ff88;">{label}</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                err = load_errors.get(comp, "Offline")
                st.markdown(
                    f'<div style="background:#ff4b4b18; border:1px solid #ff4b4b44; '
                    f'border-radius:8px; padding:12px; text-align:center; margin:4px 0;" '
                    f'title="{err}">'
                    f'<p style="margin:0; font-size:1.5rem;">❌</p>'
                    f'<p style="margin:0; font-size:0.85rem; color:#ff4b4b;">{label}</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if load_errors:
        st.divider()
        st.subheader("Load Errors")
        for comp, err in load_errors.items():
            st.error(f"**{comp.replace('_', ' ').title()}:** {err}")

    st.divider()

    # ── Observability links ───────────────────────────────────────────────────
    st.subheader("Observability Tools")

    tools = [
        ("📈 Grafana", "http://localhost:3000", "Dashboards · admin / vortex123"),
        ("🔬 Prometheus", "http://localhost:9090", "Raw metrics & alerting"),
        ("🧪 MLflow", "http://localhost:5000", "Experiment tracking & model registry"),
        ("🔍 Jaeger", "http://localhost:16686", "Distributed tracing"),
        ("📨 Kafka UI", "http://localhost:8080", "Topic browser & consumer lag"),
        ("🚨 AlertManager", "http://localhost:9093", "Alert routing & silencing"),
    ]

    tool_cols = st.columns(3)
    for i, (name, url, desc) in enumerate(tools):
        with tool_cols[i % 3]:
            st.markdown(
                f'<div style="background:#161b22; border:1px solid #30363d; '
                f'border-radius:8px; padding:14px; margin:4px 0;">'
                f'<p style="margin:0 0 4px 0; font-weight:600;">{name}</p>'
                f'<p style="margin:0; color:#8b949e; font-size:0.8rem;">{desc}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.link_button(f"Open {name.split()[1]}", url, use_container_width=True)

    st.divider()

    # ── API health raw ────────────────────────────────────────────────────────
    st.subheader("Raw API Response")
    with st.expander("View /health/deep response"):
        st.json(health)

    # ── Quick actions ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("📖 API Docs (Swagger)", "http://localhost:8000/docs", use_container_width=True)
    with col2:
        st.link_button("📖 API Docs (ReDoc)", "http://localhost:8000/redoc", use_container_width=True)
    with col3:
        if st.button("🔄 Refresh Health", use_container_width=True):
            st.rerun()
