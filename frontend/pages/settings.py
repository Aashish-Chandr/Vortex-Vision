"""
Settings page — auth, API config, demo mode, about.
"""
import streamlit as st

from frontend.utils import API_BASE, login


def render():
    st.title("⚙️ Settings")

    # ── Authentication ────────────────────────────────────────────────────────
    st.subheader("Authentication")

    if st.session_state.token:
        st.success(f"✅ Signed in as **{st.session_state.username}**")
        if st.button("Sign out", type="secondary"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
    else:
        with st.form("settings_login"):
            col1, col2 = st.columns(2)
            with col1:
                uname = st.text_input("Username", value="admin")
            with col2:
                pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary"):
                if login(uname, pwd):
                    st.success("Signed in!")
                    st.rerun()
                else:
                    st.error("Invalid credentials or API offline.")

    st.divider()

    # ── API connection ────────────────────────────────────────────────────────
    st.subheader("API Connection")
    st.markdown(f"**Current API base:** `{API_BASE}`")
    st.caption(
        "To change the API URL, set the `API_BASE` environment variable before starting the frontend."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Test connection", use_container_width=True):
            import requests

            try:
                r = requests.get(f"{API_BASE}/health/", timeout=3)
                if r.ok:
                    st.success(f"✅ API reachable — status: {r.json().get('status')}")
                else:
                    st.error(f"API returned {r.status_code}")
            except Exception as e:
                st.error(f"Cannot reach API: {e}")

    st.divider()

    # ── Demo mode ─────────────────────────────────────────────────────────────
    st.subheader("Demo Mode")
    st.markdown(
        "When **Demo mode** is enabled, the frontend generates synthetic data "
        "so you can explore all features without running the backend stack."
    )
    demo = st.toggle(
        "Enable Demo mode",
        value=st.session_state.demo_mode,
        key="settings_demo_toggle",
    )
    if demo != st.session_state.demo_mode:
        st.session_state.demo_mode = demo
        st.rerun()

    st.divider()

    # ── Default credentials ───────────────────────────────────────────────────
    st.subheader("Default Credentials")
    st.markdown(
        """
        | Username | Password | Role |
        |---|---|---|
        | `admin` | `vortex-admin-pass` | Full access |
        | `viewer` | `vortex-viewer-pass` | Read-only |
        """
    )
    st.caption("Change these in `api/routers/auth.py` before deploying to production.")

    st.divider()

    # ── About ─────────────────────────────────────────────────────────────────
    st.subheader("About VortexVision")
    st.markdown(
        """
        **VortexVision** is a production-grade real-time multimodal video analytics platform.

        | Component | Technology |
        |---|---|
        | Object Detection | YOLO26 + ByteTrack |
        | Anomaly Detection | ConvAutoencoder + TemporalTransformer |
        | NL Query | Qwen2.5-VL |
        | Streaming | Apache Kafka |
        | Backend | FastAPI + PostgreSQL |
        | Frontend | Streamlit |
        | MLOps | MLflow + DVC + Kubeflow |
        | Infra | Kubernetes + Terraform + ArgoCD |
        | Monitoring | Prometheus + Grafana + Loki + Jaeger |
        """
    )
