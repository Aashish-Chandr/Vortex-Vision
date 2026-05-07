"""
VortexVision — Real-Time Multimodal Video Analytics Platform
Main Streamlit application entry point.
"""
import streamlit as st

st.set_page_config(
    page_title="VortexVision",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/vortexvision",
        "Report a bug": "https://github.com/your-org/vortexvision/issues",
        "About": "VortexVision — Production-grade real-time video analytics platform",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }
    /* Alert badges */
    .alert-fight    { background:#ff4b4b22; border-left:3px solid #ff4b4b; padding:6px 10px; border-radius:4px; margin:4px 0; }
    .alert-weapon   { background:#ff000033; border-left:3px solid #ff0000; padding:6px 10px; border-radius:4px; margin:4px 0; }
    .alert-accident { background:#ffa50033; border-left:3px solid #ffa500; padding:6px 10px; border-radius:4px; margin:4px 0; }
    .alert-crowd    { background:#ffff0022; border-left:3px solid #ffff00; padding:6px 10px; border-radius:4px; margin:4px 0; }
    .alert-unknown  { background:#ffffff11; border-left:3px solid #888888; padding:6px 10px; border-radius:4px; margin:4px 0; }
    /* Status dot */
    .dot-green { color: #00ff88; font-size: 10px; }
    .dot-red   { color: #ff4b4b; font-size: 10px; }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "token": None,
    "username": None,
    "alert_log": [],          # list of recent alert dicts
    "demo_mode": False,       # generate synthetic data when API is offline
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Navigation ────────────────────────────────────────────────────────────────
from frontend.pages import dashboard, streams, events, query, monitoring, settings  # noqa: E402

PAGES = {
    "🏠 Dashboard":   dashboard,
    "📹 Live Streams": streams,
    "🚨 Events":       events,
    "🔍 NL Query":     query,
    "📊 Monitoring":   monitoring,
    "⚙️ Settings":     settings,
}

with st.sidebar:
    # Logo
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <span style="font-size:2rem;">🎯</span>
            <h2 style="margin:0; color:#00d4ff; font-weight:700; letter-spacing:1px;">
                VortexVision
            </h2>
            <p style="margin:0; color:#8b949e; font-size:0.75rem;">
                Real-Time Video Analytics
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Page selector
    page_name = st.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.divider()

    # Auth status
    if st.session_state.token:
        st.markdown(
            f'<p style="color:#00ff88; font-size:0.85rem;">● Signed in as '
            f'<b>{st.session_state.username}</b></p>',
            unsafe_allow_html=True,
        )
        if st.button("Sign out", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
    else:
        st.markdown(
            '<p style="color:#ff4b4b; font-size:0.85rem;">● Not authenticated</p>',
            unsafe_allow_html=True,
        )

    # Demo mode toggle
    st.session_state.demo_mode = st.toggle(
        "Demo mode (no API needed)",
        value=st.session_state.demo_mode,
        help="Generates synthetic data so you can explore the UI without running the backend.",
    )

    st.divider()
    st.caption("v1.0.0 · VortexVision")

# ── Render selected page ──────────────────────────────────────────────────────
PAGES[page_name].render()
