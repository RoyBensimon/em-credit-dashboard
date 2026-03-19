"""
EM Credit Analytics Dashboard — Main Entry Point

Run with:
    streamlit run app.py

Architecture:
  - Top navigation bar replaces the sidebar radio.
  - Active page is stored in st.session_state["nav_page"].
  - Each page module in pages/ exposes a render() function.
  - Pages that need filter controls (correlation, RV, curve, trade ideas)
    still use st.sidebar for those controls — the sidebar is available
    but no longer carries the navigation.
  - All shared data is cached in st.session_state via src/data/session.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from config.settings import APP_TITLE, APP_ICON, COLORS
from config.theme import STREAMLIT_CSS, PLOTLY_TEMPLATE  # registers plotly template


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help":  None,
        "Report a bug": None,
        "About": f"**{APP_TITLE}** — EM Credit analytics tool.",
    },
)

st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)


# ── Navigation definition ─────────────────────────────────────────────────────
# (display_label, page_module_key)
NAV_ITEMS: list[tuple[str, str]] = [
    ("Overview",      "overview"),
    ("Correlation",   "correlation_beta"),
    ("Rel. Value",    "relative_value"),
    ("Curve Trades",  "curve_trades"),
    ("Trade Ideas",   "trade_ideas"),
    ("Book Hedging",  "book_hedging"),
    ("Data Upload",   "data_upload"),
    ("Settings",      "settings"),
]

_LABEL_TO_KEY = {label: key for label, key in NAV_ITEMS}
_LABELS       = [label for label, _ in NAV_ITEMS]

# Initialise active page
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = _LABELS[0]

current = st.session_state["nav_page"]


# ── Top header bar ────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([5, 1])

with hdr_left:
    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:12px;padding:4px 0 6px 0;">
  <span style="font-size:24px">📊</span>
  <div>
    <div style="font-size:16px;font-weight:800;color:{COLORS['text']};
                letter-spacing:0.01em;">EM Credit Analytics</div>
    <div style="font-size:10px;color:{COLORS['text_muted']};
                letter-spacing:0.1em;text-transform:uppercase;">Trading Desk Tool</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with hdr_right:
    data_mode = (
        "🟢 Internal Data" if "uploaded_oas" in st.session_state else "🟡 Demo Mode"
    )
    st.markdown(
        f"<div style='text-align:right;font-size:11px;color:{COLORS['text_muted']};"
        f"padding-top:6px;padding-bottom:4px;'>{data_mode}</div>",
        unsafe_allow_html=True,
    )
    if st.button("↺ Reload Data", use_container_width=True, key="reload_data"):
        from src.data.session import clear_cache
        clear_cache()
        st.rerun()


# ── Navigation button bar ─────────────────────────────────────────────────────
_surface2 = COLORS["surface2"]
st.markdown(
    f"<div style='height:2px;background:{_surface2};margin:0 0 6px 0;'></div>",
    unsafe_allow_html=True,
)

nav_cols = st.columns(len(NAV_ITEMS))
for col, (label, _) in zip(nav_cols, NAV_ITEMS):
    is_active = current == label
    if col.button(
        label,
        key=f"nav_{label}",
        type="primary" if is_active else "secondary",
        use_container_width=True,
    ):
        st.session_state["nav_page"] = label
        st.rerun()

st.markdown(
    f"<div style='height:2px;background:{_surface2};margin:6px 0 16px 0;'></div>",
    unsafe_allow_html=True,
)


# ── Page routing ──────────────────────────────────────────────────────────────
page_key = _LABEL_TO_KEY.get(current, "overview")

if page_key == "overview":
    from pages.overview import render
elif page_key == "correlation_beta":
    from pages.correlation_beta import render
elif page_key == "relative_value":
    from pages.relative_value import render
elif page_key == "curve_trades":
    from pages.curve_trades import render
elif page_key == "trade_ideas":
    from pages.trade_ideas import render
elif page_key == "book_hedging":
    from pages.book_hedging import render
elif page_key == "data_upload":
    from pages.data_upload import render
elif page_key == "settings":
    from pages.settings_page import render
else:
    def render():
        st.error(f"Unknown page: {page_key}")

render()
