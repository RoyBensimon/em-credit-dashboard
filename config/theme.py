"""
Plotly and Streamlit theme definitions for the EM Credit Analytics Dashboard.

All charts import PLOTLY_TEMPLATE and call apply_chart_style() to enforce
a consistent dark finance aesthetic across the app.
"""

import plotly.graph_objects as go
import plotly.io as pio
from config.settings import COLORS

# ── Custom Plotly template ────────────────────────────────────────────────────
_FONT_FAMILY = "Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family=_FONT_FAMILY, color=COLORS["text"], size=12),
        title=dict(font=dict(family=_FONT_FAMILY, size=15, color=COLORS["text"]), x=0.01),
        xaxis=dict(
            gridcolor=COLORS["surface2"],
            zerolinecolor=COLORS["surface2"],
            tickfont=dict(color=COLORS["text_muted"], size=11),
            linecolor=COLORS["surface2"],
        ),
        yaxis=dict(
            gridcolor=COLORS["surface2"],
            zerolinecolor=COLORS["surface2"],
            tickfont=dict(color=COLORS["text_muted"], size=11),
            linecolor=COLORS["surface2"],
        ),
        legend=dict(
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["surface2"],
            borderwidth=1,
            font=dict(color=COLORS["text_muted"], size=11),
        ),
        margin=dict(l=50, r=20, t=50, b=40),
        hovermode="x unified",
        colorway=[
            COLORS["primary"],
            "#FF6B6B",
            "#FFE66D",
            "#A8E6CF",
            "#D4A5FF",
            "#FFB347",
            "#87CEEB",
            "#FF69B4",
            "#98FB98",
            "#DEB887",
        ],
    )
)

# Register as default template
pio.templates["em_credit"] = PLOTLY_TEMPLATE
pio.templates.default = "em_credit"


def apply_chart_style(
    fig: go.Figure,
    title: str = "",
    height: int = 420,
    show_legend: bool = True,
) -> go.Figure:
    """Apply consistent dark-finance style to any Plotly figure."""
    fig.update_layout(
        title_text=title,
        height=height,
        showlegend=show_legend,
        template="em_credit",
    )
    return fig


# ── Diverging colour scales ───────────────────────────────────────────────────

# Correlation heatmaps: vivid blue (strong negative) → neutral → vivid red (strong positive)
# Five-stop scale gives a clear, visible gradient across the full [-1, +1] range.
DIVERGING_CORR = [
    [0.00, "#1D4ED8"],   # -1.0  deep blue
    [0.30, "#93C5FD"],   # -0.4  light blue
    [0.50, "#1E293B"],   # 0.0   dark neutral
    [0.70, "#FCA5A5"],   # +0.4  light red
    [1.00, "#B91C1C"],   # +1.0  deep red
]

# Z-score heatmaps: red = A rich vs B, green = A cheap vs B (RV convention)
DIVERGING_RdGn = [
    [0.0,  "#EF4444"],
    [0.25, "#F97316"],
    [0.5,  "#374151"],
    [0.75, "#059669"],
    [1.0,  "#10B981"],
]

DIVERGING_RdBl = [
    [0.0,  "#EF4444"],
    [0.5,  "#374151"],
    [1.0,  "#3B82F6"],
]

# Sequential scale for z-scores
SEQUENTIAL_CYAN = [
    [0.0,  "#0F1117"],
    [0.5,  "#0E7490"],
    [1.0,  "#00D4FF"],
]

# ── Streamlit CSS injection ───────────────────────────────────────────────────
STREAMLIT_CSS = f"""
<style>
    /* ── Hide Streamlit auto-generated sidebar page list ────────────────── */
    [data-testid="stSidebarNav"] {{
        display: none !important;
    }}

    /* ── Global background ───────────────────────────────────────────────── */
    .stApp {{
        background-color: {COLORS["background"]};
        color: {COLORS["text"]};
    }}

    /* ── Sidebar (page-specific controls only, no nav) ───────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {COLORS["surface"]};
        border-right: 1px solid {COLORS["surface2"]};
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label {{
        color: {COLORS["text_muted"]};
        font-size: 13px;
    }}

    /* ── Buttons — high-contrast, always readable ────────────────────────── */
    /* Secondary / default buttons */
    .stButton > button {{
        background-color: {COLORS["surface2"]} !important;
        color: {COLORS["text"]} !important;
        border: 1px solid #3D4663 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        transition: background 0.15s, border-color 0.15s, color 0.15s !important;
    }}
    .stButton > button:hover {{
        background-color: #2D3348 !important;
        border-color: {COLORS["primary"]} !important;
        color: {COLORS["primary"]} !important;
    }}
    /* Primary buttons — cyan background, dark text for maximum contrast */
    .stButton > button[kind="primary"] {{
        background-color: {COLORS["primary"]} !important;
        color: #0A0E18 !important;
        border: 1px solid {COLORS["primary"]} !important;
        font-weight: 700 !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #00B8D9 !important;
        border-color: #00B8D9 !important;
        color: #0A0E18 !important;
    }}
    /* Form submit buttons */
    [data-testid="stFormSubmitButton"] > button {{
        background-color: {COLORS["primary"]} !important;
        color: #0A0E18 !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
    }}
    [data-testid="stFormSubmitButton"] > button:hover {{
        background-color: #00B8D9 !important;
        color: #0A0E18 !important;
    }}
    /* Download buttons */
    [data-testid="stDownloadButton"] > button {{
        background-color: {COLORS["surface2"]} !important;
        color: {COLORS["text"]} !important;
        border: 1px solid #3D4663 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }}
    [data-testid="stDownloadButton"] > button:hover {{
        border-color: {COLORS["primary"]} !important;
        color: {COLORS["primary"]} !important;
    }}

    /* ── Top navigation bar ──────────────────────────────────────────────── */
    .em-topnav {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: {COLORS["surface"]};
        border-bottom: 1px solid {COLORS["surface2"]};
        padding: 10px 20px;
        margin-bottom: 0;
    }}
    /* Make nav buttons taller and more prominent */
    div[data-testid="stHorizontalBlock"] .stButton > button {{
        padding: 10px 4px !important;
        font-size: 12px !important;
        letter-spacing: 0.02em !important;
        border-radius: 5px !important;
    }}

    /* ── Metric cards ────────────────────────────────────────────────────── */
    [data-testid="metric-container"] {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["surface2"]};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {COLORS["text_muted"]} !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    [data-testid="stMetricValue"] {{
        color: {COLORS["text"]} !important;
        font-size: 22px !important;
        font-weight: 700;
    }}

    /* ── Section headers ─────────────────────────────────────────────────── */
    h1, h2, h3, h4 {{
        color: {COLORS["text"]} !important;
    }}

    /* ── Data frames ─────────────────────────────────────────────────────── */
    .stDataFrame {{
        border: 1px solid {COLORS["surface2"]};
        border-radius: 6px;
    }}

    /* ── Expander ────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {{
        border: 1px solid {COLORS["surface2"]};
        border-radius: 6px;
        background-color: {COLORS["surface"]};
    }}

    /* ── Tabs ────────────────────────────────────────────────────────────── */
    [data-testid="stTabs"] .stTab {{
        color: {COLORS["text_muted"]};
    }}

    /* ── Alert boxes ─────────────────────────────────────────────────────── */
    .stAlert {{
        border-radius: 6px;
    }}

    /* ── Trade idea cards ────────────────────────────────────────────────── */
    .trade-card {{
        background-color: {COLORS["surface"]};
        border: 1px solid {COLORS["surface2"]};
        border-left: 4px solid {COLORS["primary"]};
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}
    .trade-card.rich  {{ border-left-color: {COLORS["negative"]}; }}
    .trade-card.cheap {{ border-left-color: {COLORS["positive"]}; }}
    .trade-card.curve {{ border-left-color: {COLORS["warning"]}; }}
    .trade-card.hedge {{ border-left-color: {COLORS["secondary"]}; }}

    /* ── Confidence badges ───────────────────────────────────────────────── */
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }}
    .badge-high   {{ background: #065f46; color: #6ee7b7; }}
    .badge-medium {{ background: #78350f; color: #fde68a; }}
    .badge-low    {{ background: #3b1212; color: #fca5a5; }}

    /* ── Reduce top padding so nav bar sits flush ────────────────────────── */
    .block-container {{
        padding-top: 1rem !important;
    }}
</style>
"""
