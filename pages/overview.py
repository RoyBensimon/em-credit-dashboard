"""
Overview / Dashboard Page

Layout:
  1. Market Snapshot — KPI cards (EMB, HYG, SPY, UUP, TLT)
  2. Bond × Macro Correlation (10-bond sample) | Top OAS Movers (1-week)
  3. Macro Factor Performance (rebased)
  4. Trade Ideas — 5 diverse ideas + "View All" button
  5. Latest Market News (5 articles by default)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from config.settings import COLORS, MACRO_TICKERS
from config.theme import STREAMLIT_CSS
from src.data.session import get_app_data
from src.data.preprocessor import oas_change
from src.plotting.charts import (
    plot_bond_macro_matrix,
    plot_macro_performance,
    plot_top_movers_bar,
)
from src.data.news import get_news, is_cache_fresh, cache_age_minutes

# Bond sample for the correlation matrix — excludes ISR, ZAF, MEX, CHL
_SAMPLE_BONDS = [
    "BRL_5Y",  "BRL_10Y",
    "COL_5Y",  "COL_10Y",
    "PER_5Y",  "PER_10Y",
    "IDN_5Y",  "IDN_10Y",
    "TUR_5Y",  "TUR_10Y",
    "EGY_5Y",  "EGY_10Y",
]


def render() -> None:
    # Auto-refresh every 60 seconds to update news
    st_autorefresh(interval=60_000, key="overview_autorefresh")

    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)

    st.markdown(
        "# EM Credit Analytics Dashboard",
        help="Public-proxy demo mode. All bond data is synthetic.",
    )
    st.caption("Data mode: **Public Demo** (synthetic bond universe + yfinance ETF proxies)")

    st.markdown(
        f"""
<div style="background:{COLORS['surface']};border-radius:10px;padding:20px 24px;
            border-left:4px solid {COLORS['primary']};margin:16px 0 8px 0;">
  <div style="font-size:15px;color:{COLORS['text']};line-height:1.7;">
    I built this dashboard by trying to put myself in a trader's shoes.<br>
    The idea was simple: <b>What mental load can I remove during a trading day, and how can I make
    sure we don't miss anything?</b>
    <br><br>
    As a trader, you're constantly juggling multiple dimensions at once — correlations, relative
    value, macro drivers, hedging. This tool doesn't try to replace that experience.
    It's designed to <b>cut through the noise and take some of the mental load off</b>,
    so you can get to decisions faster.
    <br><br>
    The dashboard highlights:
    <ul style="margin:6px 0 6px 16px;padding:0;color:{COLORS['text']};">
      <li>Relative value opportunities between bonds</li>
      <li>Curve trades and dislocations</li>
      <li>Correlations with macro factors</li>
      <li>Simple and actionable hedge ideas</li>
    </ul>
  </div>
  <div style="margin-top:14px;padding-top:12px;border-top:1px solid {COLORS['surface2']};
              font-size:12px;color:{COLORS['text_muted']};">
    <b style="color:{COLORS['warning']};">Important note</b> &nbsp;—&nbsp;
    This is a first version (MVP) of the tool. Some market data used here is synthetic or
    simplified, as I don't yet have access to all real-time data sources. For real analysis it
    would simply require plugging in additional APIs from market data providers to operate in real
    conditions. Macro indicators are already integrated using
    <b style="color:{COLORS['text']};">yfinance</b> (market data proxies) and
    <b style="color:{COLORS['text']};">FRED API</b> (macro data).
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    data = get_app_data()

    macro_prices  = data["macro_prices"]
    oas_df        = data["oas_df"]
    corr_bonds    = data["corr_bonds"]   # bonds × macro factors
    trade_ideas   = data["trade_ideas"]

    # ── 1. Market Snapshot ────────────────────────────────────────────────────
    st.subheader("Market Snapshot")
    kpi_cols = st.columns(5)
    _kpi_card(kpi_cols[0], macro_prices, "EMB",  "EM Bond ETF (EMB)",  pct=True)
    _kpi_card(kpi_cols[1], macro_prices, "HYG",  "HY Credit (HYG)",   pct=True)
    _kpi_card(kpi_cols[2], macro_prices, "SPY",  "US Equities (SPY)", pct=True)
    _kpi_card(kpi_cols[3], macro_prices, "UUP",  "DXY Proxy (UUP)",  pct=True)
    _kpi_card(kpi_cols[4], macro_prices, "TLT",  "Rates (TLT)",       pct=True)

    st.divider()

    # ── 2. Bond × Macro Correlation | Top OAS Movers ─────────────────────────
    col_corr, col_movers = st.columns([1.5, 1])

    with col_corr:
        st.subheader("Bond × Macro Correlation (60d)")
        _render_bond_macro_corr(corr_bonds)

    with col_movers:
        st.subheader("Top OAS Movers (1-Week)")
        oas_chg_1w = oas_change(oas_df, periods=5)
        fig_movers = plot_top_movers_bar(oas_chg_1w, n=10, height=420)
        st.plotly_chart(fig_movers, use_container_width=True)

    st.divider()

    # ── 3. Macro Factor Performance ───────────────────────────────────────────
    st.subheader("Macro Factor Performance (Rebased)")
    show_tickers = [t for t in ["EMB", "HYG", "SPY", "EEM", "TLT", "GLD"]
                    if t in macro_prices.columns]
    fig_perf = plot_macro_performance(macro_prices, tickers=show_tickers, height=340)
    st.plotly_chart(fig_perf, use_container_width=True)

    st.divider()

    # ── 4. Trade Ideas ────────────────────────────────────────────────────────
    st.subheader("Trade Ideas")
    _render_trade_ideas(trade_ideas)

    st.divider()

    # ── 5. Latest Market News ─────────────────────────────────────────────────
    _render_news_section()


# ═════════════════════════════════════════════════════════════════════════════
# Section renderers
# ═════════════════════════════════════════════════════════════════════════════

def _render_bond_macro_corr(corr_bonds: pd.DataFrame) -> None:
    if corr_bonds.empty:
        st.info("Not enough data to compute correlations.")
        return

    # Filter to sample bonds that exist in the matrix
    sample = [b for b in _SAMPLE_BONDS if b in corr_bonds.index]
    if not sample:
        sample = corr_bonds.index.tolist()[:10]

    subset = corr_bonds.loc[sample]
    fig = plot_bond_macro_matrix(subset, title="", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rows = bonds · Columns sorted most-negative → most-positive · Labels at top")


def _render_trade_ideas(trade_ideas: list[dict]) -> None:
    if not trade_ideas:
        st.info("No trade ideas generated. Check the Trade Ideas page.")
        return

    # Show 3 diverse ideas: mix trade types and countries
    diverse = _pick_diverse(trade_ideas, n=3)
    for idea in diverse:
        _render_idea_card(idea)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # "View All" button — navigates to Trade Ideas page via nav state
    if st.button("View All Trade Ideas →", type="primary"):
        st.session_state["nav_page"] = "Trade Ideas"
        st.rerun()


def _pick_diverse(ideas: list[dict], n: int = 3) -> list[dict]:
    """
    Return up to n ideas with diversity across both trade type AND country.

    Strategy:
      1. Split into RV and Curve buckets, each sorted by confidence desc.
      2. Pick alternating RV / Curve.
      3. After assembling candidates, swap duplicated-country ideas for
         different-country alternatives so the final list spans ≥2 regions
         when possible.
    """
    rv_ideas    = sorted(
        [i for i in ideas if i.get("trade_type") == "Relative Value"],
        key=lambda x: x.get("confidence_score", 0), reverse=True,
    )
    curve_ideas = sorted(
        [i for i in ideas if i.get("trade_type") == "Curve Trade"],
        key=lambda x: x.get("confidence_score", 0), reverse=True,
    )

    target_curve = n // 2
    target_rv    = n - target_curve

    selected_rv    = rv_ideas[:target_rv]
    selected_curve = curve_ideas[:target_curve]

    shortfall_rv    = target_rv    - len(selected_rv)
    shortfall_curve = target_curve - len(selected_curve)
    if shortfall_rv > 0:
        selected_curve += curve_ideas[target_curve : target_curve + shortfall_rv]
    if shortfall_curve > 0:
        selected_rv    += rv_ideas[target_rv : target_rv + shortfall_curve]

    # Interleave RV / Curve for visual variety
    result: list[dict] = []
    for rv, cu in zip(selected_rv, selected_curve):
        result.append(rv)
        result.append(cu)
    for idea in selected_rv[len(selected_curve):] + selected_curve[len(selected_rv):]:
        result.append(idea)
    result = result[:n]

    # ── Country diversification pass ─────────────────────────────────────────
    # If two ideas share the same country, swap the lower-ranked one for a
    # different-country candidate from the remaining pool.
    used_countries: set[str] = set()
    final: list[dict] = []
    for idea in result:
        final.append(idea)
        used_countries.add(idea.get("country", ""))

    remaining = [i for i in ideas if i not in final]
    for idx, idea in enumerate(final):
        country = idea.get("country", "")
        if list(i.get("country", "") for i in final).count(country) > 1:
            # Try to find a replacement from a different country
            for alt in remaining:
                if alt.get("country", "") not in used_countries:
                    final[idx] = alt
                    remaining.remove(alt)
                    used_countries.add(alt.get("country", ""))
                    break

    return final


def _render_news_section() -> None:
    st.subheader("Latest Market News")

    col_ctrl, col_src = st.columns([2, 1])
    max_items = col_ctrl.slider("Number of articles", 5, 30, 5, 5, key="news_max")

    if not is_cache_fresh():
        with st.spinner("Fetching latest news…"):
            articles = get_news(max_per_source=25, max_total=60)
    else:
        articles = get_news(max_per_source=25, max_total=60)

    if not articles:
        st.warning(
            "No articles could be fetched. "
            "(sources may be temporarily unreachable)."
        )
        return

    sources = sorted({a["source"] for a in articles})
    sel_src = col_src.selectbox("Filter by source", ["All"] + sources, key="news_src")
    if sel_src != "All":
        articles = [a for a in articles if a["source"] == sel_src]

    articles = articles[:max_items]

    age_min = cache_age_minutes()
    age_txt = f"{int(age_min)}m ago" if age_min >= 1 else "just now"
    st.caption(
        f"{len(articles)} articles · sorted by most recent · "
        f"updated {age_txt} (auto-refreshes every 1 min)"
    )

    for art in articles:
        _render_news_card(art)


# ═════════════════════════════════════════════════════════════════════════════
# Card helpers
# ═════════════════════════════════════════════════════════════════════════════

def _kpi_card(
    col,
    macro_prices: pd.DataFrame,
    ticker: str,
    label: str,
    pct: bool = True,
) -> None:
    if ticker not in macro_prices.columns:
        col.metric(label, "N/A")
        return
    prices = macro_prices[ticker].dropna()
    if len(prices) < 2:
        col.metric(label, "N/A")
        return
    last  = prices.iloc[-1]
    prev  = prices.iloc[-2]
    delta = (last - prev) / prev * 100 if pct else last - prev
    col.metric(
        label,
        f"{last:.2f}",
        delta=f"{delta:+.2f}%",
        delta_color="normal" if delta >= 0 else "inverse",
    )


def _render_idea_card(idea: dict) -> None:
    trade_type    = idea.get("trade_type", "")
    conf          = idea.get("confidence_score", 0)
    direction     = idea.get("direction", "")
    border_colour = {
        "Long":        COLORS["positive"],
        "Short":       COLORS["negative"],
        "steepener":   COLORS["warning"],
        "flattener":   COLORS["secondary"],
    }.get(direction, COLORS["primary"])

    badge_html   = _confidence_badge(conf)
    metrics_html = ""
    for k, v in list(idea.get("supporting_metrics", {}).items())[:3]:
        metrics_html += (
            f'<span style="margin-right:16px;color:{COLORS["text_muted"]};font-size:11px;">'
            f'<b style="color:{COLORS["text"]}">{k}:</b> {v}</span>'
        )

    st.markdown(
        f"""
<div style="background:{COLORS['surface']};border:1px solid {COLORS['surface2']};
     border-left:4px solid {border_colour};border-radius:6px;
     padding:12px 16px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-weight:700;font-size:14px;color:{COLORS['text']}">{idea['title']}</span>
    <span>{badge_html}
      <span style="font-size:11px;color:{COLORS['text_muted']};margin-left:8px">{trade_type}</span>
    </span>
  </div>
  <div style="margin-top:6px;font-size:12px;color:{COLORS['text_muted']}">
    {idea['rationale'][:200]}{'…' if len(idea['rationale']) > 200 else ''}
  </div>
  <div style="margin-top:6px">{metrics_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_news_card(art: dict) -> None:
    from datetime import datetime, timezone
    dt    = art["date"]
    now   = datetime.now(tz=timezone.utc)
    age_h = (now - dt).total_seconds() / 3600
    if age_h < 1:
        age_str = f"{int(age_h * 60)}m ago"
    elif age_h < 24:
        age_str = f"{int(age_h)}h ago"
    else:
        age_str = dt.strftime("%d %b %Y")

    badge_text  = art["country"] if art["country"] else art["region"]
    badge_color = {
        "LatAm":            "#1D4ED8",
        "CEEMEA":           "#7C3AED",
        "Asia":             "#065F46",
        "Emerging Markets": "#374151",
    }.get(art["region"], "#374151")
    badge_html = (
        f'<span style="background:{badge_color};color:#E2E8F0;padding:2px 8px;'
        f'border-radius:10px;font-size:10px;font-weight:600;white-space:nowrap">'
        f'{badge_text}</span>'
    ) if badge_text else ""

    summary_html = (
        f'<div style="margin-top:4px;font-size:12px;color:{COLORS["text_muted"]}">'
        f'{art["summary"]}</div>'
    ) if art["summary"] else ""

    st.markdown(
        f"""<div style="background:{COLORS['surface']};border:1px solid {COLORS['surface2']};
border-radius:6px;padding:10px 14px;margin-bottom:6px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
    <a href="{art['url']}" target="_blank"
       style="font-weight:600;font-size:13px;color:{COLORS['primary']};
              text-decoration:none;line-height:1.4;flex:1">{art['title']}</a>
    <span style="white-space:nowrap;flex-shrink:0">{badge_html}</span>
  </div>
  {summary_html}
  <div style="margin-top:6px;font-size:11px;color:{COLORS['text_muted']}">
    <span style="font-weight:600">{art['source']}</span>&nbsp;·&nbsp;{age_str}
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def _confidence_badge(conf: float) -> str:
    if conf >= 0.70:
        return (
            f'<span style="background:#065f46;color:#6ee7b7;padding:2px 8px;'
            f'border-radius:12px;font-size:11px;font-weight:600">HIGH {conf:.0%}</span>'
        )
    elif conf >= 0.50:
        return (
            f'<span style="background:#78350f;color:#fde68a;padding:2px 8px;'
            f'border-radius:12px;font-size:11px;font-weight:600">MED {conf:.0%}</span>'
        )
    return (
        f'<span style="background:#3b1212;color:#fca5a5;padding:2px 8px;'
        f'border-radius:12px;font-size:11px;font-weight:600">LOW {conf:.0%}</span>'
    )
