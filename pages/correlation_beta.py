"""
Correlation & Hedge Finder
==========================
Three-tab tool for EM sovereign bond macro analysis:

  Tab 1 — Hedge Recommender
      Enter a position (e.g. "Long Brazil 2Y"), click "Find Best Hedge",
      get an instantly actionable recommendation with scoring, alternatives,
      rationale, and the full Bond × Macro correlation matrix.

  Tab 2 — Rolling Correlation Explorer
      Pick any bond + macro factor, choose a rolling window, and inspect
      the stability of the relationship over time.

  Tab 3 — Lead-Lag Analysis
      Cross-correlation at multiple time displacements to detect whether
      a macro factor leads or lags a bond's spread movements.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import COLORS
from config.theme import STREAMLIT_CSS, apply_chart_style
from src.analytics.correlation import (
    MONTHLY_INTERPOLATED_BONDS,
    compute_rolling_corr_stability,
    rank_factors_for_bond,
    rolling_beta,
)
from src.data.session import get_app_data
from src.plotting.charts import (
    plot_bond_macro_matrix,
    plot_rolling_corr_with_stats,
)
from src.plotting.tables import style_correlation_table

# ── Lookback window map ───────────────────────────────────────────────────────
_WINDOW_MAP: dict[str, int | None] = {
    "Last 5D":   5,
    "Last 1M":   21,
    "Last 3M":   63,
    "Last 6M":   126,
    "Last 1Y":   252,
    "Full":      None,
}

# ── Hedge universe by asset class ────────────────────────────────────────────
_FACTOR_LABELS: dict[str, str] = {
    "EMB":  "EMB  ·  EM Bond ETF (CDX EM proxy)",
    "HYG":  "HYG  ·  US High Yield",
    "LQD":  "LQD  ·  US Investment Grade",
    "TLT":  "TLT  ·  20Y Treasury (duration)",
    "UUP":  "UUP  ·  USD Index (DXY proxy)",
    "SPY":  "SPY  ·  S&P 500",
    "EEM":  "EEM  ·  EM Equities",
    "VIXY": "VIXY ·  VIX Futures (volatility)",
    "GLD":  "GLD  ·  Gold",
    "PDBC": "PDBC ·  Diversified Commodities",
}

_FACTOR_CLASS: dict[str, str] = {
    "EMB": "Credit", "HYG": "Credit", "LQD": "Credit",
    "TLT": "Rates",
    "UUP": "FX",
    "SPY": "Equities", "EEM": "Equities", "VIXY": "Equities",
    "GLD": "Commodities", "PDBC": "Commodities",
}

_HEDGE_UNIVERSE_FILTER: dict[str, list[str]] = {
    "All":         ["EMB", "HYG", "LQD", "TLT", "UUP", "SPY", "EEM", "VIXY", "GLD", "PDBC"],
    "Credit":      ["EMB", "HYG", "LQD"],
    "Rates":       ["TLT"],
    "FX":          ["UUP"],
    "Equities":    ["SPY", "EEM", "VIXY"],
    "Commodities": ["GLD", "PDBC"],
}

# ── Position parser ───────────────────────────────────────────────────────────
# Maps various country name spellings → bond ID prefix used in BOND_UNIVERSE
_COUNTRY_ALIAS: dict[str, str] = {
    "brazil": "BRL",     "brl": "BRL",
    "mexico": "MEX",     "mex": "MEX",
    "colombia": "COL",   "col": "COL",
    "chile": "CHL",      "chl": "CHL",
    "peru": "PER",       "per": "PER",
    "indonesia": "IDN",  "idn": "IDN",
    "turkey": "TUR",     "tur": "TUR",
    "south africa": "ZAF", "southafrica": "ZAF", "zaf": "ZAF",
    "egypt": "EGY",      "egy": "EGY",
    "israel": "ISR",     "isr": "ISR",
    "ukraine": "UKR",    "ukr": "UKR",
}

_MATURITY_ALIAS: dict[str, str] = {
    "2y": "2Y",  "2": "2Y",
    "5y": "5Y",  "5": "5Y",
    "10y": "10Y","10": "10Y",
    "30y": "30Y","30": "30Y",
}


def _parse_position(text: str) -> tuple[str, str, str]:
    """
    Parse a free-text position into (direction, bond_id, error_msg).

    Accepts formats like:
        "Long Brazil 2Y"  →  ("Long",  "BRL_2Y",  "")
        "Short EGY_10Y"   →  ("Short", "EGY_10Y", "")
        "BRL_5Y"          →  ("Long",  "BRL_5Y",  "")   ← direction defaults to Long

    Returns ("", "", error_msg) if parsing fails.
    """
    raw = text.strip()
    if not raw:
        return "", "", "Please enter a position."

    # Normalise
    s = raw.lower().replace("-", " ").replace("_", " ")

    # Direction
    direction = "Long"
    if s.startswith("short"):
        direction = "Short"
        s = s[5:].strip()
    elif s.startswith("long"):
        s = s[4:].strip()

    # Try bond ID directly (e.g. "BRL 2Y" or "BRL_2Y" already)
    # After normalisation: "brl 2y"
    direct_match = re.match(r"^([a-z]{3})\s+(\d+y?)$", s)
    if direct_match:
        prefix  = direct_match.group(1).upper()
        mat_raw = direct_match.group(2).lower()
        maturity = _MATURITY_ALIAS.get(mat_raw)
        if prefix in {v for v in _COUNTRY_ALIAS.values()} and maturity:
            return direction, f"{prefix}_{maturity}", ""

    # Fuzzy country match (longest match first to avoid "col" matching "colombia")
    country_code = ""
    for alias in sorted(_COUNTRY_ALIAS, key=len, reverse=True):
        if alias in s:
            country_code = _COUNTRY_ALIAS[alias]
            s = s.replace(alias, "").strip()
            break

    if not country_code:
        known = ", ".join(sorted({v for v in _COUNTRY_ALIAS.values()}))
        return "", "", (
            f'Could not identify a country in "{raw}". '
            f"Try: Long Brazil 2Y, Short EGY_10Y, BRL_5Y. "
            f"Supported: {known}."
        )

    # Maturity
    mat_raw = re.search(r"(\d+\s*y?)", s)
    if mat_raw:
        maturity = _MATURITY_ALIAS.get(mat_raw.group(1).replace(" ", "").lower())
    else:
        maturity = None

    if not maturity:
        return "", "", (
            f'Could not identify a maturity in "{raw}". '
            "Add 2Y, 5Y, 10Y, or 30Y."
        )

    return direction, f"{country_code}_{maturity}", ""


def _hedge_score(corr: float, r2: float, roll_std: float) -> float:
    """
    Composite hedge score [0, 1].
    Weights: |correlation| 50%, R² 30%, stability 20%.
    stability = max(0, 1 - rolling_std / 0.5), capped at 1.
    """
    c = abs(corr) if not np.isnan(corr) else 0.0
    r = r2        if not np.isnan(r2)   else 0.0
    s_raw = roll_std if not np.isnan(roll_std) else 0.5
    stability = max(0.0, 1.0 - s_raw / 0.5)
    return 0.50 * c + 0.30 * r + 0.20 * stability


def _confidence(score: float) -> str:
    if score >= 0.60:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def _confidence_html(label: str) -> str:
    css = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}
    return f'<span class="badge {css.get(label, "badge-low")}">{label}</span>'


# ── Main render ───────────────────────────────────────────────────────────────

_CORR_EXTRA_CSS = """
<style>
/* Larger, more visible input labels on the Correlation page */
.stTextInput label, .stSelectbox label, .stNumberInput label,
.stSlider label, .stRadio label, .stMultiSelect label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #CBD5E1 !important;
}
/* Slightly larger body text */
.stMarkdown p, .stCaption { font-size: 13px !important; }
</style>
"""


def render() -> None:
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    st.markdown(_CORR_EXTRA_CSS, unsafe_allow_html=True)
    st.title("Correlation & Hedge Finder")

    data          = get_app_data()
    macro_returns = data["macro_returns"]
    oas_changes   = data["oas_changes"]
    all_bond_ids  = oas_changes.columns.tolist()
    all_factors   = macro_returns.columns.tolist()
    daily_bonds   = [b for b in all_bond_ids
                     if b not in MONTHLY_INTERPOLATED_BONDS and not b.endswith("_30Y")]

    # ── Sidebar (shared by Rolling Corr + Lead-Lag tabs) ──────────────────────
    st.sidebar.markdown("### Window & Factors")
    sidebar_window_label = st.sidebar.selectbox(
        "Lookback Window",
        ["30 days", "60 days", "90 days", "180 days", "1 year", "Full sample"],
        index=2,
        key="cb_sidebar_window",
    )
    _sidebar_w_map = {
        "30 days": 30, "60 days": 60, "90 days": 90,
        "180 days": 180, "1 year": 252, "Full sample": None,
    }
    sidebar_window = _sidebar_w_map[sidebar_window_label]

    default_facs = [f for f in ["EMB", "HYG", "SPY", "EEM", "TLT", "UUP", "VIXY", "LQD", "PDBC", "GLD"]
                    if f in all_factors]
    selected_factors = st.sidebar.multiselect(
        "Macro Factor Basket",
        all_factors,
        default=default_facs,
        format_func=lambda f: _FACTOR_LABELS.get(f, f),
    )

    if not selected_factors:
        st.warning("Select at least one macro factor in the sidebar.")
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs([
        "Hedge Recommender",
        "Rolling Correlation",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 1 — Hedge Recommender
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        _render_hedge_tab(
            oas_changes, macro_returns, all_bond_ids, daily_bonds,
            all_factors, selected_factors,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 2 — Rolling Correlation Explorer
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        _render_rolling_tab(
            oas_changes, macro_returns, daily_bonds, all_factors,
            selected_factors, sidebar_window, sidebar_window_label,
        )


# ── Tab 1: Hedge Recommender ──────────────────────────────────────────────────

def _render_hedge_tab(
    oas_changes, macro_returns, all_bond_ids, daily_bonds,
    all_factors, selected_factors,
) -> None:
    st.subheader("Hedge Recommender")
    st.caption("Enter a bond position and click Find Best Hedge to get an immediately actionable recommendation.")

    # ── Input form ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 1.5, 1.5, 1])
    pos_input = c1.text_input(
        "Position",
        placeholder="e.g. Long Brazil 2Y  ·  Short EGY_5Y  ·  BRL_10Y",
        key="hr_pos",
    )
    lookback_label = c2.selectbox(
        "Lookback", list(_WINDOW_MAP.keys()), index=2, key="hr_window"
    )
    universe_label = c3.selectbox(
        "Hedge Universe", list(_HEDGE_UNIVERSE_FILTER.keys()), index=0, key="hr_universe"
    )
    notional_m = c4.number_input(
        "Notional ($M)", min_value=1.0, max_value=500.0, value=10.0, step=1.0, key="hr_notional"
    )

    run = st.button("Find Best Hedge", type="primary", use_container_width=False)

    if run:
        direction, bond_id, err = _parse_position(pos_input)
        if err:
            st.error(err)
            st.info(
                "**Expected formats:** `Long Brazil 2Y` · `Short EGY_5Y` · `BRL_10Y` · `Short MEX 10Y`  \n"
                "Supported countries: BRL, MEX, COL, CHL, PER, IDN, TUR, ZAF, EGY, ISR, UKR"
            )
        elif bond_id not in all_bond_ids:
            # Bond parsed but not in universe
            parsed_ids = sorted([b for b in all_bond_ids if b.startswith(bond_id.split("_")[0])])
            st.error(f"Bond **{bond_id}** not found in the universe.")
            if parsed_ids:
                st.info(f"Available bonds for this country: {', '.join(parsed_ids)}")
        else:
            st.session_state["hr_result"] = {
                "bond_id":       bond_id,
                "direction":     direction,
                "lookback":      lookback_label,
                "universe":      universe_label,
                "notional_m":    notional_m,
            }

    # ── Results ───────────────────────────────────────────────────────────────
    result = st.session_state.get("hr_result")

    if result is None:
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["surface2"]};'
            f'border-radius:8px;padding:32px;text-align:center;color:{COLORS["text_muted"]};margin-top:24px">'
            f'Enter a bond position above and click <strong>Find Best Hedge</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        _render_hedge_result(
            result, oas_changes, macro_returns, all_bond_ids,
            daily_bonds, all_factors, selected_factors,
        )


def _render_hedge_result(
    result: dict, oas_changes, macro_returns, all_bond_ids,
    daily_bonds, all_factors, selected_factors,
) -> None:
    bond_id    = result["bond_id"]
    direction  = result["direction"]
    lookback   = result["lookback"]
    universe   = result["universe"]
    notional_m = result["notional_m"]
    notional   = notional_m * 1_000_000

    # Window
    window_days = _WINDOW_MAP[lookback]
    if window_days is not None:
        w_oas = oas_changes.tail(window_days)
        w_mac = macro_returns.tail(window_days)
    else:
        w_oas = oas_changes
        w_mac = macro_returns

    common_idx = w_oas.index.intersection(w_mac.index)
    w_oas = w_oas.loc[common_idx]
    w_mac = w_mac.loc[common_idx]
    n_obs = len(common_idx)

    if n_obs < 10:
        st.error(f"Only {n_obs} observations for {lookback}. Choose a longer lookback.")
        return

    # Filter hedge universe to available factors
    universe_factors = [f for f in _HEDGE_UNIVERSE_FILTER[universe] if f in all_factors]
    if not universe_factors:
        st.warning(f"No factors available for universe '{universe}'.")
        return

    # Monthly bond warning
    is_monthly = bond_id in MONTHLY_INTERPOLATED_BONDS
    if is_monthly:
        st.warning(
            f"**{bond_id}** uses monthly-interpolated FRED data. "
            "Daily OAS changes are artificially smooth — use results directionally."
        )

    # Compute factor ranking for hedge universe only
    w_mac_u = w_mac[universe_factors]
    factor_df = rank_factors_for_bond(bond_id, w_oas, w_mac_u)

    if factor_df.empty:
        st.error(f"Insufficient data for {bond_id} over {lookback}.")
        return

    # Compute rolling stability for each factor + build score
    rows = []
    for _, row in factor_df.iterrows():
        fac = row["factor"]
        if fac in macro_returns.columns and bond_id in oas_changes.columns:
            stats = compute_rolling_corr_stability(
                oas_changes[bond_id], macro_returns[fac], window=min(60, n_obs // 2)
            )
            roll_std = stats.get("std", np.nan)
        else:
            roll_std = np.nan

        score = _hedge_score(row["correlation"], row["r_squared"], roll_std)
        rows.append({
            "factor":      fac,
            "asset_class": _FACTOR_CLASS.get(fac, ""),
            "corr":        float(row["correlation"]),
            "beta":        float(row["beta"]),
            "r2":          float(row["r_squared"]),
            "roll_std":    roll_std,
            "score":       score,
            "n_obs":       int(row["n_obs"]),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)

    # ── Header ─────────────────────────────────────────────────────────────
    pos_sign = "Long" if direction == "Long" else "Short"
    st.markdown(
        f'<h4 style="color:{COLORS["text"]};margin:8px 0 4px 0">'
        f'{pos_sign} <span style="color:{COLORS["primary"]}">${notional_m:.0f}M</span>'
        f' {bond_id}  &rarr;  {lookback}  ·  {universe}</h4>',
        unsafe_allow_html=True,
    )
    st.caption(f"{n_obs} observations · {len(rows)} hedge candidates scored")
    st.divider()

    # ── Primary recommendation card ─────────────────────────────────────────
    best = rows[0]
    conf_label = _confidence(best["score"])
    hedge_dir  = "Short" if best["corr"] > 0 else "Long"
    # Flip hedge direction if position is Short
    if direction == "Short":
        hedge_dir = "Long" if hedge_dir == "Short" else "Short"
    hedge_n    = abs(best["beta"]) * notional if not np.isnan(best["beta"]) else np.nan
    hedge_n_str = f"${hedge_n / 1e6:.1f}M" if not np.isnan(hedge_n) else "N/A"

    border_col = COLORS["positive"] if conf_label == "High" else (
        COLORS["warning"] if conf_label == "Medium" else COLORS["negative"]
    )

    st.markdown(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["surface2"]};'
        f'border-left:5px solid {border_col};border-radius:8px;padding:18px 22px;margin-bottom:12px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<div>'
        f'<div style="font-size:11px;color:{COLORS["text_muted"]};text-transform:uppercase;letter-spacing:.06em">Best Hedge</div>'
        f'<div style="font-size:22px;font-weight:700;color:{COLORS["text"]};margin:2px 0">'
        f'{hedge_dir} {best["factor"]}</div>'
        f'<div style="font-size:12px;color:{COLORS["text_muted"]}">'
        f'{_FACTOR_LABELS.get(best["factor"], best["factor"])}</div>'
        f'</div>'
        f'<div style="text-align:right">'
        f'{_confidence_html(conf_label)}'
        f'<div style="font-size:11px;color:{COLORS["text_muted"]};margin-top:6px">Score {best["score"]:.2f}</div>'
        f'</div>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px">'
        f'<div><div style="font-size:10px;color:{COLORS["text_muted"]}">CORRELATION</div>'
        f'<div style="font-size:18px;font-weight:600;color:{COLORS["primary"]}">{best["corr"]:+.3f}</div></div>'
        f'<div><div style="font-size:10px;color:{COLORS["text_muted"]}">BETA (β)</div>'
        f'<div style="font-size:18px;font-weight:600;color:{COLORS["text"]}">{best["beta"]:+.3f}</div></div>'
        f'<div><div style="font-size:10px;color:{COLORS["text_muted"]}">R²</div>'
        f'<div style="font-size:18px;font-weight:600;color:{COLORS["text"]}">{best["r2"]:.0%}</div></div>'
        f'<div><div style="font-size:10px;color:{COLORS["text_muted"]}">HEDGE NOTIONAL</div>'
        f'<div style="font-size:18px;font-weight:600;color:{COLORS["warning"]}">'
        f'{hedge_dir} {hedge_n_str}</div></div>'
        f'</div>'
        f'<div style="margin-top:12px;font-size:12px;color:{COLORS["text_muted"]};'
        f'border-top:1px solid {COLORS["surface2"]};padding-top:10px">'
        f'<em>{_hedge_rationale(best, bond_id, direction, n_obs)}</em>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Alternatives ────────────────────────────────────────────────────────
    if len(rows) > 1:
        st.markdown("**Alternative Hedges**")
        alt_cols = st.columns(min(3, len(rows) - 1))
        for i, row in enumerate(rows[1:4]):
            alt_dir = "Short" if row["corr"] > 0 else "Long"
            if direction == "Short":
                alt_dir = "Long" if alt_dir == "Short" else "Short"
            alt_n   = abs(row["beta"]) * notional if not np.isnan(row["beta"]) else np.nan
            alt_n_str = f"${alt_n / 1e6:.1f}M" if not np.isnan(alt_n) else "N/A"
            alt_conf  = _confidence(row["score"])
            with alt_cols[i % len(alt_cols)]:
                st.markdown(
                    f'<div style="background:{COLORS["surface2"]};border:1px solid {COLORS["surface2"]};'
                    f'border-radius:6px;padding:12px 14px">'
                    f'<div style="font-size:13px;font-weight:700;color:{COLORS["text"]}">'
                    f'{alt_dir} {row["factor"]}</div>'
                    f'<div style="font-size:11px;color:{COLORS["text_muted"]};margin:2px 0 6px 0">'
                    f'{_FACTOR_LABELS.get(row["factor"], row["factor"])}</div>'
                    f'<div style="font-size:11px;color:{COLORS["text_muted"]}">'
                    f'Corr: <b style="color:{COLORS["text"]}">{row["corr"]:+.3f}</b>'
                    f'&nbsp;·&nbsp;β: <b style="color:{COLORS["text"]}">{row["beta"]:+.3f}</b>'
                    f'&nbsp;·&nbsp;{alt_n_str}</div>'
                    f'<div style="margin-top:6px">{_confidence_html(alt_conf)}'
                    f'<span style="font-size:10px;color:{COLORS["text_muted"]};margin-left:6px">'
                    f'Score {row["score"]:.2f}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Full table ───────────────────────────────────────────────────────────
    with st.expander("Full Hedge Candidate Table"):
        tbl_data = []
        for row in rows:
            hd     = "Short" if row["corr"] > 0 else "Long"
            if direction == "Short":
                hd = "Long" if hd == "Short" else "Short"
            hn = abs(row["beta"]) * notional if not np.isnan(row["beta"]) else np.nan
            tbl_data.append({
                "Instrument":        row["factor"],
                "Asset Class":       row["asset_class"],
                "Direction":         hd,
                "Correlation":       f"{row['corr']:+.3f}",
                "Beta (β)":          f"{row['beta']:+.3f}",
                "R²":                f"{row['r2']:.0%}",
                f"Hedge (${notional_m:.0f}M)": f"{hd} ${hn/1e6:.1f}M" if not np.isnan(hn) else "N/A",
                "Score":             f"{row['score']:.3f}",
                "Confidence":        _confidence(row["score"]),
            })
        st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

    # ── Rolling beta chart ───────────────────────────────────────────────────
    top_fac = rows[0]["factor"]
    if top_fac in macro_returns.columns and bond_id in oas_changes.columns:
        st.markdown(f"**Rolling 60d Beta — {bond_id} vs {top_fac}**")
        roll_b = rolling_beta(oas_changes[bond_id], macro_returns[top_fac], window=60)
        fig_rb = go.Figure(
            go.Scatter(
                x=roll_b.index, y=roll_b.values,
                mode="lines",
                line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.07)",
                hovertemplate="%{x|%Y-%m-%d}<br>Beta: %{y:.3f}<extra></extra>",
            )
        )
        fig_rb.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])
        fig_rb = apply_chart_style(
            fig_rb, title=f"Rolling 60d Beta: {bond_id} vs {top_fac}", height=280
        )
        st.plotly_chart(fig_rb, use_container_width=True)

        clean_b = roll_b.dropna()
        if len(clean_b) > 20 and clean_b.std() > 0.4:
            st.warning(
                f"Hedge relationship is **unstable** (beta std = {clean_b.std():.2f}). "
                "Revisit hedge ratios regularly."
            )

    # ── Bond × Macro matrix ──────────────────────────────────────────────────
    st.divider()
    st.markdown(f"**Bond × Macro Correlation Matrix — {lookback}**")
    st.caption(
        "Pearson correlation between each bond's daily OAS changes and each macro factor. "
        "Monthly-interpolated bonds excluded. Selected bond is highlighted."
    )
    daily_in_w = [b for b in oas_changes.columns
                  if b not in MONTHLY_INTERPOLATED_BONDS and not b.endswith("_30Y")]
    avail_factors = [f for f in selected_factors if f in macro_returns.columns]
    if daily_in_w and avail_factors:
        combined  = pd.concat(
            [w_oas[daily_in_w], w_mac[avail_factors]], axis=1
        ).dropna(how="all")
        corr_full = combined.corr(min_periods=10)
        bm_matrix = corr_full.loc[daily_in_w, avail_factors]
        fig_bm = plot_bond_macro_matrix(
            bm_matrix,
            title="",
            height=max(400, len(daily_in_w) * 24),
        )
        st.plotly_chart(fig_bm, use_container_width=True)
        with st.expander("Data table"):
            st.dataframe(
                style_correlation_table(bm_matrix.round(2)),
                use_container_width=True,
                height=min(600, len(daily_in_w) * 38 + 80),
            )
    else:
        st.info("No valid daily bonds or factors available.")


def _hedge_rationale(row: dict, bond_id: str, direction: str, n_obs: int) -> str:
    """Generate a short plain-English rationale for the top hedge."""
    fac  = row["factor"]
    corr = row["corr"]
    conf = _confidence(row["score"])
    cls  = _FACTOR_CLASS.get(fac, "macro")

    if abs(corr) >= 0.65:
        strength = "strong and consistent"
    elif abs(corr) >= 0.45:
        strength = "moderate"
    else:
        strength = "weak"

    move = "moves together with" if corr > 0 else "moves inversely to"

    rationale = (
        f"{bond_id} shows a {strength} {cls.lower()} relationship with {fac} "
        f"over the last {n_obs} trading days (r = {corr:+.2f}). "
        f"The bond's spread {move} {fac}. "
    )
    if conf == "High":
        rationale += "The relationship is stable and reliable for hedging."
    elif conf == "Medium":
        rationale += "The relationship is partially stable — review before sizing."
    else:
        rationale += "The relationship is weak or unstable; use with caution."
    return rationale


# ── Tab 2: Rolling Correlation Explorer ──────────────────────────────────────

def _render_rolling_tab(
    oas_changes, macro_returns, daily_bonds, all_factors,
    selected_factors, window_days, window_label,
) -> None:
    st.subheader("Rolling Correlation Explorer")
    st.caption(
        "Inspect how the correlation between a bond and a macro factor "
        "evolves over time. Useful for detecting regime changes and assessing hedge durability."
    )

    c1, c2, c3 = st.columns(3)
    roll_bond = c1.selectbox(
        "Bond",
        daily_bonds,
        index=0,
        format_func=lambda b: f"{b} *" if b in MONTHLY_INTERPOLATED_BONDS else b,
        key="t2_bond",
    )
    roll_factor = c2.selectbox(
        "Macro Factor",
        selected_factors,
        index=0,
        key="t2_factor",
    )
    roll_window = c3.selectbox(
        "Rolling Window",
        [10, 15, 20, 30, 45, 60, 90],
        index=2,
        format_func=lambda x: f"{x}d  (~{x // 5}w)",
        key="t2_roll_window",
    )

    if roll_bond not in oas_changes.columns or roll_factor not in macro_returns.columns:
        st.info("Selected bond or factor not available in the dataset.")
        return

    stats = compute_rolling_corr_stability(
        oas_changes[roll_bond],
        macro_returns[roll_factor],
        window=roll_window,
    )

    fig_roll = plot_rolling_corr_with_stats(
        stats,
        y_name=roll_bond,
        x_name=roll_factor,
        window=roll_window,
        height=420,
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    _fmt = lambda v: f"{v:+.3f}" if not np.isnan(v) else "N/A"
    s1.metric("Latest",  _fmt(stats.get("latest", np.nan)))
    s2.metric("Average", _fmt(stats.get("mean",   np.nan)))
    s3.metric("Max",     _fmt(stats.get("max",    np.nan)))
    s4.metric("Min",     _fmt(stats.get("min",    np.nan)))
    s5.metric("Std",     f"{stats.get('std', np.nan):.3f}" if not np.isnan(stats.get("std", np.nan)) else "N/A")

    if stats.get("stable") is False:
        st.warning(
            f"Correlation between **{roll_bond}** and **{roll_factor}** appears "
            f"**unstable** (rolling std = {stats.get('std', 0):.2f} > 0.30). "
            "The relationship changes across periods — use with caution for hedging."
        )
    elif stats.get("stable") is True:
        st.success(
            f"Correlation appears **stable** (rolling std = {stats.get('std', 0):.2f} < 0.30). "
            "The hedge relationship is consistent over this sample."
        )

    with st.expander("Why rolling correlation matters"):
        st.markdown(
            "Static correlation hides regime changes. A bond that was macro-driven during "
            "a risk-off episode may become idiosyncratic once the catalyst fades. "
            "Rolling correlation reveals whether the relationship is persistent or episodic — "
            "critical for assessing hedge durability."
        )
