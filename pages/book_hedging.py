"""
Book Hedging — Portfolio Risk & Hedge

Single-page trader tool:
  1. Interactive position builder (Add / Remove positions with notional)
  2. "Analyze Book" triggers the full analysis inline
  3. Correlation matrix — book bonds × macro factors (60-day Pearson)
  4. Risk overview — country / maturity / macro beta exposures
  5. Best hedge per position — ranked by OLS beta and correlation
  6. Global hedge — DV01-weighted book-level recommendation
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import BOND_UNIVERSE, COLORS
from config.theme import DIVERGING_CORR, STREAMLIT_CSS, apply_chart_style
from src.data.session import get_app_data


# ══════════════════════════════════════════════════════════════════════════════
# Constants / lookup tables
# ══════════════════════════════════════════════════════════════════════════════

# All countries present in BOND_UNIVERSE
_ALL_COUNTRIES: list[str] = sorted({b["country"] for b in BOND_UNIVERSE})

# country → sorted list of available maturities
_COUNTRY_MATURITIES: dict[str, list[int]] = defaultdict(list)
for _b in BOND_UNIVERSE:
    _COUNTRY_MATURITIES[_b["country"]].append(_b["maturity"])
_COUNTRY_MATURITIES = {k: sorted(v) for k, v in _COUNTRY_MATURITIES.items()}

# (country, maturity) → full bond metadata dict
_BOND_LOOKUP: dict[tuple[str, int], dict] = {
    (b["country"], b["maturity"]): b for b in BOND_UNIVERSE
}

_FACTOR_LABEL: dict[str, str] = {
    "EMB":  "EM Bond ETF (EMB)",
    "HYG":  "HY Corp Bond (HYG)",
    "SPY":  "S&P 500 (SPY)",
    "EEM":  "EM Equity (EEM)",
    "TLT":  "20Y+ Treasury (TLT)",
    "GLD":  "Gold (GLD)",
    "UUP":  "US Dollar / DXY (UUP)",
    "VIXY": "VIX Futures (VIXY)",
    "LQD":  "IG Corp Bond (LQD)",
    "PDBC": "Commodity (PDBC)",
}

# Only country name + demonym — no currency codes, city names, or politician names.
# Short / ambiguous tokens (3-letter currency codes, common words) cause false positives.
_NEWS_KEYWORDS: dict[str, list[str]] = {
    "Brazil":       ["Brazil", "Brazilian"],
    "Mexico":       ["Mexico", "Mexican"],
    "Colombia":     ["Colombia", "Colombian"],
    "Chile":        ["Chile", "Chilean"],
    "Peru":         ["Peru", "Peruvian"],
    "Indonesia":    ["Indonesia", "Indonesian"],
    "Turkey":       ["Turkey", "Turkish"],
    "South Africa": ["South Africa", "South African"],
    "Egypt":        ["Egypt", "Egyptian"],
    "Israel":       ["Israel", "Israeli"],
    "Ukraine":      ["Ukraine", "Ukrainian"],
}

# Pre-compiled word-boundary patterns per country (case-insensitive)
_NEWS_PATTERNS: dict[str, list[re.Pattern]] = {
    country: [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in kws]
    for country, kws in _NEWS_KEYWORDS.items()
}


def _match_article(article: dict, country: str) -> dict | None:
    """
    Return match metadata if the article is relevant to the country, else None.
    Rules:
      - Word-boundary regex only (prevents 'ILS' matching 'skills').
      - Title match = high confidence (checked first).
      - Summary match = medium confidence (checked only if title misses).
      - No match → None (article is excluded).
    """
    patterns = _NEWS_PATTERNS.get(country, [])
    title    = article.get("title",   "") or ""
    summary  = article.get("summary", "") or ""

    for pat in patterns:
        if pat.search(title):
            return {"matched_in": "title", "matched_kw": pat.pattern[2:-2]}  # strip \b

    for pat in patterns:
        if pat.search(summary):
            return {"matched_in": "summary", "matched_kw": pat.pattern[2:-2]}

    return None

_DEFAULT_POSITIONS = [
    {"direction": "Long",  "country": "Brazil",       "maturity": 5,  "notional_mm": 10.0},
    {"direction": "Short", "country": "Mexico",       "maturity": 10, "notional_mm": 10.0},
    {"direction": "Long",  "country": "Colombia",     "maturity": 5,  "notional_mm": 5.0},
    {"direction": "Short", "country": "South Africa", "maturity": 10, "notional_mm": 5.0},
    {"direction": "Long",  "country": "Israel",       "maturity": 10, "notional_mm": 8.0},
]


# ══════════════════════════════════════════════════════════════════════════════
# Session-state helpers
# ══════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    if "bh_positions" not in st.session_state:
        st.session_state["bh_positions"] = _DEFAULT_POSITIONS.copy()
    if "bh_analyzed" not in st.session_state:
        st.session_state["bh_analyzed"] = False


def _enrich(raw_pos: dict) -> dict | None:
    """Attach bond metadata to a raw {direction, country, maturity, notional_mm} dict."""
    meta = _BOND_LOOKUP.get((raw_pos["country"], raw_pos["maturity"]))
    if meta is None:
        return None
    sign = 1 if raw_pos["direction"] == "Long" else -1
    notional = raw_pos.get("notional_mm", 10.0)
    return {
        **raw_pos,
        **meta,
        "sign":        sign,
        "notional_mm": notional,
        # Actual DV01 scaled by notional ($k total per 1bp)
        "total_dv01_k": meta["dv01"] * notional / 1_000,
    }


def _get_enriched_positions() -> list[dict]:
    raw = st.session_state.get("bh_positions", [])
    result = []
    for r in raw:
        e = _enrich(r)
        if e:
            result.append(e)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Analytics
# ══════════════════════════════════════════════════════════════════════════════

def _compute_corr_matrix(
    positions: list[dict],
    oas_changes: pd.DataFrame,
    macro_returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    60-day Pearson correlation of each book bond's OAS changes
    against each macro factor's daily return.
    Returns: DataFrame with bonds as rows, macro factors as columns.
    """
    bond_ids = [p["id"] for p in positions if p["id"] in oas_changes.columns]
    if not bond_ids or macro_returns.empty:
        return pd.DataFrame()

    bond_ch  = oas_changes[bond_ids].tail(window)
    macro_rt = macro_returns.tail(window)
    combined = pd.concat([bond_ch, macro_rt], axis=1).dropna()
    if len(combined) < 20:
        return pd.DataFrame()

    full_corr = combined.corr()
    # Extract rows=bonds, cols=macro factors
    macro_cols = [c for c in macro_rt.columns if c in full_corr.columns]
    avail_bonds = [b for b in bond_ids if b in full_corr.index]
    return full_corr.loc[avail_bonds, macro_cols]


def _compute_book_risk(
    positions: list[dict],
    beta_summary: pd.DataFrame,
) -> dict:
    """
    Aggregate DV01-based risk metrics and DV01-weighted macro beta for the book.
    beta_summary: wide DataFrame, index=bond_id, columns include beta_EMB, beta_HYG, …
    """
    if not positions:
        return {}

    country_dv01: dict[str, float] = {}
    mat_dv01:     dict[str, float] = {}
    for p in positions:
        country_dv01[p["country"]] = (
            country_dv01.get(p["country"], 0.0) + p["sign"] * p["total_dv01_k"]
        )
        key = f"{p['maturity']}Y"
        mat_dv01[key] = mat_dv01.get(key, 0.0) + p["sign"] * p["total_dv01_k"]

    # DV01-weighted portfolio beta per macro factor
    macro_beta: dict[str, float] = {}
    if not beta_summary.empty:
        beta_cols = [c for c in beta_summary.columns if c.startswith("beta_")]
        for bc in beta_cols:
            factor = bc[5:]
            port_beta = 0.0
            for p in positions:
                if p["id"] in beta_summary.index:
                    b = beta_summary.at[p["id"], bc]
                    if pd.notna(b):
                        port_beta += p["sign"] * p["total_dv01_k"] * b
            macro_beta[factor] = round(port_beta, 2)

    gross = sum(p["total_dv01_k"] for p in positions)
    net   = sum(p["sign"] * p["total_dv01_k"] for p in positions)

    return {
        "country_dv01":     country_dv01,
        "maturity_dv01":    mat_dv01,
        "macro_beta":       macro_beta,
        "total_gross_dv01": gross,
        "total_net_dv01":   net,
    }


def _best_hedge_per_position(
    positions: list[dict],
    beta_summary: pd.DataFrame,
    corr_matrix: pd.DataFrame,
) -> list[dict]:
    """
    For each position, pick the macro factor with the highest absolute beta.
    Hedge direction = opposite of (position_sign × beta_sign).
    """
    results = []
    beta_cols = [c for c in beta_summary.columns if c.startswith("beta_")] if not beta_summary.empty else []

    for p in positions:
        if beta_summary.empty or p["id"] not in beta_summary.index or not beta_cols:
            results.append({"bond_id": p["id"], "hedge": None})
            continue

        row = beta_summary.loc[p["id"]]
        # Pick factor with highest |beta|
        best_bc  = max(beta_cols, key=lambda c: abs(row.get(c, 0) or 0))
        factor   = best_bc[5:]
        beta_val = float(row.get(best_bc, 0) or 0)

        # Correlation of this bond with this factor
        corr_val = np.nan
        if not corr_matrix.empty and p["id"] in corr_matrix.index and factor in corr_matrix.columns:
            corr_val = corr_matrix.at[p["id"], factor]

        # Net exposure = sign_of_position × beta → hedge in opposite direction
        net_exposure = p["sign"] * beta_val
        hedge_dir = "Short" if net_exposure > 0 else "Long"

        results.append({
            "bond_id":   p["id"],
            "country":   p["country"],
            "maturity":  p["maturity"],
            "direction": p["direction"],
            "notional":  p["notional_mm"],
            "hedge": {
                "factor":    factor,
                "label":     _FACTOR_LABEL.get(factor, factor),
                "direction": hedge_dir,
                "beta":      beta_val,
                "corr":      corr_val,
                "why":       _hedge_why(p, factor, beta_val, corr_val),
            },
        })
    return results


def _global_hedge(macro_beta: dict[str, float]) -> list[dict]:
    """Top-3 global hedges sorted by absolute DV01-weighted portfolio beta."""
    ranked = sorted(macro_beta.items(), key=lambda x: abs(x[1]), reverse=True)
    hedges = []
    for factor, exposure in ranked[:3]:
        if abs(exposure) < 0.01:
            continue
        direction = "Short" if exposure > 0 else "Long"
        hedges.append({
            "factor":    factor,
            "label":     _FACTOR_LABEL.get(factor, factor),
            "direction": direction,
            "exposure":  exposure,
            "ratio":     abs(exposure),
            "why":       _global_why(factor, exposure),
        })
    return hedges


def _hedge_why(p: dict, factor: str, beta: float, corr: float) -> str:
    dir_str  = "long" if p["direction"] == "Long" else "short"
    corr_str = f"{corr:.2f}" if not np.isnan(corr) else "N/A"
    return (
        f"{p['country']} {p['maturity']}Y ({dir_str}) has β={beta:+.2f} to {factor} "
        f"(corr={corr_str}). Hedge neutralizes this exposure."
    )


def _global_why(factor: str, exposure: float) -> str:
    sign = "positive" if exposure > 0 else "negative"
    msgs = {
        "EMB":  f"Book has {sign} EM credit beta → EMB hedge offsets broad EM spread moves.",
        "HYG":  f"Book has {sign} HY credit beta → HYG hedge covers global credit sell-off.",
        "TLT":  f"Book has {sign} rates duration → TLT hedge offsets UST rate moves.",
        "SPY":  f"Book has {sign} risk-on equity beta → SPY hedge offsets risk-off events.",
        "UUP":  f"Book has {sign} USD sensitivity → UUP hedge offsets DXY strength.",
        "VIXY": f"Book has {sign} vol beta → VIXY hedge covers sudden VIX spike risk.",
        "EEM":  f"Book has {sign} EM equity beta → EEM hedge covers EM risk-off flows.",
        "GLD":  f"Book has {sign} gold correlation → GLD hedge covers commodity risk.",
        "LQD":  f"Book has {sign} IG credit beta → LQD hedge offsets IG spread widening.",
        "PDBC": f"Book has {sign} commodity beta → PDBC hedge covers commodity risk.",
    }
    return msgs.get(factor, f"Book has {sign} exposure to {factor}.")


# ══════════════════════════════════════════════════════════════════════════════
# Chart builders
# ══════════════════════════════════════════════════════════════════════════════

def _corr_heatmap(corr_df: pd.DataFrame, positions: list[dict], height: int = 380) -> go.Figure:
    """Bonds × macro factors heatmap with direction labels on rows."""
    pos_map = {p["id"]: p for p in positions}

    # Sort columns most-negative → most-positive (same convention as correlation page)
    col_means = corr_df.mean(axis=0)
    sorted_cols = col_means.sort_values(ascending=True).index.tolist()
    df = corr_df[sorted_cols]

    row_labels = []
    for bond_id in df.index:
        p = pos_map.get(bond_id, {})
        prefix = "L" if p.get("direction") == "Long" else "S"
        row_labels.append(f"{prefix} {bond_id}")

    z    = df.values.tolist()
    text = [[f"{v:.2f}" for v in row] for row in df.values]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=df.columns.tolist(),
            y=row_labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10, "color": COLORS["text"]},
            colorscale=DIVERGING_CORR,
            zmin=-1, zmax=1,
            colorbar=dict(tickfont=dict(color=COLORS["text_muted"], size=10), thickness=12),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Corr: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(side="top")
    return apply_chart_style(fig, title="Book Bonds × Macro Factors — Correlation (60d)", height=height)


def _hbar(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str = "",
    height: int = 300,
) -> go.Figure:
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}" for v in values],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color=COLORS["text_muted"], line_width=1)
    fig.update_layout(margin=dict(l=0, r=60, t=28, b=20), xaxis_title=xlabel)
    return apply_chart_style(fig, title=title, height=height)


# ══════════════════════════════════════════════════════════════════════════════
# Position input UI
# ══════════════════════════════════════════════════════════════════════════════

def _render_position_input() -> None:
    """Form to add a new position to the book."""
    st.markdown(
        f"<div style='font-size:13px;font-weight:700;color:{COLORS['text_muted']};"
        f"letter-spacing:0.06em;margin-bottom:8px;'>ADD POSITION</div>",
        unsafe_allow_html=True,
    )

    with st.form("add_position_form", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns([1, 2, 1.2, 1.2, 1])

        direction = c1.selectbox("Direction", ["Long", "Short"], key="form_dir")
        country   = c2.selectbox("Country",   _ALL_COUNTRIES,    key="form_country")

        # Maturities available for selected country
        avail_mats = _COUNTRY_MATURITIES.get(country, [5, 10])
        mat_labels = [f"{m}Y" for m in avail_mats]
        mat_sel    = c3.selectbox("Maturity", mat_labels, key="form_mat")
        maturity   = int(mat_sel.replace("Y", ""))

        notional = c4.number_input(
            "Notional ($mm)", min_value=0.5, max_value=500.0,
            value=10.0, step=0.5, key="form_notional",
        )

        submitted = c5.form_submit_button("Add", use_container_width=True)

    if submitted:
        new_pos = {
            "direction":   direction,
            "country":     country,
            "maturity":    maturity,
            "notional_mm": notional,
        }
        if _enrich(new_pos) is not None:
            st.session_state["bh_positions"].append(new_pos)
            st.session_state["bh_analyzed"] = False
            st.rerun()
        else:
            st.error(f"{country} {maturity}Y not found in universe.")


def _render_book_table(positions: list[dict], oas_df: pd.DataFrame) -> None:
    """Display the current book as a styled table with remove buttons."""
    if not positions:
        st.info("No positions yet. Use the form above to add bonds to your book.")
        return

    st.markdown(
        f"<div style='font-size:13px;font-weight:700;color:{COLORS['text_muted']};"
        f"letter-spacing:0.06em;margin-bottom:8px;'>CURRENT BOOK</div>",
        unsafe_allow_html=True,
    )

    header = st.columns([0.8, 1.8, 1, 1, 1, 1, 1, 0.6])
    for col, label in zip(header, ["Dir", "Bond", "Rating", "Notional", "DV01 ($k)", "Duration", "OAS", "Del"]):
        col.markdown(
            f"<div style='font-size:11px;color:{COLORS['text_muted']};font-weight:600'>{label}</div>",
            unsafe_allow_html=True,
        )

    raw_list = st.session_state.get("bh_positions", [])
    for i, (_, p) in enumerate(zip(raw_list, positions)):
        dir_color = COLORS["positive"] if p["direction"] == "Long" else COLORS["negative"]
        latest_oas = "—"
        if not oas_df.empty and p["id"] in oas_df.columns:
            v = oas_df[p["id"]].dropna()
            if not v.empty:
                latest_oas = f"{v.iloc[-1]:.0f}"

        cols = st.columns([0.8, 1.8, 1, 1, 1, 1, 1, 0.6])
        cols[0].markdown(
            f"<b style='color:{dir_color}'>{p['direction']}</b>",
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f"<span style='color:{COLORS['text']}'>{p['country']} {p['maturity']}Y</span>"
            f"<br><span style='font-size:10px;color:{COLORS['text_muted']}'>{p['id']}</span>",
            unsafe_allow_html=True,
        )
        cols[2].markdown(f"<span style='font-size:12px'>{p['rating']}</span>", unsafe_allow_html=True)
        cols[3].markdown(f"<span style='font-size:12px'>${p['notional_mm']:.0f}mm</span>", unsafe_allow_html=True)
        cols[4].markdown(f"<span style='font-size:12px'>{p['total_dv01_k']:.1f}</span>", unsafe_allow_html=True)
        cols[5].markdown(f"<span style='font-size:12px'>{p['duration']:.1f}y</span>", unsafe_allow_html=True)
        cols[6].markdown(f"<span style='font-size:12px'>{latest_oas}</span>", unsafe_allow_html=True)
        if cols[7].button("✕", key=f"del_{i}", help="Remove position"):
            st.session_state["bh_positions"].pop(i)
            st.session_state["bh_analyzed"] = False
            st.rerun()

        st.markdown(
            f"<hr style='margin:2px 0;border-color:{COLORS['surface2']}'>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Analysis sections
# ══════════════════════════════════════════════════════════════════════════════

def _render_kpi_strip(positions: list[dict], risk: dict) -> None:
    n_long  = sum(1 for p in positions if p["direction"] == "Long")
    n_short = sum(1 for p in positions if p["direction"] == "Short")
    gross   = risk.get("total_gross_dv01", 0)
    net     = risk.get("total_net_dv01", 0)
    countries = len(risk.get("country_dv01", {}))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Positions",           f"{len(positions)}",        f"{n_long}L / {n_short}S")
    k2.metric("Countries",           f"{countries}")
    k3.metric("Total Notional",      f"${sum(p['notional_mm'] for p in positions):.0f}mm")
    k4.metric("Gross DV01 ($k)",     f"{gross:,.0f}")
    k5.metric("Net DV01 ($k)",       f"{net:+,.0f}")


def _render_risk_section(risk: dict) -> None:
    st.markdown(
        f"<div style='font-size:15px;font-weight:700;color:{COLORS['text']};"
        f"margin:20px 0 4px 0;'>Risk Overview</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        cd = risk.get("country_dv01", {})
        if cd:
            fig = _hbar(list(cd.keys()), list(cd.values()),
                        "Country DV01 ($k, signed)", xlabel="DV01 ($k)")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        md = risk.get("maturity_dv01", {})
        if md:
            fig = _hbar(list(md.keys()), list(md.values()),
                        "Maturity Bucket DV01 ($k, signed)", xlabel="DV01 ($k)")
            st.plotly_chart(fig, use_container_width=True)

    # Concentration alert
    cd    = risk.get("country_dv01", {})
    gross = risk.get("total_gross_dv01", 1) or 1
    if cd:
        top_c, top_v = max(cd.items(), key=lambda x: abs(x[1]))
        share = abs(top_v) / gross * 100
        if share > 40:
            st.warning(
                f"Concentration: **{top_c}** = **{share:.0f}%** of gross DV01. Consider reducing.",
                icon="⚠️",
            )


def _render_corr_section(corr_matrix: pd.DataFrame, positions: list[dict]) -> None:
    st.markdown(
        f"<div style='font-size:15px;font-weight:700;color:{COLORS['text']};"
        f"margin:20px 0 4px 0;'>Correlation Matrix — Book Bonds × Macro Factors (60d)</div>",
        unsafe_allow_html=True,
    )
    if corr_matrix.empty:
        st.info("Not enough data to compute correlations (need ≥ 20 observations).")
        return

    n_bonds  = len(corr_matrix)
    height   = max(280, 60 + n_bonds * 50)
    fig = _corr_heatmap(corr_matrix, positions, height=height)
    fig.update_layout(title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "L = Long, S = Short. Columns sorted from most-negative to most-positive correlation. "
        "Negative correlation to a factor = natural hedge."
    )


def _render_position_hedges(pos_hedges: list[dict]) -> None:
    st.markdown(
        f"<div style='font-size:15px;font-weight:700;color:{COLORS['text']};"
        f"margin:20px 0 4px 0;'>Best Hedge per Position</div>",
        unsafe_allow_html=True,
    )

    for ph in pos_hedges:
        h = ph.get("hedge")
        dir_color = COLORS["positive"] if ph["direction"] == "Long" else COLORS["negative"]

        with st.container():
            c1, c2 = st.columns([1.2, 3])
            with c1:
                st.markdown(
                    f"""
<div style="background:{COLORS['surface']};border-radius:8px;padding:12px 14px;
            border-left:3px solid {dir_color};height:100%;">
  <div style="font-size:12px;font-weight:700;color:{dir_color};">{ph['direction'].upper()}</div>
  <div style="font-size:14px;font-weight:700;color:{COLORS['text']};margin-top:2px;">
    {ph['country']} {ph['maturity']}Y
  </div>
  <div style="font-size:11px;color:{COLORS['text_muted']};margin-top:2px;">
    {ph['bond_id']} &nbsp;·&nbsp; ${ph['notional']:.0f}mm
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with c2:
                if h is None:
                    st.markdown(
                        f"<span style='color:{COLORS['text_muted']}'>No beta data available.</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    hedge_color = COLORS["negative"] if h["direction"] == "Short" else COLORS["positive"]
                    corr_str = f"{h['corr']:.2f}" if not np.isnan(h["corr"]) else "N/A"
                    st.markdown(
                        f"""
<div style="background:{COLORS['surface2']};border-radius:8px;padding:12px 16px;">
  <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
    <div>
      <div style="font-size:10px;color:{COLORS['text_muted']}">HEDGE</div>
      <div style="font-size:14px;font-weight:700;color:{hedge_color};">
        {h['direction']} {h['label']}
      </div>
    </div>
    <div>
      <div style="font-size:10px;color:{COLORS['text_muted']}">β (OLS)</div>
      <div style="font-size:14px;font-weight:700;color:{COLORS['text']}">{h['beta']:+.2f}</div>
    </div>
    <div>
      <div style="font-size:10px;color:{COLORS['text_muted']}">Correlation</div>
      <div style="font-size:14px;font-weight:700;color:{COLORS['text']}">{corr_str}</div>
    </div>
  </div>
  <div style="font-size:11px;color:{COLORS['text_muted']};margin-top:8px;">{h['why']}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def _render_global_hedge(hedges: list[dict], macro_beta: dict[str, float]) -> None:
    st.markdown(
        f"<div style='font-size:15px;font-weight:700;color:{COLORS['text']};"
        f"margin:20px 0 4px 0;'>Global Hedge for the Book</div>",
        unsafe_allow_html=True,
    )

    if not hedges:
        st.info("No meaningful macro exposures found.")
        return

    primary   = hedges[0]
    dir_color = COLORS["negative"] if primary["direction"] == "Short" else COLORS["positive"]

    # Primary hedge card
    st.markdown(
        f"""
<div style="background:{COLORS['surface']};border-radius:12px;padding:20px 24px;
            border-left:5px solid {COLORS['warning']};margin-bottom:16px;">
  <div style="font-size:11px;color:{COLORS['warning']};font-weight:700;
              letter-spacing:0.08em;margin-bottom:6px;">PRIMARY HEDGE</div>
  <div style="font-size:20px;font-weight:800;color:{COLORS['text']};margin-bottom:10px;">
    {primary['direction']} {primary['label']}
  </div>
  <div style="display:flex;gap:32px;margin-bottom:10px;">
    <div>
      <div style="font-size:10px;color:{COLORS['text_muted']}">Hedge Ratio</div>
      <div style="font-size:17px;font-weight:700;color:{dir_color}">{primary['ratio']:.2f}x</div>
    </div>
    <div>
      <div style="font-size:10px;color:{COLORS['text_muted']}">Portfolio Exposure</div>
      <div style="font-size:17px;font-weight:700;color:{COLORS['text']}">{primary['exposure']:+.2f}</div>
    </div>
  </div>
  <div style="font-size:12px;color:{COLORS['text_muted']}">{primary['why']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Alternative hedges
    if len(hedges) > 1:
        alt_cols = st.columns(len(hedges) - 1)
        for col, h in zip(alt_cols, hedges[1:]):
            dc = COLORS["negative"] if h["direction"] == "Short" else COLORS["positive"]
            col.markdown(
                f"""
<div style="background:{COLORS['surface2']};border-radius:10px;padding:14px 16px;">
  <div style="font-size:10px;color:{COLORS['text_muted']};margin-bottom:4px;">ALTERNATIVE</div>
  <div style="font-size:14px;font-weight:700;color:{dc};">{h['direction']} {h['label']}</div>
  <div style="font-size:12px;margin-top:6px;">
    <b>Ratio:</b> {h['ratio']:.2f}x &nbsp;|&nbsp; <b>Exp:</b> {h['exposure']:+.2f}
  </div>
  <div style="font-size:11px;color:{COLORS['text_muted']};margin-top:6px;">{h['why']}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    # Full beta ladder chart
    if macro_beta:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        labels = [_FACTOR_LABEL.get(f, f) for f in macro_beta]
        vals   = list(macro_beta.values())
        fig = _hbar(labels, vals, "Full Macro Beta Ladder (DV01-weighted)",
                    xlabel="Exposure", height=340)
        st.plotly_chart(fig, use_container_width=True)


def _render_news_section(positions: list[dict]) -> None:
    from src.data.news import get_news

    news = get_news(max_total=80)

    # Build: {bond_id: [ {article, matched_in, matched_kw}, … ]}
    matched: dict[str, list[dict]] = {}
    for p in positions:
        hits = []
        for art in news:
            m = _match_article(art, p["country"])
            if m:
                hits.append({**m, "article": art})
        matched[p["id"]] = hits

    alerted = [(p, matched[p["id"]]) for p in positions if matched[p["id"]]]

    st.markdown(
        f"<div style='font-size:15px;font-weight:700;color:{COLORS['text']};"
        f"margin:20px 0 4px 0;'>News Alerts</div>",
        unsafe_allow_html=True,
    )

    if not news:
        st.info("News feed empty — no internet connection.")
        return

    debug_mode = st.checkbox("Show match details", value=False, key="news_debug")

    if alerted:
        items = " | ".join(
            f"{p['direction']} {p['country']} {p['maturity']}Y ({len(hits)} article{'s' if len(hits)>1 else ''})"
            for p, hits in alerted
        )
        st.error(f"Positions with relevant news: {items}", icon="🔴")
    else:
        st.success("No position-linked news in current feed.", icon="✅")

    for p in positions:
        hits      = matched[p["id"]]
        dir_color = COLORS["positive"] if p["direction"] == "Long" else COLORS["negative"]

        with st.expander(
            f"{p['direction']} {p['country']} {p['maturity']}Y — {len(hits)} article(s)",
            expanded=bool(hits),
        ):
            if not hits:
                st.caption(f"No relevant news found for {p['country']}.")
                continue

            for h in hits[:5]:
                art      = h["article"]
                where    = h["matched_in"]    # "title" or "summary"
                kw       = h["matched_kw"]
                date_str = (
                    art["date"].strftime("%d %b %H:%M")
                    if hasattr(art.get("date"), "strftime") else ""
                )
                # Badge color: green for title match, amber for summary-only
                badge_color = COLORS["positive"] if where == "title" else COLORS["warning"]
                badge_label = "title" if where == "title" else "description"

                debug_html = (
                    f"<div style='font-size:10px;color:{COLORS['text_muted']};margin-top:4px;'>"
                    f"Matched <b style='color:{badge_color}'>{badge_label}</b> "
                    f"on keyword <b>«{kw}»</b></div>"
                ) if debug_mode else ""

                st.markdown(
                    f"""
<div style="background:{COLORS['surface2']};border-radius:8px;padding:10px 14px;
            margin-bottom:6px;border-left:3px solid {dir_color};">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
    <span style="font-size:10px;background:{badge_color};color:#000;padding:1px 6px;
                 border-radius:3px;font-weight:700;">{badge_label.upper()}</span>
  </div>
  <a href="{art.get('url','#')}" target="_blank"
     style="color:{COLORS['primary']};font-size:13px;font-weight:600;text-decoration:none;">
    {art.get('title','—')}
  </a>
  <div style="font-size:11px;color:{COLORS['text_muted']};margin-top:3px;">
    {art.get('source','—')} · {date_str}
  </div>
  <div style="font-size:11px;color:{COLORS['text_muted']};margin-top:4px;">
    {(art.get('summary') or '')[:180]}…
  </div>
  {debug_html}
</div>
""",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# Page entry point
# ══════════════════════════════════════════════════════════════════════════════

def render() -> None:
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    _init_state()

    st.title("Book Hedging")
    st.caption(
        "Build your book position by position, then click **Analyze Book** "
        "to get the full risk breakdown, correlation matrix, and hedge recommendations."
    )

    data         = get_app_data()
    beta_summary = data.get("beta_summary",  pd.DataFrame())
    oas_df       = data.get("oas_df",        pd.DataFrame())
    oas_changes  = data.get("oas_changes",   pd.DataFrame())
    macro_returns = data.get("macro_returns", pd.DataFrame())

    # ── Position input ────────────────────────────────────────────────────────
    _render_position_input()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    positions = _get_enriched_positions()
    _render_book_table(positions, oas_df)

    # ── Action buttons ────────────────────────────────────────────────────────
    if positions:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        btn_cols = st.columns([2, 1, 4])

        with btn_cols[0]:
            if st.button("Analyze Book", type="primary", use_container_width=True):
                st.session_state["bh_analyzed"] = True
                st.rerun()

        with btn_cols[1]:
            if st.button("Clear Book", use_container_width=True):
                st.session_state["bh_positions"] = []
                st.session_state["bh_analyzed"]  = False
                st.rerun()

    # ── Analysis (shown only after "Analyze Book") ────────────────────────────
    if not st.session_state.get("bh_analyzed") or not positions:
        return

    st.divider()

    # Compute everything
    risk        = _compute_book_risk(positions, beta_summary)
    corr_matrix = _compute_corr_matrix(positions, oas_changes, macro_returns, window=60)
    pos_hedges  = _best_hedge_per_position(positions, beta_summary, corr_matrix)
    macro_beta  = risk.get("macro_beta", {})
    global_hedges = _global_hedge(macro_beta)

    # KPI strip
    _render_kpi_strip(positions, risk)

    st.divider()

    # 1. Risk overview
    _render_risk_section(risk)

    st.divider()

    # 2. Correlation matrix
    _render_corr_section(corr_matrix, positions)

    st.divider()

    # 3. Per-position hedges
    _render_position_hedges(pos_hedges)

    st.divider()

    # 4. Global hedge
    _render_global_hedge(global_hedges, macro_beta)

    st.divider()

    # 5. News alerts
    _render_news_section(positions)
