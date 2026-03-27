"""
Curve Trade Builder Page

Allows the user to:
  - Select a country and view its full OAS curve (multiple snapshots)
  - Compute 2-point slopes and 3-point butterfly metrics
  - View rolling z-scores and percentiles of curve metrics
  - Get DV01-neutral trade sizing
  - View all candidate curve trade opportunities (screened universe)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import COLORS
from config.theme import STREAMLIT_CSS
from src.data.session import get_app_data
from src.analytics.curve_analysis import (
    extract_country_curve,
    compute_all_slopes,
    compute_all_butterflies,
    curve_zscores,
    curve_percentiles,
    dv01_neutral_weight,
    dv01_neutral_butterfly,
    compute_inflation_context,
)
from src.plotting.charts import (
    plot_yield_curve,
    plot_slope_history,
)

# ── Macro factor labels (subset used for beta check) ──────────────────────────
_MACRO_LABELS = {
    "TLT": "US Long Rates (TLT)",
    "UUP": "USD Index (UUP)",
    "EMB": "EM Bond Index (EMB)",
    "SPY": "US Equity (SPY)",
    "HYG": "US HY Credit (HYG)",
    "VIXY": "Volatility (VIXY)",
}

# ── Region mapping — used for same-region cross-country check ─────────────────
_REGION_MAP: dict[str, str] = {
    "Brazil":       "LATAM",
    "Mexico":       "LATAM",
    "Colombia":     "LATAM",
    "Chile":        "LATAM",
    "Peru":         "LATAM",
    "Turkey":       "EMEA",
    "South Africa": "EMEA",
    "Egypt":        "EMEA",
    "Israel":       "EMEA",
    "Ukraine":      "EMEA",
    "Indonesia":    "Asia",
}


def render() -> None:
    """Render the Curve Trade Builder page."""
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    st.title("Curve Trade Builder")
    st.caption(
        "Identify steepener / flattener / butterfly opportunities using "
        "z-score analysis of OAS curve metrics across EM sovereigns."
    )

    data = get_app_data()
    meta            = data["meta"]
    oas_df          = data["oas_df"]
    curve_trades_df = data["curve_trades"]
    macro_returns   = data.get("macro_returns", pd.DataFrame())

    # ── Limitations note ──────────────────────────────────────────────────────
    with st.expander("⚠️  Important — About This Tool", expanded=False):
        st.markdown(
            f"""
<div style="background:{COLORS['surface']};border-left:4px solid {COLORS['warning']};
     border-radius:6px;padding:14px 18px;font-size:13px;color:{COLORS['text']};">
<b>This tool highlights curve dislocations and potential RV opportunities, but several
important drivers are not yet captured:</b>
<ul style="margin-top:8px;margin-bottom:4px;">
  <li>News, political events and macroeconomic developments</li>
  <li>Flows and positioning</li>
  <li>Liquidity and bid-ask conditions</li>
  <li>Upcoming supply, auctions and issuance calendars</li>
  <li>Technical factors and index rebalancing</li>
</ul>
<b>Signals should be used as decision support, not as automatic trade instructions.
Trader judgment remains essential before executing any trade.</b>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.markdown("### Curve Settings")

    all_countries = sorted(meta["country"].unique())
    sel_country   = st.sidebar.selectbox(
        "Country", all_countries, index=0
    )
    zscore_window = st.sidebar.selectbox(
        "Z-Score Window", [63, 126, 252], index=2,
        format_func=lambda x: f"{x}d (~{x//21}M)",
    )

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "Curve View", "Slope & Butterfly", "DV01 Sizing"
    ])

    # Compute country curve data
    curve_df = extract_country_curve(meta, oas_df, sel_country)

    # ── Tab 1: Curve View + Bond selector + Opportunity Screener ─────────────
    with tab1:
        # ── Bond curve selector ───────────────────────────────────────────────
        st.markdown("**Bond Rate Curve**")
        st.caption("Select a bond to display the OAS curve of its issuer, with the selected bond highlighted.")

        all_bonds = sorted(
            b for b in meta["id"].tolist()
            if b in oas_df.columns and meta.loc[meta["id"] == b, "maturity"].values[0] != 30
        )

        sel_bond_curve = st.selectbox(
            "Select Bond", all_bonds, index=0, key="cv_bond_curve_sel"
        )

        bond_country  = meta.loc[meta["id"] == sel_bond_curve, "country"].values[0]
        bond_maturity = int(meta.loc[meta["id"] == sel_bond_curve, "maturity"].values[0])
        bond_oas_now  = float(oas_df[sel_bond_curve].iloc[-1]) if sel_bond_curve in oas_df.columns else None

        sel_curve_df = extract_country_curve(meta, oas_df, bond_country)

        if sel_curve_df.empty:
            st.info(f"No curve data available for {bond_country}.")
        else:
            fig_bond_curve = plot_yield_curve(
                sel_curve_df,
                title=f"{bond_country} OAS Curve — {sel_bond_curve} highlighted",
                height=400,
            )
            if bond_oas_now is not None:
                fig_bond_curve.add_trace(
                    go.Scatter(
                        x=[bond_maturity],
                        y=[bond_oas_now],
                        mode="markers+text",
                        name=sel_bond_curve,
                        text=[sel_bond_curve],
                        textposition="top center",
                        marker=dict(
                            size=14,
                            color=COLORS["warning"],
                            symbol="star",
                            line=dict(color="#fff", width=1),
                        ),
                        showlegend=True,
                    )
                )
            st.plotly_chart(fig_bond_curve, use_container_width=True)

        # ── Cross-country comparison ──────────────────────────────────────────
        with st.expander("Cross-Country Comparison (same maturity node)"):
            sel_mat = st.selectbox(
                "Maturity Node", ["2Y", "5Y", "10Y", "30Y"], index=2, key="xc_mat"
            )
            rows: list[dict] = []
            for c in all_countries:
                cdf = extract_country_curve(meta, oas_df, c)
                if not cdf.empty and sel_mat in cdf.columns:
                    rows.append({
                        "Country": c,
                        f"OAS {sel_mat} (bps)": round(cdf[sel_mat].iloc[-1], 1),
                        "1M Chg (bps)": round(
                            cdf[sel_mat].iloc[-1] - cdf[sel_mat].iloc[-22]
                            if len(cdf) >= 22 else 0, 1
                        ),
                    })
            if rows:
                xc_df = pd.DataFrame(rows).sort_values(f"OAS {sel_mat} (bps)", ascending=False)
                st.dataframe(xc_df, use_container_width=True, hide_index=True)

        # ── Opportunity Screener ──────────────────────────────────────────────
        st.divider()
        st.markdown("**Curve Dislocation Screener — Full EM Universe**")
        st.caption(
            "Curve metrics with |z-score| > 1.5 across the EM sovereign universe, "
            "sorted by signal strength. Conviction scores incorporate z-score magnitude, "
            "percentile rank, cross-country context, and macro beta."
        )

        if curve_trades_df.empty:
            st.info("No curve dislocations flagged at this time.")
        else:
            sel_countries_ct = st.multiselect(
                "Filter by Country",
                sorted(curve_trades_df["country"].unique()),
                default=sorted(curve_trades_df["country"].unique()),
                key="cv_opp_countries",
            )
            ct_filtered = curve_trades_df[curve_trades_df["country"].isin(sel_countries_ct)]

            for _, row in ct_filtered.head(5).iterrows():
                _render_curve_opportunity_card(row, meta, oas_df, macro_returns, zscore_window)

            with st.expander("Full screener table"):
                st.dataframe(ct_filtered, use_container_width=True, height=380)
                csv_ct = ct_filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Screener CSV", csv_ct, "curve_screener.csv", "text/csv"
                )

    # ── Tab 2: Slope & Butterfly metrics ──────────────────────────────────────
    with tab2:
        t2_col1, t2_col2 = st.columns([2, 1])
        t2_country = t2_col1.selectbox(
            "Country", all_countries,
            index=all_countries.index(sel_country) if sel_country in all_countries else 0,
            key="t2_country_sel",
        )
        t2_window = t2_col2.selectbox(
            "Z-Score Window", [63, 126, 252], index=2,
            format_func=lambda x: f"{x}d (~{x//21}M)",
            key="t2_zscore_window",
        )
        t2_curve_df = extract_country_curve(meta, oas_df, t2_country)

        st.subheader(f"{t2_country} — Slope & Butterfly Metrics")

        if t2_curve_df.empty or t2_curve_df.shape[1] < 2:
            st.warning("Need at least 2 maturity nodes to compute slopes.")
        else:
            slopes_df      = compute_all_slopes(t2_curve_df)
            butterflies_df = compute_all_butterflies(t2_curve_df)
            all_metrics    = pd.concat([slopes_df, butterflies_df], axis=1)

            if all_metrics.empty:
                st.info("No metrics computable for this country.")
            else:
                zs_df  = curve_zscores(all_metrics, window=t2_window)
                pct_df = curve_percentiles(all_metrics, window=t2_window)

                st.markdown("**Current Curve Metrics**")

                # Pre-compute inflation context for all metrics
                infl_cache: dict[str, dict] = {}
                for metric in all_metrics.columns:
                    zs = zs_df[metric].iloc[-1]
                    if not np.isnan(zs) and abs(zs) >= 1.5:
                        infl_cache[metric] = compute_inflation_context(
                            metric, zs, macro_returns, window=21
                        )

                summary_rows = []
                for metric in all_metrics.columns:
                    val  = all_metrics[metric].iloc[-1]
                    zs   = zs_df[metric].iloc[-1]
                    pct  = pct_df[metric].iloc[-1]
                    is_bf = "butterfly" in metric
                    conv  = _compute_conviction(zs, pct)

                    if not np.isnan(zs) and abs(zs) >= 1.5:
                        signal_label = conv["label"]
                        infl_label   = infl_cache.get(metric, {}).get("label", "—")
                    else:
                        signal_label = "—"
                        infl_label   = "—"

                    summary_rows.append({
                        "Metric":        metric,
                        "Current (bps)": round(val, 1) if not np.isnan(val) else None,
                        "Z-Score":       round(zs, 2)  if not np.isnan(zs) else None,
                        "Percentile":    round(pct, 0) if not np.isnan(pct) else None,
                        "Conviction":    signal_label,
                        "Inflation":     infl_label,
                        "Type":          "Butterfly" if is_bf else "Slope",
                    })

                summ_df = pd.DataFrame(summary_rows)
                st.dataframe(summ_df, use_container_width=True, hide_index=True)

                st.caption(
                    "Inflation column uses a composite proxy: GLD (+35%), PDBC (+35%), "
                    "TLT (−30%). In production, real FRED breakevens (T5YIE / T10YIE / T5YIFR) "
                    "would be used instead."
                )

                sel_metric = st.selectbox(
                    "Plot History For", all_metrics.columns.tolist(), index=0,
                    key="t2_metric_sel",
                )
                if sel_metric:
                    fig_slope = plot_slope_history(
                        all_metrics[sel_metric],
                        zs_df[sel_metric] if sel_metric in zs_df else None,
                        metric_name=sel_metric,
                        height=350,
                    )
                    st.plotly_chart(fig_slope, use_container_width=True)

                    # ── Inflation context detail block ────────────────────────
                    sel_zs = zs_df[sel_metric].iloc[-1]
                    if not np.isnan(sel_zs) and abs(sel_zs) >= 1.5:
                        infl = infl_cache.get(
                            sel_metric,
                            compute_inflation_context(sel_metric, sel_zs, macro_returns),
                        )
                        _render_inflation_context_block(infl, sel_metric)
                    else:
                        st.caption(
                            "No inflation context shown — signal below threshold (|z| < 1.5)."
                        )

    # ── Tab 3: DV01-neutral trade sizing ──────────────────────────────────────
    with tab3:
        st.subheader("DV01-Neutral Trade Sizing Calculator")
        st.caption(
            "Compute the notional sizes needed to make a curve trade "
            "P&L-neutral to a parallel shift in spreads."
        )

        col_country, col_type = st.columns([1, 2])
        with col_country:
            dv01_country = st.selectbox(
                "Country", all_countries,
                index=all_countries.index(sel_country) if sel_country in all_countries else 0,
                key="dv01_country",
            )
        with col_type:
            c_type = st.radio("Trade Type", ["2-Leg (Slope)", "3-Leg (Butterfly)"], horizontal=True)

        country_meta = meta[meta["country"] == dv01_country].sort_values("maturity")

        if country_meta.empty:
            st.warning(f"No bonds found for {dv01_country}.")
        else:
            bond_options = {
                f"{row['maturity']}Y — {row['id']} (DV01=${row['dv01']:,})": row
                for _, row in country_meta.iterrows()
            }

            if c_type == "2-Leg (Slope)":
                col1, col2 = st.columns(2)
                with col1:
                    sel_short = st.selectbox(
                        "Short-end Bond (buy)", list(bond_options.keys()), index=0
                    )
                with col2:
                    sel_long = st.selectbox(
                        "Long-end Bond (sell)", list(bond_options.keys()),
                        index=min(1, len(bond_options) - 1),
                    )

                notional_short = st.number_input(
                    "Short-End Notional ($mm)", min_value=1.0, max_value=500.0,
                    value=10.0, step=1.0,
                )

                b_short = bond_options[sel_short]
                b_long  = bond_options[sel_long]

                sizing = dv01_neutral_weight(
                    b_short["dv01"],
                    b_long["dv01"],
                    notional_short=notional_short * 1_000_000,
                )

                if sizing:
                    _render_sizing_table(sizing, c_type, b_short, b_long)

                    net_carry = (
                        b_short["oas_base"] * sizing["notional_short"] / 1e6 / 10_000
                        - b_long["oas_base"]  * sizing["notional_long"]  / 1e6 / 10_000
                    ) * 10_000 / 12
                    st.info(
                        f"Approx. monthly carry on the trade: **{net_carry:+.1f} bps** "
                        f"(long {sel_short.split(' — ')[0]}, short {sel_long.split(' — ')[0]})."
                    )

            else:
                options = list(bond_options.keys())
                col1, col2, col3 = st.columns(3)
                sel_s = col1.selectbox("Short-end (wing)", options, index=0)
                sel_b = col2.selectbox("Belly (long)",     options, index=min(1, len(options)-1))
                sel_l = col3.selectbox("Long-end (wing)",  options, index=min(2, len(options)-1))

                notional_belly = st.number_input(
                    "Belly Notional ($mm)", min_value=1.0, max_value=500.0,
                    value=10.0, step=1.0,
                )

                b_s = bond_options[sel_s]
                b_b = bond_options[sel_b]
                b_l = bond_options[sel_l]

                sizing = dv01_neutral_butterfly(
                    b_s["dv01"], b_b["dv01"], b_l["dv01"],
                    notional_belly=notional_belly * 1_000_000,
                )
                if sizing:
                    _render_butterfly_sizing(sizing, b_s, b_b, b_l)


# ═════════════════════════════════════════════════════════════════════════════
# Signal quality helpers
# ═════════════════════════════════════════════════════════════════════════════

def _compute_conviction(zs: float, pct: float) -> dict:
    """
    Assign a conviction label based on z-score magnitude and percentile extremity.

    Rules:
      High     : |z| >= 2.5 AND (pct >= 90 or pct <= 10)
      Medium   : |z| >= 2.0 OR  (pct >= 85 or pct <= 15)
      Low      : everything else above the 1.5 threshold
    """
    if np.isnan(zs) or np.isnan(pct):
        return {"label": "Low conviction",  "color": COLORS["text_muted"],  "badge": "🟡"}

    abs_z    = abs(zs)
    extreme  = pct >= 90 or pct <= 10

    if abs_z >= 2.5 and extreme:
        return {"label": "High conviction",   "color": COLORS["positive"], "badge": "🟢"}
    if abs_z >= 2.0 or (pct >= 85 or pct <= 15):
        return {"label": "Medium conviction", "color": COLORS["warning"],  "badge": "🟡"}
    return     {"label": "Low conviction",    "color": COLORS["text_muted"], "badge": "⚪"}


def _cross_country_check(
    meta: pd.DataFrame,
    oas_df: pd.DataFrame,
    metric: str,
    zs_direction: float,
    zscore_window: int,
    signal_country: str = "",
) -> dict:
    """
    Check how many same-region peers show the same slope/fly signal.

    Compares only within the same geographic region (LATAM, EMEA, Asia)
    so the comparison is economically meaningful. A LATAM steepener in
    Brazil is only informative when compared to Mexico/Colombia/Peru/Chile,
    not to Turkey or Indonesia which are driven by different macro factors.

    For regions with a single country (e.g. Indonesia in Asia), falls back
    to comparing against all countries in the portfolio.

    Returns a dict with:
      - n_same           : int
      - total            : int
      - region           : str
      - interpretation   : str
    """
    is_butterfly = "butterfly" in metric
    raw_nodes    = metric.replace(" slope", "").replace(" butterfly", "").split("/")

    # Determine the region of the signal country and filter peers accordingly
    signal_region = _REGION_MAP.get(signal_country, "")
    all_countries = meta["country"].unique()
    region_peers  = [c for c in all_countries if _REGION_MAP.get(c) == signal_region and c != signal_country]

    # Fallback: if region has <2 peers (e.g. Indonesia alone in Asia), use all countries
    if len(region_peers) < 2:
        region_peers  = [c for c in all_countries if c != signal_country]
        signal_region = "Portfolio"

    count_same  = 0
    count_total = 0

    for country in region_peers:
        cdf = extract_country_curve(meta, oas_df, country)
        if cdf.empty:
            continue

        if is_butterfly and len(raw_nodes) == 3:
            s, m, l = raw_nodes
            if all(n in cdf.columns for n in [s, m, l]):
                series = 2 * cdf[m] - cdf[s] - cdf[l]
            else:
                continue
        elif not is_butterfly and len(raw_nodes) == 2:
            s, l = raw_nodes
            if s in cdf.columns and l in cdf.columns:
                series = cdf[l] - cdf[s]
            else:
                continue
        else:
            continue

        if len(series.dropna()) < zscore_window // 2:
            continue

        roll_mean = series.rolling(zscore_window, min_periods=zscore_window // 2).mean()
        roll_std  = series.rolling(zscore_window, min_periods=zscore_window // 2).std()
        zs_series = (series - roll_mean) / roll_std.replace(0, np.nan)
        zs_val    = zs_series.iloc[-1]

        if pd.isna(zs_val):
            continue

        count_total += 1
        if (zs_direction > 0 and zs_val > 1.5) or (zs_direction < 0 and zs_val < -1.5):
            count_same += 1

    if count_total == 0:
        return {"n_same": 0, "total": 0, "region": signal_region,
                "interpretation": f"No {signal_region} peers with sufficient data to compare."}

    frac        = count_same / count_total
    region_label = signal_region if signal_region else "portfolio"

    if frac >= 0.5:
        interp = (
            f"Regional move — {count_same}/{count_total} {region_label} peers show the same signal. "
            f"Likely driven by a regional factor rather than a specific opportunity."
        )
    elif frac >= 0.25:
        interp = (
            f"Partially regional — {count_same}/{count_total} {region_label} peers show similar move. "
            f"Signal retains some country-specific component."
        )
    else:
        interp = (
            f"Country-specific — only {count_same}/{count_total} {region_label} peers show a similar signal. "
            f"More likely a genuine {signal_country} dislocation."
        )

    return {"n_same": count_same, "total": count_total, "region": signal_region, "interpretation": interp}


def _macro_beta_check(
    metric_series: pd.Series,
    macro_returns: pd.DataFrame,
    window: int = 63,
) -> dict:
    """
    Regress daily changes in the metric against macro factor returns.
    Returns the dominant factor and its correlation.
    """
    if macro_returns.empty or len(metric_series) < window:
        return {"factor": None, "corr": np.nan, "interpretation": "Macro beta data unavailable."}

    metric_chg = metric_series.diff().dropna().tail(window)
    candidates = [f for f in ["TLT", "UUP", "EMB", "SPY", "VIXY"] if f in macro_returns.columns]

    if not candidates:
        return {"factor": None, "corr": np.nan, "interpretation": "Macro beta data unavailable."}

    best_factor, best_corr = None, 0.0
    for fac in candidates:
        fac_ret = macro_returns[fac].tail(window)
        aligned  = pd.concat([metric_chg, fac_ret], axis=1).dropna()
        if len(aligned) < 20:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr   = corr
            best_factor = fac

    if best_factor is None or abs(best_corr) < 0.2:
        interp = "Signal appears idiosyncratic vs macro drivers tested."
    elif abs(best_corr) >= 0.5:
        label  = _MACRO_LABELS.get(best_factor, best_factor)
        interp = f"Signal largely explained by {label} (corr={best_corr:+.2f}). Review whether it is macro-driven."
    else:
        label  = _MACRO_LABELS.get(best_factor, best_factor)
        interp = f"Partial macro link to {label} (corr={best_corr:+.2f}). Signal retains some idiosyncratic component."

    return {"factor": best_factor, "corr": best_corr, "interpretation": interp}


# ═════════════════════════════════════════════════════════════════════════════
# Render helpers
# ═════════════════════════════════════════════════════════════════════════════

def _render_curve_opportunity_card(
    row: pd.Series,
    meta: pd.DataFrame,
    oas_df: pd.DataFrame,
    macro_returns: pd.DataFrame,
    zscore_window: int,
) -> None:
    zs        = row["zscore"]
    pct       = row.get("percentile", np.nan)
    direction = row["trade_direction"]
    colour    = COLORS["positive"] if zs > 0 else COLORS["negative"]
    arrow     = "▲" if zs > 0 else "▼"

    # Extract row fields first so they are available everywhere below
    country = row["country"]
    metric  = row["metric"]

    # Conviction
    conv = _compute_conviction(zs, pct)

    # Wording based on conviction
    if conv["label"] == "High conviction":
        signal_word = "Tradable setup"
    elif conv["label"] == "Medium conviction":
        signal_word = "Signal worth reviewing"
    else:
        signal_word = "Potential dislocation"

    # Cross-country check (same-region peers only)
    xc = _cross_country_check(meta, oas_df, metric, zs, zscore_window,
                               signal_country=country)

    # Macro beta check — rebuild metric series
    is_bf    = "butterfly" in metric
    raw_nodes = metric.replace(" slope", "").replace(" butterfly", "").split("/")
    cdf       = extract_country_curve(meta, oas_df, country)
    beta_info = {"factor": None, "corr": np.nan, "interpretation": "Macro beta data unavailable."}

    if not cdf.empty:
        try:
            if is_bf and len(raw_nodes) == 3:
                s, m, l = raw_nodes
                if all(n in cdf.columns for n in [s, m, l]):
                    metric_series = 2 * cdf[m] - cdf[s] - cdf[l]
                    beta_info = _macro_beta_check(metric_series, macro_returns, window=zscore_window)
            elif not is_bf and len(raw_nodes) == 2:
                s, l = raw_nodes
                if s in cdf.columns and l in cdf.columns:
                    metric_series = cdf[l] - cdf[s]
                    beta_info = _macro_beta_check(metric_series, macro_returns, window=zscore_window)
        except Exception:
            pass

    dv01_str = (
        f" &nbsp;·&nbsp; DV01 ratio: {row['dv01_weight_ratio']:.2f}x"
        if not pd.isna(row.get("dv01_weight_ratio")) else ""
    )

    st.markdown(
        f"""
<div style="background:{COLORS['surface']};border:1px solid {COLORS['surface2']};
     border-left:4px solid {colour};border-radius:6px;padding:12px 16px;margin-bottom:10px;">

  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
    <div>
      <span style="font-size:14px;font-weight:700;color:{COLORS['text']}">
        {row['country']} — {row['metric']}
      </span>
      &nbsp;&nbsp;
      <span style="font-size:11px;font-weight:600;color:{conv['color']};">
        {conv['badge']} {conv['label']}
      </span>
    </div>
    <div style="text-align:right;">
      <span style="color:{colour};font-weight:600;font-size:13px;">
        {arrow} {signal_word}
      </span>
      <span style="color:{COLORS['text_muted']};font-size:12px;margin-left:8px;">
        z = {zs:+.2f}
      </span>
    </div>
  </div>

  <div style="font-size:11px;color:{COLORS['text_muted']};margin-top:6px;">
    Current: {row['current_bps']:.1f} bps &nbsp;·&nbsp;
    {pct:.0f}th percentile &nbsp;·&nbsp;
    Suggests: <b style="color:{COLORS['text']}">{direction.title()}</b>
    {dv01_str}
  </div>

  <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div style="background:{COLORS['surface2']};border-radius:4px;padding:7px 10px;font-size:11px;color:{COLORS['text_muted']};">
      <b style="color:{COLORS['text']}">Cross-country</b><br>{xc['interpretation']}
    </div>
    <div style="background:{COLORS['surface2']};border-radius:4px;padding:7px 10px;font-size:11px;color:{COLORS['text_muted']};">
      <b style="color:{COLORS['text']}">Macro beta</b><br>{beta_info['interpretation']}
    </div>
  </div>

</div>
""",
        unsafe_allow_html=True,
    )


def _render_sizing_table(
    sizing: dict,
    trade_type: str,
    b_short: pd.Series,
    b_long: pd.Series,
) -> None:
    cols = st.columns(2)
    with cols[0]:
        st.metric(
            f"Short-end Notional ({b_short['id']})",
            f"${sizing['notional_short']:,.0f}",
        )
        st.metric("DV01 Short Leg", f"${sizing['dv01_short_leg']:,.1f}")
    with cols[1]:
        st.metric(
            f"Long-end Notional ({b_long['id']})",
            f"${sizing['notional_long']:,.0f}",
        )
        st.metric("DV01 Long Leg", f"${sizing['dv01_long_leg']:,.1f}")

    st.metric(
        "Weight Ratio (long/short)",
        f"{sizing['weight_ratio']:.3f}x",
        help="Multiply short-end notional by this ratio to get long-end notional.",
    )


def _render_butterfly_sizing(
    sizing: dict,
    b_s: pd.Series,
    b_b: pd.Series,
    b_l: pd.Series,
) -> None:
    cols = st.columns(3)
    cols[0].metric(f"Short Wing ({b_s['id']})",  f"${sizing['notional_short']:,.0f}")
    cols[1].metric(f"Belly ({b_b['id']})",        f"${sizing['notional_belly']:,.0f}")
    cols[2].metric(f"Long Wing ({b_l['id']})",    f"${sizing['notional_long']:,.0f}")

    c2a, c2b, c2c = st.columns(3)
    c2a.metric("DV01 Short Wing", f"${sizing['dv01_short_leg']:,.1f}")
    c2b.metric("DV01 Belly",      f"${sizing['dv01_belly_leg']:,.1f}")
    c2c.metric("DV01 Long Wing",  f"${sizing['dv01_long_leg']:,.1f}")


def _render_inflation_context_block(infl: dict, metric_name: str) -> None:
    """
    Render a compact Inflation Context card below the metric history chart.
    """
    label      = infl.get("label", "Inflation-neutral")
    interp     = infl.get("interpretation", "")
    infl_z     = infl.get("infl_proxy_z", float("nan"))
    trend      = infl.get("infl_trend", "neutral")
    tickers    = infl.get("proxy_tickers", ["GLD", "PDBC", "TLT"])
    color_key  = infl.get("color", "text_muted")

    # Map color key to actual hex value
    border_color = {
        "positive":  COLORS.get("positive", "#22c55e"),
        "negative":  COLORS.get("negative", "#ef4444"),
        "text_muted": COLORS.get("text_muted", "#6b7280"),
    }.get(color_key, COLORS.get("text_muted", "#6b7280"))

    label_color = {
        "positive":  "#22c55e",
        "negative":  "#ef4444",
        "text_muted": "#94a3b8",
    }.get(color_key, "#94a3b8")

    trend_icon = {"rising": "↑", "falling": "↓", "neutral": "→"}.get(trend, "→")
    z_str = f"{infl_z:+.2f}" if not (infl_z != infl_z) else "N/A"  # NaN check

    tickers_str = " · ".join(tickers)

    st.markdown(
        f"""
<div style="background:{COLORS['surface']};border:1px solid {COLORS['surface2']};
     border-left:4px solid {border_color};border-radius:6px;
     padding:12px 16px;margin-top:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
    <span style="font-size:13px;font-weight:700;color:{COLORS['text']};">
      Inflation Context — <em>{metric_name}</em>
    </span>
    <span style="font-size:13px;font-weight:700;color:{label_color};">
      {label}
    </span>
  </div>
  <div style="margin-top:8px;font-size:12px;color:{COLORS['text_muted']};">
    {interp}
  </div>
  <div style="margin-top:8px;display:flex;gap:20px;font-size:11px;color:{COLORS['text_muted']};">
    <span>Inflation proxy z-score: <b style="color:{COLORS['text']}">{z_str}</b></span>
    <span>Trend: <b style="color:{COLORS['text']}">{trend_icon} {trend.capitalize()}</b></span>
    <span>Proxy: <b style="color:{COLORS['text']}">{tickers_str}</b></span>
  </div>
  <div style="margin-top:6px;font-size:10px;color:{COLORS['text_muted']};font-style:italic;">
    Note — This is a macro proxy (GLD/PDBC/TLT composite). In live production,
    real FRED breakevens (T5YIE · T10YIE · T5YIFR) would be used instead.
    Treat as directional context only.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
