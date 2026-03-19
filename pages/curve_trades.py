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
    compute_slope,
    compute_curvature,
    compute_all_slopes,
    compute_all_butterflies,
    curve_zscores,
    curve_percentiles,
    dv01_neutral_weight,
    dv01_neutral_butterfly,
)
from src.plotting.charts import (
    plot_yield_curve,
    plot_slope_history,
)


def render() -> None:
    """Render the Curve Trade Builder page."""
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    st.title("Curve Trade Builder")
    st.caption(
        "Identify steepener / flattener / butterfly opportunities using "
        "z-score analysis of OAS curve metrics across EM sovereigns."
    )

    data = get_app_data()
    meta   = data["meta"]
    oas_df = data["oas_df"]
    curve_trades_df = data["curve_trades"]

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
        st.caption("Sélectionnez un bond pour afficher la courbe OAS vs Maturité de son émetteur, avec le bond mis en évidence.")

        # All available bonds across the universe (excluding 30Y for clarity)
        all_bonds = sorted(
            b for b in meta["id"].tolist()
            if b in oas_df.columns and meta.loc[meta["id"] == b, "maturity"].values[0] != 30
        )

        sel_bond_curve = st.selectbox(
            "Select Bond", all_bonds, index=0, key="cv_bond_curve_sel"
        )

        # Derive country of selected bond and extract its curve
        bond_country = meta.loc[meta["id"] == sel_bond_curve, "country"].values[0]
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
            # Highlight the selected bond as a distinct marker
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
        st.markdown("**Curve Trade Opportunities — Full EM Universe**")
        st.caption(
            "Toutes les métriques de courbe avec |z-score| > 1.5 sur l'univers EM souverain. "
            "Triées par force de signal."
        )

        if curve_trades_df.empty:
            st.info("No curve trade opportunities flagged at this time.")
        else:
            sel_countries_ct = st.multiselect(
                "Filter by Country",
                sorted(curve_trades_df["country"].unique()),
                default=sorted(curve_trades_df["country"].unique()),
                key="cv_opp_countries",
            )
            ct_filtered = curve_trades_df[curve_trades_df["country"].isin(sel_countries_ct)]

            for _, row in ct_filtered.head(5).iterrows():
                _render_curve_opportunity_card(row)

            with st.expander("Full screener table"):
                st.dataframe(ct_filtered, use_container_width=True, height=380)
                csv_ct = ct_filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Screener CSV", csv_ct, "curve_screener.csv", "text/csv"
                )

    # ── Tab 2: Slope & Butterfly metrics ──────────────────────────────────────
    with tab2:
        st.subheader(f"{sel_country} — Slope & Butterfly Metrics")

        if curve_df.empty or curve_df.shape[1] < 2:
            st.warning("Need at least 2 maturity nodes to compute slopes.")
        else:
            slopes_df  = compute_all_slopes(curve_df)
            butterflies_df = compute_all_butterflies(curve_df)
            all_metrics = pd.concat([slopes_df, butterflies_df], axis=1)

            if all_metrics.empty:
                st.info("No metrics computable for this country.")
            else:
                zs_df  = curve_zscores(all_metrics, window=zscore_window)
                pct_df = curve_percentiles(all_metrics, window=zscore_window)

                # Current summary table
                st.markdown("**Current Curve Metrics**")
                summary_rows = []
                for metric in all_metrics.columns:
                    val = all_metrics[metric].iloc[-1]
                    zs  = zs_df[metric].iloc[-1]
                    pct = pct_df[metric].iloc[-1]
                    is_bf = "butterfly" in metric

                    if not np.isnan(zs) and abs(zs) >= 1.5:
                        flag = "🟢 Cheap" if zs > 0 else "🔴 Rich"
                    else:
                        flag = "—"

                    summary_rows.append({
                        "Metric":         metric,
                        "Current (bps)":  round(val, 1) if not np.isnan(val) else None,
                        "Z-Score":        round(zs, 2)  if not np.isnan(zs) else None,
                        "Percentile":     round(pct, 0) if not np.isnan(pct) else None,
                        "Signal":         flag,
                        "Type":           "Butterfly" if is_bf else "Slope",
                    })

                summ_df = pd.DataFrame(summary_rows)
                st.dataframe(summ_df, use_container_width=True, hide_index=True)

                # Metric selector for history chart
                sel_metric = st.selectbox(
                    "Plot History For", all_metrics.columns.tolist(), index=0
                )
                if sel_metric:
                    fig_slope = plot_slope_history(
                        all_metrics[sel_metric],
                        zs_df[sel_metric] if sel_metric in zs_df else None,
                        metric_name=sel_metric,
                        height=350,
                    )
                    st.plotly_chart(fig_slope, use_container_width=True)

    # ── Tab 3: DV01-neutral trade sizing ──────────────────────────────────────
    with tab3:
        st.subheader("DV01-Neutral Trade Sizing Calculator")
        st.caption(
            "Compute the notional sizes needed to make a curve trade "
            "P&L-neutral to a parallel shift in spreads."
        )

        c_type = st.radio("Trade Type", ["2-Leg (Slope)", "3-Leg (Butterfly)"], horizontal=True)
        country_meta = meta[meta["country"] == sel_country].sort_values("maturity")

        if country_meta.empty:
            st.warning(f"No bonds found for {sel_country}.")
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

            else:  # Butterfly
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
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

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


def _render_curve_opportunity_card(row: pd.Series) -> None:
    zs        = row["zscore"]
    direction = row["trade_direction"]
    colour    = COLORS["positive"] if zs > 0 else COLORS["negative"]
    arrow     = "▲" if zs > 0 else "▼"

    st.markdown(
        f"""
<div style="background:{COLORS['surface']};border:1px solid {COLORS['surface2']};
     border-left:4px solid {colour};border-radius:6px;padding:10px 14px;margin-bottom:8px;">
  <b style="color:{COLORS['text']}">{row['country']} — {row['metric']}</b>
  <span style="float:right;color:{colour};font-weight:600">{arrow} {direction.title()} | z={zs:+.2f}</span>
  <div style="font-size:11px;color:{COLORS['text_muted']};margin-top:4px">
    Current: {row['current_bps']:.1f}bps | {row['percentile']:.0f}th percentile
    {f" | DV01 ratio: {row['dv01_weight_ratio']:.2f}x" if not pd.isna(row.get('dv01_weight_ratio')) else ""}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
