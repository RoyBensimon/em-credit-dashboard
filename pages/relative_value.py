"""
Relative Value Screener Page

Allows the user to:
  - View the full RV screener table with z-scores
  - Drill into an issuer's OAS curve vs. fitted fair value
  - View time-series of OAS and residuals for a selected bond
  - Filter by country / rating / maturity bucket
  - See top cheap / top rich rankings
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config.settings import COLORS
from config.theme import STREAMLIT_CSS, apply_chart_style
from src.data.session import get_app_data
from src.analytics.relative_value import (
    fit_issuer_curve,
    top_cheap_rich,
)
from src.analytics.correlation import (
    MONTHLY_INTERPOLATED_BONDS,
    compute_bond_bond_corr,
    compute_spread_zscore_matrix,
)
from src.plotting.charts import (
    plot_rv_scatter,
    plot_rv_zscore_bar,
    plot_zscore_matrix,
)
from src.plotting.tables import style_rv_table


_BB_WINDOW_MAP: dict[str, int | None] = {
    "30 days":     30,
    "60 days":     60,
    "90 days":     90,
    "180 days":    180,
    "1 year":      252,
    "Full sample": None,
}


def _build_rv_pairs(
    corr_matrix: pd.DataFrame,
    zscore_matrix: pd.DataFrame,
    min_corr: float = 0.60,
    min_z: float = 1.5,
) -> pd.DataFrame:
    """
    Identify bond pairs with both high correlation and a significant spread
    Z-score — the core RV signal.

    Convention:
        Z > 0  →  Bond A cheap vs Bond B  →  Long A / Short B
        Z < 0  →  Bond A rich vs Bond B   →  Short A / Long B
    """
    bonds = corr_matrix.index.tolist()
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []

    for a in bonds:
        for b in bonds:
            if a >= b:
                continue
            if (a, b) in seen:
                continue

            corr = corr_matrix.loc[a, b] if (a in corr_matrix.index and b in corr_matrix.columns) else np.nan
            z_ab = zscore_matrix.loc[a, b] if (a in zscore_matrix.index and b in zscore_matrix.columns) else np.nan

            if np.isnan(corr) or np.isnan(z_ab):
                continue
            if corr < min_corr or abs(z_ab) < min_z:
                continue

            if z_ab > 0:
                signal = f"{a} cheap vs {b}"
                trade  = f"Long {a}  /  Short {b}"
            else:
                signal = f"{a} rich vs {b}"
                trade  = f"Short {a}  /  Long {b}"

            rows.append({
                "Bond A":      a,
                "Bond B":      b,
                "Correlation": round(corr, 2),
                "Spread Z":    round(z_ab, 2),
                "Signal":      signal,
                "Trade":       trade,
            })
            seen.add((a, b))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["_absz"] = df["Spread Z"].abs()
    return df.sort_values(["_absz", "Correlation"], ascending=[False, False]).drop(columns="_absz").reset_index(drop=True)


def render() -> None:
    """Render the Relative Value Screener page."""
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    st.title("Relative Value Screener")
    st.caption(
        "Rich / cheap identification across the EM sovereign bond universe. "
        "Z-scores are based on residuals from polynomial OAS-vs-maturity curve fits."
    )

    data = get_app_data()

    rv_df  = data["rv_universe"]
    meta   = data["meta"]
    oas_df = data["oas_df"]

    # ── Sidebar filters ────────────────────────────────────────────────────────
    st.sidebar.markdown("### RV Filters")

    all_countries = sorted(rv_df["country"].unique())
    sel_countries = st.sidebar.multiselect(
        "Countries", all_countries, default=all_countries
    )

    all_ratings = sorted(rv_df["rating"].unique())
    sel_ratings = st.sidebar.multiselect(
        "Ratings", all_ratings, default=all_ratings
    )

    mat_options = sorted(rv_df["maturity"].unique())
    sel_mats = st.sidebar.multiselect(
        "Maturity Buckets (Y)", mat_options, default=mat_options
    )

    rv_filter = st.sidebar.radio(
        "Show", ["All", "Cheap", "Rich", "Neutral"],
        horizontal=True,
    )

    # Apply filters
    mask = (
        rv_df["country"].isin(sel_countries) &
        rv_df["rating"].isin(sel_ratings) &
        rv_df["maturity"].isin(sel_mats)
    )
    if rv_filter != "All":
        mask &= rv_df["rv_label"] == rv_filter
    filtered = rv_df[mask].copy()

    # ── KPI row ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe Size",   len(filtered))
    c2.metric("Cheap 🟢",  int((filtered["rv_label"] == "Cheap").sum()))
    c3.metric("Rich 🔴",   int((filtered["rv_label"] == "Rich").sum()))
    avg_z = filtered["zscore"].abs().mean()
    c4.metric("Avg |Z-Score|",  f"{avg_z:.2f}" if not np.isnan(avg_z) else "—")
    avg_oas = filtered["oas_current"].mean()
    c5.metric("Avg OAS (bps)",  f"{avg_oas:.0f}" if not np.isnan(avg_oas) else "—")

    st.divider()

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "RV Pair Trades", "Universe Table", "Issuer Curve Fit", "Spread Z-Score",
    ])

    # ── Tab 1: RV Pair Trades ─────────────────────────────────────────────────
    with tab1:
        st.subheader("RV Pair Trade Signals")
        st.caption(
            "Pairs flagged by both high OAS-change correlation AND a significant spread Z-score. "
            "Only daily-data bonds are included. "
            "Convention: Z > 0 → Bond A cheap vs B → Long A / Short B."
        )

        col_p1, col_p2, col_p3 = st.columns(3)
        min_corr_inp = col_p1.slider("Min Correlation", 0.30, 0.95, 0.60, 0.05, key="rv_min_corr")
        min_z_inp    = col_p2.slider("Min |Spread Z|",  0.50, 4.00, 1.50, 0.25, key="rv_min_z")
        pair_win_lbl = col_p3.selectbox(
            "Correlation window", list(_BB_WINDOW_MAP.keys()), index=3, key="rv_pair_window"
        )
        pair_win = _BB_WINDOW_MAP[pair_win_lbl]
        pair_z_lookback = pair_win if pair_win is not None else len(oas_df)

        oas_chg_pairs = oas_df.diff().dropna()
        corr_mat_pairs  = compute_bond_bond_corr(oas_chg_pairs, window=pair_win, exclude_monthly=True)
        zscore_mat_pairs = compute_spread_zscore_matrix(
            oas_df, z_window=pair_z_lookback, exclude_monthly=True
        )

        pairs_df = _build_rv_pairs(corr_mat_pairs, zscore_mat_pairs, min_corr_inp, min_z_inp)

        if pairs_df.empty:
            st.info(
                f"No pairs found with correlation ≥ {min_corr_inp:.2f} "
                f"and |Spread Z| ≥ {min_z_inp:.2f}. Try loosening the filters."
            )
        else:
            st.markdown(f"**{len(pairs_df)} pair(s) identified**")

            for _, row in pairs_df.head(10).iterrows():
                z = row["Spread Z"]
                direction = "cheap" if z > 0 else "rich"
                card_html = (
                    f'<div class="trade-card {direction}">'
                    f'<strong>{row["Trade"]}</strong>'
                    f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                    f'Corr: <strong>{row["Correlation"]:+.2f}</strong>'
                    f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                    f'Spread Z: <strong>{z:+.2f}</strong>'
                    f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                    f'<em>{row["Signal"]}</em>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

            st.divider()

            def _color_zscore(val):
                try:
                    v = float(val)
                    if v >= 2:
                        return "background-color:#1a4731;color:#4ade80"
                    elif v <= -2:
                        return "background-color:#4a1a1a;color:#f87171"
                    elif v >= 1:
                        return "background-color:#1a3320;color:#86efac"
                    elif v <= -1:
                        return "background-color:#3a1a1a;color:#fca5a5"
                    return ""
                except (TypeError, ValueError):
                    return ""

            def _color_corr(val):
                try:
                    v = float(val)
                    if v >= 0.8:
                        return "background-color:#1a2f4a;color:#93c5fd"
                    elif v >= 0.6:
                        return "background-color:#1a2540;color:#bfdbfe"
                    return ""
                except (TypeError, ValueError):
                    return ""

            styled = pairs_df.style.map(_color_zscore, subset=["Spread Z"]).map(
                _color_corr, subset=["Correlation"]
            )
            st.dataframe(styled, use_container_width=True, height=400)

            csv_pairs = pairs_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Pair Signals", csv_pairs, "rv_pair_trades.csv", "text/csv"
            )

    # ── Tab 2: Universe table ──────────────────────────────────────────────────
    with tab2:
        st.subheader("RV Screener — Full Universe")
        col_help, col_sort = st.columns([3, 1])
        sort_by = col_sort.selectbox(
            "Sort by", ["zscore", "oas_current", "residual_bps", "country"],
            index=0,
        )
        asc = col_sort.checkbox("Ascending", value=False)
        disp = filtered.sort_values(sort_by, ascending=asc, na_position="last")

        st.dataframe(
            style_rv_table(disp),
            use_container_width=True,
            height=500,
        )

        # Download button
        csv = disp.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV", csv, "rv_screener.csv", "text/csv"
        )

        # Top cheap / rich summary
        cheap, rich = top_cheap_rich(rv_df, n=5)
        cc, cr = st.columns(2)
        with cc:
            st.markdown(f"#### Top 5 Cheap")
            if not cheap.empty:
                for _, row in cheap.iterrows():
                    st.markdown(
                        f"**{row['bond_id']}** — OAS {row['oas_current']:.0f}bps, "
                        f"z={row['zscore']:+.2f} (+{row['residual_bps']:.0f}bps wide)",
                    )
            else:
                st.info("No bonds flagged as cheap under current filters.")
        with cr:
            st.markdown(f"#### Top 5 Rich")
            if not rich.empty:
                for _, row in rich.iterrows():
                    st.markdown(
                        f"**{row['bond_id']}** — OAS {row['oas_current']:.0f}bps, "
                        f"z={row['zscore']:+.2f} ({row['residual_bps']:+.0f}bps tight)",
                    )
            else:
                st.info("No bonds flagged as rich under current filters.")


    # ── Tab 3: Issuer curve fit ────────────────────────────────────────────────
    with tab3:
        st.subheader("Issuer OAS Curve — Fair Value Fit")

        sel_country = st.selectbox(
            "Select Country", sorted(meta["country"].unique()), index=0,
            key="rv_country_sel",
        )

        oas_latest = oas_df.iloc[-1]
        curve_res  = fit_issuer_curve(meta, oas_latest, sel_country)

        if curve_res:
            fig_cv = plot_rv_scatter(curve_res, sel_country, height=420)
            st.plotly_chart(fig_cv, use_container_width=True)

            r2 = curve_res.get("r_squared", np.nan)
            st.caption(
                f"Polynomial degree-2 fit R² = {r2:.1%}. "
                "Green dots = cheap (trading wide of curve); "
                "Red dots = rich (tight)."
            )

            # Show residuals table for this country
            country_bonds = meta[meta["country"] == sel_country]["id"].tolist()
            resid_df = pd.DataFrame({
                "Bond":      curve_res["bond_ids"],
                "Maturity":  curve_res["maturities"],
                "OAS (bps)": curve_res["oas_actual"],
                "Fair (bps)":curve_res["oas_fitted"],
                "Resid (bps)":curve_res["residuals"],
            })
            resid_df["Label"] = resid_df["Resid (bps)"].apply(
                lambda r: "Cheap" if r > 10 else ("Rich" if r < -10 else "Neutral")
            )
            st.dataframe(resid_df.round(1), use_container_width=True, height=200)
        else:
            st.info(f"Not enough bonds for {sel_country} to fit a curve (need ≥2).")




    # ── Tab 4: Spread Z-Score Matrix ──────────────────────────────────────────
    with tab4:
        st.subheader("OAS Spread Z-Score Matrix")
        st.caption(
            "Z-score of the OAS_A - OAS_B spread over a rolling window. "
            "Green = A cheap vs B (spread wide vs history) → Buy A / Sell B. "
            "Red = A rich vs B → Sell A / Buy B. "
            "Matrix is antisymmetric: Z[A,B] = -Z[B,A]."
        )

        zs_window_label = st.selectbox(
            "Z-score lookback", list(_BB_WINDOW_MAP.keys()), index=4, key="bb_z_window"
        )
        zs_window = _BB_WINDOW_MAP[zs_window_label]
        z_lookback = zs_window if zs_window is not None else len(oas_df)

        zscore_mat = compute_spread_zscore_matrix(
            oas_df, z_window=z_lookback, exclude_monthly=True
        )

        if zscore_mat.empty:
            st.info("Not enough data to compute spread Z-scores.")
        else:
            fig_zmat = plot_zscore_matrix(
                zscore_mat,
                title=f"OAS Spread Z-Score — {zs_window_label} window",
                height=max(400, len(zscore_mat) * 22),
            )
            st.plotly_chart(fig_zmat, use_container_width=True)

            # KPI: count of pairs outside ±1.5
            n_cheap = int((zscore_mat.values > 1.5).sum() // 2)
            n_rich  = int((zscore_mat.values < -1.5).sum() // 2)
            kc1, kc2 = st.columns(2)
            kc1.metric("Pairs: A cheap vs B  (Z > +1.5)", n_cheap)
            kc2.metric("Pairs: A rich vs B   (Z < -1.5)", n_rich)

            csv_zs = zscore_mat.round(3).to_csv().encode("utf-8")
            st.download_button(
                "Download Z-Score Matrix", csv_zs,
                f"spread_zscore_{zs_window_label.replace(' ', '_')}.csv", "text/csv",
            )

