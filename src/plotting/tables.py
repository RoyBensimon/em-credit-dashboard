"""
Styled table builders for the EM Credit Analytics Dashboard.

These functions return pandas Styler objects or Plotly table figures
that can be directly rendered in Streamlit via st.dataframe() or st.plotly_chart().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import COLORS


# ── Colour helpers ────────────────────────────────────────────────────────────

def _red_green_diverge(val: float, vmin: float = -3, vmax: float = 3) -> str:
    """Return a CSS background-color for a diverging red-green scale."""
    if pd.isna(val):
        return ""
    ratio = (val - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))
    if ratio > 0.5:
        g = int(150 + (ratio - 0.5) * 2 * 70)
        return f"background-color: rgba(16,185,129,{(ratio-0.5)*1.4:.2f}); color: #fff"
    else:
        r = int(150 + (0.5 - ratio) * 2 * 70)
        return f"background-color: rgba(239,68,68,{(0.5-ratio)*1.4:.2f}); color: #fff"


def _label_colour(label: str) -> str:
    """CSS colour for rich/cheap/neutral labels."""
    mapping = {
        "Cheap":   f"color: {COLORS['positive']}; font-weight: 600",
        "Rich":    f"color: {COLORS['negative']}; font-weight: 600",
        "Neutral": f"color: {COLORS['text_muted']}",
    }
    return mapping.get(label, "")


# ── Table stylers ─────────────────────────────────────────────────────────────

def style_rv_table(rv_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply conditional formatting to the RV screener table.

    - Z-score column: red-green diverging colour scale.
    - rv_label column: coloured text.
    - OAS change columns: sign-based colouring.
    """
    display_cols = [
        "bond_id", "country", "maturity", "rating",
        "oas_current", "oas_fair", "residual_bps",
        "zscore", "chg_1w_bps", "chg_1m_bps",
        "carry_bps_1m", "rv_label",
    ]
    df = rv_df[[c for c in display_cols if c in rv_df.columns]].copy()

    # Rename for display
    df.columns = [
        c.replace("_bps", " (bps)")
         .replace("_", " ")
         .replace("oas", "OAS")
         .replace("chg", "Chg")
         .title()
        for c in df.columns
    ]

    def colour_zscore(col: pd.Series) -> list[str]:
        return [_red_green_diverge(v, -3, 3) for v in col]

    def colour_changes(col: pd.Series) -> list[str]:
        return [
            f"color: {COLORS['positive']}" if v < 0
            else (f"color: {COLORS['negative']}" if v > 0 else "")
            for v in col
        ]

    styler = df.style

    # Apply to z-score column
    zscore_col = [c for c in df.columns if "Zscore" in c or "Z-Score" in c]
    for col in zscore_col:
        styler = styler.apply(colour_zscore, subset=[col])

    # Apply to change columns (tighter for OAS = negative is good)
    chg_cols = [c for c in df.columns if "Chg" in c]
    for col in chg_cols:
        styler = styler.apply(colour_changes, subset=[col])

    styler = styler.set_properties(**{
        "font-size": "12px",
        "text-align": "right",
    }).format({
        c: "{:.1f}" for c in df.select_dtypes("number").columns
    }, na_rep="—")

    return styler


def style_correlation_table(corr: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply red-green diverging colour scale to a correlation matrix."""
    def colour_cell(val: float) -> str:
        return _red_green_diverge(val, -1, 1)

    return (
        corr.style
        .applymap(colour_cell)
        .format("{:.2f}", na_rep="—")
        .set_properties(**{"font-size": "12px", "text-align": "center"})
    )


def style_beta_table(beta_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Colour beta columns by magnitude and sign."""
    beta_cols = [c for c in beta_df.columns if c.startswith("beta_")]

    def colour_beta(col: pd.Series) -> list[str]:
        return [_red_green_diverge(v, -2, 2) for v in col]

    styler = beta_df.style
    for col in beta_cols:
        if col in beta_df.columns:
            styler = styler.apply(colour_beta, subset=[col])

    return styler.format({
        c: "{:.3f}" for c in beta_df.select_dtypes("number").columns
    }, na_rep="—").set_properties(**{"font-size": "12px"})


def style_trade_table(ideas_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Light styling for the trade ideas summary table."""
    def colour_confidence(col: pd.Series) -> list[str]:
        result = []
        for v in col:
            try:
                f = float(str(v).replace("%", "")) / 100
                if f >= 0.70:
                    result.append(f"color: {COLORS['positive']}; font-weight: 600")
                elif f >= 0.50:
                    result.append(f"color: {COLORS['warning']}")
                else:
                    result.append(f"color: {COLORS['negative']}")
            except (ValueError, TypeError):
                result.append("")
        return result

    styler = ideas_df.style

    conf_cols = [c for c in ideas_df.columns if "Confidence" in c]
    for col in conf_cols:
        styler = styler.apply(colour_confidence, subset=[col])

    return styler.set_properties(**{"font-size": "12px"})


def format_bond_universe_table(meta: pd.DataFrame, oas_latest: pd.Series) -> pd.DataFrame:
    """
    Produce a clean display table combining metadata and latest OAS.
    """
    df = meta.copy()
    df["OAS (bps)"] = df["id"].map(oas_latest).round(1)
    df["Yield (%)"] = (df["yield_base"] + (df["id"].map(oas_latest) - df["oas_base"]) / 100).round(2)

    display = df[[
        "id", "country", "maturity", "rating", "duration", "dv01",
        "coupon", "OAS (bps)", "Yield (%)",
    ]].rename(columns={
        "id": "Bond ID",
        "country": "Country",
        "maturity": "Mat (Y)",
        "rating": "Rating",
        "duration": "Duration",
        "dv01": "DV01 ($)",
        "coupon": "Coupon (%)",
    })

    display["DV01 ($)"] = display["DV01 ($)"].apply(lambda x: f"${x:,.0f}")
    return display
