"""
Curve Trade Builder for the EM Credit Analytics Dashboard.

Provides functions to:
  1. Extract yield / OAS curves per country (2Y, 5Y, 10Y, 30Y nodes).
  2. Compute slope (e.g. 2s10s, 5s10s) and curvature (butterfly) metrics.
  3. Compute rolling z-scores of each metric.
  4. Compute DV01-neutral trade weights for 2-leg and 3-leg curve trades.
  5. Identify candidate steepener / flattener / butterfly trade opportunities.
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd

from config.settings import (
    BOND_UNIVERSE,
    CURVE_ZSCORE_THRESHOLD,
    DEFAULT_ZSCORE_WINDOW,
)
from src.data.preprocessor import rolling_zscore, rolling_percentile


# Maturity nodes we recognise
MATURITY_NODES = [2, 5, 10, 30]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Curve data extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_country_curve(
    meta: pd.DataFrame,
    oas_df: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    """
    Build a time-series DataFrame of OAS for each maturity node of a country.

    Returns
    -------
    pd.DataFrame  index=date, columns=[2Y, 5Y, 10Y, 30Y]  (only nodes present)
    """
    bonds = meta[meta["country"] == country][["id", "maturity"]]
    bonds = bonds[bonds["id"].isin(oas_df.columns)]

    if bonds.empty:
        return pd.DataFrame()

    # Map maturity bucket → OAS column
    maturity_map = {row["maturity"]: row["id"] for _, row in bonds.iterrows()}
    curve_cols = {f"{m}Y": oas_df[bid] for m, bid in maturity_map.items() if bid in oas_df.columns}

    if not curve_cols:
        return pd.DataFrame()

    curve_df = pd.DataFrame(curve_cols, index=oas_df.index)
    curve_df.columns = [c for c in curve_df.columns]
    return curve_df.dropna(how="all")


def get_available_maturity_pairs(
    meta: pd.DataFrame,
    country: str,
) -> list[tuple[int, int]]:
    """Return all maturity pairs available for a given country."""
    mats = sorted(meta[meta["country"] == country]["maturity"].unique())
    return list(itertools.combinations(mats, 2))


# ═════════════════════════════════════════════════════════════════════════════
# 2. Slope and curvature metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_slope(
    curve_df: pd.DataFrame,
    short_node: str,
    long_node: str,
) -> pd.Series:
    """
    Compute 2-leg curve slope: long_OAS - short_OAS  (in bps).

    A positive slope means the long end trades wider (normal credit curve).
    A rising slope = curve steepening.
    """
    if short_node not in curve_df.columns or long_node not in curve_df.columns:
        return pd.Series(dtype=float)
    return curve_df[long_node] - curve_df[short_node]


def compute_curvature(
    curve_df: pd.DataFrame,
    short_node: str,
    belly_node: str,
    long_node: str,
) -> pd.Series:
    """
    Compute 3-leg butterfly curvature:
        curvature = 2 × belly_OAS - short_OAS - long_OAS

    Positive curvature → belly is cheap vs. wings.
    Negative curvature → belly is rich vs. wings.
    """
    required = [short_node, belly_node, long_node]
    if not all(n in curve_df.columns for n in required):
        return pd.Series(dtype=float)
    return 2 * curve_df[belly_node] - curve_df[short_node] - curve_df[long_node]


def compute_all_slopes(
    curve_df: pd.DataFrame,
    node_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Compute all pairwise slopes for a country's curve.

    Parameters
    ----------
    curve_df   : DataFrame with columns like '2Y', '5Y', '10Y', '30Y'
    node_pairs : if None, use all adjacent pairs

    Returns DataFrame with one column per slope metric.
    """
    if node_pairs is None:
        cols = sorted(curve_df.columns, key=lambda c: int(c.replace("Y", "")))
        node_pairs = [(cols[i], cols[i + 1]) for i in range(len(cols) - 1)]

    result: dict[str, pd.Series] = {}
    for short, long_node in node_pairs:
        key = f"{short}/{long_node} slope"
        result[key] = compute_slope(curve_df, short, long_node)

    return pd.DataFrame(result, index=curve_df.index)


def compute_all_butterflies(
    curve_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all 3-node butterfly metrics."""
    cols = sorted(curve_df.columns, key=lambda c: int(c.replace("Y", "")))
    result: dict[str, pd.Series] = {}

    for i in range(len(cols) - 2):
        s, m, l = cols[i], cols[i + 1], cols[i + 2]
        key = f"{s}/{m}/{l} butterfly"
        result[key] = compute_curvature(curve_df, s, m, l)

    return pd.DataFrame(result, index=curve_df.index)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Z-scores and historical context
# ═════════════════════════════════════════════════════════════════════════════

def curve_zscores(
    slope_df: pd.DataFrame,
    window: int = DEFAULT_ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Compute rolling z-scores for all slope/curvature metrics."""
    return slope_df.apply(lambda col: rolling_zscore(col, window=window))


def curve_percentiles(
    slope_df: pd.DataFrame,
    window: int = DEFAULT_ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Compute rolling historical percentiles for all slope/curvature metrics."""
    return slope_df.apply(lambda col: rolling_percentile(col, window=window))


# ═════════════════════════════════════════════════════════════════════════════
# 4. DV01-neutral trade sizing
# ═════════════════════════════════════════════════════════════════════════════

def dv01_neutral_weight(
    dv01_short: float,
    dv01_long: float,
    notional_short: float = 1_000_000,
) -> dict:
    """
    Compute the DV01-neutral notional for a 2-leg curve trade.

    For a steepener (long short-end, short long-end), the DV01 of both
    legs must match so that a parallel shift in the curve is P&L-neutral.

    Parameters
    ----------
    dv01_short     : DV01 per $1mm of the short-maturity bond (USD)
    dv01_long      : DV01 per $1mm of the long-maturity bond (USD)
    notional_short : Face value of the short-end position ($)

    Returns
    -------
    dict with:
      - notional_short : $
      - notional_long  : $ (DV01-neutral)
      - dv01_short_leg : $ DV01 of short leg
      - dv01_long_leg  : $ DV01 of long leg (should ≈ dv01_short_leg)
      - weight_ratio   : notional_long / notional_short
    """
    if dv01_long == 0:
        return {}

    notional_long     = notional_short * (dv01_short / dv01_long)
    dv01_short_leg    = notional_short / 1_000_000 * dv01_short
    dv01_long_leg     = notional_long  / 1_000_000 * dv01_long

    return {
        "notional_short":  round(notional_short, 0),
        "notional_long":   round(notional_long, 0),
        "dv01_short_leg":  round(dv01_short_leg, 1),
        "dv01_long_leg":   round(dv01_long_leg, 1),
        "weight_ratio":    round(notional_long / notional_short, 3),
    }


def dv01_neutral_butterfly(
    dv01_short: float,
    dv01_belly: float,
    dv01_long: float,
    notional_belly: float = 1_000_000,
) -> dict:
    """
    Compute DV01-neutral notionals for a 3-leg butterfly.

    Convention: long belly, short wings (or reverse).
    Constraint: dv01_short_leg + dv01_long_leg = 2 × dv01_belly_leg

    Returns notional for wings given belly notional.
    """
    if dv01_short == 0 or dv01_long == 0:
        return {}

    dv01_belly_leg = notional_belly / 1_000_000 * dv01_belly
    # Split the total wing DV01 proportionally
    total_wing_dv01 = 2 * dv01_belly_leg
    ratio            = dv01_short / dv01_long

    # notional_short * dv01_s/1mm + notional_long * dv01_l/1mm = total_wing_dv01
    # notional_short * dv01_s = notional_long * dv01_l (keep ratio balanced)
    notional_long  = (total_wing_dv01 / 2) / (dv01_long / 1_000_000)
    notional_short = (total_wing_dv01 / 2) / (dv01_short / 1_000_000)

    return {
        "notional_belly":  round(notional_belly, 0),
        "notional_short":  round(notional_short, 0),
        "notional_long":   round(notional_long, 0),
        "dv01_belly_leg":  round(dv01_belly_leg, 1),
        "dv01_short_leg":  round(notional_short / 1_000_000 * dv01_short, 1),
        "dv01_long_leg":   round(notional_long  / 1_000_000 * dv01_long, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. Curve trade opportunity screener
# ═════════════════════════════════════════════════════════════════════════════

def screen_curve_trades(
    meta: pd.DataFrame,
    oas_df: pd.DataFrame,
    zscore_window: int = DEFAULT_ZSCORE_WINDOW,
    threshold: float = CURVE_ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """
    Screen all countries for curve trade opportunities.

    For each country, compute slope and butterfly z-scores, then flag
    any metric that is beyond ±threshold.

    Returns a DataFrame with columns:
      [country, metric, current_value, zscore, percentile,
       trade_type, trade_direction, rationale, dv01_sizing]
    """
    rows: list[dict] = []

    for country in meta["country"].unique():
        curve_df = extract_country_curve(meta, oas_df, country)
        if curve_df.empty or curve_df.shape[1] < 2:
            continue

        slopes_df = compute_all_slopes(curve_df)
        butterflies_df = compute_all_butterflies(curve_df)

        all_metrics = pd.concat([slopes_df, butterflies_df], axis=1)
        zs_df       = curve_zscores(all_metrics, window=zscore_window)
        pct_df      = curve_percentiles(all_metrics, window=zscore_window)

        current_zs  = zs_df.iloc[-1]
        current_val = all_metrics.iloc[-1]
        current_pct = pct_df.iloc[-1]

        # Bond metadata for DV01 sizing
        country_bonds = meta[meta["country"] == country].set_index("maturity")

        for metric in all_metrics.columns:
            zs  = current_zs.get(metric, np.nan)
            val = current_val.get(metric, np.nan)
            pct = current_pct.get(metric, np.nan)

            if pd.isna(zs) or abs(zs) < threshold:
                continue

            # Determine trade type and direction
            is_butterfly = "butterfly" in metric
            if is_butterfly:
                trade_type = "butterfly"
                direction  = "long belly vs wings" if zs > 0 else "short belly vs wings"
                nodes      = [n.replace("Y", "") for n in metric.split("/")[:3]]
            else:
                trade_type = "curve"
                direction  = "steepener" if zs > 0 else "flattener"
                nodes_raw  = metric.replace(" slope", "").split("/")
                nodes      = [n.replace("Y", "") for n in nodes_raw]

            # DV01 sizing (2-leg)
            sizing_info = {}
            if len(nodes) >= 2:
                try:
                    m1, m2 = int(nodes[0]), int(nodes[-1])
                    b1 = country_bonds.loc[m1] if m1 in country_bonds.index else None
                    b2 = country_bonds.loc[m2] if m2 in country_bonds.index else None
                    if b1 is not None and b2 is not None:
                        sizing_info = dv01_neutral_weight(
                            b1["dv01"], b2["dv01"]
                        )
                except Exception:
                    pass

            rationale = (
                f"{country} {metric}: z-score={zs:+.2f} "
                f"({pct:.0f}th pct ile). "
                f"Current={val:.1f}bps. "
                f"Suggests {direction}."
            )

            rows.append({
                "country":         country,
                "metric":          metric,
                "current_bps":     round(val, 1) if not np.isnan(val) else np.nan,
                "zscore":          round(zs, 2),
                "percentile":      round(pct, 0) if not np.isnan(pct) else np.nan,
                "trade_type":      trade_type,
                "trade_direction": direction,
                "rationale":       rationale,
                "dv01_weight_ratio": sizing_info.get("weight_ratio", np.nan),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("zscore", key=abs, ascending=False)
    return df.reset_index(drop=True)
