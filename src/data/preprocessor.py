"""
Data preprocessing and feature engineering for the EM Credit Dashboard.

Converts raw prices / OAS series into the analytics-ready inputs that
the correlation engine, RV screener, and curve builder consume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import BOND_UNIVERSE, RATING_ORDER


# ── Returns and spread changes ────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """
    Compute simple (or log) returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted-close prices.  Columns = tickers, index = dates.
    log    : bool
        If True, return log returns (ln(P_t / P_{t-1})).

    Returns
    -------
    pd.DataFrame of the same shape, first row is NaN (dropped by default
    by callers who call .dropna()).
    """
    if log:
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def compute_spread_changes(oas: pd.DataFrame) -> pd.DataFrame:
    """First difference of OAS (in basis points)."""
    return oas.diff()


def align_and_clean(
    *dfs: pd.DataFrame,
    method: str = "ffill",
    dropna: bool = True,
) -> tuple[pd.DataFrame, ...]:
    """
    Inner-join multiple DataFrames on their date index, forward-fill gaps,
    and drop any remaining NaN rows.
    """
    combined = pd.concat(dfs, axis=1, join="inner")
    combined = combined.fillna(method=method)
    if dropna:
        combined = combined.dropna()
    # Split back into original DataFrames by column names
    result = []
    col_cursor = 0
    for df in dfs:
        ncols = df.shape[1]
        result.append(combined.iloc[:, col_cursor: col_cursor + ncols])
        col_cursor += ncols
    return tuple(result)


# ── Rolling statistics ────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute rolling z-score: (x - rolling_mean) / rolling_std.

    Useful for identifying when a spread or slope is historically
    stretched relative to recent history.
    """
    roll = series.rolling(window, min_periods=max(20, window // 5))
    return (series - roll.mean()) / roll.std()


def rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling historical percentile of the current value (0–100).
    Tells the user 'this spread is in the Xth percentile of history'.
    """
    def _pct(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        return float((x < x[-1]).mean() * 100)

    return series.rolling(window, min_periods=20).apply(_pct, raw=True)


def compute_carry_approx(
    oas: pd.Series,
    duration: float,
    coupon: float,
    rf_rate: float = 4.50,
) -> dict:
    """
    Approximate carry and roll-down metrics for a single bond.

    Parameters
    ----------
    oas      : current OAS in bps
    duration : modified duration in years
    coupon   : annual coupon rate (%)
    rf_rate  : risk-free rate proxy (10Y UST, %)

    Returns dict with keys:
      - carry_bps_pa  : approximate carry per annum (bps)
      - carry_bps_1m  : carry over 1 month (bps)
      - rolldown_1y   : approx roll-down assuming 1Y of passage of time (bps)
    """
    carry_pa   = float(oas)                      # OAS is already 'excess carry'
    carry_1m   = carry_pa / 12
    # Approximate roll-down: assume 1Y shortening on a roughly upward curve reduces spread
    # by a small amount per year of duration reduction.  This is a first-order proxy.
    rolldown_1y = 0.0  # We need the full curve for a proper roll-down; placeholder for now

    return {
        "carry_bps_pa": round(carry_pa, 1),
        "carry_bps_1m": round(carry_1m, 1),
        "rolldown_1y":  round(rolldown_1y, 1),
    }


# ── Bond universe helpers ─────────────────────────────────────────────────────

def bond_meta_to_df(bond_meta: list[dict] | None = None) -> pd.DataFrame:
    """Convert the BOND_UNIVERSE list of dicts to a tidy DataFrame."""
    bond_meta = bond_meta or BOND_UNIVERSE
    df = pd.DataFrame(bond_meta)

    # Add a numeric rating rank for sorting
    df["rating_rank"] = df["rating"].map(
        lambda r: RATING_ORDER.index(r) if r in RATING_ORDER else 99
    )
    return df


def latest_oas(oas_df: pd.DataFrame) -> pd.Series:
    """Return the most recent OAS level for each bond."""
    return oas_df.iloc[-1]


def oas_change(oas_df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """Return OAS change over `periods` days for each bond (in bps)."""
    return oas_df.iloc[-1] - oas_df.iloc[-1 - periods]


def oas_pct_change(oas_df: pd.DataFrame, periods: int = 20) -> pd.Series:
    """Return % OAS change over `periods` days for each bond."""
    old = oas_df.iloc[-1 - periods]
    new = oas_df.iloc[-1]
    return (new - old) / old.replace(0, np.nan) * 100


def macro_returns_aligned(
    macro_prices: pd.DataFrame,
    oas_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (macro_returns, oas_changes) aligned to the same date index,
    with the first row dropped (NaN from diff/pct_change).
    """
    mret   = compute_returns(macro_prices).dropna(how="all")
    dspread = compute_spread_changes(oas_df).dropna(how="all")

    common_idx = mret.index.intersection(dspread.index)
    return mret.loc[common_idx], dspread.loc[common_idx]
