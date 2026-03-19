"""
Relative Value (RV) Screener for the EM Credit Analytics Dashboard.

Core logic:
  1. For each issuer (country), fit a smooth OAS-vs-maturity curve using
     polynomial regression (degree 2 = quadratic = concave credit curve).
  2. Compute residuals: actual OAS minus fitted OAS.
  3. Compute rolling z-scores of residuals → identify rich / cheap bonds.
  4. Score and rank bonds across the entire universe.
  5. Compute approximate carry metrics.

All bond data is expected as a wide DataFrame (index=dates, columns=bond IDs)
plus a metadata DataFrame produced by preprocessor.bond_meta_to_df().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import BOND_UNIVERSE, RV_ZSCORE_THRESHOLD
from src.data.preprocessor import rolling_zscore, rolling_percentile


# ═════════════════════════════════════════════════════════════════════════════
# 1. Issuer curve fitting
# ═════════════════════════════════════════════════════════════════════════════

def fit_issuer_curve(
    meta: pd.DataFrame,
    oas_latest: pd.Series,
    country: str,
    degree: int = 2,
) -> dict:
    """
    Fit a polynomial OAS-vs-maturity curve for a single country.

    Parameters
    ----------
    meta       : bond metadata DataFrame (from bond_meta_to_df())
    oas_latest : pd.Series of latest OAS levels, indexed by bond_id
    country    : country name to filter
    degree     : polynomial degree (1=linear, 2=quadratic)

    Returns
    -------
    dict with keys:
      - country     : str
      - bond_ids    : list[str]
      - maturities  : np.ndarray
      - oas_actual  : np.ndarray
      - oas_fitted  : np.ndarray
      - residuals   : np.ndarray
      - coeffs      : np.ndarray  (polynomial coefficients)
      - r_squared   : float
    """
    sub = meta[meta["country"] == country].copy()
    sub = sub[sub["id"].isin(oas_latest.index)].copy()
    if len(sub) < 2:
        return {}

    sub["oas_actual"] = sub["id"].map(oas_latest)
    sub = sub.dropna(subset=["oas_actual"]).sort_values("maturity")

    x = sub["maturity"].values.astype(float)
    y = sub["oas_actual"].values.astype(float)

    if len(x) < 2:
        return {}

    # Fit polynomial; fall back to degree 1 if not enough points
    actual_degree = min(degree, len(x) - 1)
    coeffs = np.polyfit(x, y, actual_degree)
    y_hat  = np.polyval(coeffs, x)

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "country":    country,
        "bond_ids":   sub["id"].tolist(),
        "maturities": x,
        "oas_actual": y,
        "oas_fitted": y_hat,
        "residuals":  y - y_hat,
        "coeffs":     coeffs,
        "r_squared":  r2,
    }


def fit_all_issuer_curves(
    meta: pd.DataFrame,
    oas_latest: pd.Series,
    degree: int = 2,
) -> dict[str, dict]:
    """
    Fit curves for every country in the bond universe.

    Returns dict {country: curve_result_dict}.
    """
    countries = meta["country"].unique()
    return {
        c: fit_issuer_curve(meta, oas_latest, c, degree)
        for c in countries
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. Peer-group fair value (cross-issuer, same maturity bucket)
# ═════════════════════════════════════════════════════════════════════════════

def fit_peer_curve(
    meta: pd.DataFrame,
    oas_latest: pd.Series,
    maturity_bucket: int | None = None,
    rating_bucket: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute a cross-country fair-value model:
      OAS_fair = f(duration, rating_rank)

    using OLS regression across all bonds in a peer group.

    Returns a DataFrame with columns:
      [bond_id, country, maturity, rating, oas_actual, oas_fair, residual]
    """
    df = meta.copy()
    df["oas_actual"] = df["id"].map(oas_latest)
    df = df.dropna(subset=["oas_actual"])

    if maturity_bucket is not None:
        df = df[df["maturity"] == maturity_bucket]

    if rating_bucket is not None:
        df = df[df["rating"].isin(rating_bucket)]

    if len(df) < 3:
        return pd.DataFrame()

    # OLS: OAS ~ duration + rating_rank (both numeric)
    X = df[["duration", "rating_rank"]].values
    y = df["oas_actual"].values

    X_aug = np.column_stack([np.ones(len(X)), X])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_hat = X_aug @ coeffs
    except Exception:
        y_hat = np.full(len(y), y.mean())

    df = df.copy()
    df["oas_fair"]  = y_hat
    df["residual"]  = df["oas_actual"] - y_hat
    return df[["id", "country", "maturity", "rating", "duration",
               "oas_actual", "oas_fair", "residual"]].rename(columns={"id": "bond_id"})


# ═════════════════════════════════════════════════════════════════════════════
# 3. Historical z-scores of OAS residuals
# ═════════════════════════════════════════════════════════════════════════════

def compute_historical_residuals(
    meta: pd.DataFrame,
    oas_df: pd.DataFrame,
    degree: int = 2,
) -> pd.DataFrame:
    """
    For each date in oas_df, refit the issuer curves and compute residuals.

    This is computationally expensive for long histories; we use a
    rolling window approach: refit every 5 business days to save time.

    Returns
    -------
    pd.DataFrame  index=date, columns=bond_id, values=residual (bps)
    """
    # Resample to weekly frequency to keep computation tractable
    sample_dates = oas_df.index[::5]   # every 5 days
    residuals_hist: dict[str, list] = {b["id"]: [] for b in BOND_UNIVERSE}
    date_list: list = []

    for dt in sample_dates:
        if dt not in oas_df.index:
            continue
        oas_snap = oas_df.loc[dt]
        date_list.append(dt)

        for country in meta["country"].unique():
            res = fit_issuer_curve(meta, oas_snap, country, degree)
            if not res:
                continue
            for bid, resid in zip(res["bond_ids"], res["residuals"]):
                if bid in residuals_hist:
                    residuals_hist[bid].append(resid)

        # Pad bonds with no curve (single bond countries)
        for bid in residuals_hist:
            if len(residuals_hist[bid]) < len(date_list):
                residuals_hist[bid].append(np.nan)

    df_resid = pd.DataFrame(residuals_hist, index=date_list)
    return df_resid


def compute_rv_zscores(
    residuals_df: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling z-scores of OAS residuals for each bond.

    z > +1.5 → bond is cheap (trading wide of fair value)
    z < -1.5 → bond is rich  (trading tight of fair value)
    """
    return residuals_df.apply(lambda col: rolling_zscore(col, window=window))


# ═════════════════════════════════════════════════════════════════════════════
# 4. Universe-wide RV screener
# ═════════════════════════════════════════════════════════════════════════════

def build_rv_universe(
    meta: pd.DataFrame,
    oas_df: pd.DataFrame,
    residuals_df: pd.DataFrame | None = None,
    zscore_window: int = 126,
) -> pd.DataFrame:
    """
    Construct the full RV screener table for the bond universe.

    Each row = one bond.  Columns include current OAS, fair value OAS,
    residual, z-score, carry, and a richness/cheapness label.

    Returns
    -------
    pd.DataFrame sorted by z-score (most cheap first).
    """
    oas_latest = oas_df.iloc[-1]
    oas_1w_ago = oas_df.iloc[-6] if len(oas_df) >= 6 else oas_df.iloc[0]
    oas_1m_ago = oas_df.iloc[-22] if len(oas_df) >= 22 else oas_df.iloc[0]

    # Fit peer-group curve across all bonds
    peer = fit_peer_curve(meta, oas_latest)
    if peer.empty:
        peer = pd.DataFrame({"bond_id": meta["id"].tolist()})

    # Compute z-scores using full history
    if residuals_df is not None and len(residuals_df) >= 20:
        # Compute z-score from the historical residuals
        for col in residuals_df.columns:
            s = residuals_df[col].dropna()
            if len(s) >= 10:
                mean_r = s.mean()
                std_r  = s.std()
                # Current residual: latest row of residuals_df
                cur_r  = residuals_df[col].dropna().iloc[-1] if col in residuals_df.columns else np.nan
                residuals_df.loc[:, col + "_zscore"] = (residuals_df[col] - mean_r) / (std_r + 1e-9)

    # Build issuer curves for all countries
    issuer_curves = fit_all_issuer_curves(meta, oas_latest)

    rows: list[dict] = []
    for _, bond in meta.iterrows():
        bid     = bond["id"]
        country = bond["country"]

        # Fair value from issuer curve
        curve = issuer_curves.get(country, {})
        if curve and bid in curve["bond_ids"]:
            idx      = curve["bond_ids"].index(bid)
            oas_fair = curve["oas_fitted"][idx]
            residual = curve["residuals"][idx]
        else:
            oas_fair = np.nan
            residual = np.nan

        # Historical z-score of residual
        if residuals_df is not None and bid in residuals_df.columns:
            s = residuals_df[bid].dropna()
            if len(s) >= 10:
                mean_r = s.mean()
                std_r  = s.std()
                zscore = (residual - mean_r) / (std_r + 1e-9) if not np.isnan(residual) else np.nan
            else:
                zscore = residual / 20 if not np.isnan(residual) else np.nan
        else:
            # Use current cross-sectional dispersion as a fallback
            zscore = residual / 20 if not np.isnan(residual) else np.nan

        oas_now = oas_latest.get(bid, np.nan)
        chg_1w  = (oas_now - oas_1w_ago.get(bid, np.nan)) if bid in oas_1w_ago.index else np.nan
        chg_1m  = (oas_now - oas_1m_ago.get(bid, np.nan)) if bid in oas_1m_ago.index else np.nan

        # Richness / cheapness label
        if pd.isna(zscore):
            label = "Neutral"
        elif zscore > RV_ZSCORE_THRESHOLD:
            label = "Cheap"
        elif zscore < -RV_ZSCORE_THRESHOLD:
            label = "Rich"
        else:
            label = "Neutral"

        rows.append({
            "bond_id":        bid,
            "country":        country,
            "maturity":       bond["maturity"],
            "rating":         bond["rating"],
            "duration":       bond["duration"],
            "dv01":           bond["dv01"],
            "oas_current":    round(oas_now, 1) if not np.isnan(oas_now) else np.nan,
            "oas_fair":       round(oas_fair, 1) if not np.isnan(oas_fair) else np.nan,
            "residual_bps":   round(residual, 1) if not np.isnan(residual) else np.nan,
            "zscore":         round(zscore, 2) if not np.isnan(zscore) else np.nan,
            "chg_1w_bps":     round(chg_1w, 1) if not np.isnan(chg_1w) else np.nan,
            "chg_1m_bps":     round(chg_1m, 1) if not np.isnan(chg_1m) else np.nan,
            "carry_bps_1m":   round(bond["oas_base"] / 12, 1),  # approx carry
            "rv_label":       label,
        })

    rv_df = pd.DataFrame(rows)
    rv_df = rv_df.sort_values("zscore", ascending=False, na_position="last")
    return rv_df.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Helpers
# ═════════════════════════════════════════════════════════════════════════════

def top_cheap_rich(rv_df: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the top-n cheapest and top-n richest bonds."""
    cheap = rv_df[rv_df["rv_label"] == "Cheap"].head(n)
    rich  = rv_df[rv_df["rv_label"] == "Rich"].tail(n).sort_values("zscore")
    return cheap, rich
