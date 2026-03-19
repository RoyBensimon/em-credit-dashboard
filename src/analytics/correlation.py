"""
Correlation & Beta Engine for the EM Credit Analytics Dashboard.

Core analytics module that:
  1. Computes static and rolling Pearson correlation matrices.
  2. Estimates OLS betas (factor loadings) for each bond vs. a macro basket.
  3. Ranks the best hedge proxy per bond by R².
  4. Runs a simple regime-change detector using rolling volatility.

All inputs are expected as pd.DataFrames of daily changes/returns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
    _SM_AVAILABLE = True
except ImportError:
    _SM_AVAILABLE = False

from config.settings import BETA_MIN_R2, BOND_UNIVERSE, DEFAULT_ROLLING_WINDOWS

# ── Bonds with monthly-interpolated FRED OAS data ─────────────────────────────
# Daily OAS changes for these bonds are artificially smooth (interpolated between
# two monthly data points). Pearson correlations on daily changes are unreliable
# and should not be shown without a clear caveat.
MONTHLY_INTERPOLATED_BONDS: frozenset[str] = frozenset(
    b["id"] for b in BOND_UNIVERSE if "FRED:IRLT" in b.get("data_source", "")
)  # → MEX_10Y, ZAF_10Y, CHL_10Y, ISR_10Y


# ═════════════════════════════════════════════════════════════════════════════
# 1. Correlation matrix
# ═════════════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(
    changes: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Compute a pairwise Pearson correlation matrix.

    Parameters
    ----------
    changes     : DataFrame of daily returns or spread changes.
    method      : 'pearson' (default) or 'spearman'.
    min_periods : minimum non-NaN observations required per pair.

    Returns
    -------
    pd.DataFrame (n_assets × n_assets) with values in [-1, 1].
    """
    return changes.corr(method=method, min_periods=min_periods)


def compute_rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 60,
    min_periods: int = 20,
) -> pd.Series:
    """
    Rolling Pearson correlation between two series.
    """
    return series_a.rolling(window=window, min_periods=min_periods).corr(series_b)


def compute_all_rolling_correlations(
    target: pd.Series,
    factors: pd.DataFrame,
    windows: list[int] | None = None,
) -> dict[int, pd.DataFrame]:
    """
    For a single target series, compute rolling correlation with all factor
    columns at multiple window lengths.

    Returns
    -------
    dict {window: pd.DataFrame}  where each DataFrame has one column per factor.
    """
    windows = windows or DEFAULT_ROLLING_WINDOWS
    result: dict[int, pd.DataFrame] = {}

    for w in windows:
        rows: dict[str, pd.Series] = {}
        for col in factors.columns:
            rows[col] = compute_rolling_correlation(target, factors[col], window=w)
        result[w] = pd.DataFrame(rows, index=target.index)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 2. OLS Beta estimation
# ═════════════════════════════════════════════════════════════════════════════

def compute_ols_beta(
    y: pd.Series,
    X: pd.DataFrame,
    add_constant: bool = True,
) -> dict:
    """
    OLS regression of y on X.

    Returns a dict with:
      - betas       : pd.Series of beta coefficients (one per X column)
      - alpha       : intercept
      - r_squared   : R²
      - p_values    : pd.Series of p-values per coefficient
      - t_stats     : pd.Series of t-statistics
      - residuals   : pd.Series of regression residuals
    """
    # Align and drop NaNs
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < max(10, X.shape[1] + 2):
        return {"betas": pd.Series(dtype=float), "alpha": np.nan, "r_squared": np.nan,
                "p_values": pd.Series(dtype=float), "t_stats": pd.Series(dtype=float),
                "residuals": pd.Series(dtype=float)}

    y_clean = df["y"]
    X_clean = df.drop(columns="y")

    if _SM_AVAILABLE:
        X_reg = sm.add_constant(X_clean, has_constant="add") if add_constant else X_clean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model  = sm.OLS(y_clean, X_reg).fit()
        betas    = model.params.drop("const", errors="ignore")
        alpha    = model.params.get("const", np.nan)
        r2       = model.rsquared
        pvals    = model.pvalues.drop("const", errors="ignore")
        tstats   = model.tvalues.drop("const", errors="ignore")
        residuals = pd.Series(model.resid, index=y_clean.index)
    else:
        # Fallback: numpy lstsq
        X_np = np.column_stack([np.ones(len(X_clean)), X_clean.values]) if add_constant else X_clean.values
        coeffs, _, _, _ = np.linalg.lstsq(X_np, y_clean.values, rcond=None)
        alpha  = coeffs[0] if add_constant else 0.0
        betas_ = coeffs[1:] if add_constant else coeffs
        betas  = pd.Series(betas_, index=X_clean.columns)
        y_hat  = X_np @ coeffs
        ss_res = np.sum((y_clean.values - y_hat) ** 2)
        ss_tot = np.sum((y_clean.values - y_clean.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        pvals  = pd.Series(np.full(len(betas), np.nan), index=betas.index)
        tstats = pd.Series(np.full(len(betas), np.nan), index=betas.index)
        residuals = pd.Series(
            y_clean.values - X_np @ coeffs, index=y_clean.index
        )

    return {
        "betas":     betas,
        "alpha":     float(alpha),
        "r_squared": float(r2),
        "p_values":  pvals,
        "t_stats":   tstats,
        "residuals": residuals,
    }


def compute_all_betas(
    targets: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run OLS regressions for each column in `targets` against the full
    factor matrix.

    Returns a summary DataFrame with one row per target asset:
      columns: [alpha, beta_<factor1>, …, r_squared, best_factor, best_r2]
    """
    rows: list[dict] = []
    for col in targets.columns:
        res = compute_ols_beta(targets[col], factors)
        row: dict = {"bond_id": col, "alpha": res["alpha"], "r_squared": res["r_squared"]}
        for f in factors.columns:
            row[f"beta_{f}"] = res["betas"].get(f, np.nan)

        # Best single factor (highest |correlation| with target)
        if not res["betas"].empty and res["r_squared"] >= BETA_MIN_R2:
            abs_betas = res["betas"].abs()
            row["best_factor"] = abs_betas.idxmax()
            row["best_beta"]   = res["betas"][row["best_factor"]]
        else:
            row["best_factor"] = "N/A"
            row["best_beta"]   = np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("bond_id")


def rolling_beta(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
    min_periods: int = 20,
) -> pd.Series:
    """
    Rolling OLS beta of y regressed on x (univariate, no intercept in
    denominator – computed as cov(y,x)/var(x)).
    """
    roll_cov = y.rolling(window=window, min_periods=min_periods).cov(x)
    roll_var = x.rolling(window=window, min_periods=min_periods).var()
    return roll_cov / roll_var.replace(0, np.nan)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Hedge suggestions
# ═════════════════════════════════════════════════════════════════════════════

def generate_hedge_suggestions(
    beta_summary: pd.DataFrame,
    factor_labels: dict[str, str] | None = None,
    top_n: int = 2,
) -> pd.Series:
    """
    For each bond, produce a short hedge narrative based on its top factor
    betas and R².

    Returns a pd.Series indexed by bond_id with human-readable strings.
    """
    factor_labels = factor_labels or {
        "EMB": "EM credit beta (EMB)",
        "HYG": "HY credit beta (HYG)",
        "SPY": "US equity / risk-off (SPY)",
        "EEM": "EM equity beta (EEM)",
        "TLT": "duration / rates (TLT)",
        "UUP": "USD / DXY (UUP)",
        "GLD": "gold / safe-haven (GLD)",
        "VIXY": "volatility (VIX)",
    }

    suggestions: dict[str, str] = {}
    beta_cols = [c for c in beta_summary.columns if c.startswith("beta_")]

    for bond_id, row in beta_summary.iterrows():
        r2 = row.get("r_squared", np.nan)
        if pd.isna(r2) or r2 < BETA_MIN_R2:
            suggestions[bond_id] = "Insufficient data for hedge recommendation."
            continue

        # Rank factors by |beta|
        betas_named = {}
        for bc in beta_cols:
            f = bc.replace("beta_", "")
            b = row.get(bc, np.nan)
            if not pd.isna(b):
                betas_named[f] = b

        sorted_factors = sorted(betas_named.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top = sorted_factors[:top_n]

        parts = []
        for fname, fval in top:
            label = factor_labels.get(fname, fname)
            direction = "positively" if fval > 0 else "negatively"
            parts.append(f"{label} ({direction}, β={fval:+.2f})")

        suggestion = (
            f"R²={r2:.0%}. Primary drivers: {'; '.join(parts)}. "
            f"Suggest using {top[0][0]} as primary hedge instrument."
        )
        suggestions[bond_id] = suggestion

    return pd.Series(suggestions, name="hedge_suggestion")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Regime detection
# ═════════════════════════════════════════════════════════════════════════════

def compute_cross_correlation(
    y: pd.Series,
    x: pd.Series,
    max_lag: int = 15,
) -> pd.Series:
    """
    Cross-correlation between x and y at lags -max_lag to +max_lag.

    Convention
    ----------
    lag k > 0 : x leads y by k days  →  corr(y[t], x[t-k])
    lag k = 0 : contemporaneous
    lag k < 0 : y leads x by |k| days

    Used by traders to detect lead-lag relationships, e.g. DXY moves
    preceding EM spread widening by several days.

    Returns
    -------
    pd.Series indexed by integer lag, values in [-1, 1].
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    result: dict[int, float] = {}
    for k in range(-max_lag, max_lag + 1):
        result[k] = float(df["y"].corr(df["x"].shift(k)))
    return pd.Series(result, name="cross_corr")


def detect_regime(
    series: pd.Series,
    short_window: int = 20,
    long_window: int = 60,
    vol_multiplier: float = 1.5,
) -> pd.Series:
    """
    Simple two-regime classifier: 'risk-on' vs. 'risk-off' / 'stress'.

    Logic:
      - Compute rolling volatility at short and long windows.
      - Label a period as 'stress' when short-term vol > vol_multiplier × long-term vol.
      - Otherwise label as 'normal'.

    Parameters
    ----------
    series          : daily returns (e.g. EMB returns)
    short_window    : lookback for short-term vol
    long_window     : lookback for long-term vol
    vol_multiplier  : multiplier threshold

    Returns
    -------
    pd.Series of str: 'normal' | 'stress'
    """
    short_vol = series.rolling(short_window, min_periods=5).std()
    long_vol  = series.rolling(long_window,  min_periods=20).std()

    regime = pd.Series("normal", index=series.index)
    regime[short_vol > vol_multiplier * long_vol] = "stress"
    return regime


def regime_correlation_table(
    changes: pd.DataFrame,
    regime: pd.Series,
) -> dict[str, pd.DataFrame]:
    """
    Compute separate correlation matrices for each regime label.

    Returns dict {regime_label: correlation_matrix}.
    """
    result: dict[str, pd.DataFrame] = {}
    for label in regime.unique():
        idx = regime[regime == label].index
        subset = changes.loc[changes.index.intersection(idx)]
        if len(subset) >= 10:
            result[str(label)] = compute_correlation_matrix(subset)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 5. Hedge engine helpers  (window-aware, work on pre-sliced data)
# ═════════════════════════════════════════════════════════════════════════════

def rank_factors_for_bond(
    bond_id: str,
    oas_changes: pd.DataFrame,
    macro_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    For a single bond, compute Pearson correlation and OLS beta vs every macro
    factor.  Both DataFrames should already be sliced to the desired window
    before calling this function.

    Returns
    -------
    pd.DataFrame with columns [factor, correlation, beta, r_squared, n_obs]
    sorted by |correlation| descending.
    """
    if bond_id not in oas_changes.columns:
        return pd.DataFrame()

    y = oas_changes[bond_id]
    rows: list[dict] = []

    for factor in macro_returns.columns:
        x = macro_returns[factor]
        df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        n = len(df)
        if n < 10:
            continue

        corr = float(df["y"].corr(df["x"]))
        beta_res = compute_ols_beta(df["y"], df[["x"]])
        beta = beta_res["betas"].get("x", np.nan)
        r2   = beta_res["r_squared"]

        rows.append({
            "factor":      factor,
            "correlation": round(corr, 3),
            "beta":        round(float(beta), 3) if not np.isnan(beta) else np.nan,
            "r_squared":   round(float(r2), 3)   if not np.isnan(r2) else np.nan,
            "n_obs":       n,
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    df_out["_abs"] = df_out["correlation"].abs()
    df_out = df_out.sort_values("_abs", ascending=False).drop(columns="_abs")
    return df_out.reset_index(drop=True)


def compute_macro_idio_score(
    bond_id: str,
    oas_changes: pd.DataFrame,
    macro_returns: pd.DataFrame,
) -> dict:
    """
    Classify a bond as macro-driven / mixed / idiosyncratic using R² from
    multivariate OLS of its OAS changes against all macro factors.

    Both DataFrames should be pre-sliced to the desired window.

    Returns dict: r_squared, label, macro_pct, idio_pct, n_obs.
    """
    if bond_id not in oas_changes.columns:
        return {"r_squared": np.nan, "label": "N/A", "macro_pct": np.nan, "idio_pct": np.nan, "n_obs": 0}

    result = compute_ols_beta(oas_changes[bond_id], macro_returns)
    r2     = result.get("r_squared", np.nan)
    n      = len(pd.concat([oas_changes[[bond_id]], macro_returns], axis=1).dropna())

    if np.isnan(r2):
        label = "N/A"
    elif r2 >= 0.60:
        label = "Macro-Driven"
    elif r2 >= 0.30:
        label = "Mixed"
    else:
        label = "Idiosyncratic"

    return {
        "r_squared": round(r2, 3) if not np.isnan(r2) else np.nan,
        "label":     label,
        "macro_pct": round(r2, 3) if not np.isnan(r2) else np.nan,
        "idio_pct":  round(1 - r2, 3) if not np.isnan(r2) else np.nan,
        "n_obs":     n,
    }


def compute_rolling_corr_stability(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
) -> dict:
    """
    Compute a rolling correlation series and return summary statistics used to
    assess hedge stability.

    stable = True if the rolling std is below 0.30 (correlation does not
    frequently change sign or magnitude across the lookback).

    Returns dict: series, mean, std, min, max, latest, stable, n_valid.
    """
    roll  = y.rolling(window=window, min_periods=max(10, window // 4)).corr(x)
    clean = roll.dropna()

    if len(clean) < 5:
        return {
            "series": roll, "mean": np.nan, "std": np.nan,
            "min": np.nan, "max": np.nan, "latest": np.nan,
            "stable": None, "n_valid": 0,
        }

    return {
        "series":  roll,
        "mean":    round(float(clean.mean()), 3),
        "std":     round(float(clean.std()), 3),
        "min":     round(float(clean.min()), 3),
        "max":     round(float(clean.max()), 3),
        "latest":  round(float(clean.iloc[-1]), 3),
        "stable":  bool(clean.std() < 0.30),
        "n_valid": len(clean),
    }


def compute_bond_bond_corr(
    oas_changes: pd.DataFrame,
    window: int | None = None,
    exclude_monthly: bool = True,
) -> pd.DataFrame:
    """
    Pairwise bond-to-bond Pearson correlation matrix of OAS daily changes.

    Parameters
    ----------
    oas_changes      : full OAS changes DataFrame
    window           : number of most-recent trading days to use (None = full)
    exclude_monthly  : drop monthly-interpolated bonds (unreliable daily corr)
    """
    cols = [
        c for c in oas_changes.columns
        if not (exclude_monthly and c in MONTHLY_INTERPOLATED_BONDS)
    ]
    data = oas_changes[cols]
    if window is not None:
        data = data.tail(window)
    return data.corr(method="pearson", min_periods=20)


def compute_spread_zscore_matrix(
    oas_df: pd.DataFrame,
    z_window: int = 252,
    exclude_monthly: bool = True,
) -> pd.DataFrame:
    """
    For every bond pair (A, B), compute the Z-score of the current spread
    OAS_A - OAS_B relative to its history over the last z_window days.

    Convention
    ----------
    Z > 0  →  spread currently wider than average  →  A cheap vs B  (BUY A / SELL B)
    Z < 0  →  spread currently tighter than average →  A rich vs B   (SELL A / BUY B)
    Z = 0  →  on-diagonal (same bond)

    Returns
    -------
    pd.DataFrame (square, bonds × bonds).
    """
    cols = [
        c for c in oas_df.columns
        if not (exclude_monthly and c in MONTHLY_INTERPOLATED_BONDS)
    ]
    df = oas_df[cols].tail(max(z_window, 60))

    matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for a in cols:
        for b in cols:
            if a == b:
                matrix.loc[a, b] = 0.0
                continue
            spread = (df[a] - df[b]).dropna()
            if len(spread) < 20:
                continue
            std_s = float(spread.std())
            if std_s < 1e-9:
                continue
            z = (float(spread.iloc[-1]) - float(spread.mean())) / std_s
            matrix.loc[a, b] = round(z, 2)

    return matrix
