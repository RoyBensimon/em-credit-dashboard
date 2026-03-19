"""
RV Pair Screener — cross-country bond pair relative value analysis.

For each candidate pair (A, B):
  1. Correlation of daily OAS changes  (how tightly they co-move)
  2. Spread differential history: OAS_A(t) - OAS_B(t)
  3. Z-score of current diff vs rolling `window`-day history
  4. Hedge ratio: OLS beta of Δ(OAS_A) on Δ(OAS_B)
  5. DV01-neutral ratio: dv01_A / dv01_B as a cross-check

A trade signal is generated when:
  - correlation  >= min_corr          (default 0.55)
  - |z-score|    >= zscore_threshold  (default 1.5)

Same-country pairs are excluded (those are curve trades).
30Y bonds and monthly-interpolated bonds are excluded.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.correlation import MONTHLY_INTERPOLATED_BONDS


# ── Rating buckets ────────────────────────────────────────────────────────────
_RATING_BUCKET: dict[str, str] = {
    "AAA": "IG",  "AA+": "IG",  "AA": "IG",   "AA-": "IG",
    "A+":  "IG",  "A":   "IG",  "A-": "IG",
    "BBB+": "IG", "BBB": "IG",  "BBB-": "IG",
    "BB+": "HY",  "BB":  "HY",  "BB-": "HY",
    "B+":  "B",   "B":   "B",   "B-":  "B",
    "CCC+": "D",  "CCC": "D",   "CCC-": "D",
    "CC":  "D",   "C":   "D",   "D":   "D",
}


# ── Pair affinity ─────────────────────────────────────────────────────────────

def _pair_affinity(meta_a: pd.Series, meta_b: pd.Series) -> int:
    """
    Affinity score between two bonds (higher = more natural RV pair).
    Same maturity: +2, adjacent maturity (≤5Y gap): +1, same rating bucket: +1.
    """
    score = 0
    mat_diff = abs(int(meta_a["maturity"]) - int(meta_b["maturity"]))
    if mat_diff == 0:
        score += 2
    elif mat_diff <= 5:
        score += 1
    if _RATING_BUCKET.get(meta_a["rating"], "?") == _RATING_BUCKET.get(meta_b["rating"], "?"):
        score += 1
    return score


# ── Hedge ratio ───────────────────────────────────────────────────────────────

def _ols_hedge_ratio(
    changes_a: pd.Series,
    changes_b: pd.Series,
    min_obs: int = 40,
) -> float:
    """
    OLS beta of Δ(OAS_A) on Δ(OAS_B).

    Interpretation: to be OAS-change-neutral in the pair,
    for every 1bp of B shorted, hold hedge_ratio bps of A long.
    Clipped to [0.30, 3.00] to avoid extreme estimates.
    Falls back to 1.0 if data is insufficient.
    """
    aln = pd.concat([changes_a, changes_b], axis=1).dropna()
    if len(aln) < min_obs or aln.iloc[:, 1].std() < 1e-9:
        return 1.0
    slope, *_ = stats.linregress(aln.iloc[:, 1].values, aln.iloc[:, 0].values)
    return float(np.clip(slope, 0.30, 3.00))


# ── Main screener ─────────────────────────────────────────────────────────────

def screen_rv_pairs(
    oas_df: pd.DataFrame,
    meta: pd.DataFrame,
    oas_changes: pd.DataFrame,
    window: int = 126,
    min_corr: float = 0.55,
    zscore_threshold: float = 1.5,
    top_n: int = 5,
) -> list[dict]:
    """
    Screen the bond universe for top RV pair trade opportunities.

    Parameters
    ----------
    oas_df            : OAS level history (index=date, columns=bond_id)
    meta              : bond metadata DataFrame
    oas_changes       : OAS daily change history (index=date, columns=bond_id)
    window            : rolling window for z-score computation (trading days)
    min_corr          : minimum pairwise correlation to consider a pair
    zscore_threshold  : minimum |z-score| to generate a trade signal
    top_n             : maximum number of pairs to return

    Returns
    -------
    List of pair dicts sorted by signal_strength descending.
    Each dict contains all fields needed to render a trader-facing card.
    """
    # ── Eligible universe ─────────────────────────────────────────────────────
    eligible = meta[
        (meta["maturity"] != 30)
        & (~meta["id"].isin(MONTHLY_INTERPOLATED_BONDS))
        & (meta["id"].isin(oas_df.columns))
        & (meta["id"].isin(oas_changes.columns))
    ].copy()

    if len(eligible) < 2:
        return []

    bond_ids = eligible["id"].tolist()
    meta_idx = eligible.set_index("id")

    # Full-sample pairwise correlation of OAS daily changes
    corr_matrix = (
        oas_changes[bond_ids]
        .dropna(how="all")
        .corr(method="pearson", min_periods=30)
    )

    # OAS level data capped at window for z-score
    oas_w = oas_df[bond_ids].dropna(how="all").tail(window + 10)

    results: list[dict] = []

    for id_a, id_b in itertools.combinations(bond_ids, 2):
        meta_a = meta_idx.loc[id_a]
        meta_b = meta_idx.loc[id_b]

        # Skip within-country pairs (handled by curve trade screener)
        if meta_a["country"] == meta_b["country"]:
            continue

        corr = (
            corr_matrix.loc[id_a, id_b]
            if (id_a in corr_matrix.index and id_b in corr_matrix.columns)
            else np.nan
        )
        if pd.isna(corr) or corr < min_corr:
            continue

        if id_a not in oas_w.columns or id_b not in oas_w.columns:
            continue

        diff = (oas_w[id_a] - oas_w[id_b]).dropna()
        if len(diff) < max(window // 2, 30):
            continue

        hist_mean = float(diff.mean())
        hist_std  = float(diff.std())
        if hist_std < 1.0:
            continue

        current_diff = float(diff.iloc[-1])
        zscore = (current_diff - hist_mean) / hist_std

        if abs(zscore) < zscore_threshold:
            continue

        # ── Hedge ratio ───────────────────────────────────────────────────────
        hedge_ratio = _ols_hedge_ratio(
            oas_changes.get(id_a, pd.Series(dtype=float)),
            oas_changes.get(id_b, pd.Series(dtype=float)),
        )

        # DV01-neutral ratio
        dv01_a = float(meta_a.get("dv01", np.nan))
        dv01_b = float(meta_b.get("dv01", np.nan))
        dv01_ratio = (
            round(dv01_a / dv01_b, 2)
            if (not np.isnan(dv01_a) and not np.isnan(dv01_b) and dv01_b > 0)
            else None
        )

        # ── Trade direction ───────────────────────────────────────────────────
        # z > 0 → OAS_A - OAS_B wider than average → A cheap vs B → Long A / Short B
        dislocation = current_diff - hist_mean
        if zscore > 0:
            long_id, short_id = id_a, id_b
            why = (
                f"{id_a} ({meta_a['country']} {meta_a['maturity']}Y) is trading "
                f"{abs(dislocation):.0f}bps wide of {id_b} vs its {window}d average "
                f"— {id_a} appears cheap relative to {id_b}."
            )
        else:
            long_id, short_id = id_b, id_a
            why = (
                f"{id_b} ({meta_b['country']} {meta_b['maturity']}Y) is trading "
                f"{abs(dislocation):.0f}bps wide of {id_a} vs its {window}d average "
                f"— {id_b} appears cheap relative to {id_a}."
            )

        affinity       = _pair_affinity(meta_a, meta_b)
        signal_strength = abs(zscore) * corr * (1.0 + 0.2 * affinity)

        results.append({
            "pair":              f"{id_a} / {id_b}",
            "id_a":              id_a,
            "id_b":              id_b,
            "country_a":         str(meta_a["country"]),
            "country_b":         str(meta_b["country"]),
            "maturity_a":        int(meta_a["maturity"]),
            "maturity_b":        int(meta_b["maturity"]),
            "rating_a":          str(meta_a["rating"]),
            "rating_b":          str(meta_b["rating"]),
            "correlation":       round(float(corr), 3),
            "current_diff_bps":  round(current_diff, 1),
            "hist_avg_diff_bps": round(hist_mean, 1),
            "dislocation_bps":   round(dislocation, 1),
            "zscore":            round(float(zscore), 2),
            "long_id":           long_id,
            "short_id":          short_id,
            "trade":             f"Long {long_id} / Short {short_id}",
            "hedge_ratio":       round(float(hedge_ratio), 2),
            "dv01_ratio":        dv01_ratio,
            "why":               why,
            "affinity":          affinity,
            "signal_strength":   round(signal_strength, 3),
        })

    results.sort(key=lambda x: x["signal_strength"], reverse=True)
    return results[:top_n]
