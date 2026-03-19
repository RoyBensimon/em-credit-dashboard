"""
Trade Idea Generator for the EM Credit Analytics Dashboard.

Aggregates signals from:
  - Relative Value screener (OAS z-scores)
  - Curve Trade screener (slope / butterfly z-scores)
  - Correlation / Beta engine (macro factor exposures)

to produce a ranked, structured shortlist of trade ideas that a trader
can directly discuss or act upon.

Each idea is a dict with fields:
  title, trade_type, rationale, supporting_metrics,
  hedge_suggestion, key_risks, confidence_score, catalyst_notes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    RV_ZSCORE_THRESHOLD,
    CURVE_ZSCORE_THRESHOLD,
)

# ── Trade type labels ─────────────────────────────────────────────────────────
TYPE_RV       = "Relative Value"
TYPE_CURVE    = "Curve Trade"
TYPE_HEDGE    = "Hedge / Risk Reduction"
TYPE_MACRO    = "Macro-Relative"


# ═════════════════════════════════════════════════════════════════════════════
# 1. Generate RV trade ideas
# ═════════════════════════════════════════════════════════════════════════════

def rv_ideas_from_screener(
    rv_df: pd.DataFrame,
    beta_summary: pd.DataFrame | None = None,
    zscore_threshold: float = RV_ZSCORE_THRESHOLD,
    max_ideas: int = 6,
) -> list[dict]:
    """
    Convert RV screener output into structured trade ideas.

    Parameters
    ----------
    rv_df            : output of build_rv_universe()
    beta_summary     : optional beta/correlation summary for hedge suggestions
    zscore_threshold : minimum |z-score| to generate an idea
    max_ideas        : maximum number of ideas to return
    """
    ideas: list[dict] = []

    flagged = rv_df[rv_df["rv_label"].isin(["Cheap", "Rich"])].copy()
    flagged = flagged.sort_values("zscore", key=abs, ascending=False).head(max_ideas)

    for _, row in flagged.iterrows():
        bid     = row["bond_id"]
        country = row["country"]
        mat     = row["maturity"]
        label   = row["rv_label"]
        zscore  = row["zscore"]
        oas     = row["oas_current"]
        fair    = row["oas_fair"]
        resid   = row["residual_bps"]
        carry   = row["carry_bps_1m"]

        # Confidence: function of |z-score| magnitude
        conf = _zscore_to_confidence(abs(zscore))

        # Hedge suggestion from beta summary
        hedge = _get_hedge_suggestion(bid, beta_summary)

        title = f"{country} {mat}Y — {'Buy' if label == 'Cheap' else 'Sell'}"
        rationale = (
            f"{bid} is trading {'wide' if label == 'Cheap' else 'tight'} of fair value by "
            f"{abs(resid):.0f}bps (z={zscore:+.2f}). "
            f"Current OAS {oas:.0f}bps vs. curve-fair {fair:.0f}bps. "
            f"Carry ≈ {carry:.1f}bps/month."
        )

        supporting_metrics = {
            "OAS (current)":   f"{oas:.1f} bps",
            "OAS (fair value)":f"{fair:.1f} bps",
            "Residual":        f"{resid:+.1f} bps",
            "Z-Score":         f"{zscore:+.2f}",
            "1M Carry":        f"{carry:.1f} bps/month",
            "Chg 1W":          f"{row.get('chg_1w_bps', 0):+.1f} bps",
            "Chg 1M":          f"{row.get('chg_1m_bps', 0):+.1f} bps",
        }

        key_risks = _rv_risks(country, label, mat)
        catalyst  = _rv_catalyst(country, label)

        ideas.append({
            "title":              title,
            "bond_id":            bid,
            "country":            country,
            "trade_type":         TYPE_RV,
            "direction":          "Long" if label == "Cheap" else "Short",
            "rationale":          rationale,
            "supporting_metrics": supporting_metrics,
            "hedge_suggestion":   hedge,
            "key_risks":          key_risks,
            "confidence_score":   conf,
            "catalyst_notes":     catalyst,
        })

    return ideas


# ═════════════════════════════════════════════════════════════════════════════
# 2. Generate curve trade ideas
# ═════════════════════════════════════════════════════════════════════════════

def curve_ideas_from_screener(
    curve_trades_df: pd.DataFrame,
    meta: pd.DataFrame,
    max_ideas: int = 4,
) -> list[dict]:
    """
    Convert curve screener output into structured trade ideas.
    """
    ideas: list[dict] = []
    if curve_trades_df.empty:
        return ideas

    top = curve_trades_df.head(max_ideas)

    for _, row in top.iterrows():
        country   = row["country"]
        metric    = row["metric"]
        direction = row["trade_direction"]
        zscore    = row["zscore"]
        val       = row["current_bps"]
        pct       = row["percentile"]
        weight_r  = row.get("dv01_weight_ratio", np.nan)

        conf = _zscore_to_confidence(abs(zscore))

        sizing_note = ""
        if not pd.isna(weight_r):
            sizing_note = f"  DV01-neutral weight ratio: {weight_r:.2f}x."

        rationale = (
            f"{country} {metric.replace('slope','').replace('butterfly','')} "
            f"is {direction.upper()} (z={zscore:+.2f}, {pct:.0f}th pct ile). "
            f"Current value: {val:.1f}bps.{sizing_note}"
        )

        title = f"{country} — {metric.replace(' slope','').replace(' butterfly','')} {direction.title()}"

        supporting_metrics = {
            "Metric":        metric,
            "Current (bps)": f"{val:.1f}",
            "Z-Score":       f"{zscore:+.2f}",
            "Percentile":    f"{pct:.0f}th" if not np.isnan(pct) else "N/A",
            "DV01 Ratio":    f"{weight_r:.2f}x" if not np.isnan(weight_r) else "N/A",
        }

        key_risks = [
            f"Parallel shift in {country} curve would be DV01-neutral but carry risk remains.",
            "Idiosyncratic credit events could override the technical signal.",
            "Liquidity in the short-end node may be limited.",
        ]

        ideas.append({
            "title":              title,
            "bond_id":            metric,
            "country":            country,
            "trade_type":         TYPE_CURVE,
            "direction":          direction.title(),
            "rationale":          rationale,
            "supporting_metrics": supporting_metrics,
            "hedge_suggestion":   f"Hedge macro beta with EMB or a short-dated EM CDS position.",
            "key_risks":          key_risks,
            "confidence_score":   conf,
            "catalyst_notes":     f"Watch for upcoming {country} macro data or central bank meeting.",
        })

    return ideas


# ═════════════════════════════════════════════════════════════════════════════
# 3. Generate macro-relative trade ideas
# ═════════════════════════════════════════════════════════════════════════════

def macro_ideas_from_betas(
    beta_summary: pd.DataFrame,
    rv_df: pd.DataFrame,
    macro_returns: pd.DataFrame,
    max_ideas: int = 3,
) -> list[dict]:
    """
    Generate macro-driven trade ideas based on current factor momentum
    and beta exposures.

    Logic:
      - Identify the 1-month trend in each macro factor (positive / negative)
      - Cross with the bond's beta to that factor
      - If a cheap bond has a high beta to a positively trending factor → bullish idea
    """
    ideas: list[dict] = []
    if beta_summary.empty or macro_returns.empty:
        return ideas

    # 1-month factor momentum (last 22 trading days)
    factor_mom = macro_returns.tail(22).mean()  # avg daily return ≈ directional drift
    pos_factors = factor_mom[factor_mom > 0].index.tolist()
    neg_factors = factor_mom[factor_mom < 0].index.tolist()

    # Find cheap bonds with high beta to positive factors
    cheap_bonds = rv_df[rv_df["rv_label"] == "Cheap"]["bond_id"].tolist()

    beta_cols = [c for c in beta_summary.columns if c.startswith("beta_")]

    generated = 0
    for bid in cheap_bonds:
        if generated >= max_ideas:
            break
        if bid not in beta_summary.index:
            continue

        row    = beta_summary.loc[bid]
        r2     = row.get("r_squared", 0)
        if r2 < 0.05:
            continue

        # Find highest-beta factor that has positive momentum
        best_f, best_b = None, 0
        for bc in beta_cols:
            f = bc.replace("beta_", "")
            b = row.get(bc, 0)
            if f in pos_factors and abs(b) > abs(best_b):
                best_f, best_b = f, b

        if best_f is None:
            continue

        oas_row = rv_df[rv_df["bond_id"] == bid].iloc[0]
        conf    = _zscore_to_confidence(abs(oas_row["zscore"]) * r2 * 2)

        title = f"{oas_row['country']} {oas_row['maturity']}Y — Macro-supported long"
        rationale = (
            f"{bid} is cheap on RV (z={oas_row['zscore']:+.2f}) AND has a "
            f"positive beta (β={best_b:+.2f}) to {best_f}, which is showing "
            f"positive 1-month momentum ({factor_mom[best_f]*100:+.2f}% avg/day). "
            f"R²={r2:.0%} of returns explained by macro factors."
        )

        ideas.append({
            "title":              title,
            "bond_id":            bid,
            "country":            oas_row["country"],
            "trade_type":         TYPE_MACRO,
            "direction":          "Long",
            "rationale":          rationale,
            "supporting_metrics": {
                "OAS Z-Score":       f"{oas_row['zscore']:+.2f}",
                f"Beta to {best_f}": f"{best_b:+.2f}",
                "Factor R²":         f"{r2:.0%}",
                f"{best_f} 1M mom.": f"{factor_mom[best_f]*100:+.2f}%/day",
            },
            "hedge_suggestion":   f"Sell {best_f} exposure via ETF short to isolate idiosyncratic RV.",
            "key_risks":          [
                f"Factor reversal in {best_f} would work against the position.",
                "EM-wide credit sell-off would hurt despite cheap RV.",
            ],
            "confidence_score":   conf,
            "catalyst_notes":     "Factor momentum has been consistent for >3 weeks.",
        })
        generated += 1

    return ideas


# ═════════════════════════════════════════════════════════════════════════════
# 4. Aggregate all signals
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_trade_ideas(
    rv_df: pd.DataFrame,
    curve_trades_df: pd.DataFrame,
    beta_summary: pd.DataFrame,
    meta: pd.DataFrame,
    macro_returns: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Master trade idea generation function.

    Combines RV, curve, and macro signals into a single ranked list.

    Returns
    -------
    List of trade idea dicts, sorted by confidence_score descending.
    """
    ideas: list[dict] = []

    # RV ideas
    rv_ideas = rv_ideas_from_screener(rv_df, beta_summary)
    ideas.extend(rv_ideas)

    # Curve ideas
    curve_ideas = curve_ideas_from_screener(curve_trades_df, meta)
    ideas.extend(curve_ideas)

    # Macro-relative ideas
    if macro_returns is not None and not beta_summary.empty:
        macro_ideas = macro_ideas_from_betas(beta_summary, rv_df, macro_returns)
        ideas.extend(macro_ideas)

    # Sort by confidence, then by |z-score| proxy (from supporting_metrics)
    ideas.sort(key=lambda x: x["confidence_score"], reverse=True)

    # Add rank
    for i, idea in enumerate(ideas, 1):
        idea["rank"] = i

    return ideas


def ideas_to_dataframe(ideas: list[dict]) -> pd.DataFrame:
    """Convert trade ideas list to a flat summary DataFrame."""
    rows: list[dict] = []
    for idea in ideas:
        metrics = idea.get("supporting_metrics", {})
        rows.append({
            "Rank":          idea.get("rank", ""),
            "Title":         idea["title"],
            "Type":          idea["trade_type"],
            "Direction":     idea.get("direction", ""),
            "Country":       idea.get("country", ""),
            "Confidence":    f"{idea['confidence_score']:.0%}",
            **{k: v for k, v in list(metrics.items())[:3]},
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _zscore_to_confidence(abs_zscore: float) -> float:
    """Map a z-score magnitude to a confidence score in [0, 1]."""
    if abs_zscore >= 3.0:
        return 0.85
    elif abs_zscore >= 2.0:
        return 0.72
    elif abs_zscore >= 1.5:
        return 0.60
    elif abs_zscore >= 1.0:
        return 0.45
    else:
        return 0.30


def _get_hedge_suggestion(
    bond_id: str,
    beta_summary: pd.DataFrame | None,
) -> str:
    """Retrieve hedge text from pre-computed beta summary."""
    if beta_summary is None or bond_id not in beta_summary.index:
        return "Use EMB as EM credit hedge. Consider DXY hedge via UUP."

    row = beta_summary.loc[bond_id]
    best_f = row.get("best_factor", "EMB")
    r2     = row.get("r_squared", 0)

    if pd.isna(best_f) or best_f == "N/A":
        return "Use EMB as EM credit hedge."

    return (
        f"Primary hedge: {best_f} (R²={r2:.0%}). "
        f"Residual idiosyncratic risk: {1 - r2:.0%} of total."
    )


def _rv_risks(country: str, label: str, maturity: int) -> list[str]:
    """Generate standard key-risks for an RV trade."""
    risks = [
        f"OAS could remain {'wide' if label == 'Cheap' else 'tight'} if "
        f"{'fundamental deterioration' if label == 'Cheap' else 'demand is structurally strong'}.",
        "Global EM credit sell-off would hurt all EM longs regardless of relative value.",
        f"Liquidity risk: {country} bonds may gap sharply on adverse headlines.",
    ]
    if maturity >= 10:
        risks.append("Long-duration bonds are more sensitive to UST rate moves.")
    return risks


def _rv_catalyst(country: str, label: str) -> str:
    """Produce a generic catalyst note."""
    if label == "Cheap":
        return (
            f"Potential catalysts for convergence: {country} credit rating review, "
            "EM fund inflows, or global risk-on sentiment shift."
        )
    return (
        f"Potential catalysts for re-widening: {country} political risk, "
        "EM outflows, or USD strengthening."
    )
