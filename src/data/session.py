"""
Session-state management for the EM Credit Analytics Dashboard.

All expensive data loads (yfinance downloads, synthetic generation,
analytics pre-computation) are cached in st.session_state so that
navigating between pages does not re-trigger downloads.

Usage:
    from src.data.session import get_app_data
    data = get_app_data()
    macro_prices = data["macro_prices"]
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from config.settings import DEFAULT_LOOKBACK_DAYS
from src.data.loader import load_macro_prices, load_bond_oas_history
from src.data.preprocessor import (
    bond_meta_to_df,
    compute_returns,
    compute_spread_changes,
)
from src.analytics.correlation import compute_correlation_matrix, compute_all_betas
from src.analytics.relative_value import (
    build_rv_universe,
    compute_historical_residuals,
)
from src.analytics.curve_analysis import screen_curve_trades
from src.analytics.trade_ideas import generate_all_trade_ideas
from src.analytics.rv_pairs import screen_rv_pairs

logger = logging.getLogger(__name__)

_CACHE_KEY = "em_dashboard_data"


def get_app_data(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    force_reload: bool = False,
) -> dict:
    """
    Load and cache all application data in st.session_state.

    Returns a dict with keys:
      macro_prices   : pd.DataFrame  (ETF close prices)
      macro_returns  : pd.DataFrame  (ETF daily returns)
      oas_df         : pd.DataFrame  (bond OAS history)
      oas_changes    : pd.DataFrame  (bond OAS daily changes)
      meta           : pd.DataFrame  (bond metadata)
      corr_macro     : pd.DataFrame  (macro ETF correlation matrix)
      corr_bonds     : pd.DataFrame  (bond × macro correlation matrix)
      beta_summary   : pd.DataFrame  (OLS beta summary table)
      rv_universe    : pd.DataFrame  (RV screener output)
      curve_trades   : pd.DataFrame  (curve trade screener output)
      trade_ideas    : list[dict]    (structured trade ideas)
    """
    if not force_reload and _CACHE_KEY in st.session_state:
        return st.session_state[_CACHE_KEY]

    with st.spinner("Loading market data and running analytics…"):
        # ── 1. Raw data ─────────────────────────────────────────────────────
        macro_prices = load_macro_prices(lookback_days=lookback_days)
        meta         = bond_meta_to_df()
        oas_df       = load_bond_oas_history(lookback_days=lookback_days)

        # ── 2. Returns / changes ─────────────────────────────────────────────
        macro_returns = compute_returns(macro_prices).dropna(how="all")
        oas_changes   = compute_spread_changes(oas_df).dropna(how="all")

        # Align to common index
        common_idx    = macro_returns.index.intersection(oas_changes.index)
        macro_ret_aln = macro_returns.loc[common_idx]
        oas_chg_aln   = oas_changes.loc[common_idx]

        # ── 3. Correlation matrices ──────────────────────────────────────────
        corr_macro = compute_correlation_matrix(macro_ret_aln)

        # Bond × macro correlation: join oas_changes with macro_returns
        combined    = pd.concat([oas_chg_aln, macro_ret_aln], axis=1).dropna(how="all")
        corr_full   = compute_correlation_matrix(combined)
        # Extract the bond × macro sub-matrix
        bond_ids    = oas_chg_aln.columns.tolist()
        macro_ids   = macro_ret_aln.columns.tolist()
        corr_bonds  = corr_full.loc[bond_ids, macro_ids] if all(
            c in corr_full.index for c in bond_ids
        ) else pd.DataFrame()

        # ── 4. Beta estimation ───────────────────────────────────────────────
        beta_summary = compute_all_betas(oas_chg_aln, macro_ret_aln)

        # ── 5. Relative Value screener ───────────────────────────────────────
        # Compute historical residuals on downsampled data (every 5 days)
        try:
            residuals_hist = compute_historical_residuals(meta, oas_df)
        except Exception as exc:
            logger.warning("Could not compute historical residuals: %s", exc)
            residuals_hist = None

        rv_universe = build_rv_universe(meta, oas_df, residuals_hist)

        # ── 6. Curve trade screener ──────────────────────────────────────────
        curve_trades = screen_curve_trades(meta, oas_df)

        # ── 7. RV pair screener ──────────────────────────────────────────────
        rv_pairs = screen_rv_pairs(oas_df, meta, oas_chg_aln)

        # ── 8. Trade ideas ───────────────────────────────────────────────────
        trade_ideas = generate_all_trade_ideas(
            rv_universe,
            curve_trades,
            beta_summary,
            meta,
            macro_returns=macro_ret_aln,
        )

        data = {
            "macro_prices":  macro_prices,
            "macro_returns": macro_ret_aln,
            "oas_df":        oas_df,
            "oas_changes":   oas_chg_aln,
            "meta":          meta,
            "corr_macro":    corr_macro,
            "corr_bonds":    corr_bonds,
            "beta_summary":  beta_summary,
            "rv_universe":   rv_universe,
            "curve_trades":  curve_trades,
            "rv_pairs":      rv_pairs,
            "trade_ideas":   trade_ideas,
        }

        st.session_state[_CACHE_KEY] = data
        return data


def clear_cache() -> None:
    """Force a full data reload on next call to get_app_data()."""
    if _CACHE_KEY in st.session_state:
        del st.session_state[_CACHE_KEY]
