"""
Public market data loader for the EM Credit Analytics Dashboard.

Attempts to download real ETF prices from yfinance.
If the download fails (no internet, stale cache, etc.), falls back to
fully synthetic data so the app always runs end-to-end.

TODO: Replace / augment with Bloomberg BLPAPI or internal desk feed
      when running on a bank workstation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import (
    BOND_UNIVERSE,
    DEFAULT_LOOKBACK_DAYS,
    MACRO_TICKERS,
)

logger = logging.getLogger(__name__)

# ── FRED loader (optional, no API key required) ───────────────────────────────
try:
    from src.data.fred_loader import (
        load_em_sovereign_oas,
        load_israel_oas_history,
        load_ust_curve,
    )
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False
    logger.warning("fred_loader not found – sovereign OAS will be fully synthetic.")

# ── yfinance import (optional dependency) ────────────────────────────────────
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance not installed – using fully synthetic data.")


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def load_macro_prices(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily adjusted-close prices for macro ETF proxies.

    Columns : one per ticker (e.g. EMB, HYG, SPY …)
    Index   : DatetimeIndex (business days)

    Falls back to synthetic prices if yfinance fails.
    """
    tickers = tickers or list(MACRO_TICKERS.keys())
    end   = datetime.today()
    start = end - timedelta(days=int(lookback_days * 1.45))  # pad for weekends

    if _YF_AVAILABLE:
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"].dropna(how="all")
            else:
                prices = raw[["Close"]].rename(columns={"Close": tickers[0]}).dropna(how="all")

            # If download returned nothing useful, fall through to synthetic
            if prices.empty or len(prices) < 10:
                raise ValueError("Download returned insufficient data; using synthetic fallback.")

            # Keep only requested tickers that actually downloaded
            present = [t for t in tickers if t in prices.columns]
            prices = prices[present]

            # Fill any missing tickers with synthetic columns
            missing = [t for t in tickers if t not in prices.columns]
            if missing:
                logger.warning("Could not download: %s – using synthetic.", missing)
                synth = _synthetic_macro_prices(missing, prices.index)
                prices = pd.concat([prices, synth], axis=1)

            # Align to a clean business-day index and forward-fill gaps
            prices = prices.asfreq("B").ffill().dropna(how="all")
            prices = prices.tail(lookback_days)
            return prices

        except Exception as exc:
            logger.warning("yfinance download failed (%s) – using synthetic data.", exc)

    # Full synthetic fallback
    return _synthetic_macro_prices_full(tickers, lookback_days)


def load_bond_oas_history(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    bond_meta: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of daily OAS (bps) for the synthetic bond universe.

    Columns : bond IDs (e.g. BRL_5Y, MEX_10Y …)
    Index   : DatetimeIndex matching macro prices index

    Bond OAS series are generated synthetically but are correlated
    with the macro ETF factor structure (EMB, HYG, etc.) so that
    downstream analytics (correlations, betas, RV z-scores) behave
    realistically.

    TODO: Replace with live OAS feeds (Bloomberg / Refinitiv / internal)
          when available.
    """
    bond_meta = bond_meta or BOND_UNIVERSE

    # Use macro prices as the factor base so bond returns are correlated
    macro = load_macro_prices(lookback_days)
    oas_df = _synthetic_bond_oas(bond_meta, macro)

    if _FRED_AVAILABLE:
        # Map of bond_id → (fred_country, oas_base) for bonds with FRED data
        fred_bond_map = {
            "ISR_10Y": ("Israel",        next((b["oas_base"] for b in bond_meta if b["id"] == "ISR_10Y"), -30)),
            "MEX_10Y": ("Mexico",        next((b["oas_base"] for b in bond_meta if b["id"] == "MEX_10Y"), 195)),
            "ZAF_10Y": ("South Africa",  next((b["oas_base"] for b in bond_meta if b["id"] == "ZAF_10Y"), 320)),
            "CHL_10Y": ("Chile",         next((b["oas_base"] for b in bond_meta if b["id"] == "CHL_10Y"), 105)),
        }

        for bond_id, (country, oas_base) in fred_bond_map.items():
            if bond_id not in oas_df.columns:
                continue
            try:
                if country == "Israel":
                    real = load_israel_oas_history(lookback_days)
                else:
                    real = load_em_sovereign_oas(country, oas_base=oas_base, lookback_days=lookback_days)

                if real is not None and not real.empty:
                    common = oas_df.index.intersection(real.index)
                    if len(common) > 10:
                        oas_df.loc[common, bond_id] = real.loc[common].values
                        logger.info("%s OAS: using real FRED data (%d rows).", bond_id, len(common))
            except Exception as exc:
                logger.warning("%s FRED OAS failed (%s) – keeping synthetic.", bond_id, exc)

    return oas_df


def load_bond_prices_from_oas(
    oas_df: pd.DataFrame,
    bond_meta: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Approximate bond prices from OAS changes using:
        ΔP ≈ -duration × (ΔOAS / 10_000)

    Starts each bond at par (100) and compounds changes daily.
    """
    bond_meta = bond_meta or BOND_UNIVERSE
    dur_map = {b["id"]: b["duration"] for b in bond_meta}

    oas_chg = oas_df.diff()    # daily change in OAS (bps)
    price_chg = pd.DataFrame(index=oas_df.index, columns=oas_df.columns, dtype=float)

    for col in oas_df.columns:
        dur = dur_map.get(col, 7.0)
        # Approximate daily price return from OAS change and coupon carry
        daily_return = -dur * oas_chg[col] / 10_000
        price_chg[col] = daily_return

    # Start at 100 and compound
    prices = (1 + price_chg.fillna(0)).cumprod() * 100
    prices.iloc[0] = 100
    return prices


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers – synthetic data generation
# ═════════════════════════════════════════════════════════════════════════════

def _business_day_index(n: int) -> pd.DatetimeIndex:
    """Generate the last n business days ending today."""
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n)
    return dates


def _synthetic_macro_prices_full(
    tickers: list[str],
    n_days: int,
) -> pd.DataFrame:
    """
    Generate synthetic ETF price series with realistic risk parameters.
    Prices follow geometric Brownian motion with calibrated vol / drift.
    """
    rng = np.random.default_rng(42)
    dates = _business_day_index(n_days)

    # Approximate annual vol and starting price per ticker
    _params: dict[str, tuple[float, float]] = {
        "EMB":  (0.065, 90.0),
        "HYG":  (0.080, 77.0),
        "SPY":  (0.160, 450.0),
        "EEM":  (0.200, 40.0),
        "TLT":  (0.140, 95.0),
        "GLD":  (0.120, 175.0),
        "UUP":  (0.050, 28.0),
        "VIXY": (0.600, 15.0),
        "LQD":  (0.070, 108.0),
        "PDBC": (0.180, 15.0),
    }

    dt = 1 / 252
    prices: dict[str, np.ndarray] = {}

    for t in tickers:
        vol, s0 = _params.get(t, (0.15, 100.0))
        drift = 0.05 * dt  # generic 5% annual drift
        shocks = rng.normal(0, vol * np.sqrt(dt), n_days)
        log_ret = drift - 0.5 * vol**2 * dt + shocks
        prices[t] = s0 * np.exp(np.cumsum(log_ret))

    return pd.DataFrame(prices, index=dates)


def _synthetic_macro_prices(
    tickers: list[str],
    existing_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Fill-in synthetic columns aligned to an existing price index."""
    n = len(existing_index)
    full = _synthetic_macro_prices_full(tickers, n)
    full.index = existing_index
    return full


def _synthetic_bond_oas(
    bond_meta: list[dict],
    macro_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate OAS time series for each bond, correlated with macro factors.

    Each bond's daily OAS change is modelled as:
        dOAS = β_emb * dEMB_spread + β_hyg * dHYG_spread
               + country_factor * country_shock
               + idio_vol * idio_shock
               + mean_reversion_term

    where dETF_spread is approximated from ETF price returns (inverted and
    duration-adjusted to OAS space).
    """
    rng = np.random.default_rng(seed=2024)
    n   = len(macro_prices)
    dates = macro_prices.index

    # Compute approximate OAS proxy from EMB and HYG price returns
    # OAS change ≈ -price_return / duration_proxy
    def _price_to_oas_chg(col: str, dur_proxy: float = 7.0) -> np.ndarray:
        if col in macro_prices.columns:
            ret = macro_prices[col].pct_change().fillna(0).values
            return -ret * 10_000 / dur_proxy   # convert to bps
        return np.zeros(n)

    emb_oas_chg = _price_to_oas_chg("EMB", dur_proxy=7.0)
    hyg_oas_chg = _price_to_oas_chg("HYG", dur_proxy=4.0)
    spy_ret     = macro_prices["SPY"].pct_change().fillna(0).values if "SPY" in macro_prices.columns else np.zeros(n)
    uup_ret     = macro_prices["UUP"].pct_change().fillna(0).values if "UUP" in macro_prices.columns else np.zeros(n)

    # Country-specific volatility multiplier (higher for HY, lower for IG).
    # Kept deliberately modest so that the historical residual std stays
    # in the 8–15 bps range, making a 25–40 bps dislocation clearly z>2.
    country_vols: dict[str, float] = {
        "Brazil": 0.60, "Mexico": 0.40, "Colombia": 0.50,
        "Chile": 0.30,  "Peru": 0.38,   "Indonesia": 0.45,
        "Turkey": 0.90, "South Africa": 0.55, "Egypt": 0.80,
    }
    idio_vols: dict[str, float] = {
        "Brazil": 1.2, "Mexico": 0.9, "Colombia": 1.0,
        "Chile": 0.6,  "Peru": 0.7,   "Indonesia": 0.9,
        "Turkey": 1.8, "South Africa": 1.3, "Egypt": 1.9,
    }

    # Country-level correlated shocks (shared within a country)
    country_names = list({b["country"] for b in bond_meta})
    country_shocks: dict[str, np.ndarray] = {
        c: rng.normal(0, country_vols.get(c, 1.0), n)
        for c in country_names
    }

    oas_dict: dict[str, np.ndarray] = {}

    # Persistent dislocation biases added to specific bonds for demo realism.
    # These represent bonds that have been driven away from fair value by
    # idiosyncratic supply/demand technicals, rating agency action, or a
    # specific news event — the kind of rich/cheap situation the RV screener
    # is designed to flag.
    # TODO: Remove when plugging in real desk data.
    _dislocation_bias: dict[str, float] = {
        "BRL_5Y":   +28.0,   # cheap — trades 28bps wide of Brazil curve
        "MEX_10Y":  -22.0,   # rich  — trades 22bps tight of Mexico curve
        "COL_30Y":  +35.0,   # cheap — Colombia long end dislocated
        "CHL_10Y":  -18.0,   # rich  — Chile 10Y bid up by strong demand
        "IDN_5Y":   +25.0,   # cheap — Indonesia 5Y faces supply pressure
        "TUR_10Y":  -30.0,   # rich  — Turkey 10Y short covering distorts level
    }

    for bond in bond_meta:
        bid     = bond["id"]
        country = bond["country"]
        oas0    = bond["oas_base"]
        _dur    = bond["duration"]  # reserved for future roll-down / DV01 weighting

        # Factor loadings (calibrated to approximate historical correlations)
        b_emb = 0.65 + rng.uniform(-0.10, 0.10)
        b_hyg = 0.25 + rng.uniform(-0.05, 0.05)
        b_spy = -0.05 * (oas0 / 200)  # higher-spread bonds more sensitive to risk-off
        b_uup = -0.10 * (oas0 / 200)  # DXY negative for EM

        # Daily OAS changes
        dOAS = (
            b_emb * emb_oas_chg
            + b_hyg * hyg_oas_chg
            + b_spy * spy_ret * 10_000 / 10   # small equity linkage
            + b_uup * uup_ret * 10_000 / 10
            + country_shocks[country]
            + rng.normal(0, idio_vols.get(country, 3.0), n)
        )

        # Recent dislocation bias — ramps up only in the last ~20% of history.
        # This simulates a bond that has recently drifted away from fair value
        # (e.g. due to a supply/demand technical or rating action), creating a
        # z-score signal visible to the RV screener.
        bias = _dislocation_bias.get(bid, 0.0)
        if bias != 0.0:
            onset = int(n * 0.80)   # dislocation starts at the 80th percentile of history
            ramp  = np.zeros(n)
            ramp[onset:] = np.linspace(0, 1, n - onset)
            dislocation = bias * ramp  # level offset growing toward full bias
        else:
            dislocation = np.zeros(n)

        # Mean reversion: pull toward the base OAS level with moderate speed.
        # Faster mean reversion keeps the historical residual variance small,
        # which makes recent dislocations stand out clearly in z-score space.
        oas_series = np.empty(n)
        oas_series[0] = oas0
        mr_speed = 0.04  # ~4% daily mean reversion (~25-day half-life)
        for t in range(1, n):
            oas_series[t] = (
                oas_series[t - 1]
                + dOAS[t]
                - mr_speed * (oas_series[t - 1] - oas0)
            )
            # Clamp to realistic range (no negative OAS)
            oas_series[t] = max(oas_series[t], 5.0)

        # Apply persistent dislocation on top of the mean-reverting series
        oas_dict[bid] = oas_series + dislocation

    return pd.DataFrame(oas_dict, index=dates)
