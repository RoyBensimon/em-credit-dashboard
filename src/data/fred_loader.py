"""
FRED (Federal Reserve Economic Data) loader for the EM Credit Analytics Dashboard.

Uses the public FRED CSV endpoint — no API key required.
Provides real daily US Treasury yields, credit spread indices, and
monthly sovereign bond yields for select countries.

Data sources:
  - US Treasury yields : DGS2, DGS5, DGS10, DGS30 (daily)
  - Credit indices     : BAMLH0A0HYM2 (HY OAS), BAMLC0A0CM (IG OAS) (daily)
  - USD broad index    : DTWEXBGS (weekly, interpolated to daily)
  - SOFR               : SOFR (daily)
  - Israel 10Y         : IRLTLT01ILM156N (monthly, ILS local govt bond)

Note on Israel:
  The FRED Israel series is the *local-currency (ILS) govt bond* 10Y yield,
  not a USD Eurobond OAS.  The derived spread vs UST 10Y is therefore a
  sovereign creditworthiness proxy, not a direct USD Eurobond OAS.
  For proper USD Eurobond OAS, a Bloomberg or Refinitiv feed is required.

Note on Ukraine:
  No Ukraine sovereign bond yield series is available on FRED.
  Post-2024 restructuring data exists only on Bloomberg / Refinitiv.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_REQUEST_TIMEOUT = 15  # seconds

# ── Series definitions ────────────────────────────────────────────────────────

UST_SERIES: dict[str, str] = {
    "DGS2":  "US 2Y Treasury Yield (%)",
    "DGS5":  "US 5Y Treasury Yield (%)",
    "DGS10": "US 10Y Treasury Yield (%)",
    "DGS30": "US 30Y Treasury Yield (%)",
}

CREDIT_SPREAD_SERIES: dict[str, str] = {
    "BAMLH0A0HYM2": "US HY OAS (bps, ICE BofA)",
    "BAMLC0A0CM":   "US IG OAS (bps, ICE BofA)",
}

OTHER_SERIES: dict[str, str] = {
    "SOFR":     "SOFR (%)",
    "DTWEXBGS": "USD Broad Trade-Weighted Index",
}

SOVEREIGN_YIELD_SERIES: dict[str, dict] = {
    "Israel": {
        "series_id":   "IRLTLT01ILM156N",
        "freq":         "monthly",
        "description":  "Israel 10Y Govt Bond Yield, %  (ILS local bond, OECD/FRED)",
        "scale_factor": 1.00,  # ILS spread ≈ USD Eurobond OAS for this credit quality
    },
    "Mexico": {
        "series_id":   "IRLTLT01MXM156N",
        "freq":         "monthly",
        "description":  "Mexico 10Y Govt Bond Yield, %  (MXN local bond, OECD/FRED)",
        "scale_factor": 0.37,  # Local spread >> USD Eurobond OAS (high MXN inflation premium)
    },
    "South Africa": {
        "series_id":   "IRLTLT01ZAM156N",
        "freq":         "monthly",
        "description":  "South Africa 10Y Govt Bond Yield, %  (ZAR local bond, OECD/FRED)",
        "scale_factor": 0.70,  # ZAR bonds include significant currency risk premium
    },
    "Chile": {
        "series_id":   "IRLTLT01CLM156N",
        "freq":         "monthly",
        "description":  "Chile 10Y Govt Bond Yield, %  (CLP local bond, OECD/FRED)",
        "scale_factor": 0.95,  # CLP spread is close to USD Eurobond OAS for IG Chile
    },
    # Not available on FRED: Brazil, Indonesia, Turkey, Peru, Egypt, Colombia, Ukraine.
}


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def load_ust_curve(lookback_days: int = 504) -> pd.DataFrame:
    """
    Return daily US Treasury yields for 2Y / 5Y / 10Y / 30Y maturities.

    Columns : DGS2, DGS5, DGS10, DGS30 (yield in %, e.g. 4.27)
    Index   : DatetimeIndex (business days, forward-filled)
    """
    frames = []
    for sid in UST_SERIES:
        s = _fetch_fred_series(sid, lookback_days)
        if s is not None:
            frames.append(s.rename(sid))

    if not frames:
        logger.warning("FRED UST curve: all downloads failed — returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).asfreq("B").ffill()
    return df.tail(lookback_days).dropna(how="all")


def load_credit_spread_indices(lookback_days: int = 504) -> pd.DataFrame:
    """
    Return daily US HY and IG OAS from ICE BofA indices (via FRED).

    Columns : BAMLH0A0HYM2 (HY OAS, bps), BAMLC0A0CM (IG OAS, bps)
    Index   : DatetimeIndex (business days)

    Note: FRED reports these already in bps (e.g. 3.28 = 328 bps for HY).
    Multiply by 100 to convert to the bps convention used by the dashboard.
    """
    frames = []
    for sid in CREDIT_SPREAD_SERIES:
        s = _fetch_fred_series(sid, lookback_days)
        if s is not None:
            # FRED stores these as decimal (e.g. 3.28) → convert to bps
            frames.append((s * 100).rename(sid))

    if not frames:
        logger.warning("FRED credit indices: all downloads failed.")
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).asfreq("B").ffill()
    return df.tail(lookback_days).dropna(how="all")


def load_israel_oas_history(
    lookback_days: int = 504,
    ust_10y: pd.Series | None = None,
) -> pd.Series | None:
    """
    Return a daily-interpolated OAS proxy for Israel sovereign bonds.

    Methodology
    -----------
    1. Download FRED IRLTLT01ILM156N  (Israel 10Y local-currency govt bond, monthly)
    2. Download FRED DGS10            (US 10Y Treasury yield, daily) if not provided
    3. OAS_proxy = (Israel_yield - UST_10Y) * 100   [in bps]
    4. Resample to business-day frequency via cubic interpolation

    Caveat: this is a *local ILS bond* spread, not a USD Eurobond OAS.
    Treat as a sovereign creditworthiness direction indicator.

    Returns None if FRED download fails.
    """
    # Fetch Israel monthly yield
    il_monthly = _fetch_fred_series("IRLTLT01ILM156N", lookback_days=lookback_days * 3)
    if il_monthly is None or il_monthly.empty:
        logger.warning("FRED: Israel yield download failed.")
        return None

    # Always fetch UST with extended lookback so monthly resampling has enough coverage.
    # The ust_10y parameter is accepted for API compatibility but ignored here to avoid
    # the case where a short (e.g. 60-day) series yields only 1-2 monthly data points.
    ust_series = _fetch_fred_series("DGS10", lookback_days=max(lookback_days * 3, 600))
    if ust_series is None:
        logger.warning("FRED: UST 10Y download failed for Israel OAS calculation.")
        return None

    # Align on monthly index, compute spread
    ust_monthly = ust_series.resample("MS").last()
    common = il_monthly.index.intersection(ust_monthly.index)
    if len(common) < 3:
        logger.warning("FRED: insufficient Israel/UST overlap for OAS calculation.")
        return None

    oas_monthly = (il_monthly.loc[common] - ust_monthly.loc[common]) * 100  # bps

    # Interpolate to business-day frequency
    bday_index = pd.bdate_range(
        start=oas_monthly.index[0],
        end=pd.Timestamp.today().normalize(),
    )
    oas_daily = (
        oas_monthly
        .reindex(oas_monthly.index.union(bday_index))
        .interpolate(method="time")
        .reindex(bday_index)
        .ffill()
    )

    oas_daily.name = "ISR_10Y_OAS_proxy"
    return oas_daily.tail(lookback_days)


def load_em_sovereign_oas(
    country: str,
    oas_base: float,
    lookback_days: int = 504,
) -> pd.Series | None:
    """
    Return a daily-interpolated OAS proxy for a sovereign bond, derived from
    a FRED local-currency government bond yield series.

    Methodology
    -----------
    1. Fetch the local-currency 10Y yield from FRED (monthly)
    2. Fetch UST 10Y (daily), resample to monthly
    3. Compute local_spread = (local_yield - UST_10Y) * 100  [bps]
    4. Apply level-calibration so the mean of the series equals oas_base:
         OAS_proxy(t) = oas_base + (local_spread(t) - median(local_spread)) * scale_factor
       This preserves the direction / dynamics from real data while anchoring
       the level to our calibrated oas_base (which reflects USD Eurobond OAS,
       not the local bond spread which embeds FX and inflation premia).
    5. Interpolate to business-day frequency, forward-fill to today.

    Parameters
    ----------
    country    : key in SOVEREIGN_YIELD_SERIES (e.g. "Mexico")
    oas_base   : target USD Eurobond OAS level in bps (from settings.py)
    lookback_days : number of business days to return

    Returns None if the country has no FRED series or the download fails.
    """
    meta = SOVEREIGN_YIELD_SERIES.get(country)
    if meta is None:
        return None

    series_id    = meta["series_id"]
    scale_factor = meta.get("scale_factor", 1.0)

    local_monthly = _fetch_fred_series(series_id, lookback_days=max(lookback_days * 3, 600))
    if local_monthly is None or local_monthly.empty:
        logger.warning("FRED: %s yield download failed.", country)
        return None

    ust_series = _fetch_fred_series("DGS10", lookback_days=max(lookback_days * 3, 600))
    if ust_series is None:
        logger.warning("FRED: UST 10Y download failed for %s OAS calculation.", country)
        return None

    ust_monthly = ust_series.resample("MS").last()
    common = local_monthly.index.intersection(ust_monthly.index)
    if len(common) < 3:
        logger.warning("FRED: insufficient %s/UST overlap.", country)
        return None

    local_spread = (local_monthly.loc[common] - ust_monthly.loc[common]) * 100  # bps

    # Level-calibrate: anchor mean to oas_base, preserve dynamics
    spread_median = local_spread.median()
    oas_monthly   = oas_base + (local_spread - spread_median) * scale_factor

    # Interpolate to business-day frequency
    bday_index = pd.bdate_range(
        start=oas_monthly.index[0],
        end=pd.Timestamp.today().normalize(),
    )
    oas_daily = (
        oas_monthly
        .reindex(oas_monthly.index.union(bday_index))
        .interpolate(method="time")
        .reindex(bday_index)
        .ffill()
    )
    oas_daily.name = f"{country}_10Y_OAS_proxy"
    return oas_daily.tail(lookback_days)


def load_all_fred_macro(lookback_days: int = 504) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Convenience wrapper — returns a dict with all available FRED series.

    Keys:
      'ust_curve'       : pd.DataFrame  (DGS2/5/10/30)
      'credit_spreads'  : pd.DataFrame  (HY OAS / IG OAS in bps)
      'israel_oas'      : pd.Series | None
      'em_sovereign'    : dict[str, pd.Series | None]  (Mexico, South Africa, Chile)
    """
    ust = load_ust_curve(lookback_days)

    return {
        "ust_curve":     ust,
        "credit_spreads": load_credit_spread_indices(lookback_days),
        "israel_oas":    load_israel_oas_history(lookback_days),
        "em_sovereign":  {
            c: load_em_sovereign_oas(c, oas_base=0, lookback_days=lookback_days)
            for c in SOVEREIGN_YIELD_SERIES
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_fred_series(series_id: str, lookback_days: int = 504) -> pd.Series | None:
    """
    Download a single FRED series via the public CSV endpoint.
    Returns a pd.Series with DatetimeIndex, or None on failure.
    """
    start = (datetime.today() - timedelta(days=int(lookback_days * 1.6))).strftime("%Y-%m-%d")

    try:
        r = requests.get(
            _FRED_CSV_URL,
            params={"id": series_id},
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text))
        if df.shape[1] < 2:
            raise ValueError(f"Unexpected CSV shape: {df.shape}")

        df.columns = ["date", series_id]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        df = df.dropna().set_index("date")
        df = df[df.index >= start]

        if df.empty:
            raise ValueError("No data rows after filtering.")

        logger.info("FRED: downloaded %s (%d rows).", series_id, len(df))
        return df[series_id]

    except Exception as exc:
        logger.warning("FRED: failed to download %s — %s", series_id, exc)
        return None
