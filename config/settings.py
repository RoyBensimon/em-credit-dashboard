"""
Global configuration and constants for the EM Credit Analytics Dashboard.

All magic numbers, ticker lists, bond universe definitions, and default
parameters live here so that the rest of the codebase stays clean.
"""

# ── App metadata ─────────────────────────────────────────────────────────────
APP_TITLE   = "EM Credit Analytics Dashboard"
APP_ICON    = "📊"
APP_VERSION = "1.0.0"

# ── Date / lookback defaults ─────────────────────────────────────────────────
DEFAULT_LOOKBACK_DAYS     = 504   # ~2 years of trading days
DEFAULT_ROLLING_WINDOWS   = [20, 60, 120]
DEFAULT_CORRELATION_WINDOW = 60
DEFAULT_ZSCORE_WINDOW     = 252   # 1 year rolling z-score window

# ── Public proxy tickers (yfinance) ─────────────────────────────────────────
# TODO: Replace with internal desk data feed when available.
MACRO_TICKERS: dict[str, str] = {
    "EMB":  "iShares J.P. Morgan USD EM Bond ETF",
    "HYG":  "iShares iBoxx $ HY Corporate Bond ETF",
    "SPY":  "SPDR S&P 500 ETF",
    "EEM":  "iShares MSCI Emerging Markets ETF",
    "TLT":  "iShares 20+ Year Treasury Bond ETF",
    "GLD":  "SPDR Gold Shares",
    "UUP":  "Invesco DB US Dollar Index Bullish Fund (DXY Proxy)",
    "VIXY": "ProShares VIX Short-Term Futures ETF",
    "LQD":  "iShares iBoxx $ IG Corporate Bond ETF",
    "PDBC": "Invesco Optimum Yield Diversified Commodity Strategy ETF",
}

# Country ETF proxies used to generate correlated synthetic bond returns
COUNTRY_ETF_PROXIES: dict[str, str | None] = {
    "Brazil":       "EWZ",
    "Mexico":       "EWW",
    "Colombia":     "GXG",
    "Chile":        "ECH",
    "Indonesia":    "EIDO",
    "Turkey":       "TUR",
    "South Africa": "EZA",
    "Peru":         "EPU",
    "India":        "INDA",
    "Egypt":        None,  # No liquid single-country ETF; use synthetic only
}

# ── Synthetic bond universe ───────────────────────────────────────────────────
# Fields per bond:
#   id           : unique bond identifier (used as column name in time-series)
#   country      : sovereign issuer country
#   issuer       : legal issuer name (same as country for sovereigns)
#   currency     : denomination currency
#   coupon       : annual coupon rate (%)
#   maturity     : years to maturity bucket (2 / 5 / 10 / 30)
#   rating       : Moody's / S&P composite rating proxy
#   duration     : modified duration (years)
#   dv01         : dollar value of 1bp per $1mm face (USD)
#   oas_base     : starting OAS in bps (calibrated to approximate market levels)
#   yield_base   : starting yield in % (approx UST + OAS)
#
# TODO: Replace oas_base / yield_base with live desk feeds or Bloomberg pulls.
BOND_UNIVERSE: list[dict] = [
    # ── Brazil (BB-) — no FRED series; oas_base calibrated to Jan-2026 market levels
    {"id": "BRL_2Y",  "country": "Brazil",       "issuer": "Brazil",       "currency": "USD",
     "coupon": 4.625, "maturity": 2,  "rating": "BB-",  "duration": 1.90,  "dv01": 19_000,
     "oas_base": 230, "yield_base": 6.00, "sector": "Sovereign"},

    {"id": "BRL_5Y",  "country": "Brazil",       "issuer": "Brazil",       "currency": "USD",
     "coupon": 5.625, "maturity": 5,  "rating": "BB-",  "duration": 4.40,  "dv01": 44_000,
     "oas_base": 285, "yield_base": 7.10, "sector": "Sovereign"},

    {"id": "BRL_10Y", "country": "Brazil",       "issuer": "Brazil",       "currency": "USD",
     "coupon": 6.000, "maturity": 10, "rating": "BB-",  "duration": 7.80,  "dv01": 78_000,
     "oas_base": 310, "yield_base": 7.35, "sector": "Sovereign"},

    {"id": "BRL_30Y", "country": "Brazil",       "issuer": "Brazil",       "currency": "USD",
     "coupon": 6.500, "maturity": 30, "rating": "BB-",  "duration": 14.20, "dv01": 142_000,
     "oas_base": 340, "yield_base": 7.65, "sector": "Sovereign"},

    # ── Mexico (BBB-) — MEX_10Y driven by FRED IRLTLT01MXM156N (scale 0.37)
    {"id": "MEX_2Y",  "country": "Mexico",       "issuer": "Mexico",       "currency": "USD",
     "coupon": 3.500, "maturity": 2,  "rating": "BBB-", "duration": 1.95,  "dv01": 19_500,
     "oas_base": 130, "yield_base": 5.55, "sector": "Sovereign"},

    {"id": "MEX_5Y",  "country": "Mexico",       "issuer": "Mexico",       "currency": "USD",
     "coupon": 4.000, "maturity": 5,  "rating": "BBB-", "duration": 4.50,  "dv01": 45_000,
     "oas_base": 165, "yield_base": 5.90, "sector": "Sovereign"},

    {"id": "MEX_10Y", "country": "Mexico",       "issuer": "Mexico",       "currency": "USD",
     "coupon": 4.500, "maturity": 10, "rating": "BBB-", "duration": 8.00,  "dv01": 80_000,
     "oas_base": 195, "yield_base": 6.20, "sector": "Sovereign",
     "data_source": "FRED:IRLTLT01MXM156N (local MXN bond, scale=0.37)"},

    {"id": "MEX_30Y", "country": "Mexico",       "issuer": "Mexico",       "currency": "USD",
     "coupon": 5.000, "maturity": 30, "rating": "BBB-", "duration": 14.80, "dv01": 148_000,
     "oas_base": 220, "yield_base": 6.45, "sector": "Sovereign"},

    # ── Colombia (BB+) — no FRED series; oas_base reflects fiscal/political stress
    {"id": "COL_5Y",  "country": "Colombia",     "issuer": "Colombia",     "currency": "USD",
     "coupon": 4.750, "maturity": 5,  "rating": "BB+",  "duration": 4.40,  "dv01": 44_000,
     "oas_base": 235, "yield_base": 6.60, "sector": "Sovereign"},

    {"id": "COL_10Y", "country": "Colombia",     "issuer": "Colombia",     "currency": "USD",
     "coupon": 5.200, "maturity": 10, "rating": "BB+",  "duration": 7.90,  "dv01": 79_000,
     "oas_base": 265, "yield_base": 6.90, "sector": "Sovereign"},

    {"id": "COL_30Y", "country": "Colombia",     "issuer": "Colombia",     "currency": "USD",
     "coupon": 5.750, "maturity": 30, "rating": "BB+",  "duration": 14.50, "dv01": 145_000,
     "oas_base": 285, "yield_base": 7.10, "sector": "Sovereign"},

    # ── Chile (A-) — CHL_10Y driven by FRED IRLTLT01CLM156N (scale 0.95)
    {"id": "CHL_5Y",  "country": "Chile",        "issuer": "Chile",        "currency": "USD",
     "coupon": 2.750, "maturity": 5,  "rating": "A-",   "duration": 4.60,  "dv01": 46_000,
     "oas_base": 90,  "yield_base": 5.15, "sector": "Sovereign"},

    {"id": "CHL_10Y", "country": "Chile",        "issuer": "Chile",        "currency": "USD",
     "coupon": 3.500, "maturity": 10, "rating": "A-",   "duration": 8.30,  "dv01": 83_000,
     "oas_base": 105, "yield_base": 5.30, "sector": "Sovereign",
     "data_source": "FRED:IRLTLT01CLM156N (local CLP bond, scale=0.95)"},

    {"id": "CHL_30Y", "country": "Chile",        "issuer": "Chile",        "currency": "USD",
     "coupon": 4.000, "maturity": 30, "rating": "A-",   "duration": 15.10, "dv01": 151_000,
     "oas_base": 125, "yield_base": 5.50, "sector": "Sovereign"},

    # ── Peru (BBB) — no FRED series; oas_base reflects political uncertainty
    {"id": "PER_5Y",  "country": "Peru",         "issuer": "Peru",         "currency": "USD",
     "coupon": 3.300, "maturity": 5,  "rating": "BBB",  "duration": 4.55,  "dv01": 45_500,
     "oas_base": 155, "yield_base": 5.80, "sector": "Sovereign"},

    {"id": "PER_10Y", "country": "Peru",         "issuer": "Peru",         "currency": "USD",
     "coupon": 4.125, "maturity": 10, "rating": "BBB",  "duration": 8.20,  "dv01": 82_000,
     "oas_base": 185, "yield_base": 6.10, "sector": "Sovereign"},

    # ── Indonesia (BBB) — no FRED series; oas_base reflects commodity/rupiah dynamics
    {"id": "IDN_5Y",  "country": "Indonesia",    "issuer": "Indonesia",    "currency": "USD",
     "coupon": 4.100, "maturity": 5,  "rating": "BBB",  "duration": 4.50,  "dv01": 45_000,
     "oas_base": 185, "yield_base": 6.10, "sector": "Sovereign"},

    {"id": "IDN_10Y", "country": "Indonesia",    "issuer": "Indonesia",    "currency": "USD",
     "coupon": 4.650, "maturity": 10, "rating": "BBB",  "duration": 7.90,  "dv01": 79_000,
     "oas_base": 210, "yield_base": 6.35, "sector": "Sovereign"},

    {"id": "IDN_30Y", "country": "Indonesia",    "issuer": "Indonesia",    "currency": "USD",
     "coupon": 5.250, "maturity": 30, "rating": "BBB",  "duration": 14.60, "dv01": 146_000,
     "oas_base": 240, "yield_base": 6.65, "sector": "Sovereign"},

    # ── Turkey (B+) — no FRED series; oas_base reflects normalization progress
    {"id": "TUR_5Y",  "country": "Turkey",       "issuer": "Turkey",       "currency": "USD",
     "coupon": 6.375, "maturity": 5,  "rating": "B+",   "duration": 4.20,  "dv01": 42_000,
     "oas_base": 370, "yield_base": 8.95, "sector": "Sovereign"},

    {"id": "TUR_10Y", "country": "Turkey",       "issuer": "Turkey",       "currency": "USD",
     "coupon": 7.250, "maturity": 10, "rating": "B+",   "duration": 7.10,  "dv01": 71_000,
     "oas_base": 410, "yield_base": 9.35, "sector": "Sovereign"},

    # ── South Africa (BB-) — ZAF_10Y driven by FRED IRLTLT01ZAM156N (scale 0.70)
    {"id": "ZAF_5Y",  "country": "South Africa", "issuer": "South Africa", "currency": "USD",
     "coupon": 5.000, "maturity": 5,  "rating": "BB-",  "duration": 4.40,  "dv01": 44_000,
     "oas_base": 295, "yield_base": 7.20, "sector": "Sovereign"},

    {"id": "ZAF_10Y", "country": "South Africa", "issuer": "South Africa", "currency": "USD",
     "coupon": 5.875, "maturity": 10, "rating": "BB-",  "duration": 7.80,  "dv01": 78_000,
     "oas_base": 320, "yield_base": 7.45, "sector": "Sovereign",
     "data_source": "FRED:IRLTLT01ZAM156N (local ZAR bond, scale=0.70)"},

    # ── Egypt (B) — no FRED series; oas_base reflects IMF programme stress
    {"id": "EGY_5Y",  "country": "Egypt",        "issuer": "Egypt",        "currency": "USD",
     "coupon": 6.200, "maturity": 5,  "rating": "B",    "duration": 4.10,  "dv01": 41_000,
     "oas_base": 520, "yield_base": 9.45, "sector": "Sovereign"},

    {"id": "EGY_10Y", "country": "Egypt",        "issuer": "Egypt",        "currency": "USD",
     "coupon": 7.500, "maturity": 10, "rating": "B",    "duration": 7.00,  "dv01": 70_000,
     "oas_base": 570, "yield_base": 9.95, "sector": "Sovereign"},

    # ── Israel (A+) — OAS derived from FRED IRLTLT01ILM156N (ILS local bond) vs UST 10Y.
    # Note: this is the LOCAL-CURRENCY (ILS) sovereign yield spread, not a USD Eurobond OAS.
    # Treat as a sovereign creditworthiness direction indicator.
    # USD Eurobond OAS would require Bloomberg / Refinitiv.
    # oas_base is a calibrated starting level; loader.py will overwrite with real FRED data
    # when available.
    {"id": "ISR_10Y", "country": "Israel",       "issuer": "Israel",       "currency": "USD",
     "coupon": 3.875, "maturity": 10, "rating": "A+",   "duration": 8.10,  "dv01": 81_000,
     "oas_base": -30, "yield_base": 4.25, "sector": "Sovereign",
     "data_source": "FRED:IRLTLT01ILM156N"},

    # ── Ukraine (CC / Distressed) — NO free real-time data available.
    # Post-2024 restructuring Eurobonds are only priced on Bloomberg/Refinitiv.
    # The entry below uses calibrated synthetic OAS at distressed levels.
    # Remove or replace with a live feed when available.
    {"id": "UKR_10Y", "country": "Ukraine",      "issuer": "Ukraine",      "currency": "USD",
     "coupon": 7.750, "maturity": 10, "rating": "CC",   "duration": 5.80,  "dv01": 58_000,
     "oas_base": 2800, "yield_base": 32.00, "sector": "Sovereign",
     "data_source": "synthetic — no free feed post-restructuring"},
]

# ── Rating ordering for sorting ───────────────────────────────────────────────
RATING_ORDER: list[str] = [
    "AAA", "AA+", "AA", "AA-",
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-",
    "B+", "B", "B-",
    "CCC+", "CCC", "CCC-",
    "CC", "C", "D",
]

# ── Analytics thresholds ──────────────────────────────────────────────────────
RV_ZSCORE_THRESHOLD     = 1.5   # z-score to flag a bond as rich / cheap
CURVE_ZSCORE_THRESHOLD  = 1.5
BETA_MIN_R2             = 0.10  # Minimum R² to report a beta as meaningful
CONFIDENCE_HIGH         = 0.70
CONFIDENCE_MEDIUM       = 0.50
CONFIDENCE_LOW          = 0.35

# ── Color palette (matches theme.py) ────────────────────────────────────────
COLORS: dict[str, str] = {
    "primary":     "#00D4FF",
    "secondary":   "#7C3AED",
    "positive":    "#10B981",
    "negative":    "#EF4444",
    "neutral":     "#6B7280",
    "warning":     "#F59E0B",
    "background":  "#0F1117",
    "surface":     "#1E2130",
    "surface2":    "#262B3D",
    "text":        "#E2E8F0",
    "text_muted":  "#94A3B8",
}
