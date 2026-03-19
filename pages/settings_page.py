"""
Settings & Methodology Page

Explains:
  - What each metric means
  - How correlations and betas are computed
  - How RV z-scores are defined
  - How curve trades are constructed
  - Limitations of public proxies
  - How to adapt the tool to real desk data

Also exposes:
  - Global parameter controls (z-score thresholds, rolling windows)
  - Cache reset
"""

from __future__ import annotations

import streamlit as st

from config.settings import (
    COLORS,
    DEFAULT_ROLLING_WINDOWS,
    DEFAULT_ZSCORE_WINDOW,
    RV_ZSCORE_THRESHOLD,
    CURVE_ZSCORE_THRESHOLD,
)
from config.theme import STREAMLIT_CSS
from src.data.session import clear_cache


def render() -> None:
    """Render the Settings & Methodology page."""
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    st.title("Settings & Methodology")

    tab1, tab2, tab3 = st.tabs([
        "Methodology", "Data Sources & Limitations", "App Settings"
    ])

    # ── Tab 1: Methodology ────────────────────────────────────────────────────
    with tab1:
        _methodology_section()

    # ── Tab 2: Data sources ───────────────────────────────────────────────────
    with tab2:
        _data_sources_section()

    # ── Tab 3: App settings ───────────────────────────────────────────────────
    with tab3:
        _app_settings_section()


# ═════════════════════════════════════════════════════════════════════════════
# Methodology text
# ═════════════════════════════════════════════════════════════════════════════

def _methodology_section() -> None:
    st.subheader("Analytics Methodology")

    with st.expander("1. Correlation & Beta Engine", expanded=True):
        st.markdown(
            """
### Correlation Matrix

Pairwise Pearson correlations are computed between:
- Daily OAS changes of each bond in the universe
- Daily total returns of macro ETF proxies (EMB, HYG, SPY, etc.)

**Formula:**
```
ρ(X, Y) = Cov(X, Y) / (σ_X × σ_Y)
```

A positive correlation between a bond's OAS change and SPY returns means
the bond spread widens when equities rally (unusual risk-off behaviour).
More commonly, bonds with high EMB correlation are driven by broad EM credit flows.

### Rolling Correlations

Rolling windows of 20 / 60 / 120 days are used to track how correlations
evolve over time. Shorter windows capture recent regime shifts; longer windows
provide more stable estimates.

### OLS Beta Estimation

We regress each bond's daily spread change against the full factor matrix:

```
ΔSpread_t = α + β₁·ΔFactor₁_t + β₂·ΔFactor₂_t + … + ε_t
```

Betas quantify how much the bond spread changes per unit of factor return.
R² measures the total fraction of spread variance explained by the macro basket.

**Interpretation:**
- β_EMB = 1.2 → bond spread widens 1.2× more than the average EM bond when EMB sells off
- R² = 40% → 40% of daily spread moves are driven by macro; 60% is idiosyncratic

### Hedge Suggestion Logic

The primary hedge instrument is identified as the factor with the highest |beta| and
meaningful R². The hedge ratio (notional of ETF / notional of bond) is approximately:

```
hedge_ratio ≈ DV01_bond / DV01_ETF × |β|
```

Note: ETF DV01 is only approximate and should be refined with actual ETF duration.
"""
        )

    with st.expander("2. Relative Value Screener"):
        st.markdown(
            """
### Issuer Curve Fitting

For each country (e.g. Mexico), we fit a polynomial regression of OAS on maturity:

```
OAS_fair(T) = a₀ + a₁·T + a₂·T²
```

This produces a smooth "fair value" OAS curve. The degree-2 polynomial captures
the typical concave shape of credit curves (short end cheaper, belly richer).

### Residuals

```
Residual = OAS_actual - OAS_fair
```

A positive residual means the bond trades wider than fair value → **cheap**.
A negative residual means the bond trades tighter → **rich**.

### Z-Score

```
Z-Score = (Residual_current - Mean_Residual) / Std_Residual
```

Where mean and standard deviation are computed over a rolling window (typically
126–252 days). A z-score above +1.5 flags a bond as **cheap** (≥1.5σ wide).

### Limitations

- With only 2–4 bonds per country, the polynomial fit has very limited degrees
  of freedom. A 3-point quadratic fit is exact (zero residuals).
- The approach is more powerful with 6+ bonds per issuer (as in a real desk
  universe with multiple corporate issuers in the same country / sector).
- For cross-country peer analysis, we use a two-factor OLS model
  (duration + rating rank) to define peer fair value.

### Carry Approximation

Monthly carry ≈ OAS / 12 (in bps).

This ignores the time-value of coupon reinvestment and roll-down effects.
For a proper carry/roll-down calculation, you need the full OAS curve at
adjacent maturities (available when plugging in real desk data).
"""
        )

    with st.expander("3. Curve Trade Builder"):
        st.markdown(
            """
### Slope Metric

```
Slope(S, L) = OAS(L) - OAS(S)
```

Where S = short maturity, L = long maturity. A positive slope means the curve
is upward-sloping (normal). A high z-score → curve is historically steep
→ **flattener** opportunity.

### Butterfly (Curvature) Metric

```
Curvature(S, M, L) = 2·OAS(M) - OAS(S) - OAS(L)
```

Positive curvature → the belly is cheap vs. the wings → **long belly**.
Negative curvature → belly is rich → **short belly**.

### DV01-Neutral Sizing

For a 2-leg slope trade (long S, short L), the DV01-neutral constraint is:

```
Notional_L = Notional_S × (DV01_S / DV01_L)
```

This ensures that a 1bp parallel shift in the entire curve has zero P&L impact,
so the trade is a pure bet on the slope (not parallel shift).

For the 3-leg butterfly, both wings must have the same combined DV01 as the belly:

```
DV01_S·N_S + DV01_L·N_L = 2·DV01_M·N_M
```

The system splits the wing DV01 equally between the two legs by default.

### Z-Score on Curve Metrics

The same rolling z-score methodology as the RV screener is applied to each
slope / butterfly series. This tells you whether the current curve shape is
historically extreme.
"""
        )

    with st.expander("4. Trade Idea Generator"):
        st.markdown(
            """
### Signal Aggregation

Trade ideas are generated from three signal types:

1. **RV Signal** — Bonds with |z-score| ≥ 1.5 from the issuer curve fit.
2. **Curve Signal** — Curve metrics with |z-score| ≥ 1.5 from rolling history.
3. **Macro-Relative Signal** — Cheap bonds with high beta to positively
   trending macro factors (momentum crossover).

### Confidence Score

Confidence is a monotone function of signal z-score magnitude:
```
|z| ≥ 3.0  → Confidence = 85%
|z| ≥ 2.0  → Confidence = 72%
|z| ≥ 1.5  → Confidence = 60%
|z| ≥ 1.0  → Confidence = 45%
otherwise  → Confidence = 30%
```

For macro-relative ideas, the z-score is scaled by the factor R²:
```
confidence_input = |z_RV| × R² × 2
```

### What Confidence Means

Confidence is **not** a probability of profit. It reflects signal strength
relative to historical norms. High confidence = the observed dislocation is
large relative to history. Actual trade outcomes depend on market dynamics,
liquidity, and events not captured by these signals.
"""
        )


# ═════════════════════════════════════════════════════════════════════════════
# Data sources section
# ═════════════════════════════════════════════════════════════════════════════

def _data_sources_section() -> None:
    st.subheader("Data Sources & Limitations")

    st.markdown(
        """
### Public Demo Mode

In demo mode, the app uses:

| Source | Data | Notes |
|--------|------|-------|
| `yfinance` | ETF daily prices: EMB, HYG, SPY, EEM, TLT, GLD, UUP, VIXY, LQD, PDBC | Adjusted close. ~2Y history. |
| Synthetic generator | Bond OAS time-series (25 EM sovereign bonds) | Correlated random walk calibrated to approximate market levels |
| Internal | Bond metadata (country, rating, DV01, duration) | Based on publicly known EM sovereign bond parameters |

### Synthetic Data Limitations

- **Individual EM bond OAS data is not publicly available** for free.
  Bloomberg ALLQ / Refinitiv Eikon / ICE BofA indices are the standard sources
  in a sell-side environment.
- The synthetic OAS series behave realistically (correlated with EMB, mean-reverting)
  but do not reflect actual historical bond prices.
- **Treat all specific numeric outputs in demo mode as illustrative only.**

### Known Proxy Issues

| Proxy | Issue |
|-------|-------|
| UUP for DXY | UUP tracks a basket of USD vs G10 EM currencies, not the exact DXY index. |
| TLT for rates | TLT has 20Y+ duration vs UST 10Y. Adjust beta estimates accordingly. |
| VIXY for VIX | VIXY tracks VIX futures, not VIX spot. Contango drag applies. |
| ETF returns ≠ OAS changes | ETF prices reflect duration + spread; we back-out approximate OAS changes via duration scaling. |

### Adapting to Real Desk Data

1. **Replace synthetic OAS** with Bloomberg / Refinitiv pulls in `src/data/loader.py`.
   Look for the `TODO` markers.
2. **Use the Data Upload page** to import CSV exports from Bloomberg or your
   internal system.
3. **Update `BOND_UNIVERSE`** in `config/settings.py` with your actual bond universe
   (ISINs, live DV01s, actual durations).
4. **Add FRED API key** in `.env` for additional macro time-series (UST yields,
   SOFR, OFR, etc.).

```python
# Example: Loading from Bloomberg BLPAPI (TODO)
import blpapi
# ...
# Then call blp.history(tickers, fields=["YLD_YTM_MID"], dt_start=start)
```
"""
    )


# ═════════════════════════════════════════════════════════════════════════════
# App settings section
# ═════════════════════════════════════════════════════════════════════════════

def _app_settings_section() -> None:
    st.subheader("App Settings")

    st.markdown("**Current Default Parameters**")
    param_df = {
        "Parameter":          [
            "Lookback days",
            "Correlation window",
            "Z-Score window",
            "RV flag threshold",
            "Curve flag threshold",
            "Beta min R²",
        ],
        "Current Value":      [
            "504 (~2Y)",
            "60d",
            "252d (1Y)",
            "±1.5σ",
            "±1.5σ",
            "10%",
        ],
        "Description": [
            "Total history used for analytics",
            "Default rolling window for correlation",
            "Window for z-score normalisation",
            "Minimum z-score to label bond as rich/cheap",
            "Minimum z-score to flag curve opportunity",
            "Minimum R² for hedge to be reported",
        ],
    }
    st.table(param_df)

    st.markdown("---")
    st.subheader("Cache Management")
    st.info(
        "The app caches all data in session state to avoid re-downloading on "
        "every page navigation. Force a reload below if data looks stale."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reload All Data", type="primary"):
            clear_cache()
            st.success("Cache cleared. Data will reload on next page visit.")

    with col2:
        if st.button("Clear Uploaded Data Only"):
            for key in ["uploaded_oas", "uploaded_meta"]:
                if key in st.session_state:
                    del st.session_state[key]
            clear_cache()
            st.success("Uploaded data cleared. Reverted to demo mode.")

    st.markdown("---")
    st.subheader("About")
    st.markdown(
        """
**EM Credit Analytics Dashboard** v1.0.0

Built as an intern preparation tool for an EM Credit trading desk.

**Stack:** Python · Streamlit · Plotly · pandas · NumPy · SciPy · statsmodels

**Architecture:**
- `config/` — settings and theme
- `src/data/` — data loading, preprocessing, upload handling
- `src/analytics/` — correlation engine, RV screener, curve analysis, trade ideas
- `src/plotting/` — Plotly chart builders
- `pages/` — Streamlit page modules

**Contact / Issues:** Extend this tool by modifying the `TODO` markers in the source code.
"""
    )
