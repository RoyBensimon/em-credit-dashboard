# EM Credit Analytics Dashboard

A professional Streamlit analytics tool for an Emerging Markets Credit trading desk.

Designed for: *Intern preparation / junior quant work on an EM Credit desk.*

---

## What This Dashboard Does

| Module | Capability |
|--------|-----------|
| **Overview** | KPI cards, top OAS movers, macro performance, trade idea shortlist |
| **Correlation & Beta Engine** | Static + rolling correlations, OLS betas, hedge suggestions, regime analysis |
| **Relative Value Screener** | Issuer curve fitting, OAS residuals, z-scores, rich/cheap ranking |
| **Curve Trade Builder** | Slope / butterfly metrics, z-scores, DV01-neutral sizing, opportunity screener |
| **Trade Idea Generator** | Signal-aggregated, structured trade ideas with rationale, metrics, and risks |
| **Data Upload** | CSV / Excel upload, column mapping, data validation, internal mode activation |
| **Settings & Methodology** | Full methodology documentation, parameter reference, cache management |

---

## Quick Start

### 1. Clone / copy the project

```bash
cd em_credit_dashboard
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure environment variables

```bash
cp .env.example .env
# Edit .env to add your FRED API key if desired
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Demo Mode vs Internal Mode

### Demo Mode (default)

- **Macro data**: downloaded from `yfinance` (EMB, HYG, SPY, EEM, TLT, GLD, UUP, VIXY, LQD, PDBC)
- **Bond data**: 25-bond synthetic EM sovereign universe (OAS time-series generated as correlated random walks calibrated to approximate market levels)
- Works offline if yfinance download fails (fully synthetic fallback)

### Internal Mode

Upload your own bond data via the **Data Upload** page:

1. CSV or Excel file (wide format: date × bond_id columns)
2. Map columns to the internal schema
3. Click **Activate Internal Data Mode**

The analytics pipeline rebuilds automatically using your data.

---

## Project Structure

```
em_credit_dashboard/
├── app.py                         Main entry point + sidebar navigation
├── requirements.txt
├── .env.example
│
├── config/
│   ├── settings.py                Global constants, bond universe, colours
│   └── theme.py                   Plotly dark-finance theme, CSS injection
│
├── data/
│   └── sample/
│       ├── generate_sample.py     Script to generate sample CSVs
│       ├── sample_oas_wide.csv    (auto-generated)
│       └── sample_bond_metadata.csv (auto-generated)
│
├── src/
│   ├── data/
│   │   ├── loader.py              yfinance download + synthetic fallback
│   │   ├── preprocessor.py        Returns, spread changes, z-scores
│   │   ├── uploader.py            CSV/Excel upload + column mapping
│   │   └── session.py             Session-state cache manager
│   │
│   ├── analytics/
│   │   ├── correlation.py         Corr matrix, rolling corr, OLS betas, regime
│   │   ├── relative_value.py      Issuer curve fit, residuals, z-scores
│   │   ├── curve_analysis.py      Slope, butterfly, DV01-neutral sizing
│   │   └── trade_ideas.py         Signal aggregation → structured ideas
│   │
│   └── plotting/
│       ├── charts.py              All Plotly chart builders
│       └── tables.py              Styled pandas Styler builders
│
└── pages/
    ├── overview.py                KPIs, movers, trade shortlist
    ├── correlation_beta.py        Full correlation & beta engine
    ├── relative_value.py          RV screener
    ├── curve_trades.py            Curve trade builder
    ├── trade_ideas.py             Trade idea generator
    ├── data_upload.py             Upload & column mapping
    └── settings_page.py           Methodology & settings
```

---

## Adapting to Real Desk Data

### Step 1 – Bond universe

Edit `config/settings.py` → `BOND_UNIVERSE` list.  Replace synthetic `oas_base` / `yield_base` with live values, and add `isin` fields if needed.

### Step 2 – Live OAS feed

In `src/data/loader.py`, replace the `_synthetic_bond_oas()` function with a real data pull:

```python
# TODO: Bloomberg BLPAPI example
import blpapi

def load_bond_oas_history_blp(bond_ids, start, end):
    # blp.history(bond_ids, fields=["OAS_SPREAD_MID"], dt_start=start, dt_end=end)
    ...
```

### Step 3 – Macro series

Add FRED series to supplement ETF proxies (e.g. actual UST yields):

```python
# src/data/loader.py
import fredapi
fred = fredapi.Fred(api_key=os.getenv("FRED_API_KEY"))
ust_10y = fred.get_series("DGS10", observation_start=start)
```

### Step 4 – Internal CSV flow

Use the **Data Upload** page to import Bloomberg CSV exports directly.  The wide-format template matches the internal schema.

### Step 5 – CI / automation

The analytics modules are pure Python functions — they can be scheduled as a daily job (e.g. via the included `anthropic-skills:schedule` integration) to generate a fresh idea report each morning.

---

## Analytics Summary

| Metric | Description |
|--------|-------------|
| OAS (bps) | Option-adjusted spread over UST benchmark |
| RV Z-Score | (Residual − µ) / σ over rolling window; >+1.5 = cheap, <−1.5 = rich |
| Beta (β) | OLS coefficient of bond spread change on macro factor return |
| R² | Fraction of spread variance explained by the macro basket |
| Slope | OAS(long) − OAS(short) in bps; rising = steepening |
| Curvature | 2×OAS(belly) − OAS(short) − OAS(long); positive = belly cheap vs wings |
| DV01 | Dollar value of 1bp change per $1mm notional |
| Confidence | Z-score–derived signal strength score [0–1] |

---

## Running Sample Data Generation

```bash
python data/sample/generate_sample.py
```

This produces:
- `data/sample/sample_oas_wide.csv` — 1-year of synthetic OAS data (25 bonds)
- `data/sample/sample_macro_prices.csv` — 1-year of macro ETF prices
- `data/sample/sample_bond_metadata.csv` — bond metadata table

Upload these files via the **Data Upload** page to test the internal mode.

---

## Disclaimer

This dashboard uses **synthetic bond data** and **public ETF proxies** in demo mode.
All analytics outputs are for **educational / preparation purposes only** and do not constitute investment advice or actual trading signals.
