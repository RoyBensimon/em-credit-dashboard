"""
Data Upload & Mapping Page

Allows users to upload internal desk data (CSV / Excel) and wire it
into the analytics pipeline.

Workflow:
  1. Upload a file (prices / yields / OAS / spreads).
  2. Map columns interactively using dropdowns.
  3. Specify what each series represents.
  4. Preview cleaned dataset and run validation.
  5. Optionally upload bond metadata.
  6. Activate internal mode (replaces synthetic data).

TODO: When connecting to a live internal feed (Bloomberg / Refinitiv),
      route through this module's data-cleaning pipeline for consistency.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import COLORS
from config.theme import STREAMLIT_CSS
from src.data.session import clear_cache
from src.data.uploader import (
    parse_uploaded_file,
    validate_price_data,
    pivot_bond_series,
    apply_column_mapping,
    OPTIONAL_METADATA_COLS,
)


def render() -> None:
    """Render the Data Upload & Mapping page."""
    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
    st.title("Data Upload & Mapping")
    st.caption(
        "Upload your internal bond data (prices, yields, OAS, or spreads) "
        "and map it to the analytics pipeline. Supports CSV and Excel files."
    )

    tab1, tab2, tab3 = st.tabs([
        "Time-Series Upload", "Metadata Upload", "Current Data Status"
    ])

    # ── Tab 1: Time-series upload ─────────────────────────────────────────────
    with tab1:
        st.subheader("Upload Bond Price / Yield / OAS Data")

        st.markdown(
            """
**Expected file format:**
- **Wide format** (recommended): one column per bond, date index in first column.
- **Long format**: columns `date`, `bond_id`, `value`.
- Dates should be in `YYYY-MM-DD`, `DD/MM/YYYY`, or `MM/DD/YYYY` format.
"""
        )

        uploaded = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="ts_upload",
        )

        if uploaded is None:
            st.info("No file uploaded. The dashboard is running in **Public Demo Mode** with synthetic bond data.")
            _show_sample_format()
            return

        # Parse
        raw_bytes = uploaded.read()
        df = parse_uploaded_file(raw_bytes, uploaded.name)

        if df is None:
            st.error(f"Could not parse '{uploaded.name}'. Check the file format.")
            return

        st.success(f"File parsed: **{uploaded.name}** — {len(df)} rows × {df.shape[1]} columns")

        # ── Column mapping ────────────────────────────────────────────────────
        st.subheader("Step 1: Identify Data Type")

        data_type = st.radio(
            "What does this file contain?",
            ["OAS / Spread (bps)", "Yield (%)", "Price (dollars)"],
            horizontal=True,
        )

        st.subheader("Step 2: Map Columns")

        if df.index.name and pd.api.types.is_datetime64_any_dtype(df.index):
            st.success(f"Date index detected: **{df.index.name}** — {df.index.dtype}")
        else:
            # Let user choose which column is the date
            date_col_candidates = ["date", "Date", "DATE", "Dates", "as_of_date"] + df.columns.tolist()
            date_col = st.selectbox(
                "Which column is the date?",
                [c for c in date_col_candidates if c in df.columns] or df.columns.tolist(),
                index=0,
            )
            try:
                df.index = pd.to_datetime(df[date_col], errors="coerce")
                df = df.drop(columns=[date_col])
                df = df.sort_index()
                st.success(f"Date column set: **{date_col}** ({df.index.dtype})")
            except Exception as e:
                st.warning(f"Could not parse date column: {e}")

        # Column preview
        st.subheader("Step 3: Preview Cleaned Data")
        st.dataframe(df.head(20), use_container_width=True)

        # Validation
        st.subheader("Step 4: Data Quality Check")
        validation = validate_price_data(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows",    validation["n_rows"])
        c2.metric("Columns", validation["n_cols"])
        c3.metric("Valid",   "✅ Yes" if validation["valid"] else "❌ Issues found")

        if validation["issues"]:
            st.warning("Data quality issues detected:")
            for issue in validation["issues"]:
                st.markdown(f"- {issue}")
        else:
            st.success("No data quality issues detected.")

        # Per-column stats
        with st.expander("Column Statistics"):
            if validation["summary"]:
                stats_df = pd.DataFrame(validation["summary"]).T
                st.dataframe(stats_df, use_container_width=True)

        # ── Activate internal mode ─────────────────────────────────────────────
        st.subheader("Step 5: Activate Internal Mode")
        st.markdown(
            "Store this dataset in the session and use it as the OAS input "
            "for all analytics pages."
        )

        series_type = data_type.split(" ")[0].lower()  # 'oas', 'yield', 'price'

        if st.button("Activate Internal Data Mode", type="primary"):
            # Store uploaded data in session state
            st.session_state["uploaded_oas"] = df
            st.session_state["uploaded_series_type"] = series_type
            # Clear the analytics cache so it rebuilds with new data
            clear_cache()
            st.success(
                "Internal data activated. Navigate to any analytics page to see results. "
                "The dashboard will rebuild the analytics using your uploaded data."
            )
            st.info(
                "**Note:** The current implementation uses the uploaded series as a "
                "direct OAS replacement if the column names match the bond IDs in the "
                "universe. For full custom column mapping, extend `src/data/uploader.py`."
            )

    # ── Tab 2: Metadata upload ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Upload Bond Metadata")
        st.markdown(
            """
Provide a CSV or Excel file with bond characteristics.
Required column: `bond_id`.
Optional columns: `country`, `issuer`, `currency`, `coupon`, `maturity`,
`rating`, `duration`, `dv01`, `sector`, `isin`.
"""
        )

        meta_upload = st.file_uploader(
            "Upload bond metadata (CSV or Excel)",
            type=["csv", "xlsx"],
            key="meta_upload",
        )

        if meta_upload is not None:
            meta_bytes = meta_upload.read()
            meta_df    = parse_uploaded_file(meta_bytes, meta_upload.name)
            if meta_df is not None:
                st.success(f"Metadata parsed: {len(meta_df)} bonds.")
                st.dataframe(meta_df, use_container_width=True)

                missing_cols = [c for c in OPTIONAL_METADATA_COLS if c not in meta_df.columns]
                if missing_cols:
                    st.warning(f"Optional columns not found: {missing_cols}. "
                               "Analytics requiring these fields will use defaults.")

                if st.button("Save Metadata to Session"):
                    st.session_state["uploaded_meta"] = meta_df
                    clear_cache()
                    st.success("Bond metadata saved to session.")
        else:
            st.info(
                "No metadata file uploaded. "
                "Using the built-in synthetic bond universe for analytics."
            )

        # Show built-in universe for reference
        with st.expander("Current Bond Universe (built-in)"):
            from config.settings import BOND_UNIVERSE
            universe_df = pd.DataFrame(BOND_UNIVERSE)
            st.dataframe(
                universe_df[[
                    "id", "country", "maturity", "rating", "duration",
                    "dv01", "coupon", "oas_base", "yield_base",
                ]],
                use_container_width=True,
            )

    # ── Tab 3: Current data status ─────────────────────────────────────────────
    with tab3:
        st.subheader("Current Data Status")

        mode = "Internal Upload" if "uploaded_oas" in st.session_state else "Public Demo (Synthetic)"
        st.markdown(
            f"**Active Mode:** "
            f"<span style='color:{COLORS['positive'] if mode == 'Internal Upload' else COLORS['warning']}'>"
            f"{mode}</span>",
            unsafe_allow_html=True,
        )

        if "uploaded_oas" in st.session_state:
            udf = st.session_state["uploaded_oas"]
            st.metric("Uploaded Series",   udf.shape[1])
            st.metric("Date Range Start",  str(udf.index.min())[:10])
            st.metric("Date Range End",    str(udf.index.max())[:10])
            st.metric("Observations",      len(udf))

            if st.button("Reset to Demo Mode", type="secondary"):
                del st.session_state["uploaded_oas"]
                if "uploaded_meta" in st.session_state:
                    del st.session_state["uploaded_meta"]
                clear_cache()
                st.success("Reset to public demo mode.")

        else:
            st.info(
                "Running in **Public Demo Mode**. "
                "Bond data is synthetic (correlated random walk). "
                "Macro ETF data is downloaded from yfinance."
            )
            if "em_dashboard_data" in st.session_state:
                d = st.session_state["em_dashboard_data"]
                mp = d.get("macro_prices", pd.DataFrame())
                oas = d.get("oas_df", pd.DataFrame())
                st.metric("Macro ETFs loaded",   mp.shape[1] if not mp.empty else 0)
                st.metric("Synthetic bonds",     oas.shape[1] if not oas.empty else 0)
                st.metric("Date range",
                          f"{str(oas.index.min())[:10]} → {str(oas.index.max())[:10]}"
                          if not oas.empty else "N/A")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _show_sample_format() -> None:
    """Show a sample expected file format."""
    with st.expander("Expected File Formats"):
        st.markdown("**Option A — Wide format (recommended)**")
        sample_wide = pd.DataFrame({
            "date":      ["2024-01-02", "2024-01-03", "2024-01-04"],
            "BRL_5Y":    [265.0, 268.5, 262.0],
            "MEX_10Y":   [185.0, 183.2, 187.4],
            "TUR_5Y":    [380.0, 375.0, 385.0],
        })
        st.dataframe(sample_wide, use_container_width=True, hide_index=True)

        st.markdown("**Option B — Long format**")
        sample_long = pd.DataFrame({
            "date":     ["2024-01-02", "2024-01-02", "2024-01-02"],
            "bond_id":  ["BRL_5Y", "MEX_10Y", "TUR_5Y"],
            "oas_bps":  [265.0, 185.0, 380.0],
        })
        st.dataframe(sample_long, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Wide-Format Template",
            _generate_wide_template(),
            "em_credit_template.csv",
            "text/csv",
        )


def _generate_wide_template() -> bytes:
    """Generate a downloadable template CSV."""
    from config.settings import BOND_UNIVERSE
    from src.data.loader import load_bond_oas_history

    try:
        oas = load_bond_oas_history(lookback_days=30)
        oas_sample = oas.tail(5)
        oas_sample.index.name = "date"
        return oas_sample.reset_index().to_csv(index=False).encode("utf-8")
    except Exception:
        cols = [b["id"] for b in BOND_UNIVERSE[:5]]
        idx  = pd.bdate_range(end=pd.Timestamp.today(), periods=5)
        df   = pd.DataFrame(np.random.uniform(100, 500, (5, 5)), columns=cols, index=idx)
        df.index.name = "date"
        return df.reset_index().to_csv(index=False).encode("utf-8")
