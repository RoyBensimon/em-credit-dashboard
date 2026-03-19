"""
CSV / Excel upload handler for the EM Credit Analytics Dashboard.

Provides functions to:
  - parse uploaded files into DataFrames
  - validate column presence and data quality
  - map user-specified columns to the internal schema
  - convert price / yield / spread columns to the correct types

This module is the integration point for internal desk data.
TODO: When real desk data is available, adapt the column mappings below
      to match your internal naming conventions.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Internal column schema ────────────────────────────────────────────────────
# These are the canonical column names used throughout the analytics layer.
# When uploading external data, users map their columns to these names.
REQUIRED_PRICE_COLS    = ["date"]          # minimum required
OPTIONAL_PRICE_COLS    = ["bond_id", "price", "yield_pct", "oas_bps"]

REQUIRED_METADATA_COLS = ["bond_id"]       # minimum required for metadata
OPTIONAL_METADATA_COLS = [
    "country", "issuer", "currency", "coupon",
    "maturity", "rating", "duration", "dv01",
    "sector", "isin",
]


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def parse_uploaded_file(
    file_bytes: bytes,
    filename: str,
    date_col: str = "date",
    date_format: str | None = None,
) -> pd.DataFrame | None:
    """
    Parse an uploaded CSV or Excel file into a DataFrame.

    Returns None and logs an error if parsing fails.
    """
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        elif filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            logger.error("Unsupported file type: %s", filename)
            return None

        # Strip whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]

        # Parse date column if present
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
            df = df.dropna(subset=[date_col])
            df = df.set_index(date_col).sort_index()

        return df

    except Exception as exc:
        logger.error("Failed to parse %s: %s", filename, exc)
        return None


def validate_price_data(df: pd.DataFrame) -> dict:
    """
    Run basic quality checks on an uploaded price/OAS DataFrame.

    Returns a dict with keys:
      - valid      : bool
      - n_rows     : int
      - n_cols     : int
      - issues     : list[str]   (warnings / errors)
      - summary    : dict        (per-column stats)
    """
    issues: list[str] = []

    if df.empty:
        return {"valid": False, "n_rows": 0, "n_cols": 0, "issues": ["DataFrame is empty"], "summary": {}}

    # Check for all-NaN columns
    all_nan = df.columns[df.isna().all()].tolist()
    if all_nan:
        issues.append(f"Columns with all NaN values: {all_nan}")

    # Check for non-numeric columns (besides index)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        issues.append(f"Non-numeric columns detected: {non_numeric}")

    # Check for large gaps (>5 consecutive NaN in any column)
    for col in df.select_dtypes(include="number").columns:
        max_gap = df[col].isna().astype(int).groupby(
            df[col].notna().astype(int).cumsum()
        ).sum().max()
        if max_gap and max_gap > 5:
            issues.append(f"Column '{col}' has a gap of {int(max_gap)} consecutive NaN values.")

    # Date index continuity
    if isinstance(df.index, pd.DatetimeIndex):
        gaps = df.index.to_series().diff().dt.days.dropna()
        big_gaps = gaps[gaps > 10]
        if not big_gaps.empty:
            issues.append(
                f"Date index has {len(big_gaps)} gaps larger than 10 calendar days."
            )

    summary: dict = {}
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        summary[col] = {
            "count":  len(s),
            "mean":   round(s.mean(), 4) if len(s) else None,
            "std":    round(s.std(), 4)  if len(s) else None,
            "min":    round(s.min(), 4)  if len(s) else None,
            "max":    round(s.max(), 4)  if len(s) else None,
            "pct_nan": round(df[col].isna().mean() * 100, 1),
        }

    return {
        "valid":   len([i for i in issues if "error" in i.lower()]) == 0,
        "n_rows":  len(df),
        "n_cols":  df.shape[1],
        "issues":  issues,
        "summary": summary,
    }


def wide_to_long_oas(
    df: pd.DataFrame,
    bond_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert a wide-format DataFrame (columns = bond IDs, index = dates)
    to a long-format DataFrame with columns [date, bond_id, oas_bps].

    Useful for internal data that arrives in a wide spread-sheet layout.
    """
    if bond_ids is not None:
        df = df[[c for c in bond_ids if c in df.columns]]
    df_long = df.reset_index().melt(
        id_vars=df.index.name or "date",
        var_name="bond_id",
        value_name="oas_bps",
    )
    df_long.columns = ["date", "bond_id", "oas_bps"]
    return df_long.dropna(subset=["oas_bps"])


def apply_column_mapping(
    df: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    """
    Rename DataFrame columns according to a user-specified mapping.

    Parameters
    ----------
    df      : input DataFrame
    mapping : {user_col_name: canonical_col_name}

    Returns a copy with columns renamed and unmapped columns dropped.
    """
    df = df.copy()
    keep = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=keep)
    # Keep only mapped columns
    return df[list(keep.values())]


def pivot_bond_series(
    df: pd.DataFrame,
    bond_id_col: str = "bond_id",
    value_col: str = "oas_bps",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Pivot a long-format bond DataFrame into wide format.

    Returns a DataFrame with index=date, columns=bond_id.
    """
    if date_col in df.columns:
        df = df.set_index(date_col)
    pivoted = df.pivot(columns=bond_id_col, values=value_col)
    return pivoted.sort_index()
