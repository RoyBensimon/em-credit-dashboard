"""
Generate a sample bond OAS CSV file for testing the Data Upload page.

Run from the project root:
    python data/sample/generate_sample.py

Produces: data/sample/sample_oas_wide.csv
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from src.data.loader import load_bond_oas_history, load_macro_prices
from config.settings import BOND_UNIVERSE


def main():
    print("Generating sample OAS data…")
    oas = load_bond_oas_history(lookback_days=252)
    out_path = os.path.join(os.path.dirname(__file__), "sample_oas_wide.csv")
    oas.index.name = "date"
    oas.to_csv(out_path)
    print(f"  → Saved {oas.shape} OAS DataFrame to {out_path}")

    print("Generating sample macro prices…")
    macro = load_macro_prices(lookback_days=252)
    macro_path = os.path.join(os.path.dirname(__file__), "sample_macro_prices.csv")
    macro.index.name = "date"
    macro.to_csv(macro_path)
    print(f"  → Saved {macro.shape} macro DataFrame to {macro_path}")

    print("Generating bond metadata CSV…")
    meta_df = pd.DataFrame(BOND_UNIVERSE)
    meta_path = os.path.join(os.path.dirname(__file__), "sample_bond_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"  → Saved {len(meta_df)} bonds to {meta_path}")

    print("\nDone.  These files can be uploaded via the 'Data Upload' page.")


if __name__ == "__main__":
    main()
