#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_clean_merged_data.py

Clean merged MIDUS dataset:
1) Add 0s for all twin_pair variables for participants from MKE2.
2) Create age at P2 (C2PAGE).
3) Create time between P2 and P5 in months (time_P2_P5).
4) Compute skewness and kurtosis for neuroscience PANAS variables and create log-transformed C5SPGN.
"""

import pandas as pd
import os
import numpy as np
from scipy.stats import skew, kurtosis

PROCESSED_DIR = "data/processed"
TABLE_DIR = "results/tables"

# Ensure output directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
MERGED_FILE = os.path.join(PROCESSED_DIR, "midus_merged.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "midus_merged_clean.csv")
STATS_OUTPUT = os.path.join(TABLE_DIR, "panas_skew_kurtosis.csv")



def main():
    """Main execution function."""
    # =========================
    # Load merged dataset
    # =========================
    df = pd.read_csv(MERGED_FILE)
    print(f"Loaded merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # =========================
    # Identify participants from MKE2
    # =========================
    mke2_markers = ["CACB1", "CACF1", "CACF7A", "CACRAGE", "CACIDATE_MO", "CACIDATE_YR", "CACRSEX"]
    is_mke2 = df[mke2_markers].notna().any(axis=1)
    print(f"Number of participants identified as MKE2: {is_mke2.sum()}")

    # =========================
    # Fill 0s for twin_pair columns
    # =========================
    twin_cols = [col for col in df.columns if col.startswith("twin_pair_")]
    print(f"Twin pair columns detected: {twin_cols}")
    df.loc[is_mke2, twin_cols] = 0

    # =========================
    # Compute age at P2 (C2PAGE)
    # =========================
    if "StartYear" not in df.columns or "StartMonth" not in df.columns:
        raise ValueError("StartYear and StartMonth columns required from daily diary data")

    df["C2PAGE"] = np.nan

    # --- M3 participants ---
    m3_mask = ~is_mke2 & df["C1PBYEAR"].notna()
    df.loc[m3_mask, "C2PAGE"] = df.loc[m3_mask, "StartYear"] - df.loc[m3_mask, "C1PBYEAR"]

    # --- MKE2 participants ---
    mke2_mask = is_mke2 & df["CACRAGE"].notna() & df["CACIDATE_YR"].notna() & df["CACIDATE_MO"].notna()
    df.loc[mke2_mask, "C2PAGE"] = df.loc[mke2_mask, "CACRAGE"] + (
        (df.loc[mke2_mask, "StartYear"] - df.loc[mke2_mask, "CACIDATE_YR"])
        + (df.loc[mke2_mask, "StartMonth"] - df.loc[mke2_mask, "CACIDATE_MO"]) / 12
    )

    # =========================
    # Compute time between P2 and P5 in months
    # =========================
    time_cols_required = ["C5PDATE_YR", "C5PDATE_MO"]
    for col in time_cols_required:
        if col not in df.columns:
            raise ValueError(f"{col} required to compute time_P2_P5")

    df["time_P2_P5"] = (df["C5PDATE_YR"] - df["StartYear"]) * 12 + (df["C5PDATE_MO"] - df["StartMonth"])
    df["time_P2_P5"] = df["time_P2_P5"].abs()  # ensure positive

    # =========================
    # Compute skewness and kurtosis for neuroscience PANAS and log-transform C5SPGN
    # =========================
    panas_vars = ["C5SPGP", "C5SPGN"]
    stats_list = []

    for var in panas_vars:
        series = df[var].dropna()
        var_skew = skew(series)
        var_kurt = kurtosis(series)
        stats_list.append({"variable": var, "skewness": var_skew, "kurtosis": var_kurt})

    # Save stats to CSV
    pd.DataFrame(stats_list).to_csv(STATS_OUTPUT, index=False)
    print(f"Skewness and kurtosis saved to {STATS_OUTPUT}")

    # Create log-transformed negative affect
    df["C5SPGN_log"] = np.log(df["C5SPGN"] + 0.001)

    # =========================
    # Save cleaned dataset
    # =========================
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned merged dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
