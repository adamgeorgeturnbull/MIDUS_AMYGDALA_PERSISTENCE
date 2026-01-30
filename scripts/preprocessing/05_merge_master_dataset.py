#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_merge_master_dataset.py

Merge MIDUS participant-level diary data with P5 and MKE2 covariates.
Ensures:
- Merge on M2ID
- Keep all variables
- For overlapping columns, combine into a single column
"""

import pandas as pd
import os

PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "midus_merged.csv")

# =========================
# Load processed datasets
# =========================
daily_diary = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_diary_processed.csv"))
m3p5_cov = pd.read_csv(os.path.join(PROCESSED_DIR, "m3p5_covariates.csv"))
mke2_cov = pd.read_csv(os.path.join(PROCESSED_DIR, "mke2_covariates.csv"))

# =========================
# Function to merge two dataframes with overlapping columns
# =========================
def merge_combine(df1, df2, key="M2ID"):
    """
    Merge df1 and df2 on key.
    For overlapping columns (other than key), combine into a single column
    preferring non-null values from df1, then df2.
    """
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    
    # Identify overlapping columns (exclude key)
    overlap_cols = list(df1_cols & df2_cols - {key})
    
    df_merged = df1.merge(df2, on=key, how="outer", suffixes=("_1", "_2"))
    
    for col in overlap_cols:
        col_1 = f"{col}_1"
        col_2 = f"{col}_2"
        df_merged[col] = df_merged[col_1].fillna(df_merged[col_2])
        df_merged.drop([col_1, col_2], axis=1, inplace=True)
    
    return df_merged



def main():
    """Main execution function."""
    # =========================
    # Merge all datasets
    # =========================
    # First merge diary with M3P5
    merged = merge_combine(daily_diary, m3p5_cov, key="M2ID")

    # Then merge in MKE2 covariates
    merged = merge_combine(merged, mke2_cov, key="M2ID")

    # =========================
    # Save final merged dataset
    # =========================
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Merged dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
