#!/usr/bin/env python3
"""
02_construct_daily_diary_affect.py

Process MIDUS daily diary data to compute participant-level positive and
negative affect summary scores, averages for raw affect items, log-transform
negative affect, and save processed participant-level data and descriptive
statistics.

Note: This script intentionally excludes demographic variables to avoid
redundancy with other datasets.

Inputs:
- data/raw/M3P2_variables.csv (Project 2: daily diary affect, long format)

Outputs:
- data/processed/daily_diary_processed.csv (participant-level affect summaries)
- data/processed/daily_diary_descriptives.csv (reliability and distribution stats)

Run from project root directory.
"""

import os

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import kurtosis, skew

# ============================================================================
# Paths and Constants
# ============================================================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

RAW_FILE = os.path.join(RAW_DIR, "M3P2_variables.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "daily_diary_processed.csv")
DESCRIPTIVES_FILE = os.path.join(PROCESSED_DIR, "daily_diary_descriptives.csv")

# MIDUS missing value codes to recode as NaN
MISSING_CODES_AFFECT = [7, 8, 9]  # Don't know, Refused, Not applicable
MISSING_CODE_MONTH = 98
MISSING_CODE_YEAR = 9998

# ============================================================================
# Helper Functions
# ============================================================================
def construct_daily_diary_affect(raw_file, output_file, descriptives_file):
    """
    Process daily diary affect data and compute summary statistics.

    Args:
        raw_file: Path to raw M3P2 daily diary data
        output_file: Path to save participant-level processed data
        descriptives_file: Path to save descriptive statistics

    Returns:
        None (saves files to disk)
    """
    # ========================================================================
    # Load and Clean Data
    # ========================================================================
    df = pd.read_csv(raw_file)
    print(f"Loaded daily diary data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Affect item columns (C2DC1 through C2DC27)
    all_items = [f"C2DC{i}" for i in range(1, 28)]

    # Recode MIDUS missing value codes to NaN
    missing_dict = {code: np.nan for code in MISSING_CODES_AFFECT}
    df[all_items] = df[all_items].replace(missing_dict)
    df["C2DIMON"] = df["C2DIMON"].replace(MISSING_CODE_MONTH, np.nan)
    df["C2DIYEAR"] = df["C2DIYEAR"].replace(MISSING_CODE_YEAR, np.nan)

    # Define positive and negative affect items
    pos_items = [f"C2DC{i}" for i in [7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27]]
    neg_items = [f"C2DC{i}" for i in [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20]]

    # Count number of diary days per participant (any data)
    n_days_any = df.groupby("M2ID").size().reset_index(name="n_days_any")

    # Keep only days with complete affect data
    df_complete = df.dropna(subset=all_items).copy()
    print(f"Rows after removing days with missing affect data: {df_complete.shape[0]}")

    n_days_complete = (
        df_complete.groupby("M2ID")
        .size()
        .reset_index(name="n_days_complete")
    )

    # ========================================================================
    # Compute Row-Level Affect Scores
    # ========================================================================
    df_complete.loc[:, "PA_score"] = df_complete[pos_items].mean(axis=1)
    df_complete.loc[:, "NA_score"] = df_complete[neg_items].mean(axis=1)

    # ========================================================================
    # Compute Participant-Level Averages
    # ========================================================================
    summary_scores = (
        df_complete
        .groupby("M2ID")[["PA_score", "NA_score"]]
        .mean()
        .reset_index()
    )

    # Log-transform negative affect
    # Use half the minimum non-zero value as offset (preserves distribution better)
    min_nonzero = summary_scores.loc[summary_scores["NA_score"] > 0, "NA_score"].min()
    log_offset = min_nonzero / 2 if not pd.isna(min_nonzero) else 0.01
    summary_scores["NA_score_log"] = np.log(summary_scores["NA_score"] + log_offset)
    print(f"Log transform offset for NA_score: {log_offset:.4f} (half of min non-zero value)")

    # Participant-level raw affect item means
    raw_means = (
        df_complete
        .groupby("M2ID")[all_items]
        .mean()
        .reset_index()
    )

    participant_means = summary_scores.merge(raw_means, on="M2ID")

    # ========================================================================
    # Extract Diary Start Date
    # ========================================================================
    start_date_info = (
        df[df["C2DDAY"] == 1][["M2ID", "C2DIMON", "C2DIYEAR"]]
        .drop_duplicates("M2ID")
        .rename(columns={
            "C2DIMON": "StartMonth",
            "C2DIYEAR": "StartYear"
        })
    )

    # ========================================================================
    # Merge All Participant-Level Data
    # ========================================================================
    final_df = (
        participant_means
        .merge(n_days_any, on="M2ID")
        .merge(n_days_complete, on="M2ID")
        .merge(start_date_info, on="M2ID", how="left")  # left join: keep all diary participants
    )

    n_missing_start = final_df["StartYear"].isna().sum()
    if n_missing_start > 0:
        print(f"WARNING: {n_missing_start} participants have no day-1 date record "
              f"(StartYear = NaN). C2PAGE and time_P2_P5 will be missing for these participants.")
        print(f"  Missing StartYear M2IDs: {final_df.loc[final_df['StartYear'].isna(), 'M2ID'].tolist()}")
    else:
        print(f"All {len(final_df)} participants have a valid StartYear.")

    final_df.to_csv(output_file, index=False)
    print(f"Processed daily diary data saved to: {output_file}")

    # ========================================================================
    # Compute Descriptive Statistics
    # ========================================================================
    alpha_pos = pg.cronbach_alpha(data=df_complete[pos_items])[0]
    alpha_neg = pg.cronbach_alpha(data=df_complete[neg_items])[0]

    summary_vars = ["PA_score", "NA_score", "NA_score_log"]

    skews = {v: skew(final_df[v], nan_policy="omit") for v in summary_vars}
    kurts = {
        v: kurtosis(final_df[v], nan_policy="omit", fisher=True)
        for v in summary_vars
    }

    days_mean = final_df[["n_days_any", "n_days_complete"]].mean()
    days_std = final_df[["n_days_any", "n_days_complete"]].std()

    raw_mean = final_df[all_items].mean()
    raw_std = final_df[all_items].std()

    gen_mean = final_df[summary_vars].mean()
    gen_std = final_df[summary_vars].std()

    desc_dict = {
        "Participants_before": df["M2ID"].nunique(),
        "Participants_after": final_df["M2ID"].nunique(),
        "Participants_removed": df["M2ID"].nunique() - final_df["M2ID"].nunique(),
        "Cronbach_alpha_PA": alpha_pos,
        "Cronbach_alpha_NA": alpha_neg,
    }

    desc_df = pd.DataFrame([desc_dict])

    for v in summary_vars:
        desc_df[f"Skew_{v}"] = skews[v]
        desc_df[f"Kurtosis_{v}"] = kurts[v]

    for col in days_mean.index:
        desc_df[f"Mean_{col}"] = days_mean[col]
        desc_df[f"Std_{col}"] = days_std[col]

    for col in raw_mean.index:
        desc_df[f"Mean_{col}"] = raw_mean[col]
        desc_df[f"Std_{col}"] = raw_std[col]

    for col in gen_mean.index:
        desc_df[f"Mean_{col}"] = gen_mean[col]
        desc_df[f"Std_{col}"] = gen_std[col]

    desc_df.to_csv(descriptives_file, index=False)
    print(f"Descriptive statistics saved to: {descriptives_file}")


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    # Ensure output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Process daily diary data
    construct_daily_diary_affect(RAW_FILE, OUTPUT_FILE, DESCRIPTIVES_FILE)


if __name__ == "__main__":
    main()
