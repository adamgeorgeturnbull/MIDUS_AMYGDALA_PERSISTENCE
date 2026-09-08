#!/usr/bin/env python3
"""
02_construct_daily_diary_affect.py (MR1)

Process MR1 daily diary data (RA2 wave) to compute participant-level PA/NA scores.

MR1 variable differences vs M3:
- ID key: MIDUSID (not M2ID)
- Affect items: RA2DC1-RA2DC27 (not C2DC1-C2DC27)
  Note: P2 file lists RA2DC7 before RA2DC6; sort numerically.
- Diary date: RA2DDAY, RA2DIMON, RA2DIYEAR

Inputs:
- data/raw/MR1_P2_variables.csv

Outputs:
- data/processed/daily_diary_processed.csv
- data/processed/daily_diary_descriptives.csv

Run from MR1_validation/ directory.
"""

import os
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import kurtosis, skew

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

RAW_FILE = os.path.join(RAW_DIR, "MR1_P2_variables.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "daily_diary_processed.csv")
DESCRIPTIVES_FILE = os.path.join(PROCESSED_DIR, "daily_diary_descriptives.csv")

MISSING_CODES_AFFECT = [7, 8, 9]
MISSING_CODE_MONTH = 98
MISSING_CODE_YEAR = 9998


def _read_raw(path):
    """Read CSV or TSV using the separator that matches the file extension."""
    actual = path
    for candidate in [path, path.replace(".csv", ".tsv"), path.replace(".tsv", ".csv")]:
        if os.path.isfile(candidate):
            actual = candidate
            break
    sep = "\t" if actual.endswith(".tsv") else ","
    return pd.read_csv(actual, encoding="utf-8-sig", sep=sep)


def construct_daily_diary_affect(raw_file, output_file, descriptives_file):
    df = _read_raw(raw_file)
    print(f"Loaded daily diary data: {df.shape[0]} rows, {df.shape[1]} columns")

    # MIDUSID is only recorded on day-1 rows; MRID is identical and always present
    df["MIDUSID"] = df["MIDUSID"].fillna(df["MRID"]).astype(int)
    print(f"MIDUSID after fill from MRID: {df['MIDUSID'].isna().sum()} missing")

    # All 27 affect items (sorted numerically — P2 file has RA2DC7 before RA2DC6)
    all_items = [f"RA2DC{i}" for i in range(1, 28)]

    missing_dict = {code: np.nan for code in MISSING_CODES_AFFECT}
    df[all_items] = df[all_items].replace(missing_dict)
    df["RA2DIMON"] = df["RA2DIMON"].replace(MISSING_CODE_MONTH, np.nan)
    df["RA2DIYEAR"] = df["RA2DIYEAR"].replace(MISSING_CODE_YEAR, np.nan)

    # Positive and negative affect item indices match M3 (same instrument)
    pos_items = [f"RA2DC{i}" for i in [7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27]]
    neg_items = [f"RA2DC{i}" for i in [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20]]

    n_days_any = df.groupby("MIDUSID").size().reset_index(name="n_days_any")

    df_complete = df.dropna(subset=all_items).copy()
    print(f"Rows after removing days with missing affect data: {df_complete.shape[0]}")

    n_days_complete = (
        df_complete.groupby("MIDUSID")
        .size()
        .reset_index(name="n_days_complete")
    )

    df_complete.loc[:, "PA_score"] = df_complete[pos_items].mean(axis=1)
    df_complete.loc[:, "NA_score"] = df_complete[neg_items].mean(axis=1)

    summary_scores = (
        df_complete
        .groupby("MIDUSID")[["PA_score", "NA_score"]]
        .mean()
        .reset_index()
    )

    min_nonzero = summary_scores.loc[summary_scores["NA_score"] > 0, "NA_score"].min()
    log_offset = min_nonzero / 2 if not pd.isna(min_nonzero) else 0.01
    summary_scores["NA_score_log"] = np.log(summary_scores["NA_score"] + log_offset)
    print(f"Log transform offset for NA_score: {log_offset:.4f} (half of min non-zero value)")

    raw_means = (
        df_complete
        .groupby("MIDUSID")[all_items]
        .mean()
        .reset_index()
    )

    participant_means = summary_scores.merge(raw_means, on="MIDUSID")

    start_date_info = (
        df[df["RA2DDAY"] == 1][["MIDUSID", "RA2DIMON", "RA2DIYEAR"]]
        .drop_duplicates("MIDUSID")
        .rename(columns={"RA2DIMON": "StartMonth", "RA2DIYEAR": "StartYear"})
    )

    final_df = (
        participant_means
        .merge(n_days_any, on="MIDUSID")
        .merge(n_days_complete, on="MIDUSID")
        .merge(start_date_info, on="MIDUSID", how="left")
    )

    n_missing_start = final_df["StartYear"].isna().sum()
    if n_missing_start > 0:
        print(f"WARNING: {n_missing_start} participants have no day-1 date record.")
    else:
        print(f"All {len(final_df)} participants have a valid StartYear.")

    final_df.to_csv(output_file, index=False)
    print(f"Processed daily diary data saved to: {output_file}")

    alpha_pos = pg.cronbach_alpha(data=df_complete[pos_items])[0]
    alpha_neg = pg.cronbach_alpha(data=df_complete[neg_items])[0]

    summary_vars = ["PA_score", "NA_score", "NA_score_log"]
    skews = {v: skew(final_df[v], nan_policy="omit") for v in summary_vars}
    kurts = {v: kurtosis(final_df[v], nan_policy="omit", fisher=True) for v in summary_vars}

    days_mean = final_df[["n_days_any", "n_days_complete"]].mean()
    days_std = final_df[["n_days_any", "n_days_complete"]].std()
    raw_mean = final_df[all_items].mean()
    raw_std = final_df[all_items].std()
    gen_mean = final_df[summary_vars].mean()
    gen_std = final_df[summary_vars].std()

    desc_dict = {
        "Participants_before": df["MIDUSID"].nunique(),
        "Participants_after": final_df["MIDUSID"].nunique(),
        "Participants_removed": df["MIDUSID"].nunique() - final_df["MIDUSID"].nunique(),
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


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    construct_daily_diary_affect(RAW_FILE, OUTPUT_FILE, DESCRIPTIVES_FILE)


if __name__ == "__main__":
    main()
