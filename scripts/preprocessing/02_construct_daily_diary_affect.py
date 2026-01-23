#!/usr/bin/env python3
"""
02_construct_daily_diary_affect.py

Process MIDUS daily diary data to compute participant-level positive and
negative affect summary scores, averages for raw affect items, log-transform
negative affect, and save processed participant-level data and descriptive
statistics for the paper.

This script intentionally excludes demographic variables to avoid redundancy
with other datasets.

Run from the project root directory.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
import pingouin as pg

# =========================
# Paths (relative to project root)
# =========================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

RAW_FILE = os.path.join(RAW_DIR, "M3P2_variables.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "daily_diary_processed.csv")
DESCRIPTIVES_FILE = os.path.join(PROCESSED_DIR, "daily_diary_descriptives.csv")

# =========================
# Function to process daily diary
# =========================
def construct_daily_diary_affect(raw_file, output_file, descriptives_file):
    # Load data
    df = pd.read_csv(raw_file)
    print(f"Loaded daily diary data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Affect item columns
    all_items = [f"C2DC{i}" for i in range(1, 28)]

    # Recode invalid responses
    df[all_items] = df[all_items].replace({7: np.nan, 8: np.nan, 9: np.nan})
    df["C2DIMON"] = df["C2DIMON"].replace(98, np.nan)
    df["C2DIYEAR"] = df["C2DIYEAR"].replace(9998, np.nan)

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

    # =========================
    # Row-level affect scores
    # =========================
    df_complete.loc[:, "PA_score"] = df_complete[pos_items].mean(axis=1)
    df_complete.loc[:, "NA_score"] = df_complete[neg_items].mean(axis=1)

    # =========================
    # Participant-level averages
    # =========================
    summary_scores = (
        df_complete
        .groupby("M2ID")[["PA_score", "NA_score"]]
        .mean()
        .reset_index()
    )

    summary_scores["NA_score_log"] = np.log(summary_scores["NA_score"] + 0.001)

    # Participant-level raw affect item means
    raw_means = (
        df_complete
        .groupby("M2ID")[all_items]
        .mean()
        .reset_index()
    )

    participant_means = summary_scores.merge(raw_means, on="M2ID")

    # =========================
    # Diary start date (Day 1)
    # =========================
    start_date_info = (
        df[df["C2DDAY"] == 1][["M2ID", "C2DIMON", "C2DIYEAR"]]
        .drop_duplicates("M2ID")
        .rename(columns={
            "C2DIMON": "StartMonth",
            "C2DIYEAR": "StartYear"
        })
    )

    # =========================
    # Merge all participant-level data
    # =========================
    final_df = (
        participant_means
        .merge(n_days_any, on="M2ID")
        .merge(n_days_complete, on="M2ID")
        .merge(start_date_info, on="M2ID")
    )

    final_df.to_csv(output_file, index=False)
    print(f"Processed daily diary data saved to: {output_file}")

    # =========================
    # Descriptive statistics
    # =========================
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


# =========================
# Run script
# =========================
if __name__ == "__main__":
    construct_daily_diary_affect(RAW_FILE, OUTPUT_FILE, DESCRIPTIVES_FILE)
