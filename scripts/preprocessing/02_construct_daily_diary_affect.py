#!/usr/bin/env python3
"""
02_construct_daily_diary_affect.py

Process MIDUS daily diary data to compute participant-level summary scores, 
averages for raw items, log-transform scores with high kurtosis, and save 
both processed participant-level data and descriptive statistics for the paper.

Run from the master folder.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
import pingouin as pg

# =========================
# Paths (relative to master folder)
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

    # Columns with affect items
    all_items = [f'C2DC{i}' for i in range(1, 28)]
    pt_items = [f'C2DY{i}' for i in range(1, 7)]  # Persistent thought variables

    # Recode invalid responses to NaN
    df[all_items + pt_items] = df[all_items + pt_items].replace({7: np.nan, 8: np.nan, 9: np.nan})
    df['C2DIMON'] = df['C2DIMON'].replace(98, np.nan)
    df['C2DIYEAR'] = df['C2DIYEAR'].replace(9998, np.nan)

    # Define positive and negative affect items
    pos_items = [f'C2DC{i}' for i in [7,8,9,10,11,12,21,22,23,24,25,26,27]]
    neg_items = [f'C2DC{i}' for i in [1,2,3,4,5,6,13,14,15,16,17,18,19,20]]

    # Count participant-level row counts (all rows)
    n_days_any = df.groupby('M2ID').size().reset_index(name='n_days_any')

    # Drop rows with ANY missing affect data for analysis
    df_complete = df.dropna(subset=all_items).copy()  # <--- MAKE COPY HERE
    print(f"Rows after removing any with missing affect data: {df_complete.shape[0]}")

    n_days_complete = df_complete.groupby('M2ID').size().reset_index(name='n_days_complete')

    # =========================
    # Compute row-level scores using .loc
    # =========================
    df_complete.loc[:, 'PA_score'] = df_complete[pos_items].mean(axis=1)
    df_complete.loc[:, 'NA_score'] = df_complete[neg_items].mean(axis=1)
    df_complete.loc[:, 'PT_score'] = df_complete[pt_items].mean(axis=1)

    # =========================
    # Participant-level averages
    # =========================
    # Summary scores
    summary_scores = df_complete.groupby('M2ID')[['PA_score','NA_score','PT_score']].mean().reset_index()
    summary_scores['NA_score_log'] = np.log(summary_scores['NA_score'] + 0.001)
    summary_scores['PT_score_log'] = np.log(summary_scores['PT_score'] + 0.001)

    # Participant-level means for all raw items
    raw_items = all_items + pt_items
    raw_means = df_complete.groupby('M2ID')[raw_items].mean().reset_index()

    # Merge summary scores with raw item means
    participant_means = summary_scores.merge(raw_means, on='M2ID')

    # =========================
    # Get static participant info
    # =========================
    static_cols = ['M2ID', 'SAMPLMAJ', 'C1PRAGE', 'C1PRSEX']
    static_info = df.drop_duplicates('M2ID')[static_cols]

    # Study start month/year per participant (from Day 1)
    start_date_info = (
        df[df['C2DDAY'] == 1][['M2ID', 'C2DIMON', 'C2DIYEAR']]
        .drop_duplicates('M2ID')
        .rename(columns={'C2DIMON': 'StartMonth', 'C2DIYEAR': 'StartYear'})
    )

    # =========================
    # Merge participant-level means with counts, static info, and start date
    # =========================
    final_df = participant_means.merge(n_days_any, on='M2ID')\
                                .merge(n_days_complete, on='M2ID')\
                                .merge(static_info, on='M2ID')\
                                .merge(start_date_info, on='M2ID')

    # Save processed participant-level dataset
    final_df.to_csv(output_file, index=False)
    print(f"Processed daily diary data saved to: {output_file}")

    # =========================
    # Descriptive statistics for whole sample
    # =========================
    # Cronbach's alpha
    alpha_pos = pg.cronbach_alpha(data=df_complete[pos_items])[0]
    alpha_neg = pg.cronbach_alpha(data=df_complete[neg_items])[0]

    # Skewness and kurtosis for summary scores
    summary_vars = ['PA_score','NA_score','NA_score_log','PT_score','PT_score_log']
    skews = {var: skew(final_df[var], nan_policy='omit') for var in summary_vars}
    kurt_vals = {var: kurtosis(final_df[var], nan_policy='omit', fisher=True) for var in summary_vars}

    # Mean and std for number of days
    days_mean = final_df[['n_days_any','n_days_complete']].mean()
    days_std = final_df[['n_days_any','n_days_complete']].std()

    # Mean and std for raw items
    raw_mean = final_df[raw_items].mean()
    raw_std = final_df[raw_items].std()

    # Mean and std for generated averages
    gen_avg_cols = ['PA_score','NA_score','NA_score_log','PT_score','PT_score_log']
    gen_mean = final_df[gen_avg_cols].mean()
    gen_std = final_df[gen_avg_cols].std()

    # Combine everything into a single descriptive dataframe (one row)
    desc_dict = {
        'Participants_before': df['M2ID'].nunique(),
        'Participants_after': final_df['M2ID'].nunique(),
        'Participants_removed': df['M2ID'].nunique() - final_df['M2ID'].nunique(),
        'Cronbach_alpha_PA': alpha_pos,
        'Cronbach_alpha_NA': alpha_neg
    }

    desc_df = pd.DataFrame([desc_dict])

    # Add skew/kurtosis
    for var in summary_vars:
        desc_df[f"Skew_{var}"] = skews[var]
        desc_df[f"Kurtosis_{var}"] = kurt_vals[var]

    # Add mean and std for days
    for col in days_mean.index:
        desc_df[f"Mean_{col}"] = days_mean[col]
        desc_df[f"Std_{col}"] = days_std[col]

    # Add mean and std for raw items
    for col in raw_mean.index:
        desc_df[f"Mean_{col}"] = raw_mean[col]
        desc_df[f"Std_{col}"] = raw_std[col]

    # Add mean and std for generated averages
    for col in gen_mean.index:
        desc_df[f"Mean_{col}"] = gen_mean[col]
        desc_df[f"Std_{col}"] = gen_std[col]

    # Save descriptive stats
    desc_df.to_csv(descriptives_file, index=False)
    print(f"Descriptive statistics saved to: {descriptives_file}")


# =========================
# Run function
# =========================
if __name__ == "__main__":
    construct_daily_diary_affect(RAW_FILE, OUTPUT_FILE, DESCRIPTIVES_FILE)
