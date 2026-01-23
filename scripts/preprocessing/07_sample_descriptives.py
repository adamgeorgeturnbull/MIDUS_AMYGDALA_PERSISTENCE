#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_sample_descriptives.py

Compute sample descriptives for 5 MIDUS samples:
1) Full daily diary sample (C2PAGE)
2) Full neuroscience sample (C5PAGE)
3) Overlapping daily diary + neuroscience (C5PAGE)
4) Neuroscience imaging sample (C5IC == 1, C5PAGE)
5) Imaging + daily diary sample (C5IC == 1 & diary, C5PAGE)

Outputs: age, sex, education, ethnicity, race in publication-ready table.
"""

import pandas as pd
import numpy as np
import os

# =========================
# Paths
# =========================
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results/tables"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLEAN_FILE = os.path.join(PROCESSED_DIR, "midus_merged_clean.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "sample_descriptives.csv")

# =========================
# Load cleaned merged data
# =========================
df = pd.read_csv(CLEAN_FILE)
print(f"Loaded cleaned merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# =========================
# Harmonize key demographic variables
# =========================
# Sex: 1 = Male, 2 = Female
df["sex"] = df["sex"].replace({1: 1, 2: 2})

# Ethnicity: Hispanic/Latino binary (0 = no, 1 = yes)
df["ethnicity"] = df["ethnicity"].replace({0: 0, 1: 1})  # already harmonized

# =========================
# Define samples
# =========================
samples = {
    "daily_diary_full": df[df["StartYear"].notna()],
    "neuro_full": df[df["C5PDATE_YR"].notna()],
    "daily_neuro_overlap": df[df["StartYear"].notna() & df["C5PDATE_YR"].notna()],
    "neuro_imaging": df[df["C5IC"] == 1],  # only completed imaging
    "imaging_daily_overlap": df[(df["C5IC"] == 1) & df["StartYear"].notna()]
}

# =========================
# Function to summarize sample
# =========================
def summarize_sample(df_sample, use_age_col):
    summary = {}
    
    # Age
    age_series = df_sample[use_age_col].dropna()
    summary["N"] = len(age_series)
    summary["Age_mean"] = f"{age_series.mean():.1f}"
    summary["Age_SD"] = f"{age_series.std():.1f}"
    summary["Age_range"] = f"{age_series.min():.0f}-{age_series.max():.0f}"
    
    # Sex
    sex_counts = df_sample["sex"].value_counts(dropna=False)
    n_total = len(df_sample)
    summary["%Female"] = f"{(sex_counts.get(2,0)/n_total*100):.1f}"
    
    # Education
    educ_series = df_sample["educ"].dropna()
    summary["Educ_mean"] = f"{educ_series.mean():.1f}"
    summary["Educ_SD"] = f"{educ_series.std():.1f}"
    
    # Ethnicity
    eth_counts = df_sample["ethnicity"].value_counts(dropna=False)
    summary["%Hispanic"] = f"{(eth_counts.get(1,0)/len(df_sample)*100):.1f}"
    
    # Race
    race_counts = df_sample["race"].value_counts(dropna=False)
    summary["%White"] = f"{(race_counts.get(1,0)/len(df_sample)*100):.1f}"
    summary["%Black"] = f"{(race_counts.get(2,0)/len(df_sample)*100):.1f}"
    summary["%NativeAmerican"] = f"{(race_counts.get(3,0)/len(df_sample)*100):.1f}"
    summary["%Asian"] = f"{(race_counts.get(4,0)/len(df_sample)*100):.1f}"
    summary["%PacificIslander"] = f"{(race_counts.get(5,0)/len(df_sample)*100):.1f}"
    summary["%Other"] = f"{(race_counts.get(6,0)/len(df_sample)*100):.1f}"
    
    return pd.Series(summary)

# =========================
# Compute descriptives for all samples
# =========================
rows = []
for name, sample_df in samples.items():
    age_col = "C2PAGE" if name == "daily_diary_full" else "C5PAGE"
    rows.append(summarize_sample(sample_df, age_col).rename(name))

descriptives_table = pd.DataFrame(rows).T
descriptives_table.to_csv(OUTPUT_FILE)
print(f"Sample descriptives saved to: {OUTPUT_FILE}")
