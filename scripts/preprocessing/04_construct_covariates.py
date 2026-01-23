#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_construct_covariates.py

Construct covariate dummies for MIDUS datasets:
- Race dummies (consistent across datasets, reference=White)
- Twin pair dummies (only for P5 twin sample)

Input: processed demographics files:
- MKE2: mke2_demos_id.csv
- M3P5: m3p5_demos_id.csv

Output: processed covariates files:
- mke2_covariates.csv
- m3p5_covariates.csv
"""

import pandas as pd
import os

# =========================
# Paths
# =========================
PROCESSED_DIR = "data/processed"

MKE2_FILE = os.path.join(PROCESSED_DIR, "mke2_demos.csv")
P5_FILE = os.path.join(PROCESSED_DIR, "m3p5_demos.csv")

MKE2_OUT = os.path.join(PROCESSED_DIR, "mke2_covariates.csv")
P5_OUT = os.path.join(PROCESSED_DIR, "m3p5_covariates.csv")

# =========================
# Race coding
# =========================
# Adjust to match the actual MIDUS race coding
RACE_CODES = [1, 2, 3, 4, 5, 6]  # 1 = White (reference), 2 = Black, 3 = Asian, 4 = Native American, 5 = Pacific Islander, 6 = Other
RACE_REF = 1

def construct_race_dummies(df):
    df = df.copy()
    for code in RACE_CODES:
        if code == RACE_REF:
            continue  # skip reference category
        col_name = f"race_{code}"
        df[col_name] = (df["race"] == code).astype(int)
    return df

# =========================
# Twin pair dummies (P5 only)
# =========================
def construct_twin_dummies(df):
    df = df.copy()
    # Only assign twins where SAMPLMAJ == 3 (twin sample)
    twins = df[df["SAMPLMAJ"] == 3].copy()
    twin_pairs = twins["M2FAMNUM"].unique()
    
    for fam in twin_pairs:
        col_name = f"twin_pair_{fam}"
        df[col_name] = ((df["SAMPLMAJ"] == 3) & (df["M2FAMNUM"] == fam)).astype(int)
    
    return df

# =========================
# Process MKE2
# =========================
df_mke2 = pd.read_csv(MKE2_FILE)
df_mke2 = construct_race_dummies(df_mke2)
# No twins in MKE2, so skip twin dummies
df_mke2.to_csv(MKE2_OUT, index=False)
print(f"MKE2 covariates saved to {MKE2_OUT}")

# =========================
# Process M3P5
# =========================
df_p5 = pd.read_csv(P5_FILE)
df_p5 = construct_race_dummies(df_p5)
df_p5 = construct_twin_dummies(df_p5)
df_p5.to_csv(P5_OUT, index=False)
print(f"M3P5 covariates saved to {P5_OUT}")
