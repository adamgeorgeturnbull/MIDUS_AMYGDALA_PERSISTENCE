#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_construct_covariates.py

Construct covariate dummies for MIDUS datasets:
- Race dummies (consistent across datasets, reference=White)
- Twin pair dummies (only for P5 twin sample)

Ensures all variable names are valid Python identifiers for analysis.

Input: processed demographics files:
- MKE2: mke2_demos.csv
- M3P5: m3p5_demos.csv

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
RACE_CODES = [1, 2, 3, 4, 5, 6]  # 1 = White (reference)
RACE_REF = 1

def construct_race_dummies(df):
    df = df.copy()
    for code in RACE_CODES:
        if code == RACE_REF:
            continue
        col_name = f"race_{code}"
        df[col_name] = (df["race"] == code).astype(int)
    return df

# =========================
# Twin pair dummies (P5 only)
# =========================
def construct_twin_dummies(df):
    df = df.copy()

    # Only consider participants in the twin sample
    twins = df[df["SAMPLMAJ"] == 3].copy()

    # Count how many participants per family
    fam_counts = twins["M2FAMNUM"].value_counts()

    # Keep only families with more than 1 participant
    twin_fams = fam_counts[fam_counts > 1].index

    for fam in twin_fams:
        # make safe Python identifier for family number
        fam_str = str(fam).replace(".", "_")
        if fam_str[0].isdigit():
            fam_str = f"fam_{fam_str}"
        col_name = f"twin_pair_{fam_str}"
        
        # Only assign 1 if participant is in this family AND SAMPLMAJ == 3
        df[col_name] = ((df["SAMPLMAJ"] == 3) & (df["M2FAMNUM"] == fam)).astype(int)

    return df

# =========================
# Process MKE2
# =========================
df_mke2 = pd.read_csv(MKE2_FILE)
df_mke2 = construct_race_dummies(df_mke2)
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
