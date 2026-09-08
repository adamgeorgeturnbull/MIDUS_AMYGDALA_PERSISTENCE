#!/usr/bin/env python3
"""
04_construct_covariates.py (MR1)

Construct analysis-ready covariates for MR1.

MR1 vs M3 differences:
- Race dummies (race_2 through race_6) constructed from harmonized 'race' field
  (RA1PF7A primary, RAACF7A fallback from MKER1); White (race=1) is reference
- Family and twin indicators are not constructed here; related-family clustering
  will be assessed separately from the real analysis dataset before choosing
  OLS versus clustered/multilevel inference

Inputs:
- data/processed/mr1p5_demos.csv

Outputs:
- data/processed/mr1p5_covariates.csv

Run from MR1_validation/ directory.
"""

import os
import sys

import pandas as pd

PROCESSED_DIR = "data/processed"
MR1P5_FILE = os.path.join(PROCESSED_DIR, "mr1p5_demos.csv")
MR1P5_OUT = os.path.join(PROCESSED_DIR, "mr1p5_covariates.csv")


RACE_CODES = [1, 2, 3, 4, 5, 6]
RACE_REF = 1  # White is reference category


def construct_race_dummies(df):
    df = df.copy()
    for code in RACE_CODES:
        if code == RACE_REF:
            continue
        col_name = f"race_{code}"
        df[col_name] = (df["race"] == code).astype(float).where(df["race"].notna())
    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.isfile(MR1P5_FILE):
        sys.exit(f"FATAL: input not found: {MR1P5_FILE}. Run 03_construct_demographics.py first.")

    df = pd.read_csv(MR1P5_FILE)

    for col in ["MIDUSID", "race"]:
        if col not in df.columns:
            sys.exit(f"FATAL: required column '{col}' not found in {MR1P5_FILE}.")

    n_missing_id = int(df["MIDUSID"].isna().sum())
    if n_missing_id > 0:
        sys.exit(f"FATAL: {n_missing_id} missing MIDUSID value(s) — abort.")
    n_dup_id = int(df["MIDUSID"].duplicated().sum())
    if n_dup_id > 0:
        sys.exit(f"FATAL: {n_dup_id} duplicate MIDUSID value(s) — abort.")

    nonmissing_race = df["race"].dropna()
    invalid_mask = ~nonmissing_race.isin(RACE_CODES)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        invalid_vals = sorted(nonmissing_race[invalid_mask].unique().tolist())
        sys.exit(
            f"FATAL: {n_invalid} row(s) have race values outside {RACE_CODES}: "
            f"{invalid_vals}"
        )

    df = construct_race_dummies(df)
    df.to_csv(MR1P5_OUT, index=False)
    print(f"MR1 covariates saved to {MR1P5_OUT}")
    print(
        "Note: this script constructs race dummies only and does not construct "
        "family or twin indicators. Related-family clustering will be assessed "
        "separately from the real analysis dataset before choosing OLS versus "
        "clustered/multilevel inference."
    )


if __name__ == "__main__":
    main()
