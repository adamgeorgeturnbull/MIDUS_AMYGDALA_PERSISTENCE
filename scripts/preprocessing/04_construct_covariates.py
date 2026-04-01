#!/usr/bin/env python3
"""
04_construct_covariates.py

Construct covariate dummy variables for MIDUS datasets for use in regression models.
Creates race dummies (reference: White) and twin pair dummies (M3P5 only).

Inputs:
- data/processed/mke2_demos.csv (Milwaukee sample demographics)
- data/processed/m3p5_demos.csv (Project 5 demographics)

Outputs:
- data/processed/mke2_covariates.csv (MKE2 with race dummies)
- data/processed/m3p5_covariates.csv (M3P5 with race and twin pair dummies)

Run from project root directory.
"""

import os

import pandas as pd

# ============================================================================
# Paths and Constants
# ============================================================================
PROCESSED_DIR = "data/processed"

# Input files
MKE2_FILE = os.path.join(PROCESSED_DIR, "mke2_demos.csv")
M3P5_FILE = os.path.join(PROCESSED_DIR, "m3p5_demos.csv")

# Output files
MKE2_OUT = os.path.join(PROCESSED_DIR, "mke2_covariates.csv")
M3P5_OUT = os.path.join(PROCESSED_DIR, "m3p5_covariates.csv")

# Race coding: 1=White, 2=Black, 3=Native American, 4=Asian, 5=Pacific Islander, 6=Other
RACE_CODES = [1, 2, 3, 4, 5, 6]
RACE_REF = 1  # White is reference category (not included as dummy)

# ============================================================================
# Helper Functions
# ============================================================================
def construct_race_dummies(df):
    """
    Create dummy variables for race categories (excluding reference category).

    Args:
        df: DataFrame with 'race' column

    Returns:
        DataFrame with added race_2, race_3, race_4, race_5, race_6 columns
    """
    df = df.copy()
    for code in RACE_CODES:
        if code == RACE_REF:
            continue  # Skip reference category
        col_name = f"race_{code}"
        # Preserve NaN when race is unknown so merge_combine can fill from MKE2
        df[col_name] = (df["race"] == code).astype(float).where(df["race"].notna())
    return df


def construct_twin_dummies(df):
    """
    Create dummy variables for twin pairs to control for family clustering.

    For each twin family with 2+ participants in the sample, creates a
    dummy variable (twin_pair_fam_XXXXX) that equals 1 for members of
    that family and 0 otherwise. Omits one family as reference.

    Args:
        df: DataFrame with 'SAMPLMAJ' and 'M2FAMNUM' columns

    Returns:
        DataFrame with added twin_pair_fam_* columns
    """
    df = df.copy()

    # Identify participants in the twin sample (SAMPLMAJ == 3)
    twins = df[df["SAMPLMAJ"] == 3].copy()

    # Count participants per twin family
    fam_counts = twins["M2FAMNUM"].value_counts()

    # Keep only families with 2+ participants (actual twin pairs in sample)
    twin_fams = fam_counts[fam_counts > 1].index

    for fam in twin_fams:
        # Create safe Python identifier from family number
        # (e.g., 5063.0 → "fam_5063_0")
        fam_str = str(fam).replace(".", "_")
        if fam_str[0].isdigit():
            fam_str = f"fam_{fam_str}"
        col_name = f"twin_pair_{fam_str}"

        # Dummy = 1 if participant is in this twin family, 0 otherwise
        df[col_name] = ((df["SAMPLMAJ"] == 3) & (df["M2FAMNUM"] == fam)).astype(int)

    return df


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    # Ensure output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Process MKE2 (race dummies only - no twins in Milwaukee sample)
    print("Processing MKE2 covariates...")
    df_mke2 = pd.read_csv(MKE2_FILE)
    df_mke2 = construct_race_dummies(df_mke2)
    df_mke2.to_csv(MKE2_OUT, index=False)
    print(f"✓ MKE2 covariates saved to {MKE2_OUT}")

    # Process M3P5 (race dummies + twin pair dummies)
    print("\nProcessing M3P5 covariates...")
    df_m3p5 = pd.read_csv(M3P5_FILE)
    df_m3p5 = construct_race_dummies(df_m3p5)
    df_m3p5 = construct_twin_dummies(df_m3p5)
    df_m3p5.to_csv(M3P5_OUT, index=False)
    print(f"✓ M3P5 covariates saved to {M3P5_OUT}")


if __name__ == "__main__":
    main()
