#!/usr/bin/env python3

"""
Construct demographics variables for MIDUS datasets independently (MKE2, M3P5).
Keeps all original variables and appends cleaned demographics columns.
"""

import pandas as pd
import numpy as np
import os


PROCESSED_DIR = "data/processed"
MKE2_FILE = os.path.join(PROCESSED_DIR, "mke2_ids.csv")
M3P5_FILE = os.path.join(PROCESSED_DIR, "m3p5_ids.csv")


def get_clean_col(df, col):
    """
    Return a cleaned Series if column exists, otherwise NA Series.
    Empty strings are converted to NA to avoid pandas FutureWarnings.
    """
    if col in df.columns:
        s = df[col].copy()
        s = s.replace("", pd.NA)
        return s
    return pd.Series(pd.NA, index=df.index)


def clean_demographics(df):
    """
    Add harmonized demographics variables while keeping all original columns.
    """
    df = df.copy()

    # enforce one row per participant
    df = df.drop_duplicates(subset="M2ID")

    # harmonized demographics
    sex_1 = get_clean_col(df, "C1PRSEX")
    sex_2 = get_clean_col(df, "CACRSEX")
    df["sex"] = sex_1.fillna(sex_2)

    educ_1 = get_clean_col(df, "C1PB1")
    educ_2 = get_clean_col(df, "CACB1")
    df["educ"] = educ_1.fillna(educ_2)

    eth_1 = get_clean_col(df, "C1PF1")
    eth_2 = get_clean_col(df, "CACF1")
    df["ethnicity"] = eth_1.fillna(eth_2)

    race_1 = get_clean_col(df, "C1PF7A")
    race_2 = get_clean_col(df, "CACF7A")
    df["race"] = race_1.fillna(race_2)

    # MIDUS missing codes → NA
    df["educ"] = df["educ"].replace({97: pd.NA, 98: pd.NA})
    df["ethnicity"] = df["ethnicity"].replace({97: pd.NA, 98: pd.NA})
    df["race"] = df["race"].replace({7: pd.NA, 8: pd.NA})

    return df




def main():
    """Main execution function."""
# ------------------------------
# Load data
# ------------------------------
    df_mke2 = pd.read_csv(MKE2_FILE)
    df_m3p5 = pd.read_csv(M3P5_FILE)

# ------------------------------
# Clean demographics independently
# ------------------------------
    df_mke2_demos = clean_demographics(df_mke2)
    df_m3p5_demos = clean_demographics(df_m3p5)

# ------------------------------
# Save outputs
# ------------------------------
    df_mke2_demos.to_csv(
        os.path.join(PROCESSED_DIR, "mke2_demos.csv"),
        index=False
    )

    df_m3p5_demos.to_csv(
        os.path.join(PROCESSED_DIR, "m3p5_demos.csv"),
        index=False
    )

    print("Demographics constructed and saved for MKE2 and M3P5.")


if __name__ == "__main__":
    main()
