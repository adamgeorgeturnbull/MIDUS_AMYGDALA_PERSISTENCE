#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_clean_merged_data.py

Clean merged MIDUS dataset:
1) Add 0s for all twin_pair variables for participants from MKE2.
2) Create age at P2 (C2PAGE).
3) Create time between P2 and P5 in months (time_P2_P5).
4) Compute skewness and kurtosis for neuroscience PANAS variables and create log-transformed C5SPGN.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

PROCESSED_DIR = "data/processed"
TABLE_DIR = "results/tables"

# Ensure output directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
MERGED_FILE = os.path.join(PROCESSED_DIR, "midus_merged.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "midus_merged_clean.csv")
STATS_OUTPUT        = os.path.join(TABLE_DIR, "panas_skew_kurtosis.csv")
PANAS_AUDIT_OUTPUT  = os.path.join(TABLE_DIR, "panas_missing_code_recode.csv")



def main():
    """Main execution function."""
    # =========================
    # Load merged dataset
    # =========================
    df = pd.read_csv(MERGED_FILE)
    print(f"Loaded merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # =========================
    # Identify participants from MKE2
    # =========================
    mke2_markers = ["CACB1", "CACF1", "CACF7A", "CACRAGE", "CACIDATE_MO", "CACIDATE_YR", "CACRSEX"]
    is_mke2 = df[mke2_markers].notna().any(axis=1)
    print(f"Number of participants identified as MKE2: {is_mke2.sum()}")

    # =========================
    # Fill 0s for twin_pair columns
    # =========================
    twin_cols = [col for col in df.columns if col.startswith("twin_pair_")]
    print(f"Twin pair columns detected: {twin_cols}")
    df.loc[is_mke2, twin_cols] = 0

    # =========================
    # Compute age at P2 (C2PAGE)
    # =========================
    if "StartYear" not in df.columns or "StartMonth" not in df.columns:
        raise ValueError("StartYear and StartMonth columns required from daily diary data")

    df["C2PAGE"] = np.nan

    # --- M3 participants ---
    m3_mask = ~is_mke2 & df["C1PBYEAR"].notna()
    df.loc[m3_mask, "C2PAGE"] = df.loc[m3_mask, "StartYear"] - df.loc[m3_mask, "C1PBYEAR"]

    # --- MKE2 participants ---
    mke2_mask = is_mke2 & df["CACRAGE"].notna() & df["CACIDATE_YR"].notna() & df["CACIDATE_MO"].notna()
    df.loc[mke2_mask, "C2PAGE"] = df.loc[mke2_mask, "CACRAGE"] + (
        (df.loc[mke2_mask, "StartYear"] - df.loc[mke2_mask, "CACIDATE_YR"])
        + (df.loc[mke2_mask, "StartMonth"] - df.loc[mke2_mask, "CACIDATE_MO"]) / 12
    )

    # =========================
    # Compute time between P2 and P5 in months
    # =========================
    time_cols_required = ["C5PDATE_YR", "C5PDATE_MO"]
    for col in time_cols_required:
        if col not in df.columns:
            raise ValueError(f"{col} required to compute time_P2_P5")

    df["time_P2_P5"] = (df["C5PDATE_YR"] - df["StartYear"]) * 12 + (df["C5PDATE_MO"] - df["StartMonth"])
    df["time_P2_P5"] = df["time_P2_P5"].abs()  # ensure positive

    # =========================
    # Recode MIDUS missing value codes to NaN for ERQ subscales
    # 98 = MISSING, 99 = INAPPLICABLE
    # =========================
    for var in ["C5SER", "C5SES"]:
        if var in df.columns:
            n_missing = (df[var] >= 97).sum()
            df[var] = df[var].where(df[var] < 97)
            print(f"Recoded {n_missing} missing-code values to NaN in {var}")

    # =========================
    # Recode PANAS missing-value codes to NaN (MIDUS 3 codebook: 8, 98, 99)
    # Must occur before skewness/kurtosis and log-transform calculations.
    # =========================
    panas_vars = ["C5SPGP", "C5SPGN"]
    PANAS_MISSING_CODES = [8, 98, 99]
    # C5SPGP and C5SPGN are mean item scores on the 1-5 PANAS response scale,
    # not summed scale scores.
    PANAS_VALID_MIN, PANAS_VALID_MAX = 1, 5

    for var in panas_vars:
        if var not in df.columns:
            raise ValueError(f"Required PANAS variable {var} not found in dataset")

    print(f"\nRecoding PANAS missing-value codes {PANAS_MISSING_CODES} to NaN...")
    audit_rows = []
    for var in panas_vars:
        # Preserve the raw series so numeric-conversion losses can be detected
        raw = df[var].copy()
        n_missing_before = int(raw.isna().sum())

        numeric = pd.to_numeric(raw, errors="coerce")

        # Nonmissing raw values that fail numeric conversion must not be silently
        # absorbed into the missing category.
        unparseable = raw[raw.notna() & numeric.isna()]
        if len(unparseable) > 0:
            print(f"ERROR: {var} has {len(unparseable)} nonmissing value(s) that could not "
                  f"be converted to numeric:")
            print(unparseable.value_counts().sort_index().to_string())
            raise ValueError(
                f"{var}: {len(unparseable)} nonmissing value(s) lost during numeric "
                f"conversion; inspect and correct the source data before proceeding"
            )

        df[var] = numeric

        counts = {code: int((df[var] == code).sum()) for code in PANAS_MISSING_CODES}
        for code in PANAS_MISSING_CODES:
            df.loc[df[var] == code, var] = np.nan

        n_missing_after = int(df[var].isna().sum())
        n_total_recoded = sum(counts.values())
        valid = df[var].dropna()

        if len(valid) == 0:
            raise ValueError(f"{var}: no valid observations remain after recoding")

        out_of_range = valid[(valid < PANAS_VALID_MIN) | (valid > PANAS_VALID_MAX)]
        if len(out_of_range) > 0:
            print(f"ERROR: {var} has {len(out_of_range)} value(s) outside valid range "
                  f"{PANAS_VALID_MIN}–{PANAS_VALID_MAX} after recoding:")
            print(out_of_range.value_counts().sort_index().to_string())
            sys.exit(1)

        audit_rows.append({
            "variable":         var,
            "n_code_8":         counts[8],
            "n_code_98":        counts[98],
            "n_code_99":        counts[99],
            "n_total_recoded":  n_total_recoded,
            "n_missing_before": n_missing_before,
            "n_missing_after":  n_missing_after,
            "n_valid":          len(valid),
            "valid_min":        float(valid.min()),
            "valid_max":        float(valid.max()),
        })
        print(f"  {var}: code 8={counts[8]}, code 98={counts[98]}, code 99={counts[99]}, "
              f"total recoded={n_total_recoded}, valid N={len(valid)}, "
              f"range=[{valid.min():.2f}, {valid.max():.2f}]")

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(PANAS_AUDIT_OUTPUT, index=False)
    print(f"PANAS missing-code audit saved to {PANAS_AUDIT_OUTPUT}")

    # =========================
    # Compute skewness and kurtosis for neuroscience PANAS and log-transform C5SPGN
    # =========================
    stats_list = []

    for var in panas_vars:
        series = df[var].dropna()
        var_skew = skew(series)
        var_kurt = kurtosis(series)
        stats_list.append({"variable": var, "skewness": var_skew, "kurtosis": var_kurt})

    # Save stats to CSV
    pd.DataFrame(stats_list).to_csv(STATS_OUTPUT, index=False)
    print(f"Skewness and kurtosis saved to {STATS_OUTPUT}")

    # Cleaned PANAS scores are validated above as 1–5: no offset is needed.
    # Use the natural log of the original positive scale; diary NA is separate.
    df["C5SPGN_log"] = np.log(df["C5SPGN"])
    print("PANAS negative affect: natural log, no added constant")

    # =========================
    # Save cleaned dataset
    # =========================
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned merged dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
