#!/usr/bin/env python3
"""
06_clean_merged_data.py (MR1)

Clean merged MR1 dataset:
1. Compute approximate age at diary wave (RA2PAGE).
2. Compute time between P2 diary and P5 neuroscience visit (months).
3. Recode MIDUS missing codes in RA5SER, RA5SES (ERQ).
4. Recode PANAS missing codes (8, 98, 99 → NaN) and validate RA5SPGP, RA5SPGN.
5. Compute skewness, kurtosis, and log-transform RA5SPGN.

MR1 vs M3 differences:
- No MKE2 markers / twin_pair fill step
- Age at diary (RA2PAGE): computed from harmonized birth_year (RA1PBYEAR primary,
  RAACBYEAR fallback) where available. For MKE participants without birth_year,
  uses the MKE interview-age formula:
    RAACRAGE + (StartYear - RAACIDATE_YR) + (StartMonth - RAACIDATE_MO)/12
  RA5PAGE (age at neuroscience visit) remains the primary age covariate in analyses.
- PANAS: RA5SPGP, RA5SPGN (not C5SPGP, C5SPGN)
- P5 date: RA5PDATE_MO, RA5PDATE_YR
- ERQ: RA5SER, RA5SES

Inputs:
- data/processed/mr1_merged.csv

Outputs:
- data/processed/mr1_merged_clean.csv
- results/tables/panas_skew_kurtosis.csv
- results/tables/panas_missing_code_recode.csv

Privacy: prints only aggregate counts — never prints participant IDs or rows.

Run from MR1_validation/ directory.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

PROCESSED_DIR = "data/processed"
TABLE_DIR     = "results/tables"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

MERGED_FILE        = os.path.join(PROCESSED_DIR, "mr1_merged.csv")
OUTPUT_FILE        = os.path.join(PROCESSED_DIR, "mr1_merged_clean.csv")
STATS_OUTPUT       = os.path.join(TABLE_DIR, "panas_skew_kurtosis.csv")
PANAS_AUDIT_OUTPUT = os.path.join(TABLE_DIR, "panas_missing_code_recode.csv")

REQUIRED_COLS = [
    "MIDUSID", "StartYear", "StartMonth", "birth_year",
    "RA5PDATE_YR", "RA5PDATE_MO", "RA5SER", "RA5SES", "RA5SPGP", "RA5SPGN",
]

PANAS_MISSING_CODES  = [8, 98, 99]
PANAS_VALID_MIN, PANAS_VALID_MAX = 1, 5

DATE_MV_YEAR  = [9997, 9998, 9999]
DATE_MV_MONTH = [97, 98, 99]


def main():
    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.isfile(MERGED_FILE):
        sys.exit(f"FATAL: {MERGED_FILE} not found. Run 05_merge_master_dataset.py first.")

    df = pd.read_csv(MERGED_FILE)
    print(f"Loaded merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        sys.exit(f"FATAL: missing required column(s): {sorted(missing_cols)}")

    if df["MIDUSID"].isna().any():
        sys.exit(
            f"FATAL: {int(df['MIDUSID'].isna().sum())} missing MIDUSID value(s) — abort."
        )
    if df["MIDUSID"].duplicated().any():
        sys.exit(
            f"FATAL: {int(df['MIDUSID'].duplicated().sum())} duplicate MIDUSID value(s) "
            "— abort."
        )

    # ── Age at diary wave (RA2PAGE) ───────────────────────────────────────────
    # Primary: harmonized birth_year (RA1PBYEAR or RAACBYEAR) → StartYear - birth_year
    start_year  = pd.to_numeric(df.get("StartYear"),  errors="coerce")
    start_month = pd.to_numeric(df.get("StartMonth"), errors="coerce")

    if "birth_year" in df.columns:
        byear = pd.to_numeric(df["birth_year"], errors="coerce")
        df["RA2PAGE"] = start_year - byear
        n_byear = df["RA2PAGE"].notna().sum()
        print(f"RA2PAGE from birth_year: {n_byear} participants")
    elif "RA1PBYEAR" in df.columns:
        df["RA2PAGE"] = start_year - pd.to_numeric(df["RA1PBYEAR"], errors="coerce")
        n_byear = df["RA2PAGE"].notna().sum()
        print(f"RA2PAGE from RA1PBYEAR: {n_byear} participants")
    else:
        df["RA2PAGE"] = np.nan
        print("WARNING: birth_year / RA1PBYEAR not found — RA2PAGE will use MKE formula only")

    # MKE interview-age formula fallback for participants still missing RA2PAGE
    mke_vars = ["RAACRAGE", "RAACIDATE_YR", "RAACIDATE_MO"]
    if all(v in df.columns for v in mke_vars) and df["RA2PAGE"].isna().any():
        raacrage = pd.to_numeric(df["RAACRAGE"], errors="coerce")
        raacrage = raacrage.where(~raacrage.isin([97, 98, 99]))
        raacidate_yr = pd.to_numeric(df["RAACIDATE_YR"], errors="coerce")
        raacidate_yr = raacidate_yr.where(~raacidate_yr.isin([9997, 9998, 9999]))
        raacidate_mo = pd.to_numeric(df["RAACIDATE_MO"], errors="coerce")
        raacidate_mo = raacidate_mo.where(~raacidate_mo.isin([97, 98, 99]))

        mke_age = (raacrage
                   + (start_year - raacidate_yr)
                   + (start_month - raacidate_mo) / 12)
        needs_fill = df["RA2PAGE"].isna() & mke_age.notna()
        df.loc[needs_fill, "RA2PAGE"] = mke_age[needs_fill]
        print(
            f"RA2PAGE from MKE interview-age formula: "
            f"{needs_fill.sum()} additional participants"
        )

    if df["RA2PAGE"].notna().any():
        print(
            f"RA2PAGE final: mean={df['RA2PAGE'].mean():.1f}, "
            f"range {df['RA2PAGE'].min():.0f}–{df['RA2PAGE'].max():.0f}"
        )
    else:
        print("WARNING: RA2PAGE could not be computed for any participant")

    # ── Time between P2 diary and P5 neuroscience visit ──────────────────────
    # Convert to numeric and recode date nonresponse codes before computing.
    # Missing dates yield missing time_P2_P5, not extreme numeric values.
    p2_yr = pd.to_numeric(df["StartYear"],   errors="coerce")
    p2_yr = p2_yr.where(~p2_yr.isin(DATE_MV_YEAR))
    p2_mo = pd.to_numeric(df["StartMonth"],  errors="coerce")
    p2_mo = p2_mo.where(~p2_mo.isin(DATE_MV_MONTH))
    p5_yr = pd.to_numeric(df["RA5PDATE_YR"], errors="coerce")
    p5_yr = p5_yr.where(~p5_yr.isin(DATE_MV_YEAR))
    p5_mo = pd.to_numeric(df["RA5PDATE_MO"], errors="coerce")
    p5_mo = p5_mo.where(~p5_mo.isin(DATE_MV_MONTH))

    df["time_P2_P5"] = ((p5_yr - p2_yr) * 12 + (p5_mo - p2_mo)).abs()

    # ── ERQ missing codes ─────────────────────────────────────────────────────
    for var in ["RA5SER", "RA5SES"]:
        raw     = df[var].copy()
        numeric = pd.to_numeric(raw, errors="coerce")
        unparseable = raw[raw.notna() & numeric.isna()]
        if len(unparseable) > 0:
            sys.exit(
                f"FATAL: {var} has {len(unparseable)} nonmissing value(s) that could not "
                f"be converted to numeric. Inspect and correct the source data."
            )
        df[var]   = numeric
        n_missing = int((df[var] >= 97).sum())
        df[var]   = df[var].where(df[var] < 97)
        print(f"Recoded {n_missing} missing-code values to NaN in {var}")

    # ── PANAS missing-code recode ─────────────────────────────────────────────
    panas_vars = ["RA5SPGP", "RA5SPGN"]

    print(f"\nRecoding PANAS missing-value codes {PANAS_MISSING_CODES} to NaN...")
    audit_rows = []
    for var in panas_vars:
        raw              = df[var].copy()
        n_missing_before = int(raw.isna().sum())

        numeric     = pd.to_numeric(raw, errors="coerce")
        unparseable = raw[raw.notna() & numeric.isna()]
        if len(unparseable) > 0:
            sys.exit(
                f"FATAL: {var} has {len(unparseable)} nonmissing value(s) that could not "
                f"be converted to numeric. Inspect and correct the source data."
            )

        df[var] = numeric

        counts = {code: int((df[var] == code).sum()) for code in PANAS_MISSING_CODES}
        for code in PANAS_MISSING_CODES:
            df.loc[df[var] == code, var] = np.nan

        n_missing_after = int(df[var].isna().sum())
        n_total_recoded = sum(counts.values())
        valid           = df[var].dropna()

        if len(valid) == 0:
            sys.exit(f"FATAL: {var}: no valid observations remain after recoding.")

        out_of_range = valid[(valid < PANAS_VALID_MIN) | (valid > PANAS_VALID_MAX)]
        if len(out_of_range) > 0:
            sys.exit(
                f"FATAL: {var} has {len(out_of_range)} value(s) outside valid range "
                f"{PANAS_VALID_MIN}–{PANAS_VALID_MAX} after recoding."
            )

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
        print(
            f"  {var}: code 8={counts[8]}, code 98={counts[98]}, "
            f"code 99={counts[99]}, total recoded={n_total_recoded}, "
            f"valid N={len(valid)}, range=[{valid.min():.2f}, {valid.max():.2f}]"
        )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(PANAS_AUDIT_OUTPUT, index=False)
    print(f"PANAS missing-code audit saved to {PANAS_AUDIT_OUTPUT}")

    # ── PANAS skew/kurtosis + log transform ──────────────────────────────────
    stats_list = []
    for var in panas_vars:
        series = df[var].dropna()
        stats_list.append({
            "variable": var,
            "skewness": skew(series),
            "kurtosis": kurtosis(series),
        })
    pd.DataFrame(stats_list).to_csv(STATS_OUTPUT, index=False)
    print(f"Skewness and kurtosis saved to {STATS_OUTPUT}")

    min_nonzero = df.loc[df["RA5SPGN"] > 0, "RA5SPGN"].min()
    log_offset  = min_nonzero / 2 if not pd.isna(min_nonzero) else 0.01
    df["RA5SPGN_log"] = np.log(df["RA5SPGN"] + log_offset)
    print(f"Log transform offset for RA5SPGN: {log_offset:.4f} (half of min non-zero value)")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned merged dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
