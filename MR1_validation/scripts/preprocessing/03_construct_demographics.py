#!/usr/bin/env python3
"""
03_construct_demographics.py (MR1)

Construct cleaned, harmonized demographics for MR1.

Primary source:  MR1 P5 (RA1P* variables)
Fallback source: MKE Refresher 1 aggregate (RAAC* variables)
                 Used only when P5 values are missing.

Harmonized output fields
  sex        : 1=Male, 2=Female       (RA1PRSEX, then RAACRSEX)
  educ       : 1–12 scale             (RA1PB1,   then RAACB1)
  ethnicity  : 0=No, 1=Yes Hispanic   (RA1PF1,   then RAACF1)
  race       : 1–6 category           (RA1PF7A,  then RAACF7A)
  birth_year : 4-digit year           (RA1PBYEAR, then RAACBYEAR)

Source indicators
  is_mker1        : 1 if participant appears in MKER1 file (regardless of fallback use)
  <field>_source  : 'p5' | 'mker1' | missing for each harmonized field

Original RA1P* and RAAC* columns are retained for auditability.

Missing-value codes applied before harmonization
  RAACRSEX   : 7, 8, 9  → NaN
  RAACB1     : 97, 98   → NaN
  RAACF1     : 97, 98   → NaN
  RAACF7A    : 7, 8, 9  → NaN
  RAACBYEAR  : 9997, 9998, 9999 → NaN
  RAACRAGE   : 97, 98, 99 → NaN
  RAACIDATE_YR : 9997, 9998, 9999 → NaN
  RAACIDATE_MO : 97, 98, 99  → NaN

  RA1PRSEX   : 7, 8, 9  → NaN
  RA1PB1     : 97, 98   → NaN
  RA1PF1     : 97, 98   → NaN
  RA1PF7A    : 7, 8, 9  → NaN
  RA1PBYEAR  : 9997, 9998, 9999 → NaN

Privacy: prints only aggregate counts — never prints participant IDs or rows.

Inputs:
  data/processed/mr1p5_ids.csv
  data/processed/mker1_ids.csv   (required — supplies Refresher demographic fallback
                                  for P5 missing values)

Output:
  data/processed/mr1p5_demos.csv

Run from MR1_validation/ directory.
"""

import os
import sys

import numpy as np
import pandas as pd

PROCESSED_DIR = "data/processed"
P5_FILE = os.path.join(PROCESSED_DIR, "mr1p5_ids.csv")
MKER1_FILE = os.path.join(PROCESSED_DIR, "mker1_ids.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "mr1p5_demos.csv")

# ── Missing-value codes ───────────────────────────────────────────────────────
MV_RAACRSEX = [7, 8, 9]
MV_RAACB1 = [97, 98]
MV_RAACF1 = [97, 98]
MV_RAACF7A = [7, 8, 9]
MV_RAACBYEAR = [9997, 9998, 9999]
MV_RAACRAGE = [97, 98, 99]
MV_RAACIDATE_YR = [9997, 9998, 9999]
MV_RAACIDATE_MO = [97, 98, 99]

MV_RA1PRSEX = [7, 8, 9]
MV_RA1PB1 = [97, 98]
MV_RA1PF1 = [97, 98]
MV_RA1PF7A = [7, 8, 9]
MV_RA1PBYEAR = [9997, 9998, 9999]

P5_REQUIRED_COLS = ["MIDUSID", "RA1PRSEX", "RA1PB1", "RA1PF1", "RA1PF7A", "RA1PBYEAR"]
MKER1_REQUIRED_COLS = [
    "MIDUSID", "RAACRSEX", "RAACB1", "RAACF1", "RAACF7A", "RAACBYEAR",
    "RAACRAGE", "RAACIDATE_YR", "RAACIDATE_MO", "SAMPLMAJ",
]


def _require_columns(df, required, label):
    """Exit with an aggregate error message if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: {label} is missing required column(s): {sorted(missing)}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)


def get_col(df, col):
    """Return numeric series for col; NaN series if absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    print(f"  NOTE: {col} not found in dataset — will be missing")
    return pd.Series(np.nan, index=df.index, dtype=float)


def apply_missing_codes(series, codes):
    """Replace listed codes with NaN."""
    return series.where(~series.isin(codes))


def harmonize_field(primary, fallback, field_name):
    """
    Combine primary and fallback series using primary precedence.

    Returns (harmonized, source_indicator):
      source_indicator: 'p5' where primary is valid, 'mker1' where fallback used, NaN elsewhere
    """
    harmonized = primary.copy()
    source = pd.Series(np.nan, index=primary.index, dtype=object)
    source[primary.notna()] = "p5"

    if fallback is not None:
        needs_fill = primary.isna() & fallback.notna()
        harmonized[needs_fill] = fallback[needs_fill]
        source[needs_fill] = "mker1"

    return harmonized, source


def clean_p5_demographics(df):
    """Apply P5 missing codes and numeric conversion to RA1P* fields."""
    df = df.copy()

    df["_ra1pb1_raw"] = get_col(df, "RA1PB1")
    df["_ra1pf1_raw"] = get_col(df, "RA1PF1")
    df["_ra1pf7a_raw"] = get_col(df, "RA1PF7A")

    df["_ra1pb1"] = apply_missing_codes(df["_ra1pb1_raw"], MV_RA1PB1)
    df["_ra1pf1"] = apply_missing_codes(df["_ra1pf1_raw"], MV_RA1PF1)
    df["_ra1pf7a"] = apply_missing_codes(df["_ra1pf7a_raw"], MV_RA1PF7A)
    df["_ra1prsex"] = apply_missing_codes(get_col(df, "RA1PRSEX"), MV_RA1PRSEX)
    df["_ra1pbyear"] = apply_missing_codes(get_col(df, "RA1PBYEAR"), MV_RA1PBYEAR)
    df["_ra1prage"] = get_col(df, "RA1PRAGE")

    return df


def clean_mker1_demographics(df):
    """Apply MKER1 missing codes to RAAC* fields."""
    df = df.copy()

    df["_raacrsex"] = apply_missing_codes(get_col(df, "RAACRSEX"), MV_RAACRSEX)
    df["_raacb1"] = apply_missing_codes(get_col(df, "RAACB1"), MV_RAACB1)
    df["_raacf1"] = apply_missing_codes(get_col(df, "RAACF1"), MV_RAACF1)
    df["_raacf7a"] = apply_missing_codes(get_col(df, "RAACF7A"), MV_RAACF7A)
    df["_raacbyear"] = apply_missing_codes(get_col(df, "RAACBYEAR"), MV_RAACBYEAR)
    df["_raacrage"] = apply_missing_codes(get_col(df, "RAACRAGE"), MV_RAACRAGE)
    df["_raacidate_yr"] = apply_missing_codes(get_col(df, "RAACIDATE_YR"), MV_RAACIDATE_YR)
    df["_raacidate_mo"] = apply_missing_codes(get_col(df, "RAACIDATE_MO"), MV_RAACIDATE_MO)

    return df


def report_conflict_check(df_p5, df_mker1, field_p5, field_mk, label):
    """
    For participants with both sources, report agreements/disagreements/fallbacks.
    Never prints IDs.
    """
    merged = df_p5[["MIDUSID", field_p5]].merge(
        df_mker1[["MIDUSID", field_mk]], on="MIDUSID", how="inner"
    )
    both_valid = merged[merged[field_p5].notna() & merged[field_mk].notna()]
    agree = (both_valid[field_p5] == both_valid[field_mk]).sum()
    disagree = len(both_valid) - agree
    fallback = (merged[field_p5].isna() & merged[field_mk].notna()).sum()
    print(
        f"  {label}: {len(both_valid)} participants with both sources — "
        f"{agree} agree, {disagree} disagree, {fallback} P5-missing filled from MKER1"
    )


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("03_construct_demographics.py (MR1)")
    print("=" * 60)

    # ── Load P5 ───────────────────────────────────────────────────────────────
    if not os.path.isfile(P5_FILE):
        sys.exit(f"FATAL: P5 file not found: {P5_FILE}. Run 01_harmonize_ids.py first.")
    df_p5 = pd.read_csv(P5_FILE)
    _require_columns(df_p5, P5_REQUIRED_COLS, "mr1p5_ids.csv")
    if df_p5["MIDUSID"].isna().any():
        sys.exit(f"FATAL: {int(df_p5['MIDUSID'].isna().sum())} missing MIDUSID in P5 — abort.")
    if df_p5["MIDUSID"].duplicated().any():
        sys.exit("FATAL: Duplicate MIDUSID in P5 — abort.")
    df_p5 = clean_p5_demographics(df_p5)
    print(f"P5 participants loaded: {len(df_p5)}")

    # ── Load MKER1 (required) ─────────────────────────────────────────────────
    if not os.path.isfile(MKER1_FILE):
        sys.exit(f"FATAL: MKER1 file not found: {MKER1_FILE}. Run 01_harmonize_ids.py first.")
    df_mker1 = pd.read_csv(MKER1_FILE)
    _require_columns(df_mker1, MKER1_REQUIRED_COLS, "mker1_ids.csv")
    if df_mker1["MIDUSID"].isna().any():
        sys.exit(f"FATAL: {int(df_mker1['MIDUSID'].isna().sum())} missing MIDUSID in MKER1 — abort.")
    if df_mker1["MIDUSID"].duplicated().any():
        sys.exit("FATAL: Duplicate MIDUSID in MKER1 — abort.")
    df_mker1 = clean_mker1_demographics(df_mker1)
    print(f"MKER1 participants loaded: {len(df_mker1)}")
    overlap_n = df_p5["MIDUSID"].isin(df_mker1["MIDUSID"]).sum()
    print(f"MKER1 ∩ P5: {overlap_n} participants")

    # Mark participants in MKER1
    df_p5["is_mker1"] = df_p5["MIDUSID"].isin(df_mker1["MIDUSID"]).astype(int)
    print(f"Participants flagged as is_mker1=1: {df_p5['is_mker1'].sum()}")

    # ── Merge MKER1 columns for fallback ─────────────────────────────────────
    n_before_merge = len(df_p5)
    mker1_cols = ["MIDUSID",
                  "_raacrsex", "_raacb1", "_raacf1", "_raacf7a", "_raacbyear",
                  "_raacrage", "_raacidate_yr", "_raacidate_mo",
                  # retain original columns for auditability
                  "RAACRSEX", "RAACB1", "RAACF1", "RAACF7A", "RAACBYEAR",
                  "RAACRAGE", "RAACIDATE_YR", "RAACIDATE_MO", "SAMPLMAJ"]
    mker1_cols = [c for c in mker1_cols if c in df_mker1.columns]
    df_p5 = df_p5.merge(
        df_mker1[mker1_cols], on="MIDUSID", how="left",
        suffixes=("", "_mker1")
    )
    if len(df_p5) != n_before_merge:
        sys.exit(
            f"FATAL: MKER1 merge changed row count "
            f"({n_before_merge} → {len(df_p5)}). Check for duplicate join keys."
        )

    # Conflict checks before harmonization
    print("\nConflict checks (participants with both P5 and MKER1 values):")
    report_conflict_check(df_p5, df_mker1, "_ra1prsex", "_raacrsex", "sex")
    report_conflict_check(df_p5, df_mker1, "_ra1pb1",   "_raacb1",   "education")
    report_conflict_check(df_p5, df_mker1, "_ra1pf1",   "_raacf1",   "ethnicity")
    report_conflict_check(df_p5, df_mker1, "_ra1pf7a",  "_raacf7a",  "race")
    report_conflict_check(df_p5, df_mker1, "_ra1pbyear","_raacbyear","birth_year")

    fb_sex   = df_p5.get("_raacrsex")
    fb_educ  = df_p5.get("_raacb1")
    fb_eth   = df_p5.get("_raacf1")
    fb_race  = df_p5.get("_raacf7a")
    fb_byear = df_p5.get("_raacbyear")

    # ── Harmonize fields ──────────────────────────────────────────────────────
    print("\nHarmonizing fields (P5 primary, MKER1 fallback):")

    df_p5["sex"],        df_p5["sex_source"]        = harmonize_field(df_p5["_ra1prsex"],  fb_sex,   "sex")
    df_p5["educ"],       df_p5["educ_source"]       = harmonize_field(df_p5["_ra1pb1"],    fb_educ,  "educ")
    df_p5["ethnicity"],  df_p5["ethnicity_source"]  = harmonize_field(df_p5["_ra1pf1"],    fb_eth,   "ethnicity")
    df_p5["race"],       df_p5["race_source"]       = harmonize_field(df_p5["_ra1pf7a"],   fb_race,  "race")
    df_p5["birth_year"], df_p5["birth_year_source"] = harmonize_field(df_p5["_ra1pbyear"], fb_byear, "birth_year")

    for field in ["sex", "educ", "ethnicity", "race", "birth_year"]:
        n_total = len(df_p5)
        n_p5    = (df_p5[f"{field}_source"] == "p5").sum()
        n_mk    = (df_p5[f"{field}_source"] == "mker1").sum()
        n_miss  = df_p5[field].isna().sum()
        print(f"  {field:12s}: {n_p5} from P5 | {n_mk} from MKER1 | {n_miss}/{n_total} missing")

    # ── Drop private intermediate columns ────────────────────────────────────
    internal = [c for c in df_p5.columns if c.startswith("_")]
    df_p5 = df_p5.drop(columns=internal)

    df_p5.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDemographics saved to {OUTPUT_FILE}  (shape: {df_p5.shape})")


if __name__ == "__main__":
    main()
