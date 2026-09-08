#!/usr/bin/env python3
"""
05_merge_master_dataset.py (MR1)

Build the MR1 analytic master dataset by combining daily diary affect summaries
and P5 covariates, then supplementing with MKER1 columns.

Analytic universe:
  The master contains the union of diary and P5-covariate participants. A
  participant is retained even if they appear in only one of the two files.
  MKER1 is required and used as a demographic supplement but does not
  independently expand the analytic universe; participants who appear only
  in the full MKER1 file are not added.

Merge order:
  1. diary ∪ P5 outer join (merge_combine, MIDUSID key):
       daily_diary_processed.csv  ⊕  mr1p5_covariates.csv
     For columns present in both files, the diary value is preferred when
     nonmissing; the P5 value fills in otherwise.
  2. MKER1 left-join supplement onto the diary ∪ P5 result:
       mker1_ids.csv → brings only columns not already in the master.
     This does not change the row count or ID set.

Structural validations:
  - MIDUSID present, nonmissing, and unique in every input.
  - Post-outer-join ID set equals the exact union of diary and P5 IDs.
  - Post-outer-join row count equals the union size.
  - MKER1 supplement does not change the master row count or ID set.

Privacy: prints only aggregate counts — never prints participant IDs or rows.

Inputs:
  data/processed/daily_diary_processed.csv
  data/processed/mr1p5_covariates.csv
  data/processed/mker1_ids.csv           (required — demographic supplement only)

Output:
  data/processed/mr1_merged.csv

Run from MR1_validation/ directory.
"""

import os
import sys

import pandas as pd

PROCESSED_DIR = "data/processed"
DIARY_FILE   = os.path.join(PROCESSED_DIR, "daily_diary_processed.csv")
COV_FILE     = os.path.join(PROCESSED_DIR, "mr1p5_covariates.csv")
MKER1_FILE   = os.path.join(PROCESSED_DIR, "mker1_ids.csv")
OUTPUT_FILE  = os.path.join(PROCESSED_DIR, "mr1_merged.csv")

KEY = "MIDUSID"


def assert_unique(df, label):
    n_dup = df[KEY].duplicated().sum()
    if n_dup > 0:
        sys.exit(
            f"FATAL: {n_dup} duplicate {KEY} values in {label}. "
            "Expected one row per participant. Investigate before proceeding."
        )


def merge_combine(df1, df2, key=KEY):
    """
    Outer-join df1 and df2 on key.

    For columns present in both files (other than key), the df1 value is
    preferred when nonmissing; the df2 value fills in otherwise.  Suffix
    columns (_1, _2) are resolved and dropped so no duplicates remain.
    """
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    overlap_cols = list((df1_cols & df2_cols) - {key})

    merged = df1.merge(df2, on=key, how="outer", suffixes=("_1", "_2"))

    for col in overlap_cols:
        merged[col] = merged[f"{col}_1"].fillna(merged[f"{col}_2"])
        merged.drop([f"{col}_1", f"{col}_2"], axis=1, inplace=True)

    return merged


def left_join(base, supplement, label):
    """
    Left-join supplement onto base on KEY, bringing only new columns.

    Protects against many-to-many by asserting row count is unchanged.
    """
    n_before = len(base)
    new_cols = [c for c in supplement.columns if c == KEY or c not in base.columns]
    new_cols = list(dict.fromkeys(new_cols))

    merged = base.merge(supplement[new_cols], on=KEY, how="left")

    if len(merged) != n_before:
        sys.exit(
            f"FATAL: many-to-many merge detected merging {label}. "
            f"Row count changed {n_before} → {len(merged)}. "
            "Assert uniqueness in the supplement file before proceeding."
        )

    matched = merged[KEY].isin(supplement[KEY]).sum()
    print(f"  {label}: {matched}/{n_before} master participants matched")
    return merged


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("05_merge_master_dataset.py (MR1)")
    print("=" * 60)

    # ── Load and validate daily diary ─────────────────────────────────────────
    if not os.path.isfile(DIARY_FILE):
        sys.exit(f"FATAL: {DIARY_FILE} not found. Run preprocessing scripts 01-04 first.")
    diary = pd.read_csv(DIARY_FILE)
    if KEY not in diary.columns:
        sys.exit(f"FATAL: {KEY} column not found in {DIARY_FILE}.")
    if diary[KEY].isna().any():
        sys.exit(
            f"FATAL: {int(diary[KEY].isna().sum())} missing {KEY} in "
            "daily_diary_processed.csv."
        )
    assert_unique(diary, "daily_diary_processed.csv")
    diary_ids = set(diary[KEY])
    print(f"Daily diary participants: {len(diary)}")

    # ── Load and validate P5 covariates ──────────────────────────────────────
    if not os.path.isfile(COV_FILE):
        sys.exit(f"FATAL: {COV_FILE} not found. Run 04_construct_covariates.py first.")
    cov = pd.read_csv(COV_FILE)
    if KEY not in cov.columns:
        sys.exit(f"FATAL: {KEY} column not found in {COV_FILE}.")
    if cov[KEY].isna().any():
        sys.exit(
            f"FATAL: {int(cov[KEY].isna().sum())} missing {KEY} in "
            "mr1p5_covariates.csv."
        )
    assert_unique(cov, "mr1p5_covariates.csv")
    cov_ids = set(cov[KEY])
    print(f"P5 covariate participants: {len(cov)}")

    # ── Outer-join diary and P5 covariates → diary ∪ P5 universe ─────────────
    print("\nBuilding diary ∪ P5 master (outer join):")
    expected_union = diary_ids | cov_ids
    merged = merge_combine(diary, cov, key=KEY)

    # Structural validation
    assert_unique(merged, "diary ∪ P5 merged result")
    merged_ids = set(merged[KEY])
    if merged_ids != expected_union:
        sys.exit(
            f"FATAL: merged ID set does not equal the union of diary and P5 IDs. "
            f"Extra: {len(merged_ids - expected_union)}, "
            f"Missing: {len(expected_union - merged_ids)}"
        )
    if len(merged) != len(expected_union):
        sys.exit(
            f"FATAL: merged row count ({len(merged)}) ≠ union size "
            f"({len(expected_union)})."
        )
    print(f"  diary ∪ P5: {len(merged)} participants")

    # ── Load and validate MKER1 supplement (required) ────────────────────────
    if not os.path.isfile(MKER1_FILE):
        sys.exit(f"FATAL: {MKER1_FILE} not found. Run 01_harmonize_ids.py first.")
    mker1 = pd.read_csv(MKER1_FILE)
    if KEY not in mker1.columns:
        sys.exit(f"FATAL: {KEY} column not found in {MKER1_FILE}.")
    if mker1[KEY].isna().any():
        sys.exit(
            f"FATAL: {int(mker1[KEY].isna().sum())} missing {KEY} in mker1_ids.csv."
        )
    assert_unique(mker1, "mker1_ids.csv")
    print(f"MKER1 participants: {len(mker1)}")

    print("\nMerging MKER1 supplement (left join — does not expand analytic universe):")
    merged = left_join(merged, mker1, "mker1_ids.csv (supplement)")

    # Validate supplement merge did not add participants
    if set(merged[KEY]) != merged_ids:
        sys.exit("FATAL: MKER1 supplement merge changed the master ID set — abort.")

    # ── Refresh is_mker1 for all master participants ───────────────────────────
    merged["is_mker1"] = merged[KEY].isin(mker1[KEY]).astype(int)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_in_diary = merged[KEY].isin(diary_ids).sum()
    n_in_p5    = merged[KEY].isin(cov_ids).sum()
    n_in_both  = merged[KEY].isin(diary_ids & cov_ids).sum()
    n_mker1    = (merged["is_mker1"] == 1).sum()

    print(f"\nMerged dataset summary:")
    print(f"  Total rows (diary ∪ P5):   {len(merged)}")
    print(f"  Present in diary:           {n_in_diary}")
    print(f"  Present in P5 covariates:   {n_in_p5}")
    print(f"  Present in both:            {n_in_both}")
    print(f"  Flagged is_mker1=1:         {n_mker1}")
    print(f"  Columns:                    {merged.shape[1]}")

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMerged dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
