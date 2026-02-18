#!/usr/bin/env python3
"""
check_mke2_data_consistency.py

Compare the MKE2 variables currently used in the project against
the newly downloaded ICPSR 37120 dataset to verify data consistency.

Checks:
  1. Sample size (N) for SAMPLMAJ == 13 in new data vs N in old data
  2. M2ID overlap between old and new
  3. Value-level consistency for every shared variable (merged on M2ID)

Inputs:
  - data/raw/MKE2_variables.csv                          (old/current)
  - data/raw/14874535/ICPSR_37120/DS0001/37120-0001-Data-REST.tsv  (new)

Run from project root:
    python scripts/check_mke2_data_consistency.py
"""

import os
import sys

import numpy as np
import pandas as pd

# ============================================================================
# Paths
# ============================================================================
OLD_FILE = "data/raw/MKE2_variables.csv"
NEW_FILE = "data/raw/14874535/ICPSR_37120/DS0001/37120-0001-Data-REST.tsv"

# Variables we use from MKE2 (columns in the old file)
USED_VARS = [
    "M2ID",
    "SAMPLMAJ",
    "CACRSEX",
    "CACB1",
    "CACF7A",
    "CACF1",
    "CACRAGE",
    "CACIDATE_YR",
    "CACIDATE_MO",
    "C5SPGP",
    "C5SPGN",
]

# SAMPLMAJ value that identifies MKE2 participants in the new ICPSR file
NEW_SAMPLMAJ = 13

# ============================================================================
# Load data
# ============================================================================
print("=" * 70)
print("MKE2 Data Consistency Check")
print("=" * 70)

if not os.path.exists(OLD_FILE):
    sys.exit(f"ERROR: Old file not found: {OLD_FILE}")
if not os.path.exists(NEW_FILE):
    sys.exit(f"ERROR: New file not found: {NEW_FILE}")

old_raw = pd.read_csv(OLD_FILE)
new_full = pd.read_csv(NEW_FILE, sep="\t")

print(f"\nOld file: {OLD_FILE}")
print(f"  Total rows (raw): {len(old_raw)}")
print(f"  Columns: {list(old_raw.columns)}")

# Filter old data to rows that actually have MKE2 data (non-null CAC* columns).
# The old file may contain IDs from other datasets with empty rows.
mke2_data_cols = [c for c in old_raw.columns if c.startswith("CAC")]
print(f"  MKE2-specific columns (CAC*): {mke2_data_cols}")
old = old_raw.dropna(subset=mke2_data_cols, how="all").copy()
n_dropped = len(old_raw) - len(old)
if n_dropped > 0:
    print(f"  Dropped {n_dropped} rows with all-empty MKE2 columns")
print(f"  Rows with MKE2 data: {len(old)}")

print(f"\nNew file: {NEW_FILE}")
print(f"  Total rows: {len(new_full)}")
print(f"  Total columns: {len(new_full.columns)}")

# ============================================================================
# Filter new data to MKE2 sample
# ============================================================================
# Find SAMPLMAJ column (case-insensitive)
samplmaj_col = None
for col in new_full.columns:
    if col.upper() == "SAMPLMAJ":
        samplmaj_col = col
        break

if samplmaj_col is None:
    print("\nWARNING: SAMPLMAJ column not found in new file.")
    print("Available columns containing 'SAMPL':")
    for col in new_full.columns:
        if "SAMPL" in col.upper():
            print(f"  {col}: unique values = {sorted(new_full[col].dropna().unique())}")
    sys.exit("Cannot proceed without SAMPLMAJ column.")

print(f"\nSAMPLMAJ column in new file: '{samplmaj_col}'")
print(f"  Unique values: {sorted(new_full[samplmaj_col].dropna().unique())}")

new = new_full[new_full[samplmaj_col] == NEW_SAMPLMAJ].copy()
print(f"  Rows with SAMPLMAJ == {NEW_SAMPLMAJ}: {len(new)}")

# ============================================================================
# Check 1: Sample size
# ============================================================================
print("\n" + "=" * 70)
print("CHECK 1: Sample Size")
print("=" * 70)
print(f"  Old MKE2 data:  N = {len(old)}")
print(f"  New (SAMPLMAJ == {NEW_SAMPLMAJ}): N = {len(new)}")
if len(old) == len(new):
    print("  PASS: Sample sizes match")
else:
    print(f"  MISMATCH: Difference of {abs(len(old) - len(new))}")

# ============================================================================
# Check 2: M2ID overlap
# ============================================================================
print("\n" + "=" * 70)
print("CHECK 2: M2ID Overlap")
print("=" * 70)

# Find M2ID column in new data (case-insensitive)
m2id_col = None
for col in new.columns:
    if col.upper() == "M2ID":
        m2id_col = col
        break

if m2id_col is None:
    print("WARNING: M2ID column not found in new file.")
    print("Columns containing 'M2' or 'ID':")
    for col in new.columns:
        if "M2" in col.upper() or "ID" in col.upper():
            print(f"  {col}")
    sys.exit("Cannot proceed without M2ID column.")

old_ids = set(old["M2ID"].astype(int))
new_ids = set(new[m2id_col].astype(int))

in_both = old_ids & new_ids
in_old_only = old_ids - new_ids
in_new_only = new_ids - old_ids

print(f"  Old M2IDs: {len(old_ids)}")
print(f"  New M2IDs: {len(new_ids)}")
print(f"  In both:   {len(in_both)}")

if in_old_only:
    print(f"  In old only ({len(in_old_only)}): {sorted(in_old_only)}")
if in_new_only:
    print(f"  In new only ({len(in_new_only)}): {sorted(in_new_only)}")

if old_ids == new_ids:
    print("  PASS: M2ID sets are identical")
else:
    print("  MISMATCH: M2ID sets differ")

# ============================================================================
# Check 3: Variable-level consistency
# ============================================================================
print("\n" + "=" * 70)
print("CHECK 3: Variable-Level Consistency")
print("=" * 70)

# Build column mapping (case-insensitive match)
new_col_lookup = {col.upper(): col for col in new.columns}

# Merge on M2ID for matched comparison
old_merge = old.copy()
old_merge["M2ID"] = old_merge["M2ID"].astype(int)
new_merge = new.copy()
new_merge[m2id_col] = new_merge[m2id_col].astype(int)
new_merge = new_merge.rename(columns={m2id_col: "M2ID"})

merged = old_merge.merge(new_merge, on="M2ID", how="inner", suffixes=("_old", "_new"))
print(f"\nMerged on M2ID: {len(merged)} matched rows\n")

all_pass = True
for var in USED_VARS:
    if var == "M2ID":
        continue  # already checked

    # Find matching column in new data
    new_var = new_col_lookup.get(var.upper())
    if new_var is None:
        print(f"  {var:15s}  MISSING in new file")
        all_pass = False
        continue

    # Get old and new columns from merged dataframe
    old_col_name = f"{var}_old" if f"{var}_old" in merged.columns else var
    new_col_name = f"{new_var}_new" if f"{new_var}_new" in merged.columns else new_var

    if old_col_name not in merged.columns:
        print(f"  {var:15s}  Column '{old_col_name}' not in merged (check suffix)")
        all_pass = False
        continue
    if new_col_name not in merged.columns:
        print(f"  {var:15s}  Column '{new_col_name}' not in merged (check suffix)")
        all_pass = False
        continue

    old_vals = merged[old_col_name]
    new_vals = merged[new_col_name]

    # Check for NaN patterns
    old_nan = old_vals.isna()
    new_nan = new_vals.isna()
    nan_mismatch = (old_nan != new_nan).sum()

    # Compare non-NaN values
    both_valid = ~old_nan & ~new_nan
    if both_valid.sum() == 0:
        print(f"  {var:15s}  No valid pairs to compare")
        continue

    # Numeric comparison with tolerance for floats
    try:
        old_num = old_vals[both_valid].astype(float)
        new_num = new_vals[both_valid].astype(float)
        close = np.isclose(old_num, new_num, atol=1e-6, rtol=1e-6)
        n_match = close.sum()
        n_total = both_valid.sum()
        n_diff = n_total - n_match
    except (ValueError, TypeError):
        # Fall back to string comparison
        old_str = old_vals[both_valid].astype(str)
        new_str = new_vals[both_valid].astype(str)
        n_match = (old_str == new_str).sum()
        n_total = both_valid.sum()
        n_diff = n_total - n_match

    status = "PASS" if n_diff == 0 and nan_mismatch == 0 else "MISMATCH"
    if status == "MISMATCH":
        all_pass = False

    print(f"  {var:15s}  {status:8s}  "
          f"({n_match}/{n_total} values match"
          f"{f', {nan_mismatch} NaN mismatches' if nan_mismatch > 0 else ''})")

    # Show mismatched values if any
    if n_diff > 0:
        try:
            diff_mask = both_valid & ~np.isclose(
                old_vals.astype(float), new_vals.astype(float),
                atol=1e-6, rtol=1e-6, equal_nan=True
            )
        except (ValueError, TypeError):
            diff_mask = both_valid & (old_vals.astype(str) != new_vals.astype(str))

        diff_rows = merged.loc[diff_mask, ["M2ID", old_col_name, new_col_name]]
        print(f"{'':19s}Differing rows:")
        for _, r in diff_rows.iterrows():
            print(f"{'':19s}  M2ID {int(r['M2ID'])}: "
                  f"old={r[old_col_name]}, new={r[new_col_name]}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if all_pass and len(old) == len(new) and old_ids == new_ids:
    print("ALL CHECKS PASSED: Old and new MKE2 data are consistent.")
else:
    print("SOME CHECKS FAILED: Review mismatches above.")
print("=" * 70)
