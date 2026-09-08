#!/usr/bin/env python3
"""
combine_condition_fd.py  (MR1 / MIDUS Refresher)

Combine per-subject condition FD CSVs into a single wide-format file.

Run on Sherlock after extract_condition_fd.sh completes for all subjects.

Input:
    /scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/condition_fd/
        <subid>_condition_fd.csv

Output:
    /scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/condition_fd_wide.csv
        One row per subject. Columns: MIDUSID, fd_neg_mean, fd_neu_mean, fd_pos_mean
        (mean FD across the full image->ITI->face window, averaged across runs).
"""

import sys
import pandas as pd
from pathlib import Path

in_dir   = Path("/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/condition_fd")
out_file = Path("/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/condition_fd_wide.csv")

EXPECTED_VALENCES = {'neg', 'neu', 'pos'}
REQUIRED_COLS     = ['subject', 'run', 'valence', 'mean_fd_condition']

if not in_dir.is_dir():
    print(f"ERROR: input directory not found: {in_dir}")
    sys.exit(1)

csv_files = sorted(in_dir.glob("*_condition_fd.csv"))
if not csv_files:
    print(f"ERROR: no condition FD CSVs found in {in_dir}")
    sys.exit(1)

print(f"Found {len(csv_files)} subject files")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    print(f"ERROR: missing required column(s): {missing_cols}")
    sys.exit(1)

unexpected_valences = set(df['valence'].unique()) - EXPECTED_VALENCES
if unexpected_valences:
    print(f"ERROR: unexpected valence value(s): {sorted(unexpected_valences)}")
    sys.exit(1)

dup_mask = df.duplicated(subset=['subject', 'run', 'valence'])
if dup_mask.any():
    print(f"ERROR: {int(dup_mask.sum())} duplicate subject/run/valence row(s)")
    sys.exit(1)

df["MIDUSID"] = df["subject"].str.replace("sub-", "", regex=False).astype(int)

# Average across runs per subject x valence
by_valence = (
    df.groupby(["MIDUSID", "valence"])["mean_fd_condition"]
    .mean()
    .reset_index()
)

# Pivot to wide: one column per valence
wide = by_valence.pivot(index="MIDUSID", columns="valence", values="mean_fd_condition")
wide.columns = [f"fd_{v}_mean" for v in wide.columns]
wide = wide.reset_index()

wide.to_csv(out_file, index=False)
print(f"Saved: {out_file}  ({len(wide)} subjects, {wide.shape[1]} columns)")
print(f"Columns: {wide.columns.tolist()}")
