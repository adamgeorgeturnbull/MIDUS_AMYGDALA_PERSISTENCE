#!/usr/bin/env python3
"""
combine_condition_fd.py

Combine per-subject condition FD CSVs into a single wide-format file.

Run on Sherlock after extract_condition_fd.sh completes for all subjects.

Input:
    /scratch/groups/fvlin/MIDUS/M3/condition_fd/
        <subid>_condition_fd.csv

Output:
    /scratch/groups/fvlin/MIDUS/M3/condition_fd_wide.csv
        One row per subject. Columns: M2ID, fd_neg_mean, fd_neu_mean, fd_pos_mean
        (mean FD across the full image→ITI→face window, averaged across runs).
"""

import sys
import pandas as pd
from pathlib import Path

in_dir   = Path("/scratch/groups/fvlin/MIDUS/M3_stc_rerun/condition_fd")
out_file = Path("/scratch/groups/fvlin/MIDUS/M3_stc_rerun/condition_fd_wide.csv")

if not in_dir.is_dir():
    print(f"ERROR: input directory not found: {in_dir}")
    sys.exit(1)

csv_files = sorted(in_dir.glob("*_condition_fd.csv"))
if not csv_files:
    raise FileNotFoundError(f"No condition FD CSVs found in {in_dir}")

print(f"Found {len(csv_files)} subject files")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df["M2ID"] = df["subject"].str.replace("sub-", "", regex=False).astype(int)

# Average across runs per subject x valence
by_valence = (
    df.groupby(["M2ID", "valence"])["mean_fd_condition"]
    .mean()
    .reset_index()
)

# Pivot to wide: one column per valence
wide = by_valence.pivot(index="M2ID", columns="valence", values="mean_fd_condition")
wide.columns = [f"fd_{v}_mean" for v in wide.columns]
wide = wide.reset_index()

wide.to_csv(out_file, index=False)
print(f"Saved: {out_file}  ({len(wide)} subjects, {wide.shape[1]} columns)")
print(f"Columns: {wide.columns.tolist()}")
