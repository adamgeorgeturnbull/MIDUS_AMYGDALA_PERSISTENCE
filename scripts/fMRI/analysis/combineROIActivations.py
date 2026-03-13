#!/usr/bin/env python3
"""
combineROIActivations.py

Combine per-subject ROI activation CSVs into a single group-level file.

Run after extract_roi_activations.sh completes on Sherlock.
Output is used by 09_merge_fmri_data.py.

Input:
    ROI_activations_output/<subid>/<subid>_roi_activations.csv

Output:
    ROI_activations_output/all_subjects_roi_activations.csv
"""

import pandas as pd
from pathlib import Path

input_dir = Path("/scratch/groups/fvlin/MIDUS/ROI_activations_output")
output_file = input_dir / "all_subjects_roi_activations.csv"

all_dfs = []

for sub_dir in sorted(input_dir.iterdir()):
    if not sub_dir.is_dir():
        continue
    csv_files = list(sub_dir.glob("*_roi_activations.csv"))
    if not csv_files:
        continue
    df = pd.read_csv(csv_files[0])
    if "subid" in df.columns:
        df["M2ID"] = df["subid"].str.replace("sub-", "", regex=False).astype(int)
        df = df.drop(columns="subid")
    all_dfs.append(df)

if not all_dfs:
    print("No ROI activation files found.")
else:
    combined = pd.concat(all_dfs, ignore_index=True)
    # Average across runs
    if "run" in combined.columns:
        combined = combined.drop(columns="run").groupby("M2ID").mean().reset_index()
    combined.to_csv(output_file, index=False)
    print(f"Saved {len(combined)} subjects to {output_file}")
    print(f"Columns: {list(combined.columns)}")
