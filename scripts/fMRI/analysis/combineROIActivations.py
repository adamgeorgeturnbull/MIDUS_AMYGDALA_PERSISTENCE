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

input_dir = Path("/scratch/groups/fvlin/MIDUS/M3/ROI_activations_output")
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
    # Average across runs — only include subjects with all 3 runs present
    if "run" in combined.columns:
        run_counts = combined.groupby("M2ID")["run"].count()
        complete = run_counts[run_counts == 3].index
        n_incomplete = (run_counts < 3).sum()
        if n_incomplete > 0:
            print(f"WARNING: excluding {n_incomplete} subject(s) with fewer than 3 valid runs: "
                  f"{sorted(run_counts[run_counts < 3].index.tolist())}")
        combined = combined[combined["M2ID"].isin(complete)]
        combined = combined.drop(columns="run").groupby("M2ID").mean().reset_index()
    combined.to_csv(output_file, index=False)
    print(f"Saved {len(combined)} subjects to {output_file}")
    print(f"Columns: {list(combined.columns)}")
