#!/usr/bin/env python3
"""
combineBTS_LSS.py

Combine per-subject LSS beta-series connectivity CSVs into a single
group-level file with harmonized MIDUS IDs.

Run this locally (or as a short non-array Sherlock job) after
runBStaskFC_LSS.sh completes for all subjects.

Input:
    BetaSeries_LSS_output/
        sub-XXXXX/
            <subid>_betaSeries_LSS_ROI_all_conditions.csv

Output:
    BetaSeries_LSS_output/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv
"""

import pandas as pd
from pathlib import Path

input_dir = Path("/scratch/groups/fvlin/MIDUS/M3_stc_rerun/BetaSeries_LSS_output")
output_file = input_dir / "all_subjects_betaSeries_LSS_all_conditions_M2ID.csv"

all_dfs = []

for sub_dir in sorted(input_dir.iterdir()):
    if not sub_dir.is_dir():
        continue
    csv_files = list(sub_dir.glob("*betaSeries_LSS_ROI_all_conditions.csv"))
    if not csv_files:
        continue
    df = pd.read_csv(csv_files[0])
    if 'subid' in df.columns:
        df['M2ID'] = df['subid'].str.replace('^sub-', '', regex=True)
        df = df.drop(columns='subid')
    all_dfs.append(df)

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"Saved {len(combined)} subjects to {output_file}")
    print(f"Columns: {list(combined.columns)}")
else:
    print("No LSS output files found.")
