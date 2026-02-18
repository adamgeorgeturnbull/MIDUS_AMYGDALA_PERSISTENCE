#!/usr/bin/env python3
"""
combineBTS.py

Combine per-subject beta-series connectivity CSVs into single group-level
files with harmonized MIDUS IDs.

Produces two output files:
1. all_subjects_betaSeries_M2ID.csv - Backward-compatible neg > neu contrast only
2. all_subjects_betaSeries_all_conditions_M2ID.csv - Per-condition (neg, neu, pos)
   Fisher z correlations plus contrasts

Input:
    BetaSeries_output/
        sub-XXXXX/
            *betaSeries_ROI_contrast_neg_vs_neu.csv
            *betaSeries_ROI_all_conditions.csv

@author: aturnbu2
"""

import pandas as pd
from pathlib import Path

# Directory containing per-subject beta-series output folders
input_dir = Path("/scratch/groups/fvlin/MIDUS/M3/BetaSeries_output")


def combine_files(input_dir, glob_pattern, output_file):
    """Combine per-subject CSVs matching glob_pattern into one file."""
    all_dfs = []

    for sub_dir in sorted(input_dir.iterdir()):
        if sub_dir.is_dir():
            csv_files = list(sub_dir.glob(glob_pattern))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                # Convert BIDS subject ID (sub-XXXXX) to numeric M2ID
                if 'subid' in df.columns:
                    df['M2ID'] = df['subid'].str.replace('^sub-', '', regex=True)
                    df = df.drop(columns='subid')
                all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Saved {len(combined_df)} subjects to {output_file}")
        print(f"  Columns: {list(combined_df.columns)}")
    else:
        print(f"No files matching '{glob_pattern}' found.")


# 1) Backward-compatible: neg > neu contrast only
combine_files(
    input_dir,
    "*betaSeries_ROI_contrast_neg_vs_neu.csv",
    input_dir / "all_subjects_betaSeries_M2ID.csv",
)

# 2) New: per-condition correlations + contrasts
combine_files(
    input_dir,
    "*betaSeries_ROI_all_conditions.csv",
    input_dir / "all_subjects_betaSeries_all_conditions_M2ID.csv",
)
