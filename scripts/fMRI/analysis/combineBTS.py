#!/usr/bin/env python3
"""
combineBTS.py

Combine per-subject beta-series connectivity CSVs into a single group-level
file with harmonized MIDUS IDs.

Each subject's beta-series analysis (run via runBStaskFC.sh / nilearn) produces a CSV with
ROI-level connectivity estimates for the negative > neutral contrast. This
script concatenates all subjects into one file and converts BIDS subject IDs
(sub-XXXXX) to MIDUS M2ID format for merging with behavioral data.

Input:
    BetaSeries_output/
        sub-XXXXX/
            *betaSeries_ROI_contrast_neg_vs_neu.csv  (one per subject)

Output:
    all_subjects_betaSeries_M2ID.csv - Combined CSV with columns from the
        beta-series analysis plus M2ID (BIDS 'subid' prefix stripped)

@author: aturnbu2
"""

import pandas as pd
from pathlib import Path

# Directory containing per-subject beta-series output folders
input_dir = Path("/scratch/groups/fvlin/MIDUS/BetaSeries_output")

output_file = input_dir / "all_subjects_betaSeries_M2ID.csv"

all_dfs = []

for sub_dir in input_dir.iterdir():
    if sub_dir.is_dir():
        # Each subject folder contains one neg_vs_neu contrast CSV
        csv_files = list(sub_dir.glob("*betaSeries_ROI_contrast_neg_vs_neu.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            # Convert BIDS subject ID (sub-XXXXX) to numeric M2ID
            if 'subid' in df.columns:
                df['M2ID'] = df['subid'].str.replace('^sub-', '', regex=True)
                df = df.drop(columns='subid')
            all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

combined_df.to_csv(output_file, index=False)
print(f"Saved combined beta-series CSV to {output_file}")
