#!/usr/bin/env python3
"""
combine_persistence.py  (MR1 / MIDUS Refresher)

Bridge step: convert run_cross_corr.py summary output into the analysis-ready
persistence CSVs that 09_merge_fmri_data.py expects.

run_cross_corr.py writes, in voxelwise_betas_summary/:
    results_summary_neg_image_vs_neg_face.csv
    results_summary_pos_image_vs_pos_face.csv
each with columns: subject, hemisphere, mean_r, median_r, std_r, n_pairs

This script renames `subject` -> `MIDUSID` (stripping the BIDS `sub-` prefix) and
writes the two files the merge step reads:
    negative_persistence_cross_run.csv
    positive_persistence_cross_run.csv
with columns: MIDUSID, hemisphere, mean_r, median_r, std_r, n_pairs

MR1 participant IDs: the `sub-XXXX` BIDS label is the MIDUSID (confirmed by
matching the behavioral data variable extract, which uses MIDUSID as primary key).

Run on Sherlock after run_cross_corr.py, or locally after scp-ing
voxelwise_betas_summary/ back. Override SUMMARY_DIR / OUT_DIR as needed.
"""

import os
from pathlib import Path
import pandas as pd

SUMMARY_DIR = Path(os.environ.get(
    "MR1_SUMMARY_DIR",
    "/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/voxelwise_betas_summary",
))
OUT_DIR = Path(os.environ.get("MR1_FMRI_OUT_DIR", str(SUMMARY_DIR)))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("results_summary_neg_image_vs_neg_face.csv", "negative_persistence_cross_run.csv"),
    ("results_summary_pos_image_vs_pos_face.csv", "positive_persistence_cross_run.csv"),
]

KEEP_COLS = ["MIDUSID", "hemisphere", "mean_r", "median_r", "std_r", "n_pairs"]

for src_name, out_name in PAIRS:
    src = SUMMARY_DIR / src_name
    if not src.exists():
        print(f"MISSING: {src} — did run_cross_corr.py run? skipping")
        continue
    df = pd.read_csv(src)
    df = df.rename(columns={"subject": "MIDUSID"})
    df["MIDUSID"] = df["MIDUSID"].astype(str).str.replace("^sub-", "", regex=True).astype(int)
    df = df[[c for c in KEEP_COLS if c in df.columns]]
    out = OUT_DIR / out_name
    df.to_csv(out, index=False)
    print(f"Wrote {out}  ({df['MIDUSID'].nunique()} subjects, {len(df)} rows)")

print("Done.")
