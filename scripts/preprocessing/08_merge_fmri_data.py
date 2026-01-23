#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_merge_fmri_data.py

Merge fMRI-derived participant-level measures into the cleaned MIDUS master dataset.

Inputs:
- Cleaned master dataset from 06_clean_merged_data.py
- fMRI-derived CSV files in data/fMRI/

fMRI files handled:
- betaSeries_neg_vs_neu.csv (1 row per participant)
- positive_persistence_cross_run.csv (3 rows per participant: hemisphere)
- negative_persistence_cross_run.csv (3 rows per participant: hemisphere)
- negative_persistence_concat.csv (3 rows per participant: hemisphere)
- fd_summary.csv (row per run)

Outputs:
- data/processed/merged_with_fmri.csv

Notes:
- All merges are left joins on M2ID
- No participant exclusions are applied
- Multi-row fMRI files are aggregated or pivoted to one row per participant
"""

import pandas as pd
import os

# =========================
# Paths
# =========================
PROCESSED_DIR = "data/processed"
FMRI_DIR = "data/fMRI"

MASTER_FILE = os.path.join(PROCESSED_DIR, "midus_merged_clean.csv")
OUT_FILE = os.path.join(PROCESSED_DIR, "midus_with_fmri.csv")

# fMRI files
BETA_FILE = os.path.join(FMRI_DIR, "betaSeries_neg_vs_neu.csv")
NEG_PERSIST_CONCAT_FILE = os.path.join(FMRI_DIR, "negative_persistence_concat.csv")
NEG_PERSIST_CROSS_FILE = os.path.join(FMRI_DIR, "negative_persistence_cross_run.csv")
POS_PERSIST_CROSS_FILE = os.path.join(FMRI_DIR, "positive_persistence_cross_run.csv")
FD_FILE = os.path.join(FMRI_DIR, "fd_summary.csv")

# =========================
# Helper functions
# =========================
def pivot_persistence_long(
    df,
    value_cols,
    prefix
):
    """
    Pivot long-format persistence data (hemisphere rows) to wide format.

    Parameters
    ----------
    df : DataFrame
        Must contain M2ID and hemisphere columns
    value_cols : list of str
        Columns to pivot (e.g., ['r', 'z'] or ['mean_r', 'median_r'])
    prefix : str
        Prefix for output column names

    Returns
    -------
    DataFrame with one row per M2ID
    """
    wide_dfs = []

    for col in value_cols:
        wide = (
            df.pivot(index="M2ID", columns="hemisphere", values=col)
            .add_prefix(f"{prefix}_{col}_")
            .reset_index()
        )
        wide_dfs.append(wide)

    out = wide_dfs[0]
    for w in wide_dfs[1:]:
        out = out.merge(w, on="M2ID", how="outer")

    return out


# =========================
# Load master dataset
# =========================
master = pd.read_csv(MASTER_FILE)
print(f"Loaded master dataset: {master.shape[0]} rows")

# =========================
# Beta-series connectivity
# =========================
beta = pd.read_csv(BETA_FILE)

beta = beta.rename(columns={
    "l_amyg-vmPFC": "conn_l_amyg_vmPFC_neg_vs_neu",
    "l_amyg-sgACC": "conn_l_amyg_sgACC_neg_vs_neu",
    "r_amyg-vmPFC": "conn_r_amyg_vmPFC_neg_vs_neu",
    "r_amyg-sgACC": "conn_r_amyg_sgACC_neg_vs_neu"
})

# =========================
# Negative persistence (concat)
# =========================
neg_concat = pd.read_csv(NEG_PERSIST_CONCAT_FILE)

neg_concat_wide = pivot_persistence_long(
    df=neg_concat,
    value_cols=["r", "z", "n_vox"],
    prefix="neg_persist_concat"
)

# =========================
# Negative persistence (cross-run)
# =========================
neg_cross = pd.read_csv(NEG_PERSIST_CROSS_FILE)

neg_cross_wide = pivot_persistence_long(
    df=neg_cross,
    value_cols=["mean_r", "median_r", "std_r", "n_pairs"],
    prefix="neg_persist_crossrun"
)

# =========================
# Positive persistence (cross-run)
# =========================
pos_cross = pd.read_csv(POS_PERSIST_CROSS_FILE)

pos_cross_wide = pivot_persistence_long(
    df=pos_cross,
    value_cols=["mean_r", "median_r", "std_r", "n_pairs"],
    prefix="pos_persist_crossrun"
)

# =========================
# Framewise displacement (FD)
# =========================
fd = pd.read_csv(FD_FILE)

# Extract M2ID from subject column (format: sub-M2ID)
fd["M2ID"] = fd["subject"].str.replace("sub-", "", regex=False).astype(int)

fd_summary = (
    fd.groupby("M2ID")
    .agg(
        fd_mean_across_runs=("mean_fd", "mean"),
        fd_max_across_runs=("mean_fd", "max"),
        fd_any_flagged=("flagged", "max"),
        fd_n_runs=("mean_fd", "count")
    )
    .reset_index()
)

# =========================
# Merge all fMRI data
# =========================
merged = master.merge(beta, on="M2ID", how="left")
merged = merged.merge(neg_concat_wide, on="M2ID", how="left")
merged = merged.merge(neg_cross_wide, on="M2ID", how="left")
merged = merged.merge(pos_cross_wide, on="M2ID", how="left")
merged = merged.merge(fd_summary, on="M2ID", how="left")

# =========================
# Availability flags
# =========================
merged["has_beta_series"] = merged[
    [
        "conn_l_amyg_vmPFC_neg_vs_neu",
        "conn_r_amyg_vmPFC_neg_vs_neu"
    ]
].notna().any(axis=1).astype(int)

merged["has_neg_persistence"] = merged.filter(
    like="neg_persist"
).notna().any(axis=1).astype(int)

merged["has_pos_persistence"] = merged.filter(
    like="pos_persist"
).notna().any(axis=1).astype(int)

merged["has_fd_data"] = merged["fd_n_runs"].notna().astype(int)

merged["has_imaging_data"] = (
    merged[[
        "has_beta_series",
        "has_neg_persistence",
        "has_pos_persistence"
    ]].any(axis=1)
).astype(int)

# =========================
# Save output
# =========================
merged.to_csv(OUT_FILE, index=False)
print(f"Merged dataset with fMRI data saved to: {OUT_FILE}")
print(f"Final dataset shape: {merged.shape}")
