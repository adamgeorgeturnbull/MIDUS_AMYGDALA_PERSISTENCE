#!/usr/bin/env python3
"""
08_merge_fmri_data.py

Merge fMRI-derived participant-level measures into the cleaned MIDUS master dataset.

Inputs:
- data/processed/midus_merged_clean.csv (cleaned master dataset)
- data/fMRI/betaSeries_neg_vs_neu.csv (beta-series connectivity, 1 row/participant)
- data/fMRI/negative_persistence_concat.csv (negative persistence, 3 rows/participant)
- data/fMRI/negative_persistence_cross_run.csv (neg persistence cross-run, 3 rows/participant)
- data/fMRI/positive_persistence_cross_run.csv (pos persistence cross-run, 3 rows/participant)
- data/fMRI/fd_summary.csv (framewise displacement, multiple rows/participant)
- data/fMRI/fmri_qc_processed.csv (quality control flags, 1 row/participant)

Outputs:
- data/processed/midus_with_fmri.csv (master dataset + fMRI measures + availability flags)

Notes:
- All merges are left joins on M2ID (no participant exclusions)
- Multi-row fMRI files are pivoted/aggregated to one row per participant
- Availability flags indicate which fMRI modalities are available per participant

Run from project root directory.
"""

import os

import pandas as pd

# ============================================================================
# Paths
# ============================================================================
PROCESSED_DIR = "data/processed"
FMRI_DIR = "data/fMRI"

# Input files
MASTER_FILE = os.path.join(PROCESSED_DIR, "midus_merged_clean.csv")

# fMRI input files
BETA_FILE = os.path.join(FMRI_DIR, "betaSeries_neg_vs_neu.csv")
NEG_PERSIST_CONCAT_FILE = os.path.join(FMRI_DIR, "negative_persistence_concat.csv")
NEG_PERSIST_CROSS_FILE = os.path.join(FMRI_DIR, "negative_persistence_cross_run.csv")
POS_PERSIST_CROSS_FILE = os.path.join(FMRI_DIR, "positive_persistence_cross_run.csv")
FD_FILE = os.path.join(FMRI_DIR, "fd_summary.csv")
QC_FILE = os.path.join(FMRI_DIR, "fmri_qc_processed.csv")

# Output file
OUT_FILE = os.path.join(PROCESSED_DIR, "midus_with_fmri.csv")

# ============================================================================
# Helper Functions
# ============================================================================
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
    # Standardize hemisphere values (BI → bilateral)
    df = df.copy()
    df["hemisphere"] = df["hemisphere"].replace({"BI": "bilateral"})

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


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    # Ensure output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ========================================================================
    # Load Master Dataset
    # ========================================================================
    master = pd.read_csv(MASTER_FILE)
    print(f"Loaded master dataset: {master.shape[0]} rows")

    # ========================================================================
    # Load and Process fMRI Data
    # ========================================================================
    # Beta-series connectivity
    beta = pd.read_csv(BETA_FILE)
    beta = beta.rename(columns={
        "l_amyg-vmPFC": "conn_l_amyg_vmPFC_neg_vs_neu",
        "l_amyg-sgACC": "conn_l_amyg_sgACC_neg_vs_neu",
        "r_amyg-vmPFC": "conn_r_amyg_vmPFC_neg_vs_neu",
        "r_amyg-sgACC": "conn_r_amyg_sgACC_neg_vs_neu"
    })

    # Negative persistence (concatenated runs)
    neg_concat = pd.read_csv(NEG_PERSIST_CONCAT_FILE)
    neg_concat_wide = pivot_persistence_long(
        df=neg_concat,
        value_cols=["r", "z", "n_vox"],
        prefix="neg_persist_concat"
    )

    # Negative persistence (cross-run)
    neg_cross = pd.read_csv(NEG_PERSIST_CROSS_FILE)
    neg_cross_wide = pivot_persistence_long(
        df=neg_cross,
        value_cols=["mean_r", "median_r", "std_r", "n_pairs"],
        prefix="neg_persist_crossrun"
    )

    # Positive persistence (cross-run)
    pos_cross = pd.read_csv(POS_PERSIST_CROSS_FILE)
    pos_cross_wide = pivot_persistence_long(
        df=pos_cross,
        value_cols=["mean_r", "median_r", "std_r", "n_pairs"],
        prefix="pos_persist_crossrun"
    )

    # Framewise displacement (FD) - aggregate across runs
    fd = pd.read_csv(FD_FILE)
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

    # Quality control (QC) flags
    qc = pd.read_csv(QC_FILE)

    # ========================================================================
    # Merge All fMRI Data with Master Dataset
    # ========================================================================
    merged = master.merge(beta, on="M2ID", how="left")
    merged = merged.merge(neg_concat_wide, on="M2ID", how="left")
    merged = merged.merge(neg_cross_wide, on="M2ID", how="left")
    merged = merged.merge(pos_cross_wide, on="M2ID", how="left")
    merged = merged.merge(fd_summary, on="M2ID", how="left")
    merged = merged.merge(qc, on="M2ID", how="left")

    # ========================================================================
    # Create Availability Flags
    # ========================================================================
    # Flag participants with each fMRI modality
    merged["has_beta_series"] = merged[
        ["conn_l_amyg_vmPFC_neg_vs_neu", "conn_r_amyg_vmPFC_neg_vs_neu"]
    ].notna().any(axis=1).astype(int)

    merged["has_neg_persistence"] = merged.filter(
        like="neg_persist"
    ).notna().any(axis=1).astype(int)

    merged["has_pos_persistence"] = merged.filter(
        like="pos_persist"
    ).notna().any(axis=1).astype(int)

    merged["has_fd_data"] = merged["fd_n_runs"].notna().astype(int)

    merged["has_imaging_data"] = (
        merged[["has_beta_series", "has_neg_persistence", "has_pos_persistence"]]
        .any(axis=1)
    ).astype(int)

    # ========================================================================
    # Save Output
    # ========================================================================
    merged.to_csv(OUT_FILE, index=False)
    print(f"✓ Merged dataset with fMRI data saved to: {OUT_FILE}")
    print(f"  Final dataset shape: {merged.shape}")
    print(f"  Participants with imaging data: {merged['has_imaging_data'].sum()}")


if __name__ == "__main__":
    main()
