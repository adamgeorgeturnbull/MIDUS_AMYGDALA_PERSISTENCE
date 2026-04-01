#!/usr/bin/env python3
"""
08_merge_fmri_data.py

Merge fMRI-derived participant-level measures into the cleaned MIDUS master dataset.

Inputs:
- data/processed/midus_merged_clean.csv (cleaned master dataset)
- data/fMRI/betaSeries_neg_vs_neu_threat_safety.csv (beta-series connectivity, 1 row/participant)
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
BETA_FILE = os.path.join(FMRI_DIR, "betaSeries_neg_vs_neu_threat_safety.csv")
NEG_PERSIST_CONCAT_FILE = os.path.join(FMRI_DIR, "negative_persistence_concat.csv")
NEG_PERSIST_CROSS_FILE = os.path.join(FMRI_DIR, "negative_persistence_cross_run.csv")
POS_PERSIST_CROSS_FILE = os.path.join(FMRI_DIR, "positive_persistence_cross_run.csv")
FD_FILE = os.path.join(FMRI_DIR, "fd_summary.csv")
QC_FILE = os.path.join(FMRI_DIR, "fmri_qc_processed.csv")

# Optional new data sources (merged when available)
VMRFC_PERSIST_FILE = os.path.join(FMRI_DIR, "vmPFC_persistence_wide.csv")        # from run_cross_corr_vmPFC.py
ROI_ACTIVATIONS_FILE = os.path.join(FMRI_DIR, "all_subjects_roi_activations.csv") # from extract_roi_activations.sh

# aCompCor preprocessing comparison persistence files
ACC_NEG_PERSIST_FILE = os.path.join(FMRI_DIR, "aCompCor_persistence_summary", "results_summary_neg_image_vs_neg_face.csv")
ACC_POS_PERSIST_FILE = os.path.join(FMRI_DIR, "aCompCor_persistence_summary", "results_summary_pos_image_vs_pos_face.csv")

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
        "l_amyg-ant_vmPFC": "conn_l_amyg_ant_vmPFC_neg_vs_neu",
        "l_amyg-post_vmPFC": "conn_l_amyg_post_vmPFC_neg_vs_neu",
        "r_amyg-ant_vmPFC": "conn_r_amyg_ant_vmPFC_neg_vs_neu",
        "r_amyg-post_vmPFC": "conn_r_amyg_post_vmPFC_neg_vs_neu"
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
    # aCompCor persistence (preprocessing comparison)
    # ========================================================================
    def _load_acc_persist(filepath, prefix):
        df = pd.read_csv(filepath)
        df["M2ID"] = df["subject"].str.replace("sub-", "", regex=False).astype(int)
        df = df.drop(columns="subject")
        return pivot_persistence_long(
            df=df,
            value_cols=["mean_r", "median_r", "std_r", "n_pairs"],
            prefix=prefix,
        )

    if os.path.exists(ACC_NEG_PERSIST_FILE) and os.path.exists(ACC_POS_PERSIST_FILE):
        acc_neg_wide = _load_acc_persist(ACC_NEG_PERSIST_FILE, "neg_persist_acc_crossrun")
        acc_pos_wide = _load_acc_persist(ACC_POS_PERSIST_FILE, "pos_persist_acc_crossrun")
        merged = merged.merge(acc_neg_wide, on="M2ID", how="left")
        merged = merged.merge(acc_pos_wide, on="M2ID", how="left")
        print(f"✓ Merged aCompCor persistence ({acc_neg_wide['M2ID'].notna().sum()} subjects)")
    else:
        print(f"  (skipping aCompCor persistence — files not found in {os.path.dirname(ACC_NEG_PERSIST_FILE)})")

    # ========================================================================
    # Optional: vmPFC persistence (from run_cross_corr_vmPFC.py)
    # ========================================================================
    if os.path.exists(VMRFC_PERSIST_FILE):
        vmPFC_persist = pd.read_csv(VMRFC_PERSIST_FILE)
        vmPFC_persist["M2ID"] = vmPFC_persist["subject"].str.replace("sub-", "", regex=False).astype(int)
        vmPFC_persist = vmPFC_persist.drop(columns="subject")
        merged = merged.merge(vmPFC_persist, on="M2ID", how="left")
        print(f"✓ Merged vmPFC persistence ({len(vmPFC_persist)} subjects)")
    else:
        print(f"  (skipping vmPFC persistence — {VMRFC_PERSIST_FILE} not found)")

    # ========================================================================
    # Optional: ROI activations (from extract_roi_activations.sh + combine)
    # ========================================================================
    if os.path.exists(ROI_ACTIVATIONS_FILE):
        roi_act = pd.read_csv(ROI_ACTIVATIONS_FILE)
        if "subid" in roi_act.columns:
            roi_act["M2ID"] = roi_act["subid"].str.replace("sub-", "", regex=False).astype(int)
            roi_act = roi_act.drop(columns="subid")
        # Average across runs if per-run rows present
        if "run" in roi_act.columns:
            roi_act = roi_act.drop(columns="run").groupby("M2ID").mean().reset_index()
        merged = merged.merge(roi_act, on="M2ID", how="left")
        print(f"✓ Merged ROI activations ({len(roi_act)} subjects)")
    else:
        print(f"  (skipping ROI activations — {ROI_ACTIVATIONS_FILE} not found)")

    # ========================================================================
    # Create Availability Flags
    # ========================================================================
    # Flag participants with each fMRI modality
    merged["has_beta_series"] = merged[
        ["conn_l_amyg_ant_vmPFC_neg_vs_neu", "conn_r_amyg_ant_vmPFC_neg_vs_neu"]
    ].notna().any(axis=1).astype(int)

    merged["has_neg_persistence"] = merged.filter(
        regex="^neg_persist_crossrun"
    ).notna().any(axis=1).astype(int)

    merged["has_pos_persistence"] = merged.filter(
        regex="^pos_persist_crossrun"
    ).notna().any(axis=1).astype(int)

    merged["has_acc_persistence"] = merged.filter(
        regex="^neg_persist_acc_crossrun"
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
