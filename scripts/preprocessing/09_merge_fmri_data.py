#!/usr/bin/env python3
"""
09_merge_fmri_data.py

Merge corrected-STC fMRI-derived participant-level measures into the cleaned
MIDUS master dataset.

Required inputs:
    data/processed/midus_merged_clean.csv
        Cleaned master dataset (one row per participant).

    data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv
        Corrected-STC LSS beta-series file.  Used only to identify which
        participants have LSS FC data (has_beta_series flag).  FC measurement
        columns are NOT merged here because analysis_utils.py loads this file
        directly when fc=True; merging the columns here would produce
        duplicate/suffixed FC columns in downstream analyses.

    data/fMRI/negative_persistence_cross_run.csv
    data/fMRI/positive_persistence_cross_run.csv
        Corrected-STC cross-run voxelwise spatial-correlation persistence
        (one row per subject × hemisphere).

    data/fMRI/fd_summary.csv
        Corrected-STC per-subject × per-run mean framewise displacement
        (one row per subject × run).  The "flagged" column is ignored.

    data/fMRI/fmri_qc_processed.csv
        Processed QC flags including qc_conservative (from 08_process_fmri_qc.py).

Optional inputs (merged when present):
    data/fMRI/vmPFC_persistence_wide.csv
    data/fMRI/all_subjects_roi_activations.csv
    data/fMRI/condition_fd_wide.csv

Output:
    data/processed/midus_with_fmri.csv

Notes:
    - All merges are left joins on M2ID (no participant exclusions).
    - Every merge is validated to preserve the master row count.
    - subject (sub-XXXXX) values are converted to integer M2ID before merging.

Run from the project root directory.
"""

import os
import sys

import pandas as pd

# ============================================================================
# Paths
# ============================================================================
PROCESSED_DIR = "data/processed"
FMRI_DIR      = "data/fMRI"

# Required inputs
MASTER_FILE      = os.path.join(PROCESSED_DIR, "midus_merged_clean.csv")
LSS_FILE         = os.path.join(FMRI_DIR, "all_subjects_betaSeries_LSS_all_conditions_M2ID.csv")
NEG_PERSIST_FILE = os.path.join(FMRI_DIR, "negative_persistence_cross_run.csv")
POS_PERSIST_FILE = os.path.join(FMRI_DIR, "positive_persistence_cross_run.csv")
FD_FILE          = os.path.join(FMRI_DIR, "fd_summary.csv")
QC_FILE          = os.path.join(FMRI_DIR, "fmri_qc_processed.csv")

# Optional inputs
VMRFC_PERSIST_FILE   = os.path.join(FMRI_DIR, "vmPFC_persistence_wide.csv")
ROI_ACTIVATIONS_FILE = os.path.join(FMRI_DIR, "all_subjects_roi_activations.csv")
COND_FD_FILE         = os.path.join(FMRI_DIR, "condition_fd_wide.csv")

# Output
OUT_FILE = os.path.join(PROCESSED_DIR, "midus_with_fmri.csv")


# ============================================================================
# Helper Functions
# ============================================================================

def _require_columns(df, required, label):
    """Exit with a clear message if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: {label} is missing required column(s): {missing}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)


def _normalize_run(val):
    """
    Normalise run identifier to integer 1–3.
    Accepts '1', '01', 'run-01', etc.  Returns None for anything out of range.
    """
    s = str(val).strip()
    if s.startswith("run-"):
        s = s[4:]
    try:
        n = int(s)
    except ValueError:
        return None
    return n if 1 <= n <= 3 else None


def _check_merge(df, n_expected, label):
    """Exit if a merge changed the master row count (indicates duplicate keys)."""
    if len(df) != n_expected:
        print(f"ERROR: merge with {label} changed row count "
              f"({n_expected} → {len(df)}).  Check for duplicate join keys.")
        sys.exit(1)


def _pivot_persistence(df, prefix):
    """
    Pivot long-format persistence (one row per M2ID × hemisphere) to wide.

    Expects df to already have an integer M2ID column.
    BI hemisphere values are standardised to 'bilateral'.
    Pivot columns: mean_r, median_r, std_r, n_pairs.
    Column naming: <prefix>_<value_col>_<hemisphere>
    """
    df = df.copy()
    df["hemisphere"] = df["hemisphere"].replace({"BI": "bilateral"})
    value_cols = ["mean_r", "median_r", "std_r", "n_pairs"]

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


def _load_persistence(filepath, label, prefix):
    """Load, validate, convert subject → M2ID, and pivot a persistence file."""
    required = ["subject", "hemisphere", "mean_r", "median_r", "std_r", "n_pairs"]
    df = pd.read_csv(filepath)
    _require_columns(df, required, label)

    blanks = df["subject"].isna() | (df["subject"].astype(str).str.strip() == "")
    if blanks.any():
        print(f"ERROR: {label}: {blanks.sum()} blank subject value(s)")
        sys.exit(1)

    df["M2ID"] = df["subject"].str.replace("sub-", "", regex=False).astype(int)

    if df.duplicated(subset=["M2ID", "hemisphere"], keep=False).any():
        print(f"ERROR: {label}: duplicate subject × hemisphere records")
        sys.exit(1)

    n_subj = df["M2ID"].nunique()
    hemis  = sorted(df["hemisphere"].unique())
    print(f"  {label}: {n_subj} subjects, {len(df)} rows, hemispheres={hemis}")
    return _pivot_persistence(df, prefix)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 70)
    print("09_merge_fmri_data.py — merge corrected-STC fMRI measures")
    print("=" * 70)

    # ========================================================================
    # Master dataset
    # ========================================================================
    master   = pd.read_csv(MASTER_FILE)
    n_master = len(master)
    print(f"\nMaster dataset: {n_master} rows × {master.shape[1]} columns")

    # ========================================================================
    # LSS beta-series: availability flag only
    # FC columns are NOT merged here (analysis_utils.py loads the file directly)
    # ========================================================================
    print(f"\nLoading LSS beta-series (availability only): {LSS_FILE}")
    lss = pd.read_csv(LSS_FILE)
    _require_columns(lss, ["M2ID"], "LSS beta-series file")
    if lss["M2ID"].isna().any():
        print("ERROR: blank M2ID in LSS beta-series file")
        sys.exit(1)
    if lss["M2ID"].duplicated().any():
        print("ERROR: duplicate M2ID in LSS beta-series file")
        sys.exit(1)
    lss_ids = set(lss["M2ID"])
    print(f"  {len(lss_ids)} participants with LSS beta-series data")

    # ========================================================================
    # Persistence files
    # ========================================================================
    print(f"\nLoading persistence files...")
    neg_wide = _load_persistence(
        NEG_PERSIST_FILE, "negative persistence", "neg_persist_crossrun"
    )
    pos_wide = _load_persistence(
        POS_PERSIST_FILE, "positive persistence", "pos_persist_crossrun"
    )

    # ========================================================================
    # Corrected FD summary (aggregate across runs; "flagged" column ignored)
    # ========================================================================
    print(f"\nLoading corrected FD summary: {FD_FILE}")
    fd = pd.read_csv(FD_FILE)
    _require_columns(fd, ["subject", "run", "mean_fd"], "fd_summary.csv")

    fd["run_int"] = fd["run"].apply(_normalize_run)
    n_bad = fd["run_int"].isna().sum()
    if n_bad:
        print(f"  WARNING: {n_bad} row(s) with invalid run values — dropped")
        fd = fd[fd["run_int"].notna()].copy()

    fd["M2ID"] = fd["subject"].str.replace("sub-", "", regex=False).astype(int)

    if fd.duplicated(subset=["M2ID", "run_int"], keep=False).any():
        print("ERROR: duplicate subject × run records in fd_summary.csv")
        sys.exit(1)

    fd_agg = (
        fd.groupby("M2ID")
        .agg(
            fd_mean_across_runs=("mean_fd", "mean"),
            fd_max_across_runs=("mean_fd", "max"),
            fd_n_runs=("mean_fd", "count"),
        )
        .reset_index()
    )
    print(f"  {len(fd_agg)} participants with FD data")

    # ========================================================================
    # QC flags (from 08_process_fmri_qc.py)
    # ========================================================================
    print(f"\nLoading QC flags: {QC_FILE}")
    qc = pd.read_csv(QC_FILE)
    _require_columns(qc, ["M2ID", "qc_conservative"], "fmri_qc_processed.csv")
    if qc["M2ID"].duplicated().any():
        print("ERROR: duplicate M2ID in fmri_qc_processed.csv")
        sys.exit(1)
    print(f"  {len(qc)} participants with QC flags")

    # ========================================================================
    # Merge required inputs
    # ========================================================================
    print(f"\nMerging onto master ({n_master} rows)...")

    merged = master.merge(qc, on="M2ID", how="left")
    _check_merge(merged, n_master, "QC flags")
    print(f"  QC flags:           {merged['qc_conservative'].notna().sum()} matched")

    merged = merged.merge(neg_wide, on="M2ID", how="left")
    _check_merge(merged, n_master, "negative persistence")
    print(f"  Neg persistence:    {merged['neg_persist_crossrun_mean_r_L'].notna().sum()} matched")

    merged = merged.merge(pos_wide, on="M2ID", how="left")
    _check_merge(merged, n_master, "positive persistence")
    print(f"  Pos persistence:    {merged['pos_persist_crossrun_mean_r_L'].notna().sum()} matched")

    merged = merged.merge(fd_agg, on="M2ID", how="left")
    _check_merge(merged, n_master, "FD")
    print(f"  FD:                 {merged['fd_n_runs'].notna().sum()} matched")

    # ========================================================================
    # Optional: vmPFC persistence
    # ========================================================================
    if os.path.exists(VMRFC_PERSIST_FILE):
        vmPFC = pd.read_csv(VMRFC_PERSIST_FILE)
        if "subject" in vmPFC.columns:
            vmPFC["M2ID"] = vmPFC["subject"].str.replace("sub-", "", regex=False).astype(int)
            vmPFC = vmPFC.drop(columns="subject")
        _require_columns(vmPFC, ["M2ID"], "vmPFC persistence file")
        if vmPFC["M2ID"].duplicated().any():
            print("ERROR: duplicate M2ID in vmPFC persistence file")
            sys.exit(1)
        merged = merged.merge(vmPFC, on="M2ID", how="left")
        _check_merge(merged, n_master, "vmPFC persistence")
        print(f"  vmPFC persistence:  {len(vmPFC)} subjects")
    else:
        print(f"  (vmPFC persistence not found — skipped)")

    # ========================================================================
    # Optional: ROI activations
    # ========================================================================
    if os.path.exists(ROI_ACTIVATIONS_FILE):
        roi_act = pd.read_csv(ROI_ACTIVATIONS_FILE)
        if "subid" in roi_act.columns:
            roi_act["M2ID"] = roi_act["subid"].str.replace("sub-", "", regex=False).astype(int)
            roi_act = roi_act.drop(columns="subid")
        _require_columns(roi_act, ["M2ID"], "ROI activations file")
        if "run" in roi_act.columns:
            roi_act = roi_act.drop(columns="run").groupby("M2ID").mean().reset_index()
        if roi_act["M2ID"].duplicated().any():
            print("ERROR: duplicate M2ID in ROI activations file after aggregation")
            sys.exit(1)
        merged = merged.merge(roi_act, on="M2ID", how="left")
        _check_merge(merged, n_master, "ROI activations")
        print(f"  ROI activations:    {len(roi_act)} subjects")
    else:
        print(f"  (ROI activations not found — skipped)")

    # ========================================================================
    # Optional: condition FD
    # ========================================================================
    if os.path.exists(COND_FD_FILE):
        cond_fd = pd.read_csv(COND_FD_FILE)
        _require_columns(cond_fd, ["M2ID"], "condition FD file")
        if cond_fd["M2ID"].duplicated().any():
            print("ERROR: duplicate M2ID in condition FD file")
            sys.exit(1)
        merged = merged.merge(cond_fd, on="M2ID", how="left")
        _check_merge(merged, n_master, "condition FD")
        print(f"  Condition FD:       {len(cond_fd)} subjects")
    else:
        print(f"  (Condition FD not found — skipped)")

    # ========================================================================
    # Availability flags
    # ========================================================================
    # has_beta_series: participant appears in LSS file (FC columns not merged here)
    merged["has_beta_series"] = merged["M2ID"].isin(lss_ids).astype(int)

    # has_neg/pos_persistence: actual left-amygdala estimate is nonmissing
    # (n_pairs alone is insufficient — n_pairs=0 is still nonmissing)
    merged["has_neg_persistence"] = merged["neg_persist_crossrun_mean_r_L"].notna().astype(int)
    merged["has_pos_persistence"] = merged["pos_persist_crossrun_mean_r_L"].notna().astype(int)

    merged["has_fd_data"] = merged["fd_n_runs"].notna().astype(int)

    merged["has_imaging_data"] = (
        (merged["has_beta_series"] == 1) |
        (merged["has_neg_persistence"] == 1) |
        (merged["has_pos_persistence"] == 1)
    ).astype(int)

    # ========================================================================
    # Summary and save
    # ========================================================================
    print(f"\nAvailability summary:")
    print(f"  has_beta_series:     {merged['has_beta_series'].sum()}")
    print(f"  has_neg_persistence: {merged['has_neg_persistence'].sum()}")
    print(f"  has_pos_persistence: {merged['has_pos_persistence'].sum()}")
    print(f"  has_fd_data:         {merged['has_fd_data'].sum()}")
    print(f"  has_imaging_data:    {merged['has_imaging_data'].sum()}")
    if "qc_conservative" in merged.columns:
        print(f"  qc_conservative N:   {merged['qc_conservative'].sum()}")

    merged.to_csv(OUT_FILE, index=False)
    print(f"\n✓ Saved to {OUT_FILE}")
    print(f"  Final shape: {merged.shape[0]} rows × {merged.shape[1]} columns")


if __name__ == "__main__":
    main()
