#!/usr/bin/env python3
"""
09_merge_fmri_data.py  (MR1 / MIDUS Refresher)

Merge reproduced fMRI-derived participant-level measures into the MR1 master
behavioral dataset.

IMPORTANT — QC provenance:
    Visual QC decisions come from historical MR1 manual inspections (carried
    forward; no new visual QC was performed for the reproduction).
    Reproduced FD replicates historical MR1 values.  The conservative sample
    also requires the 231-volume completeness criterion.

Required inputs:
    data/processed/mr1_merged_clean.csv
        Cleaned MR1 master dataset (one row per participant, keyed on MIDUSID).

    data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv
        Reproduced LSS beta-series file (legacy filename; identifier column is
        MIDUSID).  Used only to identify which participants have LSS FC data
        (has_beta_series flag).  FC measurement columns are NOT merged here;
        analysis_utils.py loads this file directly when fc=True to avoid
        duplicate column conflicts in downstream analyses.

    data/fMRI/negative_persistence_cross_run.csv
    data/fMRI/positive_persistence_cross_run.csv
        Reproduced cross-run voxelwise spatial-correlation persistence
        (one row per subject × hemisphere; MIDUSID is the identifier).

    data/fMRI/fd_summary.csv
        Per-subject × per-run mean framewise displacement (one row per
        subject × run).  subject values such as sub-30024 are converted to
        integer MIDUSID.  The "flagged" column is ignored.

    data/fMRI/fmri_qc_processed.csv
        Processed QC flags from 08_process_fmri_qc.py, including
        qc_conservative and run-volume completeness fields.  All columns
        are merged so completeness fields remain available downstream.

Optional inputs (merged when present):
    data/fMRI/condition_fd_wide.csv

Output:
    data/processed/mr1_with_fmri.csv

Notes:
    - All merges are left joins on MIDUSID (no participant exclusions).
    - Every merge is validated to preserve the master row count.
    - Persistence and QC files already use integer MIDUSID; no conversion.

Run from the MR1_validation/ directory.
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
MASTER_FILE      = os.path.join(PROCESSED_DIR, "mr1_merged_clean.csv")
LSS_FILE         = os.path.join(FMRI_DIR, "all_subjects_betaSeries_LSS_all_conditions_M2ID.csv")
NEG_PERSIST_FILE = os.path.join(FMRI_DIR, "negative_persistence_cross_run.csv")
POS_PERSIST_FILE = os.path.join(FMRI_DIR, "positive_persistence_cross_run.csv")
FD_FILE          = os.path.join(FMRI_DIR, "fd_summary.csv")
QC_FILE          = os.path.join(FMRI_DIR, "fmri_qc_processed.csv")

# Optional inputs
COND_FD_FILE = os.path.join(FMRI_DIR, "condition_fd_wide.csv")

# Output
OUT_FILE = os.path.join(PROCESSED_DIR, "mr1_with_fmri.csv")


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
    Pivot long-format persistence (one row per MIDUSID × hemisphere) to wide.

    Expects df to already have an integer MIDUSID column.
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
            df.pivot(index="MIDUSID", columns="hemisphere", values=col)
            .add_prefix(f"{prefix}_{col}_")
            .reset_index()
        )
        wide_dfs.append(wide)

    out = wide_dfs[0]
    for w in wide_dfs[1:]:
        out = out.merge(w, on="MIDUSID", how="outer")
    return out


def _load_persistence(filepath, label, prefix):
    """
    Load, validate, and pivot a persistence file.

    MR1 persistence files already use integer MIDUSID; no subject→ID conversion.
    """
    required = ["MIDUSID", "hemisphere", "mean_r", "median_r", "std_r", "n_pairs"]
    df = pd.read_csv(filepath)
    _require_columns(df, required, label)

    if df["MIDUSID"].isna().any():
        print(f"ERROR: {label}: {int(df['MIDUSID'].isna().sum())} missing MIDUSID value(s)")
        sys.exit(1)

    if df.duplicated(subset=["MIDUSID", "hemisphere"], keep=False).any():
        print(f"ERROR: {label}: duplicate MIDUSID × hemisphere records")
        sys.exit(1)

    n_subj = df["MIDUSID"].nunique()
    hemis  = sorted(df["hemisphere"].unique())
    print(f"  {label}: {n_subj} subjects, {len(df)} rows, hemispheres={hemis}")
    return _pivot_persistence(df, prefix)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 70)
    print("09_merge_fmri_data.py — merge reproduced MR1 fMRI measures")
    print("=" * 70)

    # ========================================================================
    # Master dataset
    # ========================================================================
    master = pd.read_csv(MASTER_FILE)
    _require_columns(master, ["MIDUSID"], "mr1_merged_clean.csv")
    if master["MIDUSID"].isna().any():
        print("ERROR: mr1_merged_clean.csv has missing MIDUSID values")
        sys.exit(1)
    if master["MIDUSID"].duplicated().any():
        print("ERROR: mr1_merged_clean.csv has duplicate MIDUSID values")
        sys.exit(1)
    n_master = len(master)
    print(f"\nMaster dataset: {n_master} rows × {master.shape[1]} columns")

    # ========================================================================
    # LSS beta-series: availability flag only
    # FC columns are NOT merged here (analysis_utils.py loads the file directly)
    # ========================================================================
    print(f"\nLoading LSS beta-series (availability only): {LSS_FILE}")
    lss = pd.read_csv(LSS_FILE)
    _require_columns(lss, ["MIDUSID"], "LSS beta-series file")
    if lss["MIDUSID"].isna().any():
        print("ERROR: missing MIDUSID in LSS beta-series file")
        sys.exit(1)
    if lss["MIDUSID"].duplicated().any():
        print("ERROR: duplicate MIDUSID in LSS beta-series file")
        sys.exit(1)
    lss_ids = set(lss["MIDUSID"])
    print(f"  {len(lss_ids)} participants with LSS beta-series data")

    # ========================================================================
    # Persistence files (required)
    # ========================================================================
    print(f"\nLoading persistence files...")
    neg_wide = _load_persistence(
        NEG_PERSIST_FILE, "negative persistence", "neg_persist_crossrun"
    )
    pos_wide = _load_persistence(
        POS_PERSIST_FILE, "positive persistence", "pos_persist_crossrun"
    )

    # ========================================================================
    # FD summary (aggregate across runs; "flagged" column ignored)
    # ========================================================================
    print(f"\nLoading FD summary: {FD_FILE}")
    fd = pd.read_csv(FD_FILE)
    _require_columns(fd, ["subject", "run", "mean_fd"], "fd_summary.csv")

    fd["run_int"] = fd["run"].apply(_normalize_run)
    n_bad = fd["run_int"].isna().sum()
    if n_bad:
        print(f"  WARNING: {n_bad} row(s) with invalid run values — dropped")
        fd = fd[fd["run_int"].notna()].copy()

    fd["MIDUSID"] = fd["subject"].str.replace("sub-", "", regex=False).astype(int)

    if fd.duplicated(subset=["MIDUSID", "run_int"], keep=False).any():
        print("ERROR: duplicate subject × run records in fd_summary.csv")
        sys.exit(1)

    fd_agg = (
        fd.groupby("MIDUSID")
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
    # All columns merged so run-volume completeness fields remain available.
    # ========================================================================
    print(f"\nLoading QC flags: {QC_FILE}")
    qc = pd.read_csv(QC_FILE)
    _require_columns(qc, ["MIDUSID", "qc_conservative"], "fmri_qc_processed.csv")
    if qc["MIDUSID"].isna().any():
        print("ERROR: missing MIDUSID in fmri_qc_processed.csv")
        sys.exit(1)
    if qc["MIDUSID"].duplicated().any():
        print("ERROR: duplicate MIDUSID in fmri_qc_processed.csv")
        sys.exit(1)
    print(f"  {len(qc)} participants with QC flags")

    # ========================================================================
    # Merge required inputs
    # ========================================================================
    print(f"\nMerging onto master ({n_master} rows)...")

    merged = master.merge(qc, on="MIDUSID", how="left")
    _check_merge(merged, n_master, "QC flags")
    print(f"  QC flags:           {merged['qc_conservative'].notna().sum()} matched")

    merged = merged.merge(neg_wide, on="MIDUSID", how="left")
    _check_merge(merged, n_master, "negative persistence")
    print(f"  Neg persistence:    {merged['neg_persist_crossrun_mean_r_L'].notna().sum()} matched")

    merged = merged.merge(pos_wide, on="MIDUSID", how="left")
    _check_merge(merged, n_master, "positive persistence")
    print(f"  Pos persistence:    {merged['pos_persist_crossrun_mean_r_L'].notna().sum()} matched")

    merged = merged.merge(fd_agg, on="MIDUSID", how="left")
    _check_merge(merged, n_master, "FD")
    print(f"  FD:                 {merged['fd_n_runs'].notna().sum()} matched")

    # ========================================================================
    # Optional: condition FD
    # ========================================================================
    if os.path.exists(COND_FD_FILE):
        cond_fd = pd.read_csv(COND_FD_FILE)
        _require_columns(
            cond_fd, ["MIDUSID", "fd_neg_mean", "fd_neu_mean", "fd_pos_mean"],
            "condition FD file"
        )
        if cond_fd["MIDUSID"].isna().any():
            print("ERROR: missing MIDUSID in condition FD file")
            sys.exit(1)
        if cond_fd["MIDUSID"].duplicated().any():
            print("ERROR: duplicate MIDUSID in condition FD file")
            sys.exit(1)
        merged = merged.merge(cond_fd, on="MIDUSID", how="left")
        _check_merge(merged, n_master, "condition FD")
        print(f"  Condition FD:       {len(cond_fd)} subjects")
    else:
        print(f"  (Condition FD not found — skipped)")

    # ========================================================================
    # Availability flags
    # ========================================================================
    # has_beta_series: participant's MIDUSID occurs in the LSS file
    # (FC columns are not merged; this is a per-participant membership flag)
    merged["has_beta_series"] = merged["MIDUSID"].isin(lss_ids).astype(int)

    # has_neg/pos_persistence: actual left-amygdala mean_r estimate is nonmissing
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
