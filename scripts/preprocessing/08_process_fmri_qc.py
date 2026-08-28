#!/usr/bin/env python3
"""
08_process_fmri_qc.py

Process fMRI quality control (QC) data from manual inspection and corrected-STC
framewise-displacement (FD) summary.

Creates binary QC flags and criteria for defining conservative analysis samples:
- All 3 runs pass visual QC  (from task_fMRI_QC.xlsx, run1/run2/run3 columns only)
- Mean FD < 0.5 mm           (from fd_summary.csv, corrected-STC pipeline)
- n_pairs = 6                (all 3 cross-run persistence pairs valid,
                              from negative_persistence_cross_run.csv)

Inputs:
    data/fMRI/task_fMRI_QC.xlsx
        Manual anatomical/functional visual QC.  Uses subject, run1, run2, run3
        columns only.  Any FD columns present in this file are ignored.

    data/fMRI/fd_summary.csv
        Corrected-STC per-subject × per-run mean FD (469 rows, one per subject ×
        run).  Authoritative FD source.  Run values are normalised to integers 1–3.
        The "flagged" column, if present, is ignored.

    data/fMRI/negative_persistence_cross_run.csv
        Cross-run persistence results.  Uses subject, hemisphere, and n_pairs.
        Left-hemisphere row (hemisphere == "L") supplies n_pairs for each subject.

Outputs:
    data/fMRI/fmri_qc_processed.csv

Run from the project root directory.
"""

import os
import sys

import pandas as pd

# ============================================================================
# Paths
# ============================================================================
FMRI_DIR = "data/fMRI"

QC_FILE      = os.path.join(FMRI_DIR, "task_fMRI_QC.xlsx")
FD_FILE      = os.path.join(FMRI_DIR, "fd_summary.csv")
PERSIST_FILE = os.path.join(FMRI_DIR, "negative_persistence_cross_run.csv")
OUTPUT_FILE  = os.path.join(FMRI_DIR, "fmri_qc_processed.csv")

FD_THRESHOLD = 0.5   # mm — do not change without pre-registration amendment


# ============================================================================
# Helpers
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
    Accepts '1', '01', 'run-01' etc.  Returns None for anything unparseable
    or outside 1–3.
    """
    s = str(val).strip()
    if s.startswith("run-"):
        s = s[4:]
    try:
        n = int(s)
    except ValueError:
        return None
    return n if 1 <= n <= 3 else None


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 80)
    print("Processing fMRI QC data")
    print("=" * 80)

    # ========================================================================
    # 1. Visual-QC Excel file
    #    Use subject, run1, run2, run3 columns only.  Ignore any FD columns.
    # ========================================================================
    print(f"\nLoading visual-QC data from {QC_FILE}...")
    qc = pd.read_excel(QC_FILE)
    _require_columns(qc, ["subject", "run1", "run2", "run3"], "task_fMRI_QC.xlsx")
    # Drop all other Excel columns (e.g. old mean_fd, run1_fd/run2_fd/run3_fd)
    # so they cannot produce pandas merge suffixes with corrected-STC FD columns.
    qc = qc[["subject", "run1", "run2", "run3"]].copy()
    print(f"  Loaded {len(qc)} rows")

    # Convert subject (sub-XXXXX) → integer M2ID
    qc["M2ID"] = qc["subject"].str.replace("sub-", "", regex=False).astype(int)

    # Require one unique QC row per participant
    dup_ids = qc["M2ID"][qc["M2ID"].duplicated()].unique()
    if len(dup_ids):
        print(f"ERROR: duplicate M2ID rows in task_fMRI_QC.xlsx: {dup_ids[:5].tolist()}")
        sys.exit(1)

    # Run-level visual QC: Pass → 1, anything else or NaN → 0
    qc["run1_pass"] = (qc["run1"] == "Pass").astype(int)
    qc["run2_pass"] = (qc["run2"] == "Pass").astype(int)
    qc["run3_pass"] = (qc["run3"] == "Pass").astype(int)

    qc["all_runs_pass"] = (
        (qc["run1_pass"] == 1) &
        (qc["run2_pass"] == 1) &
        (qc["run3_pass"] == 1)
    ).astype(int)

    # ========================================================================
    # 2. Corrected-STC FD summary  (authoritative FD source)
    # ========================================================================
    print(f"\nLoading corrected-STC FD data from {FD_FILE}...")
    fd = pd.read_csv(FD_FILE)
    _require_columns(fd, ["subject", "run", "mean_fd"], "fd_summary.csv")
    print(f"  FD records loaded: {len(fd)}")

    # Normalise run values to integers 1–3
    fd["run_int"] = fd["run"].apply(_normalize_run)
    n_bad = fd["run_int"].isna().sum()
    if n_bad:
        print(f"  WARNING: {n_bad} row(s) with unparseable/out-of-range run values — dropped")
        fd = fd[fd["run_int"].notna()].copy()
    fd["run_int"] = fd["run_int"].astype(int)

    # Convert subject (sub-XXXXX) → integer M2ID
    fd["M2ID"] = fd["subject"].str.replace("sub-", "", regex=False).astype(int)

    # Reject duplicate subject × run records
    dup_mask = fd.duplicated(subset=["M2ID", "run_int"], keep=False)
    if dup_mask.any():
        dup_pairs = fd.loc[dup_mask, ["M2ID", "run_int"]].drop_duplicates()
        print(f"ERROR: {len(dup_pairs)} duplicate subject × run pair(s) in fd_summary.csv")
        print(f"  Sample: {dup_pairs.head(5).to_dict('records')}")
        sys.exit(1)

    n_fd_participants = fd["M2ID"].nunique()
    run_counts = fd.groupby("M2ID")["run_int"].count()
    print(f"  FD participants: {n_fd_participants}")
    print(f"  FD run-count distribution:")
    for n_runs, n_subj in run_counts.value_counts().sort_index().items():
        print(f"    {n_runs} run(s): {n_subj} participant(s)")

    # Pivot to wide format: run1_fd, run2_fd, run3_fd
    fd_wide = fd.pivot(index="M2ID", columns="run_int", values="mean_fd")
    fd_wide.columns = [f"run{int(c)}_fd" for c in fd_wide.columns]
    for col in ["run1_fd", "run2_fd", "run3_fd"]:   # ensure all three exist
        if col not in fd_wide.columns:
            fd_wide[col] = float("nan")
    fd_wide = fd_wide[["run1_fd", "run2_fd", "run3_fd"]].reset_index()

    # mean_fd = arithmetic mean of available run-level values
    fd_wide["mean_fd"] = fd_wide[["run1_fd", "run2_fd", "run3_fd"]].mean(axis=1, skipna=True)

    missing_by_run = fd_wide[["run1_fd", "run2_fd", "run3_fd"]].isna().sum()
    print(f"  Missing FD values by run: {missing_by_run.to_dict()}")

    # ========================================================================
    # 3. Persistence file  (n_pairs from left-hemisphere rows)
    # ========================================================================
    print(f"\nLoading persistence data from {PERSIST_FILE}...")
    persist = pd.read_csv(PERSIST_FILE)
    _require_columns(
        persist, ["subject", "hemisphere", "mean_r", "n_pairs"],
        "negative_persistence_cross_run.csv"
    )

    # Convert subject (sub-XXXXX) → integer M2ID
    persist["M2ID"] = persist["subject"].str.replace("sub-", "", regex=False).astype(int)

    # Reject duplicate subject × hemisphere records
    if persist.duplicated(subset=["M2ID", "hemisphere"], keep=False).any():
        print("ERROR: duplicate subject × hemisphere records in negative_persistence_cross_run.csv")
        sys.exit(1)

    # Left-hemisphere n_pairs for each subject
    persist_left = persist[persist["hemisphere"] == "L"][["M2ID", "n_pairs"]].copy()
    print(f"  Left-hemisphere n_pairs records: {len(persist_left)}")

    # ========================================================================
    # 4. Merge — validate cardinality at each step
    # ========================================================================
    n_ref = len(qc)

    qc = qc.merge(fd_wide, on="M2ID", how="left")
    if len(qc) != n_ref:
        print(f"ERROR: FD merge changed row count ({n_ref} → {len(qc)}). "
              f"fd_wide has duplicate M2ID values.")
        sys.exit(1)

    qc = qc.merge(persist_left, on="M2ID", how="left")
    if len(qc) != n_ref:
        print(f"ERROR: n_pairs merge changed row count ({n_ref} → {len(qc)}). "
              f"persist_left has duplicate M2ID values.")
        sys.exit(1)

    # ========================================================================
    # 5. Derive QC flags
    # ========================================================================
    # fd_pass: mean_fd < threshold; NaN (no FD data) fails
    qc["fd_pass"] = (qc["mean_fd"].notna() & (qc["mean_fd"] < FD_THRESHOLD)).astype(int)

    # task_complete: n_pairs == 6; NaN (no persistence data) fails
    qc["task_complete"] = (qc["n_pairs"].notna() & (qc["n_pairs"] == 6)).astype(int)

    # Conservative sample: visual QC AND FD AND task completeness
    qc["qc_conservative"] = (
        (qc["all_runs_pass"] == 1) &
        (qc["fd_pass"] == 1) &
        (qc["task_complete"] == 1)
    ).astype(int)

    # ========================================================================
    # 6. Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("QC Summary")
    print("=" * 80)

    n = len(qc)
    print(f"\nRun-level visual QC  (N={n}):")
    print(f"  Run 1 pass:    {qc['run1_pass'].sum()} ({qc['run1_pass'].sum()/n*100:.1f}%)")
    print(f"  Run 2 pass:    {qc['run2_pass'].sum()} ({qc['run2_pass'].sum()/n*100:.1f}%)")
    print(f"  Run 3 pass:    {qc['run3_pass'].sum()} ({qc['run3_pass'].sum()/n*100:.1f}%)")
    print(f"  All runs pass: {qc['all_runs_pass'].sum()} ({qc['all_runs_pass'].sum()/n*100:.1f}%)")

    n_fd_missing = qc["mean_fd"].isna().sum()
    print(f"\nFramewise displacement  (threshold: {FD_THRESHOLD} mm, source: fd_summary.csv):")
    print(f"  Participants with FD data: {n - n_fd_missing}")
    print(f"  Participants missing FD:   {n_fd_missing}")
    if qc["mean_fd"].notna().any():
        print(f"  Mean FD range: {qc['mean_fd'].min():.3f} – {qc['mean_fd'].max():.3f}")
    print(f"  FD pass (mean_fd < {FD_THRESHOLD}): {qc['fd_pass'].sum()} ({qc['fd_pass'].sum()/n*100:.1f}%)")
    print(f"  FD fail (includes missing): {(qc['fd_pass'] == 0).sum()}")

    n_pairs_missing = qc["n_pairs"].isna().sum()
    print(f"\nTask completeness  (n_pairs == 6):")
    print(f"  Complete:   {qc['task_complete'].sum()} ({qc['task_complete'].sum()/n*100:.1f}%)")
    print(f"  Incomplete: {(qc['task_complete'] == 0).sum()}  (includes {n_pairs_missing} missing)")

    print(f"\nConservative sample  (all_runs_pass=1 AND fd_pass=1 AND task_complete=1):")
    print(f"  N = {qc['qc_conservative'].sum()} of {n} ({qc['qc_conservative'].sum()/n*100:.1f}%)")

    excluded  = qc[qc["qc_conservative"] == 0]
    excl_vis  = excluded[(excluded["all_runs_pass"] == 0) &
                         (excluded["fd_pass"] == 1) & (excluded["task_complete"] == 1)]
    excl_fd   = excluded[(excluded["fd_pass"] == 0) &
                         (excluded["all_runs_pass"] == 1) & (excluded["task_complete"] == 1)]
    excl_task = excluded[(excluded["task_complete"] == 0) &
                         (excluded["all_runs_pass"] == 1) & (excluded["fd_pass"] == 1)]
    print(f"\nExclusion breakdown  (mutually exclusive primary reason):")
    print(f"  Failed visual QC only:     {len(excl_vis)}")
    print(f"  Failed FD only:            {len(excl_fd)}")
    print(f"  Failed task complete only: {len(excl_task)}")
    print(f"  Failed multiple criteria:  {len(excluded) - len(excl_vis) - len(excl_fd) - len(excl_task)}")
    print(f"  Total excluded:            {len(excluded)}")

    # ========================================================================
    # 7. Save
    # ========================================================================
    output_cols = [
        "M2ID",
        "run1_pass", "run2_pass", "run3_pass", "all_runs_pass",
        "run1_fd", "run2_fd", "run3_fd", "mean_fd", "fd_pass",
        "n_pairs", "task_complete",
        "qc_conservative",
    ]
    qc[output_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved {len(qc)} subjects to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
