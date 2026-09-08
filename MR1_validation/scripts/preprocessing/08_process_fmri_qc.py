#!/usr/bin/env python3
"""
08_process_fmri_qc.py  (MR1 / MIDUS Refresher)

Process fMRI quality control (QC) data for the MR1 reproduction.

IMPORTANT — Data provenance:
    Visual QC decisions (task_fMRI_QC_MR1.xlsx) are historical MR1 manual
    inspections carried forward.  No new visual QC was performed for the
    reproduction.

    Reproduced FD values (fd_summary.csv from the MR1 reproduction pipeline)
    replicate the historical MR1 FD computation; they are not expected to
    differ from the original values.

    The 231-volume completeness rule (all_three_runs_231 = 1) prevents any
    shortened acquisition from qualifying for the conservative sample through
    n_pairs = 6 alone.

Creates binary QC flags and criteria for the conservative analysis sample:
- All 3 runs pass visual QC  (from task_fMRI_QC_MR1.xlsx; run1/run2/run3 only)
- Mean FD < 0.5 mm           (from fd_summary.csv)
- n_pairs = 6                (all 3 cross-run persistence pairs valid,
                              from negative_persistence_cross_run.csv)
- all_three_runs_231 = 1     (runs 01, 02, 03 each have exactly 231 volumes,
                              from run_completeness.csv)

Inputs:
    data/fMRI/task_fMRI_QC_MR1.xlsx
        Historical MR1 manual visual QC.  Uses subject, run1, run2, run3
        columns only.  Any other columns in this file are ignored.

    data/fMRI/fd_summary.csv
        Per-subject × per-run mean FD.  Run values normalised to integers 1–3.
        The "flagged" column, if present, is ignored when defining QC.

    data/fMRI/negative_persistence_cross_run.csv
        Cross-run persistence results.  Uses MIDUSID, hemisphere, and n_pairs.
        Left-hemisphere row (hemisphere == "L") supplies n_pairs per subject.

    data/fMRI/run_completeness.csv
        Per-subject run-level volume counts (from summarize_run_completeness.py).
        All fields are used as-is; no recomputation is performed here.

Outputs:
    data/fMRI/fmri_qc_processed.csv

Run from the MR1_validation/ directory.
"""

import os
import sys

import pandas as pd

# ============================================================================
# Paths
# ============================================================================
FMRI_DIR = "data/fMRI"

QC_FILE           = os.path.join(FMRI_DIR, "task_fMRI_QC_MR1.xlsx")
FD_FILE           = os.path.join(FMRI_DIR, "fd_summary.csv")
PERSIST_FILE      = os.path.join(FMRI_DIR, "negative_persistence_cross_run.csv")
COMPLETENESS_FILE = os.path.join(FMRI_DIR, "run_completeness.csv")
OUTPUT_FILE       = os.path.join(FMRI_DIR, "fmri_qc_processed.csv")

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
    Accepts '1', '01', 'run-01', etc.  Returns None for anything unparseable
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
    print("Processing fMRI QC data (MR1 reproduction)")
    print("=" * 80)

    # ========================================================================
    # 1. Visual-QC Excel file
    #    Historical MR1 manual-QC decisions carried forward — no new QC was
    #    performed for the reproduction.
    #    Use subject, run1, run2, run3 columns only; drop all others so they
    #    cannot produce merge suffixes with FD columns.
    # ========================================================================
    print(f"\nLoading visual-QC data from {QC_FILE}...")
    qc = pd.read_excel(QC_FILE)
    _require_columns(qc, ["subject", "run1", "run2", "run3"], "task_fMRI_QC_MR1.xlsx")
    qc = qc[["subject", "run1", "run2", "run3"]].copy()
    print(f"  Loaded {len(qc)} rows")

    # Convert subject (sub-XXXXX) → integer MIDUSID
    qc["MIDUSID"] = qc["subject"].str.replace("sub-", "", regex=False).astype(int)

    # Require one unique QC row per participant
    dup_ids = qc["MIDUSID"][qc["MIDUSID"].duplicated()].unique()
    if len(dup_ids):
        print("ERROR: duplicate MIDUSID rows in task_fMRI_QC_MR1.xlsx")
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
    # 2. FD summary  (reproduced FD — replicates historical MR1 computation)
    # ========================================================================
    print(f"\nLoading FD data from {FD_FILE}...")
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

    # Convert subject (sub-XXXXX) → integer MIDUSID
    fd["MIDUSID"] = fd["subject"].str.replace("sub-", "", regex=False).astype(int)

    # Reject duplicate subject × run records
    dup_mask = fd.duplicated(subset=["MIDUSID", "run_int"], keep=False)
    if dup_mask.any():
        dup_pairs = fd.loc[dup_mask, ["MIDUSID", "run_int"]].drop_duplicates()
        print(f"ERROR: {len(dup_pairs)} duplicate subject × run pair(s) in fd_summary.csv")
        sys.exit(1)

    n_fd_participants = fd["MIDUSID"].nunique()
    run_counts = fd.groupby("MIDUSID")["run_int"].count()
    print(f"  FD participants: {n_fd_participants}")
    print(f"  FD run-count distribution:")
    for n_runs, n_subj in run_counts.value_counts().sort_index().items():
        print(f"    {n_runs} run(s): {n_subj} participant(s)")

    # Pivot to wide format: run1_fd, run2_fd, run3_fd
    fd_wide = fd.pivot(index="MIDUSID", columns="run_int", values="mean_fd")
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
    # 3. Persistence  (n_pairs from left-hemisphere rows)
    # ========================================================================
    print(f"\nLoading persistence data from {PERSIST_FILE}...")
    persist = pd.read_csv(PERSIST_FILE)
    _require_columns(
        persist, ["MIDUSID", "hemisphere", "mean_r", "n_pairs"],
        "negative_persistence_cross_run.csv"
    )

    # Reject duplicate MIDUSID × hemisphere records
    if persist.duplicated(subset=["MIDUSID", "hemisphere"], keep=False).any():
        print("ERROR: duplicate MIDUSID × hemisphere records in "
              "negative_persistence_cross_run.csv")
        sys.exit(1)

    # Left-hemisphere n_pairs for each subject
    persist_left = persist[persist["hemisphere"] == "L"][["MIDUSID", "n_pairs"]].copy()
    print(f"  Left-hemisphere n_pairs records: {len(persist_left)}")

    # ========================================================================
    # 4. Run completeness  (231-volume rule; fields used as-is from csv)
    # ========================================================================
    print(f"\nLoading run-completeness data from {COMPLETENESS_FILE}...")
    completeness = pd.read_csv(COMPLETENESS_FILE)
    _require_columns(
        completeness,
        ["MIDUSID", "run1_n_volumes", "run2_n_volumes", "run3_n_volumes",
         "n_task_runs", "n_complete_231_runs", "all_three_runs_231"],
        "run_completeness.csv"
    )

    if completeness["MIDUSID"].duplicated().any():
        print("ERROR: duplicate MIDUSID in run_completeness.csv")
        sys.exit(1)

    invalid_vol_flag = ~completeness["all_three_runs_231"].isin([0, 1])
    if invalid_vol_flag.any():
        print("ERROR: all_three_runs_231 contains values other than 0 or 1")
        sys.exit(1)

    completeness = completeness[[
        "MIDUSID", "run1_n_volumes", "run2_n_volumes", "run3_n_volumes",
        "n_task_runs", "n_complete_231_runs", "all_three_runs_231",
    ]].copy()
    print(f"  Loaded {len(completeness)} rows")

    # ========================================================================
    # 5. Merge — validate cardinality at each step
    #    Visual-QC table is the reference; all joins are left joins on MIDUSID.
    #    Missing data in any joined table fails the corresponding criterion.
    # ========================================================================
    n_ref = len(qc)

    qc = qc.merge(fd_wide, on="MIDUSID", how="left")
    if len(qc) != n_ref:
        print(f"ERROR: FD merge changed row count ({n_ref} → {len(qc)}). "
              f"fd_wide has duplicate MIDUSID values.")
        sys.exit(1)

    qc = qc.merge(persist_left, on="MIDUSID", how="left")
    if len(qc) != n_ref:
        print(f"ERROR: n_pairs merge changed row count ({n_ref} → {len(qc)}). "
              f"persist_left has duplicate MIDUSID values.")
        sys.exit(1)

    qc = qc.merge(completeness, on="MIDUSID", how="left")
    if len(qc) != n_ref:
        print(f"ERROR: run-completeness merge changed row count ({n_ref} → {len(qc)}). "
              f"completeness has duplicate MIDUSID values.")
        sys.exit(1)

    # ========================================================================
    # 6. Derive QC flags
    # ========================================================================
    # fd_pass: mean_fd < threshold; NaN (no FD data) fails
    qc["fd_pass"] = (
        qc["mean_fd"].notna() & (qc["mean_fd"] < FD_THRESHOLD)
    ).astype(int)

    # task_complete: n_pairs == 6; NaN (no persistence data) fails
    qc["task_complete"] = (
        qc["n_pairs"].notna() & (qc["n_pairs"] == 6)
    ).astype(int)

    # Conservative sample: all four criteria must be true.
    # NaN in all_three_runs_231 (participant absent from completeness file)
    # evaluates as != 1, so it correctly fails the criterion.
    qc["qc_conservative"] = (
        (qc["all_runs_pass"] == 1) &
        (qc["fd_pass"] == 1) &
        (qc["task_complete"] == 1) &
        (qc["all_three_runs_231"] == 1)
    ).astype(int)

    # ========================================================================
    # 7. Summary  (aggregate counts only — no participant identifiers printed)
    # ========================================================================
    print("\n" + "=" * 80)
    print("QC Summary")
    print("=" * 80)

    n = len(qc)
    print(f"\nRun-level visual QC  (N={n}, historical MR1 decisions carried forward):")
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
    print(f"  FD pass (mean_fd < {FD_THRESHOLD}): "
          f"{qc['fd_pass'].sum()} ({qc['fd_pass'].sum()/n*100:.1f}%)")
    print(f"  FD fail (includes missing): {(qc['fd_pass'] == 0).sum()}")

    n_pairs_missing = qc["n_pairs"].isna().sum()
    print(f"\nTask completeness  (n_pairs == 6):")
    print(f"  Complete:   {qc['task_complete'].sum()} ({qc['task_complete'].sum()/n*100:.1f}%)")
    print(f"  Incomplete: {(qc['task_complete'] == 0).sum()}  "
          f"(includes {n_pairs_missing} missing)")

    n_vol_missing = qc["all_three_runs_231"].isna().sum()
    n_vol_pass    = int((qc["all_three_runs_231"] == 1).sum())
    print(f"\nRun-volume completeness  (all_three_runs_231 == 1):")
    print(f"  All 3 runs at 231 volumes: {n_vol_pass} ({n_vol_pass/n*100:.1f}%)")
    print(f"  Missing completeness data: {n_vol_missing}")
    print(f"  Fail (includes missing):   {n - n_vol_pass}")

    print(f"\nConservative sample  "
          f"(all_runs_pass=1 AND fd_pass=1 AND task_complete=1 AND all_three_runs_231=1):")
    print(f"  N = {qc['qc_conservative'].sum()} of {n} "
          f"({qc['qc_conservative'].sum()/n*100:.1f}%)")

    excluded  = qc[qc["qc_conservative"] == 0]
    excl_vis  = excluded[
        (excluded["all_runs_pass"] == 0) &
        (excluded["fd_pass"] == 1) &
        (excluded["task_complete"] == 1) &
        (excluded["all_three_runs_231"] == 1)
    ]
    excl_fd   = excluded[
        (excluded["fd_pass"] == 0) &
        (excluded["all_runs_pass"] == 1) &
        (excluded["task_complete"] == 1) &
        (excluded["all_three_runs_231"] == 1)
    ]
    excl_task = excluded[
        (excluded["task_complete"] == 0) &
        (excluded["all_runs_pass"] == 1) &
        (excluded["fd_pass"] == 1) &
        (excluded["all_three_runs_231"] == 1)
    ]
    excl_vol  = excluded[
        (excluded["all_three_runs_231"] != 1) &
        (excluded["all_runs_pass"] == 1) &
        (excluded["fd_pass"] == 1) &
        (excluded["task_complete"] == 1)
    ]
    print(f"\nExclusion breakdown  (mutually exclusive primary reason):")
    print(f"  Failed visual QC only:        {len(excl_vis)}")
    print(f"  Failed FD only:               {len(excl_fd)}")
    print(f"  Failed task complete only:    {len(excl_task)}")
    print(f"  Failed 231-volume rule only:  {len(excl_vol)}")
    print(f"  Failed multiple criteria:     "
          f"{len(excluded) - len(excl_vis) - len(excl_fd) - len(excl_task) - len(excl_vol)}")
    print(f"  Total excluded:               {len(excluded)}")

    # ========================================================================
    # 8. Save
    # ========================================================================
    output_cols = [
        "MIDUSID",
        "run1_pass", "run2_pass", "run3_pass", "all_runs_pass",
        "run1_fd", "run2_fd", "run3_fd", "mean_fd", "fd_pass",
        "n_pairs", "task_complete",
        "run1_n_volumes", "run2_n_volumes", "run3_n_volumes",
        "n_task_runs", "n_complete_231_runs", "all_three_runs_231",
        "qc_conservative",
    ]
    qc[output_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved {len(qc)} subjects to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
