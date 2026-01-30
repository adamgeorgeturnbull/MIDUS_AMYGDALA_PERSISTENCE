#!/usr/bin/env python3
"""
09_process_fmri_qc.py

Process fMRI quality control (QC) data from manual inspection.

Creates binary QC flags and criteria for defining conservative analysis samples:
- All 3 runs pass visual QC
- Mean FD < 0.5 mm

Inputs:
- data/fMRI/task_fMRI_QC.xlsx (manual QC ratings)

Outputs:
- data/fMRI/fmri_qc_processed.csv (processed QC flags)

Run from project root directory.
"""

import os

import pandas as pd

# ============================================================================
# Paths
# ============================================================================
FMRI_DIR = "data/fMRI"

QC_FILE = os.path.join(FMRI_DIR, "task_fMRI_QC.xlsx")
OUTPUT_FILE = os.path.join(FMRI_DIR, "fmri_qc_processed.csv")

# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    print("=" * 80)
    print("Processing fMRI QC data")
    print("=" * 80)

    # ========================================================================
    # Load QC Data
    # ========================================================================
    print(f"\nLoading QC data from {QC_FILE}...")
    qc = pd.read_excel(QC_FILE)
    print(f"✓ Loaded {len(qc)} subjects")

    # ========================================================================
    # Extract Subject IDs
    # ========================================================================
    # Convert subject IDs to M2ID format
    qc["M2ID"] = qc["subject"].str.replace("sub-", "", regex=False).astype(int)

    # ========================================================================
    # Create QC Flags
    # ========================================================================
    print("\nCreating QC flags...")

    # Run-level QC: 1 if Pass, 0 otherwise (Fail or NaN)
    qc["run1_pass"] = (qc["run1"] == "Pass").astype(int)
    qc["run2_pass"] = (qc["run2"] == "Pass").astype(int)
    qc["run3_pass"] = (qc["run3"] == "Pass").astype(int)

    # All runs pass
    qc["all_runs_pass"] = (
        (qc["run1_pass"] == 1) &
        (qc["run2_pass"] == 1) &
        (qc["run3_pass"] == 1)
    ).astype(int)

    # FD QC: 1 if mean FD < 0.5, 0 otherwise
    qc["fd_pass"] = (qc["mean_fd"] < 0.5).astype(int)

    # Conservative sample: all runs pass AND mean FD < 0.5
    qc["qc_conservative"] = (
        (qc["all_runs_pass"] == 1) &
        (qc["fd_pass"] == 1)
    ).astype(int)

    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("QC Summary")
    print("=" * 80)

    print(f"\nRun-level QC:")
    print(f"  Run 1 pass: {qc['run1_pass'].sum()} ({qc['run1_pass'].sum()/len(qc)*100:.1f}%)")
    print(f"  Run 2 pass: {qc['run2_pass'].sum()} ({qc['run2_pass'].sum()/len(qc)*100:.1f}%)")
    print(f"  Run 3 pass: {qc['run3_pass'].sum()} ({qc['run3_pass'].sum()/len(qc)*100:.1f}%)")
    print(f"  All runs pass: {qc['all_runs_pass'].sum()} ({qc['all_runs_pass'].sum()/len(qc)*100:.1f}%)")

    print(f"\nFramewise displacement:")
    print(f"  Mean FD < 0.5: {qc['fd_pass'].sum()} ({qc['fd_pass'].sum()/len(qc)*100:.1f}%)")
    print(f"  Mean FD ≥ 0.5: {(qc['fd_pass'] == 0).sum()} ({(qc['fd_pass'] == 0).sum()/len(qc)*100:.1f}%)")
    print(f"  Mean FD range: {qc['mean_fd'].min():.3f} - {qc['mean_fd'].max():.3f}")

    print(f"\nConservative sample criteria:")
    print(f"  All runs pass + FD < 0.5: {qc['qc_conservative'].sum()} ({qc['qc_conservative'].sum()/len(qc)*100:.1f}%)")

    # Breakdown of exclusions
    excluded = qc[qc["qc_conservative"] == 0]
    excluded_runs = excluded[excluded["all_runs_pass"] == 0]
    excluded_fd = excluded[excluded["fd_pass"] == 0]
    both = excluded[(excluded["all_runs_pass"] == 0) & (excluded["fd_pass"] == 0)]

    print(f"\nExclusion breakdown:")
    print(f"  Failed runs only: {len(excluded_runs) - len(both)}")
    print(f"  Failed FD only: {len(excluded_fd) - len(both)}")
    print(f"  Failed both: {len(both)}")
    print(f"  Total excluded: {len(excluded)}")

    # ========================================================================
    # Save Processed QC Data
    # ========================================================================
    # Select columns to save
    output_cols = [
        "M2ID",
        "run1_pass",
        "run2_pass",
        "run3_pass",
        "all_runs_pass",
        "run1_fd",
        "run2_fd",
        "run3_fd",
        "mean_fd",
        "fd_pass",
        "qc_conservative",
    ]

    qc_output = qc[output_cols].copy()

    qc_output.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Processed QC data saved to {OUTPUT_FILE}")
    print(f"  {len(qc_output)} subjects")


if __name__ == "__main__":
    main()
