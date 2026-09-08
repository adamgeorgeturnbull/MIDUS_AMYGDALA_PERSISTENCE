#!/bin/bash
# getMotion.sh
#
# SLURM job: Extract framewise displacement QC metrics from fMRIPrep confounds.
#
# SLURM wrapper around an embedded Python script that performs the same
# computation as getMotion.py. Reads all fMRIPrep confound files, computes
# mean FD, spike counts, and flags high-motion runs.
#
# QC thresholds: mean FD > 0.5 mm or >20% of volumes above 0.9 mm
#
# Output: MR1_reproduction_20260828/fd_qc/fd_summary.csv (one row per subject x run)
#
#SBATCH -J FD_QC_MR1
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/fd_qc_%j.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/fd_qc_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL

set -euo pipefail

# Load necessary modules
ml python/3.12.1
ml py-pandas/2.2.1_py312

# Run Python script
python3 << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import glob
import pandas as pd

# Paths
confounds_dir = "/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/derivatives"
output_dir    = "/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/fd_qc"
output_path   = os.path.join(output_dir, "fd_summary.csv")
os.makedirs(output_dir, exist_ok=True)

# Parameters
mean_fd_thresh   = 0.5
spike_fd_thresh  = 0.9
spike_pct_thresh = 0.2  # 20%

# Hard validation: derivatives directory must exist
if not os.path.isdir(confounds_dir):
    sys.exit(f"ABORT: derivatives directory not found: {confounds_dir}")

# Find confound files restricted to task-EmotionRegulation, runs 01-03
confound_files = sorted(glob.glob(
    os.path.join(
        confounds_dir, "*", "func",
        "*_task-EmotionRegulation_run-0[123]_desc-confounds_timeseries.tsv"
    )
))

# Hard validation: exactly 368 confound files expected
if len(confound_files) != 368:
    sys.exit(f"ABORT: expected 368 confound files, found {len(confound_files)}")

print(f"Confound files found: {len(confound_files)}")

results = []
seen_keys = set()

for i, cf in enumerate(confound_files):
    fname = os.path.basename(cf)
    pos = i + 1  # 1-based position in sorted input list

    # Strict full-filename match — abort if pattern does not match exactly
    m = re.fullmatch(
        r'(sub-[0-9]+)_task-EmotionRegulation_run-(0[123])_desc-confounds_timeseries\.tsv',
        fname
    )
    if m is None:
        sys.exit(f"ABORT: filename at position {pos} does not match expected pattern")
    sub = m.group(1)
    run = m.group(2)

    # Hard validation: no duplicate subject x run keys
    key = (sub, run)
    if key in seen_keys:
        sys.exit(f"ABORT: duplicate key at position {pos} (run {run})")
    seen_keys.add(key)

    # Hard validation: file must be readable — exception text omitted (may contain path)
    try:
        df = pd.read_csv(cf, sep='\t')
    except Exception:
        sys.exit(f"ABORT: could not read confound file at position {pos}")

    # Hard validation: confound table must not be empty
    if df.shape[0] == 0:
        sys.exit(f"ABORT: empty confound table at position {pos} (run {run})")

    # Hard validation: framewise_displacement column must be present
    if 'framewise_displacement' not in df.columns:
        sys.exit(f"ABORT: framewise_displacement absent at position {pos} (run {run})")

    # Hard validation: all nonmissing FD values must convert to numeric
    raw_fd = df['framewise_displacement']
    fd = pd.to_numeric(raw_fd, errors='coerce')
    bad = raw_fd[raw_fd.notna() & fd.isna()]
    if len(bad) > 0:
        sys.exit(
            f"ABORT: {len(bad)} nonmissing FD value(s) not numeric "
            f"at position {pos} (run {run})"
        )

    # Hard validation: at least one valid FD observation required
    if fd.notna().sum() == 0:
        sys.exit(f"ABORT: zero valid FD observations at position {pos} (run {run})")

    mean_fd   = fd.mean(skipna=True)
    n_spikes  = int((fd > spike_fd_thresh).sum())
    pct_spikes = n_spikes / fd.shape[0]  # denominator: total confound rows

    # flagged retained for provenance; downstream QC must not use this column
    flagged = bool((mean_fd > mean_fd_thresh) or (pct_spikes > spike_pct_thresh))

    results.append({
        "subject":    sub,
        "run":        run,
        "mean_fd":    mean_fd,
        "n_spikes":   n_spikes,
        "pct_spikes": pct_spikes,
        "flagged":    flagged,
    })

# Sort by subject and run before validation and saving
results_df = (
    pd.DataFrame(results)
    .sort_values(["subject", "run"])
    .reset_index(drop=True)
)

# Hard validation: final participant count must be exactly 123
n_participants = results_df["subject"].nunique()
if n_participants != 123:
    sys.exit(f"ABORT: expected 123 participants, found {n_participants}")

# Hard validation: run-count distribution must be exactly 122 with 3 runs, 1 with 2 runs
run_counts = results_df.groupby("subject")["run"].count()
dist = run_counts.value_counts().to_dict()
expected_dist = {3: 122, 2: 1}
if dist != expected_dist:
    sys.exit(f"ABORT: unexpected run-count distribution: {dict(sorted(dist.items()))}")

# Aggregate-only console output — no participant identifiers or per-file paths
print(f"Participants: {n_participants}")
print(f"Run-count distribution: {dict(sorted(dist.items()))} (n_runs: n_participants)")
print(f"Flagged runs: {int(results_df['flagged'].sum())} of {len(results_df)}")

results_df.to_csv(output_path, index=False)
print(f"Output: {output_path}")
EOF
