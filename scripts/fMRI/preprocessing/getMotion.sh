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
# Output: fd_qc/fd_summary.csv (one row per subject x run)
#
#SBATCH -J FD_QC
#SBATCH --output=/scratch/groups/fvlin/MIDUS/log/fd_qc_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/log/fd_qc_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL

# Load necessary modules
ml python/3.12.1
ml py-pandas/2.2.1_py312

# Run Python script
python3 << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import glob

# Paths
confounds_dir = "/scratch/groups/fvlin/MIDUS/derivatives"
output_dir = "/scratch/groups/fvlin/MIDUS/fd_qc"
os.makedirs(output_dir, exist_ok=True)

# Parameters
mean_fd_thresh = 0.5
spike_fd_thresh = 0.9
spike_pct_thresh = 0.2  # 20%

# Find all confounds files
confound_files = glob.glob(os.path.join(confounds_dir, "*", "func", "*_desc-confounds_timeseries.tsv"))

results = []

for cf in confound_files:
    sub = os.path.basename(cf).split("_")[0]
    run = cf.split("_run-")[1].split("_")[0]

    try:
        df = pd.read_csv(cf, sep='\t')
    except Exception as e:
        print(f"Could not read {cf}: {e}")
        continue

    if 'framewise_displacement' not in df.columns:
        print(f"No framewise_displacement column in {cf}")
        continue

    fd = df['framewise_displacement']
    mean_fd = fd.mean(skipna=True)
    n_spikes = (fd > spike_fd_thresh).sum()
    pct_spikes = n_spikes / fd.shape[0]

    flagged = (mean_fd > mean_fd_thresh) or (pct_spikes > spike_pct_thresh)

    results.append({
        "subject": sub,
        "run": run,
        "mean_fd": mean_fd,
        "n_spikes": n_spikes,
        "pct_spikes": pct_spikes,
        "flagged": flagged
    })

# Save summary
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "fd_summary.csv"), index=False)
print("QC summary saved to:", os.path.join(output_dir, "fd_summary.csv"))
EOF
