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
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/fd_qc_%j.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/fd_qc_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL

set -euo pipefail

# Load necessary modules
ml python/3.12.1
ml py-pandas/2.2.1_py312

DERIVATIVES_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/derivatives"
OUTPUT_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/fd_qc"

if [ ! -d "$DERIVATIVES_DIR" ]; then
  echo "ERROR: derivatives directory not found: $DERIVATIVES_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export DERIVATIVES_DIR OUTPUT_DIR

# Run Python script
python3 << 'EOF'
import os
import glob
import sys
import pandas as pd

derivatives_dir = os.environ['DERIVATIVES_DIR']
output_dir      = os.environ['OUTPUT_DIR']

# Parameters
mean_fd_thresh  = 0.5
spike_fd_thresh = 0.9
spike_pct_thresh = 0.2  # 20%

# Find EmotionRegulation confounds for runs 01-03 only
confound_files = sorted(glob.glob(os.path.join(
    derivatives_dir, "*", "func",
    "*_task-EmotionRegulation_run-0[123]_desc-confounds_timeseries.tsv"
)))

if not confound_files:
    print(f"ERROR: no EmotionRegulation confound files found under {derivatives_dir}")
    sys.exit(1)

print(f"Found {len(confound_files)} confound file(s)")

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
    mean_fd    = fd.mean(skipna=True)
    n_spikes   = (fd > spike_fd_thresh).sum()
    pct_spikes = n_spikes / fd.shape[0]

    flagged = (mean_fd > mean_fd_thresh) or (pct_spikes > spike_pct_thresh)

    results.append({
        "subject":    sub,
        "run":        run,
        "mean_fd":    mean_fd,
        "n_spikes":   n_spikes,
        "pct_spikes": pct_spikes,
        "flagged":    flagged
    })

results_df = pd.DataFrame(results)
out_file   = os.path.join(output_dir, "fd_summary.csv")
results_df.to_csv(out_file, index=False)
print(f"QC summary saved to: {out_file}  ({len(results_df)} rows)")
EOF
