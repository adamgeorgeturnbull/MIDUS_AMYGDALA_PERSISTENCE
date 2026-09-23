#!/bin/bash
# extract_condition_fd.sh
#
# SLURM array job: Extract mean framewise displacement across the full
# image-through-face window for each valence condition (neg, neu, pos).
#
# For each subject and run, reads:
#   - fMRIPrep confounds TSV (per-TR framewise_displacement)
#   - GLM output clean_events.tsv (onset, duration, trial_type per condition)
#
# For each valence, image and face events are paired chronologically
# (1st neg_image with 1st neg_face, etc.). FD is extracted across all TRs
# spanning from the image onset through the ITI to the face offset
# (face onset + duration). This matches the conceptual window used for
# persistence: the spatial correlation between image and face beta maps
# within the same valence condition.
#
# Output:
#   /scratch/groups/fvlin/MIDUS/M3_stc_rerun/condition_fd/
#       <subid>_condition_fd.csv
#     Columns: subject, run, valence, mean_fd_condition, n_trs, n_pairs
#
#SBATCH -J condition_fd
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/condition_fd_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/condition_fd_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-158%20

set -euo pipefail

module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

SUBJECT_LIST="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_subject_list.txt"

if [ ! -f "$SUBJECT_LIST" ]; then
  echo "ERROR: subject list not found: $SUBJECT_LIST"; exit 1
fi

subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
if [ -z "$subid" ]; then
  echo "ERROR: empty subject ID at line ${SLURM_ARRAY_TASK_ID} of $SUBJECT_LIST"; exit 1
fi
if ! [[ "$subid" =~ ^sub-[0-9]+$ ]]; then
  echo "ERROR: subject ID '${subid}' does not match sub-[0-9]+"; exit 1
fi

derivatives_dir=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/derivatives
glm_dir=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/GLM_output
out_dir=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/condition_fd
mkdir -p "$out_dir"

export derivatives_dir
export glm_dir
export out_dir
export subid

echo "[$(date)] Condition FD extraction: $subid"

python3 << 'EOF'
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

subid        = os.environ['subid']
deriv_dir    = Path(os.environ['derivatives_dir'])
glm_dir      = Path(os.environ['glm_dir'])
out_dir      = Path(os.environ['out_dir'])

TR = 2.0
VALENCES = ['neg', 'neu', 'pos']

sub_glm_dir  = glm_dir / subid
sub_func_dir = deriv_dir / subid / 'func'

if not sub_func_dir.exists():
    print(f"No fMRIPrep func dir for {subid}: {sub_func_dir}")
    sys.exit(0)

rows = []

for run in ['01', '02', '03']:
    # fMRIPrep confounds file
    confound_files = list(sub_func_dir.glob(
        f"{subid}_task-EmotionRegulation_run-{run}_*desc-confounds_timeseries.tsv"
    ))
    if not confound_files:
        print(f"  WARNING: no confounds for {subid} run-{run}")
        continue
    conf = pd.read_csv(confound_files[0], sep='\t')
    if 'framewise_displacement' not in conf.columns:
        print(f"  WARNING: no framewise_displacement in confounds for {subid} run-{run}")
        continue
    fd_series = pd.to_numeric(conf['framewise_displacement'], errors='coerce').iloc[4:].to_numpy(dtype=float)
    n_trs_total = len(fd_series)

    # Clean events saved by runGLM.sh
    events_file = sub_glm_dir / f"run-{run}_clean_events.tsv"
    if not events_file.exists():
        print(f"  WARNING: no clean_events for {subid} run-{run}")
        continue
    events = pd.read_csv(events_file, sep='\t')

    for valence in VALENCES:
        img_events  = events[events['trial_type'] == f"{valence}_image"].sort_values('onset').reset_index(drop=True)
        face_events = events[events['trial_type'] == f"{valence}_face"].sort_values('onset').reset_index(drop=True)

        n_pairs = min(len(img_events), len(face_events))
        if n_pairs == 0:
            print(f"  WARNING: no {valence} pairs for {subid} run-{run}")
            continue
        if len(img_events) != len(face_events):
            print(f"  WARNING: {valence} image/face count mismatch for {subid} run-{run} "
                  f"({len(img_events)} img, {len(face_events)} face) — using {n_pairs} pairs")

        # For each pair: TRs from image onset to face offset (onset + duration),
        # capturing the full image → ITI → face sequence
        tr_indices = set()
        for i in range(n_pairs):
            window_start = img_events.loc[i, 'onset']
            window_end   = face_events.loc[i, 'onset'] + face_events.loc[i, 'duration']
            first_tr = int(np.floor(window_start / TR))
            last_tr  = int(np.floor((window_end - 1e-9) / TR))
            for t in range(first_tr, last_tr + 1):
                if 0 <= t < n_trs_total:
                    tr_indices.add(t)

        tr_indices = sorted(tr_indices)
        fd_vals = fd_series[tr_indices]
        fd_vals_clean = fd_vals[~np.isnan(fd_vals)]

        rows.append({
            'subject':           subid,
            'run':               f"run-{run}",
            'valence':           valence,
            'mean_fd_condition': float(np.mean(fd_vals_clean)) if len(fd_vals_clean) > 0 else np.nan,
            'n_trs':             len(tr_indices),
            'n_pairs':           n_pairs,
        })

if not rows:
    print(f"No data extracted for {subid}")
    sys.exit(0)

df = pd.DataFrame(rows)
out_file = out_dir / f"{subid}_condition_fd.csv"
df.to_csv(out_file, index=False)
print(f"Saved: {out_file}  ({len(df)} rows)")
EOF
