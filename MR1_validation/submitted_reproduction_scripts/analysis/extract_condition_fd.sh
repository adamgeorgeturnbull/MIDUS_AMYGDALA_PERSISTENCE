#!/bin/bash
# extract_condition_fd.sh  (MR1 / MIDUS Refresher)
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
#   /scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/condition_fd/
#       <subid>_condition_fd.csv
#     Columns: subject, run, valence, mean_fd_condition, n_trs, n_pairs
#
#SBATCH -J condition_fd_MR1
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/condition_fd_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/condition_fd_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-123%20

set -euo pipefail

module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

SUBJECT_LIST="/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/MR1_completed_subject_list.txt"

# Validate subject list: must exist, contain exactly 123 nonblank entries,
# all matching ^sub-[0-9]+$, and all unique.
if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "ABORT: subject list not found: $SUBJECT_LIST" >&2
    exit 1
fi
n_subjects=$(awk 'NF' "$SUBJECT_LIST" | wc -l)
if [[ "$n_subjects" -ne 123 ]]; then
    echo "ABORT: subject list has $n_subjects nonblank entries, expected 123" >&2
    exit 1
fi
n_pattern=$(awk '/^sub-[0-9]+$/' "$SUBJECT_LIST" | wc -l)
if [[ "$n_pattern" -ne "$n_subjects" ]]; then
    echo "ABORT: subject list contains entries not matching ^sub-[0-9]+$" >&2
    exit 1
fi
n_unique=$(awk '/^sub-[0-9]+$/' "$SUBJECT_LIST" | sort -u | wc -l)
if [[ "$n_unique" -ne "$n_subjects" ]]; then
    echo "ABORT: subject list contains duplicate entries" >&2
    exit 1
fi

# Validate array task ID
if [[ "$SLURM_ARRAY_TASK_ID" -lt 1 || "$SLURM_ARRAY_TASK_ID" -gt 123 ]]; then
    echo "ABORT: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [1,123]" >&2
    exit 1
fi

# Get subject and validate format — do not echo the identifier
subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
if [[ ! "$subid" =~ ^sub-[0-9]+$ ]]; then
    echo "ABORT: subject at position ${SLURM_ARRAY_TASK_ID} does not match expected pattern" >&2
    exit 1
fi

derivatives_dir=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/derivatives
glm_dir=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/GLM_output
out_dir=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/condition_fd
mkdir -p "$out_dir"

export derivatives_dir
export glm_dir
export out_dir
export subid

python3 << 'EOF'
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '?')
subid         = os.environ['subid']
deriv_dir     = Path(os.environ['derivatives_dir'])
glm_dir       = Path(os.environ['glm_dir'])
out_dir       = Path(os.environ['out_dir'])

TR       = 2.0
VALENCES = ['neg', 'neu', 'pos']

rows           = []
completed_runs = 0

for run in ['01', '02', '03']:
    conf_file   = (deriv_dir / subid / 'func' /
                   f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv")
    events_file = glm_dir / subid / f"run-{run}_clean_events.tsv"

    conf_exists   = conf_file.exists()
    events_exists = events_file.exists()

    # Both absent: run was not processed; allowed for one run
    if not conf_exists and not events_exists:
        continue
    # Exactly one present: inconsistent state
    if conf_exists != events_exists:
        sys.exit(
            f"ABORT: run {run} has exactly one input — "
            f"confounds={'present' if conf_exists else 'absent'}, "
            f"events={'present' if events_exists else 'absent'} "
            f"(array task {array_task_id})"
        )

    # ── Confounds ────────────────────────────────────────────────────────────
    try:
        conf = pd.read_csv(conf_file, sep='\t')
    except Exception:
        sys.exit(f"ABORT: run {run} confounds file is not readable (array task {array_task_id})")

    if 'framewise_displacement' not in conf.columns:
        sys.exit(f"ABORT: run {run} confounds missing framewise_displacement column")

    # Remove first 4 dummy volumes; validate post-trim FD is numeric and finite.
    # Clean-event onsets were already shifted by 8 s in runGLM.sh — do not subtract again.
    raw_fd      = conf['framewise_displacement']
    fd_coerced  = pd.to_numeric(raw_fd, errors='coerce')
    raw_trimmed = raw_fd.iloc[4:]
    fd_trimmed  = fd_coerced.iloc[4:]

    bad_nonnumeric = raw_trimmed[raw_trimmed.notna() & fd_trimmed.isna()]
    if len(bad_nonnumeric) > 0:
        sys.exit(
            f"ABORT: run {run} has {len(bad_nonnumeric)} non-numeric FD value(s) after trim"
        )
    if fd_trimmed.isna().any():
        sys.exit(
            f"ABORT: run {run} has {int(fd_trimmed.isna().sum())} missing FD value(s) after trim"
        )
    fd_series = fd_trimmed.to_numpy(dtype=float)
    if not np.isfinite(fd_series).all():
        sys.exit(f"ABORT: run {run} has non-finite FD value(s) after trim")

    n_trs_total = len(fd_series)

    # ── Events ───────────────────────────────────────────────────────────────
    try:
        events = pd.read_csv(events_file, sep='\t')
    except Exception:
        sys.exit(f"ABORT: run {run} events file is not readable (array task {array_task_id})")

    required_cols = ['onset', 'duration', 'trial_type']
    missing_cols  = [c for c in required_cols if c not in events.columns]
    if missing_cols:
        sys.exit(f"ABORT: run {run} events missing column(s): {missing_cols}")

    for col in ['onset', 'duration']:
        raw         = events[col].copy()
        events[col] = pd.to_numeric(raw, errors='coerce')
        bad_conv    = raw[raw.notna() & events[col].isna()]
        if len(bad_conv) > 0:
            sys.exit(f"ABORT: run {run} events '{col}' has {len(bad_conv)} non-numeric value(s)")
        if events[col].isna().any():
            sys.exit(
                f"ABORT: run {run} events '{col}' has "
                f"{int(events[col].isna().sum())} missing value(s)"
            )
        if not np.isfinite(events[col].values).all():
            sys.exit(f"ABORT: run {run} events '{col}' has non-finite value(s)")

    if (events['onset'] < 0).any():
        sys.exit(f"ABORT: run {run} has {int((events['onset'] < 0).sum())} negative onset(s)")

    # ── Per-valence FD extraction ─────────────────────────────────────────────
    for valence in VALENCES:
        img_events  = (events[events['trial_type'] == f"{valence}_image"]
                       .sort_values('onset').reset_index(drop=True))
        face_events = (events[events['trial_type'] == f"{valence}_face"]
                       .sort_values('onset').reset_index(drop=True))

        if len(img_events) == 0:
            sys.exit(f"ABORT: run {run} has no {valence}_image events")
        if len(face_events) == 0:
            sys.exit(f"ABORT: run {run} has no {valence}_face events")
        if len(img_events) != len(face_events):
            sys.exit(
                f"ABORT: run {run} {valence} image/face count mismatch "
                f"({len(img_events)} img, {len(face_events)} face)"
            )

        n_pairs    = len(img_events)
        tr_indices = set()

        for i in range(n_pairs):
            window_start = img_events.loc[i, 'onset']
            window_end   = face_events.loc[i, 'onset'] + face_events.loc[i, 'duration']

            if window_end <= window_start:
                sys.exit(
                    f"ABORT: run {run} {valence} pair {i} has invalid window "
                    f"(start={window_start:.3f}, end={window_end:.3f})"
                )

            first_tr = int(np.floor(window_start / TR))
            last_tr  = int(np.floor((window_end - 1e-9) / TR))

            if first_tr < 0:
                sys.exit(
                    f"ABORT: run {run} {valence} pair {i} window starts before TR 0 "
                    f"(first_tr={first_tr})"
                )
            if last_tr >= n_trs_total:
                sys.exit(
                    f"ABORT: run {run} {valence} pair {i} window extends past last TR "
                    f"(last_tr={last_tr}, n_trs={n_trs_total})"
                )

            for t in range(first_tr, last_tr + 1):
                tr_indices.add(t)

        if not tr_indices:
            sys.exit(f"ABORT: run {run} {valence} FD window is empty")

        tr_indices = sorted(tr_indices)
        fd_vals    = fd_series[tr_indices]

        if not np.isfinite(fd_vals).all():
            sys.exit(f"ABORT: run {run} {valence} FD window contains non-finite values")

        rows.append({
            'subject':           subid,
            'run':               f"run-{run}",
            'valence':           valence,
            'mean_fd_condition': float(np.mean(fd_vals)),
            'n_trs':             len(tr_indices),
            'n_pairs':           n_pairs,
        })

    completed_runs += 1

# Require 2 or 3 completed runs
if completed_runs < 2:
    sys.exit(
        f"ABORT: fewer than 2 runs completed ({completed_runs}) (array task {array_task_id})"
    )
if completed_runs > 3:
    sys.exit(
        f"ABORT: more than 3 runs completed ({completed_runs}) (array task {array_task_id})"
    )

# Validate row count: exactly completed_runs × 3
expected_rows = completed_runs * 3
if len(rows) != expected_rows:
    sys.exit(f"ABORT: expected {expected_rows} rows, got {len(rows)}")

# Validate unique run × valence keys
seen_keys = set()
for row in rows:
    key = (row['run'], row['valence'])
    if key in seen_keys:
        sys.exit(f"ABORT: duplicate output key {key}")
    seen_keys.add(key)

df       = pd.DataFrame(rows)
out_file = out_dir / f"{subid}_condition_fd.csv"
df.to_csv(out_file, index=False)

print(f"Array task {array_task_id}: completed {completed_runs} run(s), saved {len(df)} rows")
EOF
