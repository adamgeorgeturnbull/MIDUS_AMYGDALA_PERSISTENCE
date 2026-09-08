#!/bin/bash
# runGLM.sh  (MR1 / MIDUS Refresher)
#
# SLURM array job: Per-run first-level GLM for the MR1 EmotionRegulation task.
#
# Identical model to the M3 pipeline (paradigm is the same): fits a nilearn
# FirstLevelModel to each run separately using fMRIPrep-preprocessed BOLD in
# MNI space. 6 trial types (neg/neu/pos image and face) convolved with the
# Glover HRF, plus 24 motion regressors and cosine drift terms. First 4
# dummy volumes removed from BOLD and confounds.
#
# GLM parameters:
#   - TR = 2.0s, slice_time_ref = 0.5, HRF = glover
#   - Drift: cosine, high_pass = 1/128 Hz
#   - Noise model: AR(1)
#   - 24 motion parameters (6 params + derivatives + squares)
#
# Output per subject per run (in MR1_reproduction_20260828/GLM_output/<subid>/):
#   - run-XX_<condition>_beta.nii.gz   (beta maps for each trial type)
#   - run-XX_clean_events.tsv          (reformatted events used in GLM)
#   - run-XX_design_matrix.csv         (full design matrix)
#
#SBATCH -J glm_MR1
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/glm_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/glm_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-123%20

set -euo pipefail

# Load modules
module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

# Paths
SUBJECT_LIST=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/MR1_completed_subject_list.txt
bids_root_dir=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/MR_P5_ImagingSession
derivatives_dir=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/derivatives
out_dir=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/GLM_output
mkdir -p "$out_dir"
export bids_root_dir=$bids_root_dir
export derivatives_dir=$derivatives_dir
export out_dir=$out_dir

# Validate subject list: must exist and contain exactly 123 nonblank entries
if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "ABORT: subject list not found: $SUBJECT_LIST" >&2
    exit 1
fi
n_subjects=$(grep -c '[^[:space:]]' "$SUBJECT_LIST")
if [[ "$n_subjects" -ne 123 ]]; then
    echo "ABORT: subject list has $n_subjects nonblank entries, expected 123" >&2
    exit 1
fi

# Validate array task ID
if [[ "$SLURM_ARRAY_TASK_ID" -lt 1 || "$SLURM_ARRAY_TASK_ID" -gt 123 ]]; then
    echo "ABORT: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [1,123]" >&2
    exit 1
fi

# Get subject and validate format
subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
if [[ ! "$subid" =~ ^sub-[0-9]+$ ]]; then
    echo "ABORT: subject at position ${SLURM_ARRAY_TASK_ID} does not match expected pattern" >&2
    exit 1
fi
export subid=$subid

# Run the Python analysis
python3 << 'EOF'
import sys

# Nilearn import and version check — abort before any further work on failure
try:
    import nilearn
except ImportError:
    sys.exit("ABORT: nilearn is not importable")
if nilearn.__version__ != '0.12.0':
    sys.exit(f"ABORT: nilearn {nilearn.__version__} != required 0.12.0")

import os
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image

array_task_id   = os.environ.get('SLURM_ARRAY_TASK_ID', '?')
subid           = os.environ['subid']
bids_root_dir   = Path(os.environ['bids_root_dir'])
derivatives_dir = Path(os.environ['derivatives_dir'])
out_dir         = Path(os.environ['out_dir'])
sub_out_dir     = out_dir / subid
sub_out_dir.mkdir(parents=True, exist_ok=True)

N_DUMMIES = 4
TR        = 2.0

motion_columns = [
    'trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
    'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
    'trans_z','trans_z_derivative1','trans_z_power2','trans_z_derivative1_power2',
    'rot_x','rot_x_derivative1','rot_x_power2','rot_x_derivative1_power2',
    'rot_y','rot_y_derivative1','rot_y_power2','rot_y_derivative1_power2',
    'rot_z','rot_z_derivative1','rot_z_power2','rot_z_derivative1_power2',
]

# Pre-loop: count available BOLD runs; require at least two
available_bolds = []
for r in ['01', '02', '03']:
    bf = (derivatives_dir / subid / 'func' /
          f"{subid}_task-EmotionRegulation_run-{r}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    if bf.exists():
        available_bolds.append(r)

if len(available_bolds) < 2:
    sys.exit(f"ABORT: fewer than 2 BOLD runs found (array task {array_task_id})")

completed_runs = 0

for run in ['01', '02', '03']:
    bold_file      = (derivatives_dir / subid / 'func' /
                      f"{subid}_task-EmotionRegulation_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    confounds_file = (derivatives_dir / subid / 'func' /
                      f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv")
    events_file    = (bids_root_dir / subid / 'func' /
                      f"{subid}_task-EmotionRegulation_run-{run}_events.tsv")

    # Missing BOLD: skip (allowed; at-least-2 already enforced above)
    if not bold_file.exists():
        print(f"  run {run}: BOLD absent, skipping")
        continue

    # BOLD exists: confounds and events must both be present
    if not confounds_file.exists():
        sys.exit(f"ABORT: run {run} BOLD exists but confounds file is absent")
    if not events_file.exists():
        sys.exit(f"ABORT: run {run} BOLD exists but events file is absent")

    print(f"Array task {array_task_id}: run {run}")

    # ── BOLD validation ────────────────────────────────────────────────────────
    bold_nib = nib.load(str(bold_file))
    if bold_nib.ndim != 4:
        sys.exit(f"ABORT: run {run} BOLD is not 4D (shape {bold_nib.shape})")
    n_bold_vols = bold_nib.shape[3]
    if n_bold_vols <= 4:
        sys.exit(f"ABORT: run {run} BOLD has {n_bold_vols} volume(s), must be > 4")

    # ── Confound loading and alignment ─────────────────────────────────────────
    confounds = pd.read_csv(confounds_file, sep='\t')
    if len(confounds) != n_bold_vols:
        sys.exit(f"ABORT: run {run} confound rows ({len(confounds)}) != BOLD volumes ({n_bold_vols})")

    missing_motion = [c for c in motion_columns if c not in confounds.columns]
    if missing_motion:
        sys.exit(f"ABORT: run {run} missing {len(missing_motion)} motion column(s): {missing_motion}")

    # ── Events loading and validation ──────────────────────────────────────────
    events = pd.read_csv(events_file, sep='\t')

    required_event_cols = ['onset', 'duration', 'database', 'valence']
    missing_event_cols  = [c for c in required_event_cols if c not in events.columns]
    if missing_event_cols:
        sys.exit(f"ABORT: run {run} events missing column(s): {missing_event_cols}")

    for col in ['onset', 'duration']:
        raw         = events[col].copy()
        events[col] = pd.to_numeric(raw, errors='coerce')
        bad_conv    = raw[raw.notna() & events[col].isna()]
        if len(bad_conv) > 0:
            sys.exit(f"ABORT: run {run} events '{col}' has {len(bad_conv)} non-numeric value(s)")
        if events[col].isna().any():
            sys.exit(f"ABORT: run {run} events '{col}' has {events[col].isna().sum()} missing value(s)")

    # ── Valence normalization (MR1 events use full words) ──────────────────────
    val_map = {'negative': 'neg', 'neutral': 'neu', 'positive': 'pos'}
    events['valence'] = events['valence'].map(val_map).fillna(events['valence'])

    # ── Timing alignment: subtract dummy duration ──────────────────────────────
    # MR1 confirmed: N_DUMMIES = 4 (dummies in-file; BIDS metadata reports zero
    # discarded volumes; no onset_trimmed column present in MR1 events files).
    events['onset'] = events['onset'] - N_DUMMIES * TR
    if (events['onset'] < 0).any():
        sys.exit(f"ABORT: run {run} has {(events['onset'] < 0).sum()} negative onset(s) after dummy removal")

    # ── valenceFollowing derivation ────────────────────────────────────────────
    events       = events.sort_values('onset').reset_index(drop=True)
    img_val_fwd  = events['valence'].replace('n/a', np.nan).ffill()
    events['valenceFollowing'] = np.where(events['database'] == 'faces', img_val_fwd, np.nan)

    # ── Build events_glm ──────────────────────────────────────────────────────
    clean_events_list = []
    conditions = ['neg', 'neu', 'pos']
    for val in conditions:
        df_img = events[events['valence'] == val][['onset', 'duration']].copy()
        df_img['trial_type'] = f"{val}_image"
        clean_events_list.append(df_img)
        df_face = events[events['valenceFollowing'] == val][['onset', 'duration']].copy()
        df_face['trial_type'] = f"{val}_face"
        clean_events_list.append(df_face)

    events_glm = pd.concat(clean_events_list, ignore_index=True)
    events_glm = events_glm.sort_values('onset').reset_index(drop=True)

    required_types = {'neg_image', 'neu_image', 'pos_image', 'neg_face', 'neu_face', 'pos_face'}
    missing_types  = required_types - set(events_glm['trial_type'].unique())
    if missing_types:
        sys.exit(f"ABORT: run {run} missing trial type(s): {sorted(missing_types)}")

    # ── BOLD trimming ─────────────────────────────────────────────────────────
    bold_img = image.index_img(bold_file, slice(N_DUMMIES, None))

    # ── Confound trimming and post-trim validation ─────────────────────────────
    motion_regressors = confounds[motion_columns].iloc[N_DUMMIES:].reset_index(drop=True)
    n_trimmed_bold    = bold_img.shape[3]
    if len(motion_regressors) != n_trimmed_bold:
        sys.exit(
            f"ABORT: run {run} trimmed confound rows ({len(motion_regressors)}) "
            f"!= trimmed BOLD volumes ({n_trimmed_bold})"
        )
    if motion_regressors.isnull().any().any() or not np.isfinite(motion_regressors.values).all():
        sys.exit(f"ABORT: run {run} trimmed motion matrix contains missing or non-finite values")

    # ── GLM (parameters unchanged from M3) ───────────────────────────────────
    glm = FirstLevelModel(
        t_r=TR,
        slice_time_ref=0.5,
        hrf_model='glover',
        drift_model='cosine',
        high_pass=1/128,
        standardize=False,
        noise_model='ar1',
        minimize_memory=False
    )
    glm.fit(run_imgs=bold_img, events=events_glm, confounds=motion_regressors)

    # ── Outputs (filenames unchanged) ─────────────────────────────────────────
    events_glm.to_csv(sub_out_dir / f"run-{run}_clean_events.tsv", sep='\t', index=False)
    for cond in events_glm['trial_type'].unique():
        beta_map = glm.compute_contrast(cond, output_type='effect_size')
        beta_map.to_filename(sub_out_dir / f"run-{run}_{cond}_beta.nii.gz")
    glm.design_matrices_[0].to_csv(sub_out_dir / f"run-{run}_design_matrix.csv", index=False)

    completed_runs += 1

# Post-loop: verify completed run count matches available runs
if completed_runs != len(available_bolds):
    sys.exit(
        f"ABORT: expected {len(available_bolds)} completed runs, "
        f"got {completed_runs} (array task {array_task_id})"
    )
print(f"Array task {array_task_id}: completed {completed_runs} run(s)")
EOF
