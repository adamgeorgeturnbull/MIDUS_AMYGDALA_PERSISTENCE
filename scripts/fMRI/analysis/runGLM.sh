#!/bin/bash
# runGLM.sh
#
# SLURM array job: Per-run first-level GLM for MIDUS EmotionRegulation task.
#
# Fits a nilearn FirstLevelModel to each run separately using fMRIPrep-
# preprocessed BOLD data in MNI space. The model includes 6 trial types
# (neg/neu/pos image and face) convolved with the Glover HRF, plus 24
# motion regressors and cosine drift terms.
#
# The first 4 volumes (dummy scans) are removed from both BOLD and confounds.
#
# GLM parameters:
#   - TR = 2.0s, slice_time_ref = 0.5, HRF = glover
#   - Drift: cosine, high_pass = 1/128 Hz
#   - Noise model: AR(1)
#   - 24 motion parameters (6 params + derivatives + squares)
#
# Output per subject per run:
#   - run-XX_<condition>_beta.nii.gz  (beta maps for each trial type)
#   - run-XX_clean_events.tsv         (reformatted events used in GLM)
#   - run-XX_design_matrix.csv        (full design matrix)
#
#SBATCH -J glm_MIDUS
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/glm_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/glm_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-158%20

set -euo pipefail

# Load modules
module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

# Verify nilearn is available in the current environment
python3 -c "import nilearn; print(f'nilearn {nilearn.__version__} available')" || {
  echo "ERROR: nilearn is not available in the current Python environment"
  exit 1
}

BIDS_ROOT_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_ImagingSession"
DERIVATIVES_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/derivatives"
OUT_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/GLM_output"
SUBJECT_LIST="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_subject_list.txt"

# Validate top-level inputs
if [ ! -f "$SUBJECT_LIST" ]; then
  echo "ERROR: subject list not found: $SUBJECT_LIST"; exit 1
fi
if [ ! -d "$BIDS_ROOT_DIR" ]; then
  echo "ERROR: BIDS root not found: $BIDS_ROOT_DIR"; exit 1
fi
if [ ! -d "$DERIVATIVES_DIR" ]; then
  echo "ERROR: derivatives directory not found: $DERIVATIVES_DIR"; exit 1
fi

# Get subject
SUBJ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
if [ -z "$SUBJ" ]; then
  echo "ERROR: empty subject ID at line ${SLURM_ARRAY_TASK_ID} of $SUBJECT_LIST"; exit 1
fi
if ! [[ "$SUBJ" =~ ^sub-[0-9]+$ ]]; then
  echo "ERROR: subject ID '${SUBJ}' does not match sub-[0-9]+"; exit 1
fi

mkdir -p "$OUT_DIR"

export BIDS_ROOT_DIR DERIVATIVES_DIR OUT_DIR SUBJ

echo "[$(date)] GLM: $SUBJ"

# Run the Python analysis
python3 << 'EOF'
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image

subid          = os.environ['SUBJ']
bids_root_dir  = Path(os.environ['BIDS_ROOT_DIR'])
derivatives_dir = Path(os.environ['DERIVATIVES_DIR'])
out_dir        = Path(os.environ['OUT_DIR'])
sub_out_dir    = out_dir / subid
sub_out_dir.mkdir(parents=True, exist_ok=True)

runs = ['01', '02', '03']
TR   = 2.0
runs_done = 0

for run in runs:
    print(f"Processing {subid}, run {run}")

    bold_file      = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv"
    events_file    = bids_root_dir   / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_events.tsv"

    missing = [str(p) for p in [bold_file, confounds_file, events_file] if not p.exists()]
    if missing:
        print(f"  WARNING: run {run} missing file(s), skipping: {missing}")
        continue

    # Load events
    events = pd.read_csv(events_file, sep='\t')

    # Create clean events using onset_trimmed
    clean_events_list = []
    conditions = ['neg', 'neu', 'pos']
    for val in conditions:
        df_img = events[events['valence'] == val][['onset_trimmed', 'duration']].copy()
        df_img = df_img.rename(columns={'onset_trimmed': 'onset'})
        df_img['trial_type'] = f"{val}_image"
        clean_events_list.append(df_img)
        df_face = events[events['valenceFollowing'] == val][['onset_trimmed', 'duration']].copy()
        df_face = df_face.rename(columns={'onset_trimmed': 'onset'})
        df_face['trial_type'] = f"{val}_face"
        clean_events_list.append(df_face)

    events_glm = pd.concat(clean_events_list, ignore_index=True)

    # Load confounds and keep 24 motion parameters
    confounds = pd.read_csv(confounds_file, sep='\t')
    motion_columns = [
        'trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
        'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
        'trans_z','trans_z_derivative1','trans_z_power2','trans_z_derivative1_power2',
        'rot_x','rot_x_derivative1','rot_x_power2','rot_x_derivative1_power2',
        'rot_y','rot_y_derivative1','rot_y_power2','rot_y_derivative1_power2',
        'rot_z','rot_z_derivative1','rot_z_power2','rot_z_derivative1_power2'
    ]
    motion_regressors = confounds[motion_columns].iloc[4:]  # remove first 4 dummy scans

    # Remove first 4 volumes from bold
    bold_img = image.index_img(bold_file, slice(4, None))

    # Initialize GLM
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

    # Fit GLM
    glm.fit(run_imgs=bold_img, events=events_glm, confounds=motion_regressors)

    # Save the clean events file
    events_glm.to_csv(sub_out_dir / f"run-{run}_clean_events.tsv", sep='\t', index=False)

    # Save beta maps
    for cond in events_glm['trial_type'].unique():
        beta_map = glm.compute_contrast(cond, output_type='effect_size')
        beta_map.to_filename(sub_out_dir / f"run-{run}_{cond}_beta.nii.gz")

    glm.design_matrices_[0].to_csv(sub_out_dir / f"run-{run}_design_matrix.csv", index=False)
    runs_done += 1

if runs_done == 0:
    print(f"ERROR: no runs could be processed for {subid}")
    sys.exit(1)

print(f"Done: {subid} — {runs_done} run(s) processed")
EOF
