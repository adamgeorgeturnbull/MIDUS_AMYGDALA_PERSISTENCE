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
# Output per subject per run (in MR1/GLM_output/<subid>/):
#   - run-XX_<condition>_beta.nii.gz   (beta maps for each trial type)
#   - run-XX_clean_events.tsv          (reformatted events used in GLM)
#   - run-XX_design_matrix.csv         (full design matrix)
#
# TODO: set --array=1-N to match the line count of MR1_subject_list.txt
#       (inventory report: BIDS subject count; expected ~127).
#
#SBATCH -J glm_MR1
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1/log/glm_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1/log/glm_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-127

# Load modules
module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

# Activate pip install for nilearn locally
pip install --user --no-deps nilearn

# Paths ------------------------------------
bids_root_dir=/scratch/groups/fvlin/MIDUS/MR1/MR_P5_ImagingSession
derivatives_dir=/scratch/groups/fvlin/MIDUS/MR1/derivatives
out_dir=/scratch/groups/fvlin/MIDUS/MR1/GLM_output
mkdir -p $out_dir
export bids_root_dir=$bids_root_dir
export derivatives_dir=$derivatives_dir
export out_dir=$out_dir

# Get subject
subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/MR1/MR1_subject_list.txt)
export subid=$subid

# Run the Python analysis
python3 << 'EOF'
import os
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image

subid = os.environ['subid']
bids_root_dir = Path(os.environ['bids_root_dir'])
derivatives_dir = Path(os.environ['derivatives_dir'])
out_dir = Path(os.environ['out_dir'])
sub_out_dir = out_dir / subid
sub_out_dir.mkdir(parents=True, exist_ok=True)

runs = ['01','02','03']
TR = 2.0

for run in runs:
    print(f"Processing {subid}, run {run}")

    # Paths to files
    bold_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv"
    events_file = bids_root_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_events.tsv"

    # Skip runs that are missing (e.g. dropped during QC or fMRIPrep failure)
    if not bold_file.exists():
        print(f"  missing BOLD, skipping: {bold_file.name}")
        continue
    if not confounds_file.exists():
        print(f"  missing confounds, skipping: {confounds_file.name}")
        continue

    # Load events
    events = pd.read_csv(events_file, sep='\t')

    # MR1 events use full words; normalise to the abbreviations used in output names
    val_map = {'negative': 'neg', 'neutral': 'neu', 'positive': 'pos'}
    events['valence'] = events['valence'].map(val_map).fillna(events['valence'])

    # VERIFY BEFORE RUNNING: subtract dummy-TR duration only if dummies are saved in the
    # BOLD file (check bold.json for NumberOfVolumesDiscardedByUser/Scanner; count n_vols).
    # M3 had 4 dummies in-file → onset_trimmed = onset - 8s.
    # If MR1 dummies were removed before BIDS conversion, set N_DUMMIES = 0.
    N_DUMMIES = 4  # <-- confirm this matches MR1 BIDS
    events['onset'] = events['onset'] - N_DUMMIES * TR

    # MR1 events have no valenceFollowing column — derive it by forward-filling
    # the image valence into the immediately following face row.
    events = events.sort_values('onset').reset_index(drop=True)
    img_val_fwd = events['valence'].replace('n/a', np.nan).ffill()
    events['valenceFollowing'] = np.where(events['database'] == 'faces', img_val_fwd, np.nan)

    # Build clean events (3 image types + 3 face types)
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
    motion_regressors = confounds[motion_columns].iloc[N_DUMMIES:]

    # Remove dummy volumes from bold
    bold_img = image.index_img(bold_file, slice(N_DUMMIES, None))

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
    clean_events_file = sub_out_dir / f"run-{run}_clean_events.tsv"
    events_glm.to_csv(clean_events_file, sep='\t', index=False)

    # Save beta maps
    for cond in events_glm['trial_type'].unique():
        beta_map = glm.compute_contrast(cond, output_type='effect_size')
        beta_map.to_filename(sub_out_dir / f"run-{run}_{cond}_beta.nii.gz")

    design_matrix = glm.design_matrices_[0]
    design_matrix.to_csv(sub_out_dir / f"run-{run}_design_matrix.csv", index=False)
EOF
