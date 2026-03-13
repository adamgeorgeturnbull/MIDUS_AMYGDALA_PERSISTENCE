#!/bin/bash
# runGLM_concat.sh
#
# SLURM array job: Concatenated (all-runs) first-level GLM.
#
# Same GLM specification as runGLM.sh, but concatenates all 3 runs into a
# single time series before fitting. This produces one beta map per condition
# (rather than per run), used by run_cross_corr_concat.py to compute the
# concatenated persistence measure (sensitivity analysis).
#
# Event onsets are adjusted by cumulative time offset to account for run
# concatenation. Same 24 motion parameters and GLM settings as per-run version.
#
# Output per subject:
#   - allruns_<condition>_beta.nii.gz  (beta maps from concatenated GLM)
#   - allruns_clean_events.tsv         (time-adjusted events)
#   - allruns_design_matrix.csv        (full design matrix)
#
#SBATCH -J glm_MIDUS
#SBATCH --output=/scratch/groups/fvlin/MIDUS/log/glm_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/log/glm_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-160

# Load modules
module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

# Activate pip install for nilearn locally
pip install --user --no-deps nilearn

# Outputs ----------------------------------
bids_root_dir=/scratch/groups/fvlin/MIDUS/M3/M3_ImagingSession
derivatives_dir=/scratch/groups/fvlin/MIDUS/derivatives
out_dir=/scratch/groups/fvlin/MIDUS/GLM_output_concat
mkdir -p $out_dir
export bids_root_dir=$bids_root_dir
export derivatives_dir=$derivatives_dir
export out_dir=$out_dir

# Get subject
subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/M3_subject_list.txt)
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

all_bold = []
all_confounds = []
all_events = []
time_offset = 0

for run in runs:
    print(f"Processing {subid}, run {run}")

    # Paths
    bold_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv"
    events_file = bids_root_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_events.tsv"

    # --- Load and clean events ---
    events = pd.read_csv(events_file, sep='\t')
    clean_events_list = []
    conditions = ['neg','neu','pos']
    for val in conditions:
        # image trials
        df_img = events[events['valence']==val][['onset_trimmed','duration']].copy()
        df_img = df_img.rename(columns={'onset_trimmed':'onset'})
        df_img['trial_type'] = f"{val}_image"
        # face trials
        df_face = events[events['valenceFollowing']==val][['onset_trimmed','duration']].copy()
        df_face = df_face.rename(columns={'onset_trimmed':'onset'})
        df_face['trial_type'] = f"{val}_face"
        # append both
        clean_events_list.extend([df_img, df_face])

    events_glm = pd.concat(clean_events_list, ignore_index=True)

    # adjust onsets by cumulative time offset
    events_glm['onset'] = events_glm['onset'] + time_offset
    all_events.append(events_glm)

    # --- Confounds ---
    confounds = pd.read_csv(confounds_file, sep='\t')
    motion_columns = [
        'trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
        'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
        'trans_z','trans_z_derivative1','trans_z_power2','trans_z_derivative1_power2',
        'rot_x','rot_x_derivative1','rot_x_power2','rot_x_derivative1_power2',
        'rot_y','rot_y_derivative1','rot_y_power2','rot_y_derivative1_power2',
        'rot_z','rot_z_derivative1','rot_z_power2','rot_z_derivative1_power2'
    ]
    confounds_run = confounds[motion_columns].iloc[4:]  # drop dummy rows
    all_confounds.append(confounds_run)

    # --- BOLD ---
    bold_img = image.index_img(bold_file, slice(4,None))  # drop dummy vols
    all_bold.append(bold_img)

    # update offset for next run
    n_vols = bold_img.shape[-1]
    time_offset += n_vols * TR

# Concatenate across runs
concat_bold = image.concat_imgs(all_bold)
concat_confounds = pd.concat(all_confounds, ignore_index=True)
concat_events = pd.concat(all_events, ignore_index=True)

# --- Run single GLM ---
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

glm.fit(run_imgs=concat_bold, events=concat_events, confounds=concat_confounds)

# Save outputs
concat_events.to_csv(sub_out_dir / "allruns_clean_events.tsv", sep='\t', index=False)
for cond in concat_events['trial_type'].unique():
    beta_map = glm.compute_contrast(cond, output_type='effect_size')
    beta_map.to_filename(sub_out_dir / f"allruns_{cond}_beta.nii.gz")

design_matrix = glm.design_matrices_[0]
design_matrix.to_csv(sub_out_dir / "allruns_design_matrix.csv", index=False)

