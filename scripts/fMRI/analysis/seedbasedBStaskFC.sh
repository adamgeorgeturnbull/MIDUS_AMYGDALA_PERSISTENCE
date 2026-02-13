#!/bin/bash
# seedbasedBStaskFC.sh
#
# SLURM array job: Voxelwise seed-based beta-series functional connectivity.
#
# Extends runBStaskFC.sh from ROI-level to whole-brain voxelwise connectivity.
# For each amygdala seed (L, R), computes voxelwise correlations between the
# seed beta-series and every gray-matter voxel, then contrasts neg vs neu.
#
# Method:
#   1. Fit trial-wise GLM (same as runBStaskFC.sh)
#   2. Extract mean amygdala beta per trial (seed time series)
#   3. Extract voxelwise betas across all gray-matter voxels
#   4. Correlate seed with each voxel separately for neg and neu trials
#   5. Compute neg - neu contrast map in Fisher z-space
#
# Subjects in motion_exclude_ids are skipped (excessive motion).
# Gray-matter mask: MNI152 GM mask (2mm, threshold=0.2)
#
# Output per subject per seed:
#   - <subid>_<seed>_seedFC_neg_vs_neu.nii.gz
#     Whole-brain Fisher z contrast map (neg > neu connectivity)
#
#SBATCH -J betaSeriesSeedFC
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3/log/betaSeriesSeedFC_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3/log/betaSeriesSeedFC_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-160

module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
pip install --user --no-deps nilearn

bids_root_dir=/scratch/groups/fvlin/MIDUS/M3/M3_ImagingSession
derivatives_dir=/scratch/groups/fvlin/MIDUS/M3/derivatives
out_dir=/scratch/groups/fvlin/MIDUS/M3/BetaSeriesSeedFC_output
mkdir -p $out_dir

export bids_root_dir=$bids_root_dir
export derivatives_dir=$derivatives_dir
export out_dir=$out_dir

motion_exclude_ids=(sub-10294 sub-11557 sub-11694 sub-12424 sub-12540 sub-13084 sub-13129 sub-13697 sub-14391 sub-14451 sub-14527 sub-14785 sub-15196 sub-16064 sub-16716 sub-17241 sub-17477 sub-17542 sub-17660 sub-18751)

subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/M3/M3_subject_list.txt)

if [[ " ${motion_exclude_ids[@]} " =~ " ${subid} " ]]; then
    echo "Skipping excluded subject $subid"
    exit 0
fi
export subid=$subid

python3 << 'EOF'
import os
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn import image, input_data, masking
from nilearn.glm.first_level import FirstLevelModel
from scipy.stats import pearsonr
import sys
from nilearn import datasets

subid = os.environ['subid']
bids_root_dir = Path(os.environ['bids_root_dir'])
derivatives_dir = Path(os.environ['derivatives_dir'])
out_dir = Path(os.environ['out_dir'])
sub_out_dir = out_dir / subid
sub_out_dir.mkdir(parents=True, exist_ok=True)

print("Running:", os.environ.get("subid"))
print("BIDS root:", os.environ.get("bids_root_dir"))
print("Derivatives:", os.environ.get("derivatives_dir"))
sys.stdout.flush()

persistence_file = '/scratch/groups/fvlin/MIDUS/M3/voxelwise_betas_summary/results_summary.csv'
persistence = pd.read_csv(persistence_file)

rows = persistence.loc[(persistence['subject'] == subid) & (persistence['hemisphere'] == 'L'),'mean_r']

if rows.empty:
    print(f"⚠️ No persistence data for {subid} (hemisphere=L). Skipping subject.")
    sys.exit(0)
else:
    mean_r_L = rows.values[0]
    print(f"✅ Loaded mean_r={mean_r_L:.3f} for {subid}")

# ------------------------
# Define seeds (L/R amygdala)
# ------------------------
# ------------------------
# Define and resample seeds (L/R amygdala)
# ------------------------
from nilearn import datasets, image

# Load Harvard–Oxford atlas (50% threshold, 2 mm)
atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
labels = atlas.labels
atlas_img = atlas.filename

# Find amygdala indices
left_idx = labels.index('Left Amygdala')
right_idx = labels.index('Right Amygdala')

def get_hemi_mask(idx):
    return image.math_img("img == {}".format(idx), img=atlas_img)

# Use one representative functional image to set the space
example_bold = (
    derivatives_dir / subid / 'func' /
    f"{subid}_task-EmotionRegulation_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)

# Resample masks into that space (so they align exactly with subject data)
left_mask_res = image.resample_to_img(get_hemi_mask(left_idx), example_bold, interpolation="nearest")
right_mask_res = image.resample_to_img(get_hemi_mask(right_idx), example_bold, interpolation="nearest")

# Combine in a dictionary for looping later
seeds = {'l_amyg': left_mask_res, 'r_amyg': right_mask_res}

# Quick sanity check: print number of voxels in each seed
for name, mask in seeds.items():
    nvox = int((mask.get_fdata() > 0).sum())
    print(f"{name} mask has {nvox} voxels after resampling")

# ------------------------
# Run parameters
# ------------------------
runs = ['01','02','03']
TR = 2.0

# ------------------------
# Load gray-matter mask for voxelwise extraction
# ------------------------
gm_mask = datasets.load_mni152_gm_mask(resolution=2, threshold=0.2, n_iter=2)

# ------------------------
# Loop over seeds
# ------------------------
for seed_name, seed_mask in seeds.items():
    print(f"Processing {seed_name} for {subid}")
    seed_beta_series = []
    all_trial_beta_imgs = []
    all_trial_names = []

    for run in runs:
        bold_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        confounds_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv"
        events_file = bids_root_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_events.tsv"

        # Load and clean events
        events = pd.read_csv(events_file, sep='\t')
        conditions = ['neg','neu']
        clean_events_list = []
        for val in conditions:
            df_val = events[events['valence']==val][['onset_trimmed','duration']].copy()
            df_val = df_val.rename(columns={'onset_trimmed':'onset'})
            df_val['duration'] = 6.0
            df_val['trial_type'] = [f"{val}_{i+1:03d}" for i in range(len(df_val))]
            clean_events_list.append(df_val)
        events_clean = pd.concat(clean_events_list, ignore_index=True)

        # Load confounds — explicit 24 motion parameters (6 + derivatives + squares)
        confounds = pd.read_csv(confounds_file, sep='\t')
        motion_columns = [
            'trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
            'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
            'trans_z','trans_z_derivative1','trans_z_power2','trans_z_derivative1_power2',
            'rot_x','rot_x_derivative1','rot_x_power2','rot_x_derivative1_power2',
            'rot_y','rot_y_derivative1','rot_y_power2','rot_y_derivative1_power2',
            'rot_z','rot_z_derivative1','rot_z_power2','rot_z_derivative1_power2'
        ]
        motion_regressors = confounds[motion_columns].iloc[4:]

        # Drop first 4 volumes from bold
        bold_img = image.index_img(bold_file, slice(4,None))

        # ------------------------
        # Fit trial-wise GLM
        # ------------------------
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
        glm.fit(run_imgs=bold_img, events=events_clean, confounds=motion_regressors)

        # Compute beta maps per trial
        trial_names = events_clean['trial_type'].tolist()
        all_trial_names.extend(trial_names)
        beta_maps = [glm.compute_contrast(trial_name, output_type='effect_size') for trial_name in trial_names]

        # ------------------------
        # Extract seed beta-series for this run
        # ------------------------
        masker_seed = input_data.NiftiMasker(mask_img=seed_mask, standardize=True)

        run_seed_ts = []

        for i, beta_img in enumerate(beta_maps):
            # Extract voxel values within the seed mask
            data = masker_seed.fit_transform(beta_img)
            
            # Flatten to 1D in case shape varies (defensive)
            data = data.ravel()
            
            # Compute mean beta for this trial
            mean_beta = np.mean(data)
            
            run_seed_ts.append(mean_beta)

        run_seed_ts = np.array(run_seed_ts)  # shape = (n_trials,)

        print(f"{subid}, {seed_name}: extracted {len(run_seed_ts)} beta values (one per trial)")

        seed_beta_series.extend(run_seed_ts)

        all_trial_beta_imgs.extend(beta_maps)

    # ------------------------
    # Extract voxelwise beta-series across all trials
    # ------------------------
    brain_masker = input_data.NiftiMasker(mask_img=gm_mask, standardize=True)
    voxel_beta_matrix = np.array([brain_masker.fit_transform(beta_img).flatten() for beta_img in all_trial_beta_imgs]).T
    # Rows = voxels, Columns = trials

    # ------------------------
    # Compute voxelwise correlations with seed
    # ------------------------
    def fisher_z(r):
        return np.arctanh(np.clip(r, -0.9999, 0.9999))

    z_corrs = np.array([fisher_z(pearsonr(seed_beta_series, voxel_beta_matrix[v, :])[0]) for v in range(voxel_beta_matrix.shape[0])])

    # ------------------------
    # Split neg vs neu trials for contrast
    # ------------------------
    neg_idx = [i for i, t in enumerate(all_trial_names) if t.startswith('neg')]
    neu_idx = [i for i, t in enumerate(all_trial_names) if t.startswith('neu')]

    z_neg = np.array([fisher_z(pearsonr(np.array(seed_beta_series)[neg_idx], voxel_beta_matrix[v, neg_idx])[0]) for v in range(voxel_beta_matrix.shape[0])])
    z_neu = np.array([fisher_z(pearsonr(np.array(seed_beta_series)[neu_idx], voxel_beta_matrix[v, neu_idx])[0]) for v in range(voxel_beta_matrix.shape[0])])
    z_contrast = z_neg - z_neu  # neg > neu

    # ------------------------
    # Save subject-level map
    # ------------------------
    z_map_img = brain_masker.inverse_transform(z_contrast)
    z_map_img.to_filename(sub_out_dir / f"{subid}_{seed_name}_seedFC_neg_vs_neu.nii.gz")
    print(f"Saved {seed_name} neg>neu map for {subid}")
EOF

