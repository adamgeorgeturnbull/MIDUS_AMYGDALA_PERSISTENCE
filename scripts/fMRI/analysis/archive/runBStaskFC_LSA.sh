#!/bin/bash
# runBStaskFC.sh
#
# SLURM array job: ROI-level beta-series task-based functional connectivity.
#
# Computes trial-wise beta-series connectivity between amygdala seeds (L, R)
# and anterior/posterior vmPFC targets for negative vs neutral contrasts.
#
# Method:
#   1. Fit a trial-wise GLM (each neg/neu trial as its own regressor)
#   2. Extract mean beta per trial for each ROI (L/R amygdala, ant/post vmPFC)
#   3. Correlate seed-target beta-series separately for neg and neu trials
#   4. Compute neg - neu contrast in Fisher z-space
#   5. Average across runs
#
# ROI masks:
#   - L/R amygdala: Harvard-Oxford atlas (50% threshold, 2mm)
#   - Anterior vmPFC: 10mm sphere at [-2, 46, -10] (safety signaling;
#     Tashjian et al., 2021, TICS)
#   - Posterior vmPFC: 10mm sphere at [0, 26, -12] (threat signaling;
#     Tashjian et al., 2021, TICS)
#
# Trial duration: 6.0s (2s image + 4s fixation)
#
# Output per subject:
#   - <subid>_betaSeries_ROI_contrast_neg_vs_neu.csv
#     Columns: subid, l_amyg-ant_vmPFC, l_amyg-post_vmPFC,
#              r_amyg-ant_vmPFC, r_amyg-post_vmPFC
#     Values: Fisher z(neg) - Fisher z(neu), averaged across runs
#
#SBATCH -J betaSeries_MIDUS
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3/log/betaSeries_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3/log/betaSeries_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
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
out_dir=/scratch/groups/fvlin/MIDUS/M3/BetaSeries_output
mkdir -p $out_dir
export bids_root_dir=$bids_root_dir
export derivatives_dir=$derivatives_dir
export out_dir=$out_dir

subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/M3/M3_subject_list.txt)
export subid=$subid

python3 << 'EOF'
import os
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn import image, datasets
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import FirstLevelModel
from scipy.stats import pearsonr

subid = os.environ['subid']
bids_root_dir = Path(os.environ['bids_root_dir'])
derivatives_dir = Path(os.environ['derivatives_dir'])
out_dir = Path(os.environ['out_dir'])
sub_out_dir = out_dir / subid
sub_out_dir.mkdir(parents=True, exist_ok=True)

runs = ['01','02','03']
TR = 2.0

# Load Harvard-Oxford 50% threshold amygdala masks
atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
labels = atlas.labels
atlas_img = atlas.filename
left_idx = labels.index('Left Amygdala')
right_idx = labels.index('Right Amygdala')
def get_hemi_mask(idx):
    return image.math_img("img == {}".format(idx), img=atlas_img)
l_amyg = get_hemi_mask(left_idx)
r_amyg = get_hemi_mask(right_idx)

# Anterior/posterior vmPFC spherical ROIs (10mm radius)
# Based on anterior-posterior vmPFC gradient for safety vs threat signaling
# (Tashjian et al., 2021, Trends in Cognitive Sciences)
from nilearn.maskers import NiftiSpheresMasker

ant_vmPFC_coords = [(-2, 46, -10)]   # anterior vmPFC (safety signaling)
post_vmPFC_coords = [(0, 26, -12)]    # posterior vmPFC (threat signaling)

maskers = {
    'l_amyg': NiftiMasker(mask_img=l_amyg, standardize=True),
    'r_amyg': NiftiMasker(mask_img=r_amyg, standardize=True),
    'ant_vmPFC': NiftiSpheresMasker(seeds=ant_vmPFC_coords, radius=10, standardize=True),
    'post_vmPFC': NiftiSpheresMasker(seeds=post_vmPFC_coords, radius=10, standardize=True)
}

results_all_runs = []

for run in runs:
    print(f"Processing {subid}, run {run}")

    bold_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv"
    events_file = bids_root_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_events.tsv"

    # Load events
    events = pd.read_csv(events_file, sep='\t')

    # Create clean events with onset_trimmed and unique trial names
    clean_events_list = []
    available_conditions = [v for v in ['neg','neu','pos'] if v in events['valence'].values]
    for val in available_conditions:
        df_val = events[events['valence']==val][['onset_trimmed','duration']].copy()
        df_val = df_val.rename(columns={'onset_trimmed':'onset'})
        df_val['duration'] = 6.0  # 2s image + 4s fixation
        df_val = df_val.reset_index(drop=True)
        df_val['trial_type'] = [f"{val}_{i+1:03d}" for i in range(len(df_val))]
        clean_events_list.append(df_val)
    events_clean = pd.concat(clean_events_list, ignore_index=True)

    # Save clean events
    clean_events_file = sub_out_dir / f"run-{run}_clean_events.tsv"
    events_clean.to_csv(clean_events_file, sep='\t', index=False)

    # Load motion confounds
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

    # Fit beta-series GLM
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

    # Compute beta maps per trial using unique trial names
    trial_names = events_clean['trial_type'].tolist()
    beta_maps = [glm.compute_contrast(trial_name, output_type='effect_size') for trial_name in trial_names]

    # Extract beta-series per ROI
    for m in maskers.values():
        m.fit(bold_img)
    beta_series = {roi: [] for roi in maskers}
    for beta_img in beta_maps:
        for roi, masker in maskers.items():
            roi_beta = masker.transform(beta_img)
            beta_series[roi].append(np.mean(roi_beta))
    for roi in beta_series:
        beta_series[roi] = np.array(beta_series[roi])

    # Compute Fisher z-transformed correlations per condition
    cond_indices = {}
    for cond in available_conditions:
        cond_indices[cond] = [i for i, t in enumerate(trial_names) if t.startswith(cond)]
    def fisher_z(r):
        return np.arctanh(np.clip(r, -0.9999, 0.9999))
    roi_pairs = [('l_amyg','ant_vmPFC'), ('l_amyg','post_vmPFC'),
                 ('r_amyg','ant_vmPFC'), ('r_amyg','post_vmPFC')]
    results_run = {'run': run}
    for seed, target in roi_pairs:
        pair_name = f'{seed}-{target}'
        z_vals = {}
        for cond, idx in cond_indices.items():
            if len(idx) >= 3:
                r, _ = pearsonr(beta_series[seed][idx], beta_series[target][idx])
                z_vals[cond] = fisher_z(r)
                results_run[f'{pair_name}_{cond}'] = z_vals[cond]
            else:
                results_run[f'{pair_name}_{cond}'] = np.nan
        # Contrasts
        if 'neg' in z_vals and 'neu' in z_vals:
            results_run[f'{pair_name}_neg_vs_neu'] = z_vals['neg'] - z_vals['neu']
        if 'neg' in z_vals and 'pos' in z_vals:
            results_run[f'{pair_name}_neg_vs_pos'] = z_vals['neg'] - z_vals['pos']
    results_all_runs.append(results_run)

# Average across runs
results_df = pd.DataFrame(results_all_runs)
results_df_mean = results_df.drop(columns='run').mean().to_frame().T
results_df_mean['subid'] = subid
results_df_mean = results_df_mean[['subid'] + [c for c in results_df_mean.columns if c != 'subid']]

# Save comprehensive results (per-condition + contrasts)
results_file = sub_out_dir / f"{subid}_betaSeries_ROI_all_conditions.csv"
results_df_mean.to_csv(results_file, index=False)
print(f"Saved results to {results_file}")

# Also save backward-compatible neg_vs_neu file
compat_cols = ['subid'] + [c for c in results_df_mean.columns if c.endswith('_neg_vs_neu')]
if len(compat_cols) > 1:
    compat_file = sub_out_dir / f"{subid}_betaSeries_ROI_contrast_neg_vs_neu.csv"
    # Rename columns to match old format (strip _neg_vs_neu suffix)
    compat_df = results_df_mean[compat_cols].copy()
    compat_df.columns = [c.replace('_neg_vs_neu', '') if c != 'subid' else c for c in compat_df.columns]
    compat_df.to_csv(compat_file, index=False)
    print(f"Saved backward-compatible file to {compat_file}")
EOF

