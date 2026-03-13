#!/bin/bash
# seedbasedBStaskFC_LSS.sh
#
# SLURM array job: Voxelwise seed-based beta-series FC using Least Squares Separate (LSS).
#
# PRIMARY beta-series FC method. Replaces LSA (see archive/seedbasedBStaskFC_LSA.sh).
#
# LSS method (Mumford et al., 2012; Rissman et al., 2004):
#   For each trial i in condition C:
#     - Regressor 1: trial_i (modeled individually — the trial of interest)
#     - Regressor 2: all OTHER trials of condition C collapsed into one nuisance regressor
#     - Regressors for remaining conditions: one regressor per other condition (all trials)
#     - Plus 24 motion parameters
#   Beta for regressor 1 becomes trial_i's beta-series value.
#
# This yields much less correlated beta estimates than LSA (Least Squares All),
# where all trials are simultaneously modeled, inflating inter-trial correlations
# and producing random-like FC matrices for rapid event-related designs (ISI < 12s).
# See Masharipov et al. (2024) for formal comparison across methods.
#
# Seeds: L/R amygdala (Harvard-Oxford atlas, 50% threshold, 2mm)
# Conditions: neg, neu, pos (conditions separated at extraction)
#
# Output per subject per seed:
#   - <subid>_<seed>_seedFC_LSS_neg.nii.gz     (Fisher z map, neg trials)
#   - <subid>_<seed>_seedFC_LSS_neu.nii.gz
#   - <subid>_<seed>_seedFC_LSS_pos.nii.gz
#   - <subid>_<seed>_seedFC_LSS_neg_vs_neu.nii.gz
#   - <subid>_<seed>_seedFC_LSS_neg_vs_pos.nii.gz
#
# Note on compute time: LSS fits ~N_trials GLMs per run (typically 36 per run,
# 108 per subject). Allow 48h per subject. Recommend --array=1-160%20 to
# avoid overwhelming the cluster.
#
#SBATCH -J betaSeriesSeedFC_LSS
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3/log/betaSeriesSeedFC_LSS_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3/log/betaSeriesSeedFC_LSS_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-160%20

module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
pip install --user --no-deps nilearn

bids_root_dir=/scratch/groups/fvlin/MIDUS/M3/M3_ImagingSession
derivatives_dir=/scratch/groups/fvlin/MIDUS/M3/derivatives
out_dir=/scratch/groups/fvlin/MIDUS/M3/BetaSeriesSeedFC_LSS_output
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
import os, sys
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

print(f"Running LSS beta-series FC: {subid}")
sys.stdout.flush()

# -------------------------------------------------------
# Amygdala seed masks (Harvard-Oxford 50%, 2mm)
# -------------------------------------------------------
atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
labels = atlas.labels
atlas_img = atlas.filename
left_idx  = labels.index('Left Amygdala')
right_idx = labels.index('Right Amygdala')

def get_hemi_mask(idx):
    return image.math_img("img == {}".format(idx), img=atlas_img)

example_bold = (
    derivatives_dir / subid / 'func' /
    f"{subid}_task-EmotionRegulation_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)
left_mask  = image.resample_to_img(get_hemi_mask(left_idx),  example_bold, interpolation="nearest")
right_mask = image.resample_to_img(get_hemi_mask(right_idx), example_bold, interpolation="nearest")
seeds = {'l_amyg': left_mask, 'r_amyg': right_mask}

for name, mask in seeds.items():
    nvox = int((mask.get_fdata() > 0).sum())
    print(f"{name}: {nvox} voxels after resampling")

# -------------------------------------------------------
# Gray-matter mask for voxelwise extraction
# -------------------------------------------------------
gm_mask = datasets.load_mni152_gm_mask(resolution=2, threshold=0.2, n_iter=2)
brain_masker = NiftiMasker(mask_img=gm_mask, standardize=True)
# Fit on a single image to initialize (will reuse across trials)
brain_masker.fit(example_bold)

# -------------------------------------------------------
# Run parameters
# -------------------------------------------------------
runs = ['01', '02', '03']
TR   = 2.0
conditions = ['neg', 'neu', 'pos']

motion_columns = [
    'trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
    'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
    'trans_z','trans_z_derivative1','trans_z_power2','trans_z_derivative1_power2',
    'rot_x','rot_x_derivative1','rot_x_power2','rot_x_derivative1_power2',
    'rot_y','rot_y_derivative1','rot_y_power2','rot_y_derivative1_power2',
    'rot_z','rot_z_derivative1','rot_z_power2','rot_z_derivative1_power2'
]

def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

def build_lss_events(events_all, trial_of_interest, cond_of_interest, all_conditions):
    """
    Build LSS event DataFrame for a single trial.

    For the condition of interest:
      - trial_of_interest: modeled as its own regressor
      - all other trials of the same condition: collapsed into '<cond>_others'
    For all other conditions:
      - all trials collapsed into '<other_cond>_all' (one regressor per condition)
    """
    rows = []
    for _, ev in events_all.iterrows():
        t = ev['trial_type']
        if t == trial_of_interest:
            # Trial of interest — keep its unique name
            rows.append({'onset': ev['onset'], 'duration': ev['duration'],
                         'trial_type': trial_of_interest})
        elif t.startswith(cond_of_interest + '_'):
            # Other trials of the same condition — collapse into one regressor
            rows.append({'onset': ev['onset'], 'duration': ev['duration'],
                         'trial_type': f'{cond_of_interest}_others'})
        else:
            # Other conditions — collapse all trials per condition
            cond = t.rsplit('_', 1)[0]  # e.g., 'neu' from 'neu_003'
            rows.append({'onset': ev['onset'], 'duration': ev['duration'],
                         'trial_type': f'{cond}_all'})
    return pd.DataFrame(rows)


# -------------------------------------------------------
# Main loop: per seed
# -------------------------------------------------------
for seed_name, seed_mask in seeds.items():
    print(f"\n=== Seed: {seed_name} ===")

    # Accumulate beta-series per condition across all runs
    # Each list will contain (seed_beta, voxel_beta_vector) tuples
    cond_data = {cond: {'seed_betas': [], 'voxel_betas': []} for cond in conditions}

    masker_seed = NiftiMasker(mask_img=seed_mask, standardize=True)

    for run in runs:
        bold_file      = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        confounds_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv"
        events_file    = bids_root_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_events.tsv"

        if not bold_file.exists():
            print(f"  WARNING: BOLD not found for run {run}, skipping")
            continue

        # Load and trim BOLD (remove 4 dummy scans)
        bold_img = image.index_img(bold_file, slice(4, None))

        # Load motion confounds
        confounds = pd.read_csv(confounds_file, sep='\t')
        motion_regressors = confounds[motion_columns].iloc[4:]

        # Build trial-wise events (unique trial names per condition)
        events = pd.read_csv(events_file, sep='\t')
        clean_events_list = []
        for val in conditions:
            df_val = events[events['valence']==val][['onset_trimmed','duration']].copy()
            df_val = df_val.rename(columns={'onset_trimmed':'onset'})
            df_val['duration'] = 6.0
            df_val = df_val.reset_index(drop=True)
            df_val['trial_type'] = [f"{val}_{i+1:03d}" for i in range(len(df_val))]
            clean_events_list.append(df_val)
        events_all = pd.concat(clean_events_list, ignore_index=True).sort_values('onset').reset_index(drop=True)

        # Group trial names by condition
        trials_by_cond = {
            cond: [t for t in events_all['trial_type'].tolist() if t.startswith(cond + '_')]
            for cond in conditions
        }
        print(f"  Run {run}: " + ", ".join(f"{c}={len(v)} trials" for c, v in trials_by_cond.items()))

        # -------------------------------------------------------
        # LSS: fit one GLM per trial
        # -------------------------------------------------------
        for cond in conditions:
            for trial_i in trials_by_cond[cond]:

                # Build LSS event table for this trial
                events_lss = build_lss_events(events_all, trial_i, cond, conditions)

                # Fit GLM
                glm_lss = FirstLevelModel(
                    t_r=TR,
                    slice_time_ref=0.5,
                    hrf_model='glover',
                    drift_model='cosine',
                    high_pass=1/128,
                    standardize=False,
                    noise_model='ar1',
                    minimize_memory=True   # important for memory efficiency across many fits
                )
                glm_lss.fit(run_imgs=bold_img, events=events_lss, confounds=motion_regressors)

                # Extract beta map for the trial of interest
                beta_img = glm_lss.compute_contrast(trial_i, output_type='effect_size')

                # Seed beta (mean within ROI)
                masker_seed.fit(bold_img)
                seed_vals = masker_seed.transform(beta_img).ravel()
                seed_beta = float(np.mean(seed_vals))

                # Voxelwise betas (gray-matter)
                voxel_betas = brain_masker.transform(beta_img).flatten()

                cond_data[cond]['seed_betas'].append(seed_beta)
                cond_data[cond]['voxel_betas'].append(voxel_betas)

        print(f"  Run {run} complete", flush=True)

    # -------------------------------------------------------
    # Compute voxelwise FC maps per condition
    # -------------------------------------------------------
    z_maps = {}
    n_voxels = brain_masker.transform(example_bold).shape[1]

    for cond in conditions:
        seed_ts = np.array(cond_data[cond]['seed_betas'])
        n_trials = len(seed_ts)
        if n_trials < 3:
            print(f"  WARNING: only {n_trials} trials for {cond}, skipping FC map")
            continue

        voxel_matrix = np.array(cond_data[cond]['voxel_betas']).T  # shape: (n_vox, n_trials)

        print(f"  Computing {cond} FC: {n_trials} trials x {voxel_matrix.shape[0]} voxels")
        z_arr = np.array([
            fisher_z(pearsonr(seed_ts, voxel_matrix[v, :])[0])
            for v in range(voxel_matrix.shape[0])
        ])
        z_maps[cond] = z_arr

    # Contrast maps in Fisher z-space
    if 'neg' in z_maps and 'neu' in z_maps:
        z_maps['neg_vs_neu'] = z_maps['neg'] - z_maps['neu']
    if 'neg' in z_maps and 'pos' in z_maps:
        z_maps['neg_vs_pos'] = z_maps['neg'] - z_maps['pos']

    # -------------------------------------------------------
    # Save maps
    # -------------------------------------------------------
    for map_name, z_data in z_maps.items():
        z_img = brain_masker.inverse_transform(z_data)
        out_file = sub_out_dir / f"{subid}_{seed_name}_seedFC_LSS_{map_name}.nii.gz"
        z_img.to_filename(out_file)
        print(f"  Saved: {out_file.name}")

print(f"\nDone: {subid}")
EOF
