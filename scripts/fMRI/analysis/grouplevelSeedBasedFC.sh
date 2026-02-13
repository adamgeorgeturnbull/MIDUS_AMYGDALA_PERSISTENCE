#!/bin/bash
# grouplevelSeedBasedFC.sh
#
# SLURM job: Group-level analysis of seed-based beta-series FC maps.
#
# Takes subject-level neg > neu connectivity maps from seedbasedBStaskFC.sh
# and runs second-level (group) analyses using nilearn's SecondLevelModel:
#
#   1. Group mean: One-sample t-test on neg > neu connectivity maps
#   2. Covariate analysis: Tests whether left amygdala persistence (mean_r)
#      predicts voxelwise neg > neu connectivity
#
# Run separately for each seed (l_amyg, r_amyg).
#
# Output:
#   - group_mean_<seed>.nii.gz     (group mean z-map)
#   - group_mean_<seed>.png        (visualization)
#   - group_covariate_<seed>.nii.gz (persistence covariate z-map)
#   - group_covariate_<seed>.png    (visualization)
#
#SBATCH -J groupLevelSeedFC
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3/log/groupSeedFC_%A.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3/log/groupSeedFC_%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL

module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
pip install --user --no-deps nilearn

python3 << 'EOF'
import os
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn import image, plotting, masking
from nilearn.glm.second_level import SecondLevelModel

# ------------------------
# Paths
# ------------------------
output_dir = Path("/scratch/groups/fvlin/MIDUS/M3/BetaSeriesSeedFC_output")
out_dir = Path("/scratch/groups/fvlin/MIDUS/M3/GroupSeedFC_output")
out_dir.mkdir(parents=True, exist_ok=True)

persistence_file = '/scratch/groups/fvlin/MIDUS/M3/voxelwise_betas_summary/results_summary.csv'
persistence = pd.read_csv(persistence_file)
persistence = persistence[persistence['hemisphere'] == 'L'][['subject','mean_r']]
# Fisher z-transform persistence r values for use as covariate
persistence['mean_z'] = np.arctanh(persistence['mean_r'])

# Load QC data and restrict to conservative sample (all 3 runs pass + FD < 0.5mm)
qc_file = '/scratch/groups/fvlin/MIDUS/M3/fmri_qc_processed.csv'
qc = pd.read_csv(qc_file)
conservative_m2ids = set(qc.loc[qc['qc_conservative'] == 1, 'M2ID'].astype(str))
conservative_subs = {f"sub-{m}" for m in conservative_m2ids}
print(f"Conservative sample: {len(conservative_subs)} subjects pass QC")

# ------------------------
# Collect subject-level maps
# ------------------------
seeds = ['l_amyg','r_amyg']
all_subjects = sorted([d.name for d in output_dir.iterdir() if d.is_dir() and d.name in conservative_subs])
print(f"Subjects with FC output in conservative sample: {len(all_subjects)}")

maps = {seed: [] for seed in seeds}
subjects_with_maps = {seed: [] for seed in seeds}

for sub in all_subjects:
    for seed in seeds:
        map_file = output_dir / sub / f"{sub}_{seed}_seedFC_neg_vs_neu.nii.gz"
        if map_file.exists():
            maps[seed].append(str(map_file))
            subjects_with_maps[seed].append(sub)

# ------------------------
# Optional gray-matter mask
# ------------------------
gm_mask = None
# gm_mask = masking.compute_gray_matter_mask(maps[seeds[0]][0])  # optional

# ------------------------
# Group-level analysis per seed
# ------------------------
for seed in seeds:
    print(f"Processing group-level analysis for {seed}")

    # ------------------------
    # 1) Group mean map (intercept only)
    # ------------------------
    n_subj = len(maps[seed])
    if n_subj == 0:
        print(f"No maps found for seed {seed}, skipping.")
        continue

    design_matrix = pd.DataFrame({'intercept': np.ones(n_subj)})
    second_level_model = SecondLevelModel(mask_img=gm_mask)
    second_level_model = second_level_model.fit(maps[seed], design_matrix=design_matrix)
    z_map_mean = second_level_model.compute_contrast(output_type='z_score')
    z_map_mean.to_filename(out_dir / f"group_mean_{seed}.nii.gz")
    plotting.plot_stat_map(z_map_mean, display_mode='z', cut_coords=7,
                           title=f"{seed} group mean", 
                           output_file=out_dir / f"group_mean_{seed}.png")

    # ------------------------
    # 2) Covariate analysis: left amygdala persistence
    # ------------------------
    # Build a dataframe linking subjects to their map index
    df = pd.DataFrame({'subject': subjects_with_maps[seed],
                        'map_idx': range(len(subjects_with_maps[seed]))})
    df = df.merge(persistence, on='subject', how='inner')
    # Drop subjects with NaN or Inf persistence values (in z-space)
    df = df[np.isfinite(df['mean_z'])].reset_index(drop=True)
    if df.empty:
        print(f"No matching subjects with persistence for seed {seed}, skipping covariate analysis")
        continue

    # Filter maps to only include subjects in the merged dataframe
    cov_maps = [maps[seed][i] for i in df['map_idx']]
    print(f"  Covariate analysis: {len(cov_maps)} subjects with valid persistence data")

    design_matrix = pd.DataFrame({'intercept': np.ones(len(df)),
                                  'persistence': df['mean_z'].values})

    second_level_model = SecondLevelModel(mask_img=gm_mask)
    z_map_cov = second_level_model.fit(cov_maps, design_matrix=design_matrix).compute_contrast(
        second_level_contrast=[0,1],  # coefficient for 'persistence'
        output_type='z_score'
    )
    z_map_cov.to_filename(out_dir / f"group_covariate_{seed}.nii.gz")
    plotting.plot_stat_map(z_map_cov, display_mode='z', cut_coords=7,
                           title=f"{seed} covariate", 
                           output_file=out_dir / f"group_covariate_{seed}.png")
EOF

