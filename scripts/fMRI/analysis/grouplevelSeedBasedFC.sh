#!/bin/bash
# grouplevelSeedBasedFC.sh
#
# SLURM job: Group-level analysis of seed-based beta-series FC maps.
#
# Takes subject-level neg > neu connectivity maps from seedbasedBStaskFC.sh
# and runs second-level (group) analyses using TFCE-based permutation testing
# (Smith & Nichols, 2009) via nilearn's non_parametric_inference:
#
#   1. Group mean: One-sample t-test on neg > neu connectivity maps
#   2. Covariate analysis: Tests whether left amygdala persistence (mean_z)
#      predicts voxelwise neg > neu connectivity
#
# TFCE (Threshold-Free Cluster Enhancement) avoids choosing a cluster-forming
# threshold. Family-wise error is controlled via the max-statistic permutation
# distribution (Eklund et al., 2016).
#
# Run separately for each seed (l_amyg, r_amyg).
#
# Output per seed:
#   - group_mean_<seed>_tstat.nii.gz          (t-statistic map)
#   - group_mean_<seed>_logp_max_tfce.nii.gz  (-log10 p_FWE from TFCE)
#   - group_mean_<seed>.png                   (visualization)
#   - group_covariate_<seed>_tstat.nii.gz
#   - group_covariate_<seed>_logp_max_tfce.nii.gz
#   - group_covariate_<seed>.png
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
from nilearn.glm.second_level import non_parametric_inference

# ------------------------
# Settings
# ------------------------
N_PERM = 5000        # number of permutations for TFCE
TWO_SIDED = True     # test both positive and negative effects
N_JOBS = 8           # parallel permutations (match --cpus-per-task)

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
# Group-level analysis per seed
# ------------------------
for seed in seeds:
    n_subj = len(maps[seed])
    print(f"\n{'=' * 60}")
    print(f"Processing {seed}: N = {n_subj}")
    print(f"{'=' * 60}")

    if n_subj < 2:
        print(f"Need >= 2 subjects for permutation testing, skipping {seed}.")
        continue

    # -------------------------------------------------------
    # 1) Group mean (one-sample test via intercept)
    # -------------------------------------------------------
    print(f"  Running TFCE group mean ({N_PERM} permutations)...")
    design_matrix_mean = pd.DataFrame({
        'intercept': np.ones(n_subj),
    })
    out_mean = non_parametric_inference(
        maps[seed],
        design_matrix=design_matrix_mean,
        model_intercept=False,
        n_perm=N_PERM,
        two_sided_test=TWO_SIDED,
        tfce=True,
        n_jobs=N_JOBS,
        verbose=1,
    )

    # Save outputs
    out_mean['t'].to_filename(out_dir / f"group_mean_{seed}_tstat.nii.gz")
    out_mean['logp_max_tfce'].to_filename(out_dir / f"group_mean_{seed}_logp_max_tfce.nii.gz")

    # Visualization: threshold at -log10(0.05) = 1.3 for FWE significance
    plotting.plot_stat_map(
        out_mean['logp_max_tfce'], display_mode='z', cut_coords=7,
        threshold=1.3,  # -log10(0.05)
        title=f"{seed} group mean (TFCE p<.05 FWE)",
        output_file=out_dir / f"group_mean_{seed}.png",
    )
    print(f"  Saved group mean outputs for {seed}")

    # -------------------------------------------------------
    # 2) Covariate analysis: left amygdala persistence
    # -------------------------------------------------------
    df = pd.DataFrame({'subject': subjects_with_maps[seed],
                        'map_idx': range(len(subjects_with_maps[seed]))})
    df = df.merge(persistence, on='subject', how='inner')
    df = df[np.isfinite(df['mean_z'])].reset_index(drop=True)

    if len(df) < 2:
        print(f"  Need >= 2 subjects with persistence for covariate analysis, skipping.")
        continue

    cov_maps = [maps[seed][i] for i in df['map_idx']]
    print(f"  Running TFCE covariate analysis: {len(cov_maps)} subjects ({N_PERM} permutations)...")

    design_matrix = pd.DataFrame({
        'persistence': df['mean_z'].values,
        'intercept': np.ones(len(df)),
    })

    out_cov = non_parametric_inference(
        cov_maps,
        design_matrix=design_matrix,
        second_level_contrast='persistence',
        model_intercept=False,  # intercept already in design matrix
        n_perm=N_PERM,
        two_sided_test=TWO_SIDED,
        tfce=True,
        n_jobs=N_JOBS,
        verbose=1,
    )

    out_cov['t'].to_filename(out_dir / f"group_covariate_{seed}_tstat.nii.gz")
    out_cov['logp_max_tfce'].to_filename(out_dir / f"group_covariate_{seed}_logp_max_tfce.nii.gz")

    plotting.plot_stat_map(
        out_cov['logp_max_tfce'], display_mode='z', cut_coords=7,
        threshold=1.3,
        title=f"{seed} persistence covariate (TFCE p<.05 FWE)",
        output_file=out_dir / f"group_covariate_{seed}.png",
    )
    print(f"  Saved covariate outputs for {seed}")

print(f"\n{'=' * 60}")
print("Done.")
print(f"{'=' * 60}")
EOF
