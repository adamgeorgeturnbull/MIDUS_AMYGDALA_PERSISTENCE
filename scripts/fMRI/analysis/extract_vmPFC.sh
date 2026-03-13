#!/bin/bash
# extract_vmPFC.sh
#
# SLURM array job: Extract voxelwise vmPFC betas from per-run GLM output.
#
# Parallel to extract_amygdala.sh but for vmPFC spherical ROIs.
# Uses condition-level GLM output from runGLM.sh (no new GLM required).
#
# For each subject, loads beta maps and extracts voxelwise values within
# anterior and posterior vmPFC spherical ROIs (10mm radius).
#
# ROI seeds (Tashjian et al., 2021, Trends in Cognitive Sciences):
#   - ant_vmPFC:  10mm sphere at [-2, 46, -10]  (safety/anterior gradient)
#   - post_vmPFC: 10mm sphere at [0, 26, -12]   (threat/posterior gradient)
#
# Conditions extracted (image conditions only, for persistence):
#   neg_image, neu_image, pos_image
#
# Output per subject:
#   <out_dir>/<subid>/<subid>_voxelwise_vmPFC_betas.csv
#     Columns: subject, run, condition, seed, nvox_resampled, beta_0..N
#     One row per run x condition x seed
#
# These CSVs are used by run_cross_corr_vmPFC.py to compute vmPFC persistence.
# Uses existing GLM_output — can be submitted immediately.
#
#SBATCH -J vmPFC_beta_extract
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3/log/vmPFC_beta_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3/log/vmPFC_beta_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-160

module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
pip install --user --no-deps nilearn

glm_dir=/scratch/groups/fvlin/MIDUS/GLM_output
out_dir=/scratch/groups/fvlin/MIDUS/voxelwise_vmPFC_betas
mkdir -p $out_dir

export glm_dir=$glm_dir
export out_dir=$out_dir

subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/M3/M3_subject_list.txt)
export subid=$subid

python3 << 'EOF'
import os
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.maskers import NiftiSpheresMasker

subid = os.environ['subid']
glm_dir = Path(os.environ['glm_dir'])
out_dir = Path(os.environ['out_dir'])
sub_glm_dir = glm_dir / subid
sub_out_dir = out_dir / subid
sub_out_dir.mkdir(parents=True, exist_ok=True)

print(f"Extracting voxelwise vmPFC betas for {subid}")

# -------------------------------------------------------
# vmPFC spherical ROIs (10mm radius)
# Tashjian et al. (2021) anterior/posterior vmPFC gradient
# -------------------------------------------------------
seeds = {
    'ant_vmPFC':  {'coords': [(-2, 46, -10)],  'radius': 10},
    'post_vmPFC': {'coords': [(0, 26, -12)],    'radius': 10},
}

# Image conditions only (used for cross-run spatial correlation / persistence)
conditions = ['neg_image', 'neu_image', 'pos_image']
runs = ['01', '02', '03']

rows = []

for run in runs:
    for cond in conditions:
        beta_file = sub_glm_dir / f"run-{run}_{cond}_beta.nii.gz"

        if not beta_file.exists():
            print(f"  WARNING: {beta_file.name} not found — skipping")
            continue

        beta_img = image.load_img(str(beta_file))

        for seed_name, seed_params in seeds.items():
            masker = NiftiSpheresMasker(
                seeds=seed_params['coords'],
                radius=seed_params['radius'],
                standardize=False,
                allow_overlap=True
            )
            try:
                voxel_vals = masker.fit_transform(beta_img).flatten()
            except Exception as e:
                print(f"  ERROR: {seed_name} run {run} {cond}: {e}")
                continue

            nvox = len(voxel_vals)
            row = [subid, f"run-{run}", cond, seed_name, nvox] + voxel_vals.tolist()
            rows.append(row)
            print(f"  {seed_name} run-{run} {cond}: {nvox} voxels")

if not rows:
    print(f"No data extracted for {subid}")
    import sys; sys.exit(0)

# Build DataFrame with dynamic voxel columns
max_voxels = max(len(r) - 5 for r in rows)
voxel_cols = [f"beta_{i}" for i in range(max_voxels)]
columns = ["subject", "run", "condition", "seed", "nvox_resampled"] + voxel_cols

for r in rows:
    if len(r) < len(columns):
        r += [np.nan] * (len(columns) - len(r))

df = pd.DataFrame(rows, columns=columns)

out_file = sub_out_dir / f"{subid}_voxelwise_vmPFC_betas.csv"
df.to_csv(out_file, index=False)
print(f"Saved: {out_file}  ({len(df)} rows)")
EOF
