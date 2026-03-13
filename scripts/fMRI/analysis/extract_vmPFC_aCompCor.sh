#!/bin/bash
# extract_vmPFC_aCompCor.sh
#
# SLURM array job: Extract voxelwise vmPFC betas from aCompCor GLM output.
#
# Identical to extract_vmPFC.sh but uses runGLM_aCompCor.sh output
# (24 motion + 6 aCompCor regressors) for preprocessing comparison.
#
# ROI seeds (Tashjian et al., 2021, Trends in Cognitive Sciences):
#   - ant_vmPFC:  10mm sphere at [-2, 46, -10]
#   - post_vmPFC: 10mm sphere at [0, 26, -12]
#
# Conditions extracted (image conditions only, for persistence):
#   neg_image, neu_image, pos_image
#
# Output per subject:
#   <out_dir>/<subid>/<subid>_voxelwise_vmPFC_betas.csv
#
# These CSVs are used by run_cross_corr_vmPFC.py (with updated BASE_DIR)
# for the aCompCor preprocessing comparison.
#
#SBATCH -J vmPFC_beta_aCompCor
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3/log/vmPFC_beta_aCompCor_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3/log/vmPFC_beta_aCompCor_%A_%a.err
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

glm_dir=/scratch/groups/fvlin/MIDUS/M3/GLM_aCompCor_output
out_dir=/scratch/groups/fvlin/MIDUS/M3/voxelwise_vmPFC_betas_aCompCor
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

print(f"Extracting voxelwise vmPFC betas (aCompCor) for {subid}")

seeds = {
    'ant_vmPFC':  {'coords': [(-2, 46, -10)],  'radius': 10},
    'post_vmPFC': {'coords': [(0, 26, -12)],    'radius': 10},
}

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
