#!/bin/bash
# extract_amygdala.sh  (MR1 / MIDUS Refresher)
#
# SLURM array job: Extract voxelwise amygdala betas from per-run GLM output.
#
# For each subject, loads beta maps from runGLM.sh and extracts voxelwise
# values within left and right amygdala masks (Harvard-Oxford atlas, 50%
# threshold, 2mm resolution). Masks are resampled to match the beta map space.
#
# Output per subject (in MR1/voxelwise_betas/<subid>/):
#   - <subid>_voxelwise_amygdala_betas.csv
#     Columns: subject, run, condition, hemisphere, nvox_resampled, beta_0..N
#     One row per run x condition x hemisphere
#
# These CSVs are used by run_cross_corr.py to compute cross-run persistence.
#
# TODO: set --array=1-N to match MR1_subject_list.txt (expected ~127).
#
#SBATCH -J amygdala_beta_extract_MR1
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1/log/amygdala_beta_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1/log/amygdala_beta_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
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

# Subject ID for this array task
SUBJ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/MR1/MR1_subject_list.txt)

# Paths
BETA_DIR=/scratch/groups/fvlin/MIDUS/MR1/GLM_output/${SUBJ}
OUTPUT_DIR=/scratch/groups/fvlin/MIDUS/MR1/voxelwise_betas/${SUBJ}
mkdir -p $OUTPUT_DIR

export SUBJ
export BETA_DIR
export OUTPUT_DIR

# Run Python script
python3 << 'EOF'
import os
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn import image, masking, datasets

subj = os.environ["SUBJ"]
beta_dir = Path(os.environ["BETA_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(exist_ok=True)

# Load beta maps for this subject
# Filenames follow: run-01_neg_image_beta.nii.gz etc.
beta_files = sorted(beta_dir.glob("run-*_*.nii.gz"))

# Load 50% threshold Harvard-Oxford amygdala masks (2mm)
atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
labels = atlas.labels
atlas_img = atlas.filename

left_idx = labels.index('Left Amygdala')
right_idx = labels.index('Right Amygdala')

def get_hemi_mask(idx):
    return image.math_img("img == {}".format(idx), img=atlas_img)

rows = []

for beta_file in beta_files:
    # Extract run and condition from filename
    fname = beta_file.stem  # e.g., run-01_neg_image_beta
    parts = fname.split("_")
    run = parts[0]           # run-01
    cond = "_".join(parts[1:-1])  # neg_image, neu_image, pos_image, neg_face etc.

    # Resample masks into beta space
    left_mask_res = image.resample_to_img(get_hemi_mask(left_idx), beta_file, interpolation="nearest")
    right_mask_res = image.resample_to_img(get_hemi_mask(right_idx), beta_file, interpolation="nearest")

    # Extract voxelwise betas
    betas_left = masking.apply_mask(beta_file, left_mask_res)
    betas_right = masking.apply_mask(beta_file, right_mask_res)

    nvox_left = len(betas_left)
    nvox_right = len(betas_right)

    # One row per hemisphere, voxel values as columns
    row_left = [subj, run, cond, "L", nvox_left] + betas_left.tolist()
    row_right = [subj, run, cond, "R", nvox_right] + betas_right.tolist()
    rows.append(row_left)
    rows.append(row_right)

# Dynamically create column names
max_voxels = max(len(r) - 5 for r in rows)
voxel_cols = [f"beta_{i}" for i in range(max_voxels)]
columns = ["subject", "run", "condition", "hemisphere", "nvox_resampled"] + voxel_cols

# Fill shorter rows with NaN
for r in rows:
    if len(r) < len(columns):
        r += [np.nan] * (len(columns) - len(r))

df = pd.DataFrame(rows, columns=columns)

# Save to CSV
output_file = output_dir / f"{subj}_voxelwise_amygdala_betas.csv"
df.to_csv(output_file, index=False)
EOF
