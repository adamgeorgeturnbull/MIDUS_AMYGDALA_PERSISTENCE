#!/bin/bash
# extract_amygdala_concat.sh
#
# SLURM array job: Extract voxelwise amygdala betas from concatenated GLM output.
#
# Same as extract_amygdala.sh but reads from GLM_output_concat (the concatenated
# all-runs GLM). Produces one row per condition x hemisphere (no run dimension).
#
# Output per subject:
#   - <subid>_allruns_voxelwise_amygdala_betas.csv
#     Columns: subject, condition, hemisphere, nvox_resampled, beta_0..N
#
# These CSVs are used by run_cross_corr_concat.py for concatenated persistence.
#
#SBATCH -J amygdala_beta_extract
#SBATCH --output=/scratch/groups/fvlin/MIDUS/log/amygdala_beta_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/log/amygdala_beta_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
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

# Subject ID for this array task
SUBJ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/M3_subject_list.txt)

# Paths
BETA_DIR=/scratch/groups/fvlin/MIDUS/GLM_output_concat/${SUBJ}
OUTPUT_DIR=/scratch/groups/fvlin/MIDUS/voxelwise_betas/${SUBJ}
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

# Load concatenated beta maps for this subject
# Filenames follow: allruns_neg_image_beta.nii.gz, allruns_neg_face_beta.nii.gz, etc.
beta_files = sorted(beta_dir.glob("allruns_*_beta.nii.gz"))

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
    # Extract condition from filename
    # e.g., allruns_neg_image_beta.nii.gz → cond = "neg_image"
    fname = beta_file.stem
    cond = fname.replace("allruns_", "").replace("_beta", "")

    # Resample masks into beta space
    left_mask_res = image.resample_to_img(get_hemi_mask(left_idx), beta_file, interpolation="nearest")
    right_mask_res = image.resample_to_img(get_hemi_mask(right_idx), beta_file, interpolation="nearest")

    # Extract voxelwise betas
    betas_left = masking.apply_mask(beta_file, left_mask_res)
    betas_right = masking.apply_mask(beta_file, right_mask_res)

    nvox_left = len(betas_left)
    nvox_right = len(betas_right)

    # One row per hemisphere
    row_left = [subj, cond, "L", nvox_left] + betas_left.tolist()
    row_right = [subj, cond, "R", nvox_right] + betas_right.tolist()
    rows.append(row_left)
    rows.append(row_right)

# Build dataframe
max_voxels = max(len(r) - 4 for r in rows)
voxel_cols = [f"beta_{i}" for i in range(max_voxels)]
columns = ["subject", "condition", "hemisphere", "nvox_resampled"] + voxel_cols

# Pad rows with NaN
for r in rows:
    if len(r) < len(columns):
        r += [np.nan] * (len(columns) - len(r))

df = pd.DataFrame(rows, columns=columns)

# Save
output_file = output_dir / f"{subj}_allruns_voxelwise_amygdala_betas.csv"
df.to_csv(output_file, index=False)

