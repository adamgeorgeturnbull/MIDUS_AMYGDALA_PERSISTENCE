#!/bin/bash
# extract_vmPFC.sh
#
# SLURM array job: Extract voxelwise vmPFC betas from per-run GLM output.
#
# Parallel to extract_amygdala.sh but for vmPFC spherical ROIs.
# Uses condition-level GLM output from runGLM.sh (no new GLM required).
# For each subject, loads beta maps and extracts voxelwise values within
# anterior and posterior vmPFC spherical ROIs (10mm radius).
#
# ROI seeds (Tashjian et al., 2021, Trends in Cognitive Sciences):
#   - ant_vmPFC:  10mm sphere at [-2, 46, -10]  (safety/anterior gradient)
#   - post_vmPFC: 10mm sphere at [0, 26, -12]   (threat/posterior gradient)
#
# Conditions extracted (image and following-face maps for persistence):
#   neg_image, neu_image, pos_image, neg_face, neu_face, pos_face
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
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/vmPFC_beta_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/vmPFC_beta_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-158%20

set -euo pipefail

module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

# Verify nilearn is available in the current environment
python3 -c "import nilearn; print(f'nilearn {nilearn.__version__} available')" || {
  echo "ERROR: nilearn is not available in the current Python environment"
  exit 1
}

SUBJECT_LIST="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_subject_list.txt"

if [ ! -f "$SUBJECT_LIST" ]; then
  echo "ERROR: subject list not found: $SUBJECT_LIST"; exit 1
fi

subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
if [ -z "$subid" ]; then
  echo "ERROR: empty subject ID at line ${SLURM_ARRAY_TASK_ID} of $SUBJECT_LIST"; exit 1
fi
if ! [[ "$subid" =~ ^sub-[0-9]+$ ]]; then
  echo "ERROR: subject ID '${subid}' does not match sub-[0-9]+"; exit 1
fi

glm_dir=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/GLM_output
out_dir=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/voxelwise_vmPFC_betas_image_face
mkdir -p "$out_dir"

export glm_dir
export out_dir
export subid

echo "[$(date)] vmPFC beta extraction: $subid"

python3 << 'EOF'
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image, masking

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
    'ant_vmPFC':  {'coord': (-2, 46, -10),  'radius': 10},
    'post_vmPFC': {'coord': (0, 26, -12),   'radius': 10},
}

def make_sphere_mask(center_mni, radius, ref_img):
    """Build a binary sphere mask in ref_img voxel space."""
    affine = ref_img.affine
    shape  = ref_img.shape[:3]
    i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    vox_coords   = np.column_stack([i.ravel(), j.ravel(), k.ravel()])
    world_coords = nib.affines.apply_affine(affine, vox_coords)
    dists        = np.sqrt(np.sum((world_coords - np.array(center_mni))**2, axis=1))
    mask_data    = (dists <= radius).reshape(shape).astype(np.int8)
    return nib.Nifti1Image(mask_data, affine, ref_img.header)

# Matched image and following-face conditions for directional cross-run persistence
conditions = ['neg_image', 'neu_image', 'pos_image', 'neg_face', 'neu_face', 'pos_face']
runs = ['01', '02', '03']

rows = []
reference_shape = None
reference_affine = None

for run in runs:
    for cond in conditions:
        beta_file = sub_glm_dir / f"run-{run}_{cond}_beta.nii.gz"

        if not beta_file.exists():
            print(f"  WARNING: {beta_file.name} not found — skipping")
            continue

        beta_img = image.load_img(str(beta_file))

        if reference_shape is None:
            reference_shape = beta_img.shape
            reference_affine = beta_img.affine.copy()
        elif (beta_img.shape != reference_shape or
              not np.allclose(beta_img.affine, reference_affine, rtol=0, atol=1e-5)):
            raise ValueError("Beta-map grids differ; voxelwise correspondence requires review")

        for seed_name, seed_params in seeds.items():
            try:
                sphere_mask = make_sphere_mask(
                    seed_params['coord'], seed_params['radius'], beta_img
                )
                voxel_vals = masking.apply_mask(beta_img, sphere_mask)
            except Exception as e:
                print(f"  ERROR: {seed_name} run {run} {cond}: {e}")
                continue

            nvox = len(voxel_vals)
            row = [subid, f"run-{run}", cond, seed_name, nvox] + voxel_vals.tolist()
            rows.append(row)
            print(f"  {seed_name} run-{run} {cond}: {nvox} voxels")

if not rows:
    print(f"No data extracted for {subid}")
    sys.exit(0)

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
