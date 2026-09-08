#!/bin/bash
# extract_amygdala.sh  (MR1 / MIDUS Refresher)
#
# SLURM array job: Extract voxelwise amygdala betas from per-run GLM output.
#
# For each subject, loads beta maps from runGLM.sh and extracts voxelwise
# values within left and right amygdala masks (Harvard-Oxford atlas, 50%
# threshold, 2mm resolution). Masks are resampled to match the beta map space.
#
# Output per subject (in MR1_reproduction_20260828/voxelwise_betas/<subid>/):
#   - <subid>_voxelwise_amygdala_betas.csv
#     Columns: subject, run, condition, hemisphere, nvox_resampled, beta_0..N
#     One row per run x condition x hemisphere
#
# These CSVs are used by run_cross_corr.py to compute cross-run persistence.
#
#SBATCH -J amygdala_beta_extract_MR1
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/amygdala_beta_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/amygdala_beta_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-123%20

set -euo pipefail

# Load modules
module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

SUBJECT_LIST=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/MR1_completed_subject_list.txt

# Validate subject list: must exist, contain exactly 123 nonblank entries,
# all matching ^sub-[0-9]+$, and all unique.
if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "ABORT: subject list not found: $SUBJECT_LIST" >&2
    exit 1
fi
n_subjects=$(awk 'NF' "$SUBJECT_LIST" | wc -l)
if [[ "$n_subjects" -ne 123 ]]; then
    echo "ABORT: subject list has $n_subjects nonblank entries, expected 123" >&2
    exit 1
fi
n_pattern=$(awk '/^sub-[0-9]+$/' "$SUBJECT_LIST" | wc -l)
if [[ "$n_pattern" -ne "$n_subjects" ]]; then
    echo "ABORT: subject list contains entries not matching ^sub-[0-9]+$" >&2
    exit 1
fi
n_unique=$(awk '/^sub-[0-9]+$/' "$SUBJECT_LIST" | sort -u | wc -l)
if [[ "$n_unique" -ne "$n_subjects" ]]; then
    echo "ABORT: subject list contains duplicate entries" >&2
    exit 1
fi

# Validate array task ID
if [[ "$SLURM_ARRAY_TASK_ID" -lt 1 || "$SLURM_ARRAY_TASK_ID" -gt 123 ]]; then
    echo "ABORT: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [1,123]" >&2
    exit 1
fi

# Get subject and validate format — do not echo the identifier
SUBJ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
if [[ ! "$SUBJ" =~ ^sub-[0-9]+$ ]]; then
    echo "ABORT: subject at position ${SLURM_ARRAY_TASK_ID} does not match expected pattern" >&2
    exit 1
fi

BETA_DIR=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/GLM_output/${SUBJ}
OUTPUT_DIR=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/voxelwise_betas/${SUBJ}
mkdir -p "$OUTPUT_DIR"

export SUBJ
export BETA_DIR
export OUTPUT_DIR

# Run Python script
python3 << 'EOF'
import os, sys, re
from pathlib import Path

# Nilearn import and version check — abort before any further work on failure
try:
    import nilearn
except ImportError:
    sys.exit("ABORT: nilearn is not importable")
if nilearn.__version__ != '0.12.0':
    sys.exit(f"ABORT: nilearn {nilearn.__version__} != required 0.12.0")

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, masking, datasets

array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '?')
subj          = os.environ['SUBJ']
beta_dir      = Path(os.environ['BETA_DIR'])
output_dir    = Path(os.environ['OUTPUT_DIR'])
output_dir.mkdir(exist_ok=True)

EXPECTED_CONDITIONS = {'neg_image', 'neu_image', 'pos_image', 'neg_face', 'neu_face', 'pos_face'}

# Load 50% threshold Harvard-Oxford amygdala masks (2mm) — unchanged
atlas     = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
labels    = atlas.labels
atlas_img = atlas.filename

left_idx  = labels.index('Left Amygdala')
right_idx = labels.index('Right Amygdala')

def get_hemi_mask(idx):
    return image.math_img("img == {}".format(idx), img=atlas_img)

# Discover beta files using an exact filename pattern.
# Expected: run-{01|02|03}_{neg|neu|pos}_{image|face}_beta.nii.gz
FNAME_RE = re.compile(r'run-(0[123])_((?:neg|neu|pos)_(?:image|face))_beta\.nii\.gz')

if not beta_dir.is_dir():
    sys.exit(f"ABORT: GLM output directory not found (array task {array_task_id})")

files_by_run = {}  # 'run-01' -> {'neg_image': Path, ...}
for fpath in sorted(beta_dir.iterdir()):
    fname = fpath.name
    # Skip known non-beta outputs; any remaining _beta.nii.gz file must match exactly
    if not fname.endswith('_beta.nii.gz'):
        continue
    m = FNAME_RE.fullmatch(fname)
    if m is None:
        sys.exit(
            f"ABORT: unexpected beta filename does not match expected pattern "
            f"(array task {array_task_id})"
        )
    run  = 'run-' + m.group(1)
    cond = m.group(2)
    if run not in files_by_run:
        files_by_run[run] = {}
    if cond in files_by_run[run]:
        sys.exit(f"ABORT: duplicate {run} {cond} map (array task {array_task_id})")
    files_by_run[run][cond] = fpath

# Require 2 or 3 runs
n_runs = len(files_by_run)
if n_runs < 2:
    sys.exit(f"ABORT: fewer than 2 runs found ({n_runs}) (array task {array_task_id})")
if n_runs > 3:
    sys.exit(f"ABORT: more than 3 runs found ({n_runs}) (array task {array_task_id})")

# For each run, require exactly the 6 expected conditions
for run, cond_map in sorted(files_by_run.items()):
    found   = set(cond_map.keys())
    missing = EXPECTED_CONDITIONS - found
    if missing:
        sys.exit(f"ABORT: {run} missing condition(s): {sorted(missing)}")

print(f"Array task {array_task_id}: {n_runs} run(s) found, extracting amygdala betas",
      flush=True)

# Extract betas; validate 3D shape, mask voxel counts, and finiteness.
# nvox reference is established from the first map and enforced across all maps.
rows           = []
nvox_left_ref  = None
nvox_right_ref = None

for run in sorted(files_by_run.keys()):
    for cond in sorted(files_by_run[run].keys()):
        beta_file = files_by_run[run][cond]

        # Require 3D
        beta_nib = nib.load(str(beta_file))
        if beta_nib.ndim != 3:
            sys.exit(f"ABORT: {run} {cond} is not 3D (shape {beta_nib.shape})")

        # Resample masks to beta map space (nearest-neighbor — unchanged)
        left_mask_res  = image.resample_to_img(get_hemi_mask(left_idx),  beta_file,
                                               interpolation='nearest')
        right_mask_res = image.resample_to_img(get_hemi_mask(right_idx), beta_file,
                                               interpolation='nearest')

        # Extract voxelwise betas (unchanged)
        betas_left  = masking.apply_mask(beta_file, left_mask_res)
        betas_right = masking.apply_mask(beta_file, right_mask_res)

        nvox_left  = len(betas_left)
        nvox_right = len(betas_right)

        if nvox_left_ref is None:
            # First map: establish reference counts and check minimum
            if nvox_left < 10:
                sys.exit(f"ABORT: left amygdala mask has {nvox_left} voxel(s), need >= 10")
            if nvox_right < 10:
                sys.exit(f"ABORT: right amygdala mask has {nvox_right} voxel(s), need >= 10")
            nvox_left_ref  = nvox_left
            nvox_right_ref = nvox_right
        else:
            # Subsequent maps: validate voxel count constancy
            if nvox_left != nvox_left_ref:
                sys.exit(
                    f"ABORT: {run} {cond} left mask has {nvox_left} voxels, "
                    f"expected {nvox_left_ref}"
                )
            if nvox_right != nvox_right_ref:
                sys.exit(
                    f"ABORT: {run} {cond} right mask has {nvox_right} voxels, "
                    f"expected {nvox_right_ref}"
                )

        # Validate beta values: no missing or non-finite
        if not np.isfinite(betas_left).all():
            sys.exit(f"ABORT: {run} {cond} left betas contain non-finite values")
        if not np.isfinite(betas_right).all():
            sys.exit(f"ABORT: {run} {cond} right betas contain non-finite values")

        # One row per hemisphere (unchanged column order)
        row_left  = [subj, run, cond, 'L', nvox_left]  + betas_left.tolist()
        row_right = [subj, run, cond, 'R', nvox_right] + betas_right.tolist()
        rows.append(row_left)
        rows.append(row_right)

# Validate row count: n_runs × 6 conditions × 2 hemispheres
expected_rows = n_runs * 12
if len(rows) != expected_rows:
    sys.exit(f"ABORT: expected {expected_rows} rows, got {len(rows)}")

# Validate every expected run × condition × hemisphere key appears exactly once
seen_keys = set()
for row in rows:
    key = (row[1], row[2], row[3])  # run, condition, hemisphere
    if key in seen_keys:
        sys.exit(f"ABORT: duplicate output key {key}")
    seen_keys.add(key)

expected_keys = {
    (run, cond, h)
    for run  in files_by_run
    for cond in EXPECTED_CONDITIONS
    for h    in ('L', 'R')
}
missing_keys = expected_keys - seen_keys
if missing_keys:
    sys.exit(f"ABORT: missing output keys: {sorted(missing_keys)}")

# Build DataFrame with dynamic voxel columns (unchanged)
max_voxels = max(len(r) - 5 for r in rows)
voxel_cols = [f'beta_{i}' for i in range(max_voxels)]
columns    = ['subject', 'run', 'condition', 'hemisphere', 'nvox_resampled'] + voxel_cols

for r in rows:
    if len(r) < len(columns):
        r += [np.nan] * (len(columns) - len(r))

df = pd.DataFrame(rows, columns=columns)

output_file = output_dir / f"{subj}_voxelwise_amygdala_betas.csv"
df.to_csv(output_file, index=False)

print(f"Array task {array_task_id}: saved {len(df)} rows "
      f"(runs={n_runs}, left_nvox={nvox_left_ref}, right_nvox={nvox_right_ref})")
EOF
