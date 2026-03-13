#!/bin/bash
# extract_roi_activations_aCompCor.sh
#
# SLURM array job: Extract mean ROI activation (beta) per condition per subject.
#
# Identical to extract_roi_activations.sh but uses runGLM_aCompCor.sh output
# (24 motion + 6 aCompCor regressors) for preprocessing comparison.
#
# ROIs: L/R amygdala (Harvard-Oxford 50%), ant_vmPFC and post_vmPFC (10mm spheres)
# Conditions: neg_image, neu_image, pos_image, neg_face, neu_face, pos_face
#
#SBATCH -J roi_activations_aCompCor
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3/log/roi_activations_aCompCor_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3/log/roi_activations_aCompCor_%A_%a.err
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
out_dir=/scratch/groups/fvlin/MIDUS/M3/ROI_activations_aCompCor_output
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
from nilearn import image, datasets, masking
from nilearn.maskers import NiftiMasker, NiftiSpheresMasker

subid = os.environ['subid']
glm_dir = Path(os.environ['glm_dir'])
out_dir = Path(os.environ['out_dir'])
sub_glm_dir = glm_dir / subid
sub_out_dir = out_dir / subid
sub_out_dir.mkdir(parents=True, exist_ok=True)

print(f"Extracting ROI activations (aCompCor) for {subid}")

atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
labels = atlas.labels
atlas_img = atlas.filename

left_idx = labels.index('Left Amygdala')
right_idx = labels.index('Right Amygdala')

def get_atlas_mask(idx):
    return image.math_img("img == {}".format(idx), img=atlas_img)

ant_vmPFC_coords = [(-2, 46, -10)]
post_vmPFC_coords = [(0, 26, -12)]

rois = {
    'l_amyg':      NiftiMasker(mask_img=get_atlas_mask(left_idx), standardize=False),
    'r_amyg':      NiftiMasker(mask_img=get_atlas_mask(right_idx), standardize=False),
    'ant_vmPFC':   NiftiSpheresMasker(seeds=ant_vmPFC_coords, radius=10, standardize=False),
    'post_vmPFC':  NiftiSpheresMasker(seeds=post_vmPFC_coords, radius=10, standardize=False),
}

conditions = ['neg_image', 'neu_image', 'pos_image', 'neg_face', 'neu_face', 'pos_face']
runs = ['01', '02', '03']

rows = []

for run in runs:
    row = {'subid': subid, 'run': run}
    any_found = False

    for cond in conditions:
        beta_file = sub_glm_dir / f"run-{run}_{cond}_beta.nii.gz"

        if not beta_file.exists():
            print(f"  WARNING: {beta_file.name} not found — skipping")
            for roi_name in rois:
                row[f'{roi_name}_{cond}'] = np.nan
            continue

        any_found = True
        beta_img = image.load_img(str(beta_file))

        for roi_name, masker in rois.items():
            try:
                vals = masker.fit_transform(beta_img)
                row[f'{roi_name}_{cond}'] = float(np.mean(vals))
            except Exception as e:
                print(f"  ERROR extracting {roi_name} for {cond} run {run}: {e}")
                row[f'{roi_name}_{cond}'] = np.nan

    if any_found:
        rows.append(row)

if not rows:
    print(f"No beta files found for {subid} — skipping output")
    import sys; sys.exit(0)

df = pd.DataFrame(rows)

out_file = sub_out_dir / f"{subid}_roi_activations.csv"
df.to_csv(out_file, index=False)
print(f"Saved: {out_file}")
EOF
