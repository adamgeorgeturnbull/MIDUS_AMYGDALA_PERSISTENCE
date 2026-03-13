#!/usr/bin/env python3
"""
run_cross_corr_concat.py

Compute voxelwise amygdala persistence from concatenated (all-runs) GLM betas.

Unlike run_cross_corr.py which computes cross-run pairwise correlations, this
script uses beta maps from a single GLM fitted to all runs concatenated. This
produces one beta map per condition (rather than per run), so persistence is a
single within-subject spatial correlation between the image and face conditions.

This provides the "concatenated" persistence measure used as a sensitivity
analysis alongside the cross-run persistence (primary measure).

Method:
    For each subject and hemisphere (L, R, bilateral):
    1. Load voxelwise betas from the concatenated GLM
    2. Correlate condition A (images) with condition B (faces)
    3. Fisher z-transform the correlation

Input:
    Per-subject *allruns_voxelwise_amygdala_betas.csv with columns:
        condition, hemisphere, beta_0, beta_1, ..., beta_N

Output:
    results_summary_allruns.csv - One row per subject x hemisphere
        with r, Fisher z, and number of voxels

@author: aturnbu2
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ========== USER SETTINGS ==========
BASE_DIR = Path("/scratch/groups/fvlin/MIDUS/voxelwise_betas")
OUT_DIR = Path("/scratch/groups/fvlin/MIDUS/voxelwise_betas_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Condition names as they appear in the concatenated CSV 'condition' column
COND_A = "neg_image.nii"
COND_B = "neg_face.nii"
COMPUTE_BILATERAL = True
MIN_VOXELS = 10
# ===================================

def fisher_z(r):
    """Fisher z-transform: arctanh(r)."""
    return 0.5 * np.log((1 + r) / (1 - r))

summary_rows = []

subjects = sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])

for subj in subjects:
    subj_dir = BASE_DIR / subj

    # Load concatenated all-runs CSV (one beta map per condition, not per run)
    csv_files = list(subj_dir.glob("*allruns_voxelwise_amygdala_betas.csv"))
    if not csv_files:
        print(f"WARNING: no all-runs voxelwise CSV for {subj}, skipping")
        continue

    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]

    voxel_cols = [c for c in df.columns if c.startswith("beta_")]

    # Build lookup: (condition, hemisphere) -> 1D voxel beta array
    data_lookup = {}
    for _, row in df.iterrows():
        cond = row['condition']
        hemi = row['hemisphere']
        vals = row[voxel_cols].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        data_lookup[(cond, hemi)] = vals

    hemis = sorted({hemi for (_, hemi) in data_lookup.keys()})
    if COMPUTE_BILATERAL:
        hemis += ["BI"]

    for hemi in hemis:
        def get_vec(cond, hemi_local):
            """Retrieve voxel vector, concatenating L+R for bilateral."""
            if hemi_local != "BI":
                return data_lookup.get((cond, hemi_local), np.array([]))
            else:
                left = data_lookup.get((cond, "L"), np.array([]))
                right = data_lookup.get((cond, "R"), np.array([]))
                if left.size == 0 and right.size == 0:
                    return np.array([])
                return np.concatenate([left, right])

        vec_a = get_vec(COND_A, hemi)
        vec_b = get_vec(COND_B, hemi)

        if vec_a.size < MIN_VOXELS or vec_b.size < MIN_VOXELS:
            r = np.nan
            z = np.nan
            n_vox = 0
        else:
            nmin = min(vec_a.size, vec_b.size)
            v1 = vec_a[:nmin]
            v2 = vec_b[:nmin]
            if np.std(v1) == 0 or np.std(v2) == 0:
                r = np.nan
                z = np.nan
            else:
                r, _ = pearsonr(v1, v2)
                z = fisher_z(r)
            n_vox = nmin

        summary_rows.append({
            'subject': subj,
            'hemisphere': hemi,
            'r': r,
            'z': z,
            'n_vox': n_vox
        })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUT_DIR / "results_summary_allruns.csv", index=False)

print("Done. Wrote:")
print(f" - {OUT_DIR / 'results_summary_allruns.csv'}")

