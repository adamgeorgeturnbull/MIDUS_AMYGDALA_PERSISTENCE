#!/usr/bin/env python3
"""
run_cross_corr.py  (MR1 / MIDUS Refresher)

Compute cross-run voxelwise amygdala persistence using pairwise correlations
between conditions across different runs. Identical method to the M3 pipeline;
only the input/output paths point at the MR1 tree.

Persistence is operationalized as the spatial correlation between amygdala
activation patterns for emotional stimuli (images) and subsequent neutral
stimuli (faces) across different runs. High correlation indicates that the
spatial pattern of amygdala response to emotional images persists into the
subsequent neutral face presentation, even when estimated from independent data
(different runs).

Method:
    For each subject and hemisphere (L, R, bilateral):
    1. Extract voxelwise beta maps for condition A (images) and B (faces)
    2. Compute Pearson correlations between all cross-run pairs (run_i A vs run_j B, i != j)
    3. Average pairwise correlations in Fisher z-space to get a single persistence estimate

Input:
    Per-subject voxelwise amygdala beta CSVs (from extract_amygdala.sh) with columns:
        run, condition, hemisphere, beta_0, beta_1, ..., beta_N

Output (in MR1_reproduction_20260828/voxelwise_betas_summary/):
    results_summary_<COND_A>_vs_<COND_B>.csv - One row per subject x hemisphere
        with mean_r, median_r, std_r, n_pairs
    results_pairs_<COND_A>_vs_<COND_B>.csv - All pairwise cross-run correlations

@author: aturnbu2
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import math

# ========== USER SETTINGS ==========
BASE_DIR = Path("/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/voxelwise_betas")
OUT_DIR  = Path("/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/voxelwise_betas_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_PAIRS = [
    ("neg_image", "neg_face"),
    ("neu_image", "neu_face"),
    ("pos_image", "pos_face"),
]

COMPUTE_BILATERAL = True
MIN_VOXELS        = 10
# ===================================

def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def inv_fisher_z(z):
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

subjects = sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])
print(f"Processing {len(subjects)} subjects")

# Pre-load all subject data once
subject_data = {}
for subj in subjects:
    subj_dir  = BASE_DIR / subj
    csv_files = [f for f in subj_dir.glob("*voxelwise_amygdala_betas.csv") if "allruns" not in f.name]
    if not csv_files:
        print(f"WARNING: no voxelwise CSV for {subj}, skipping")
        continue
    df = pd.read_csv(csv_files[0])
    df.columns = [c.strip() for c in df.columns]
    voxel_cols = [c for c in df.columns if c.startswith("beta_")]
    lookup = {}
    for _, row in df.iterrows():
        key  = (row['run'], row['condition'], row['hemisphere'])
        vals = row[voxel_cols].to_numpy(dtype=float)
        lookup[key] = vals[~np.isnan(vals)]
    hemis = sorted({hemi for (_, _, hemi) in lookup.keys()})
    subject_data[subj] = {'lookup': lookup, 'hemis': hemis}

if not subject_data:
    print("ERROR: No data collected — check that extract_amygdala.sh produced .csv files in:")
    print(f"  {BASE_DIR}/<subid>/<subid>_voxelwise_amygdala_betas.csv")
    sys.exit(1)

for COND_A, COND_B in CONDITION_PAIRS:
    print(f"\n--- {COND_A} vs {COND_B} ---")
    summary_rows = []
    pairs_rows   = []

    for subj, sdata in subject_data.items():
        lookup = sdata['lookup']
        hemis  = sdata['hemis'] + (["BI"] if COMPUTE_BILATERAL else [])

        runs_A = sorted({run for (run, cond, hemi) in lookup.keys() if cond == COND_A})
        runs_B = sorted({run for (run, cond, hemi) in lookup.keys() if cond == COND_B})

        def get_vec(run, cond, hemi_local):
            if hemi_local != "BI":
                return lookup.get((run, cond, hemi_local), np.array([]))
            left  = lookup.get((run, cond, "L"), np.array([]))
            right = lookup.get((run, cond, "R"), np.array([]))
            if left.size == 0 and right.size == 0:
                return np.array([])
            return np.concatenate([left, right])

        for hemi in hemis:
            pair_rs   = []
            pair_meta = []

            for run_a in runs_A:
                for run_b in runs_B:
                    if run_a == run_b:
                        continue
                    vec_a = get_vec(run_a, COND_A, hemi)
                    vec_b = get_vec(run_b, COND_B, hemi)
                    if vec_a.size < MIN_VOXELS or vec_b.size < MIN_VOXELS:
                        continue
                    nmin = min(vec_a.size, vec_b.size)
                    v1, v2 = vec_a[:nmin], vec_b[:nmin]
                    if np.std(v1) == 0 or np.std(v2) == 0:
                        continue
                    r, p = pearsonr(v1, v2)
                    pair_rs.append(r)
                    pair_meta.append({
                        'subject':    subj,
                        'hemisphere': hemi,
                        'run_A':      run_a,
                        'cond_A':     COND_A,
                        'run_B':      run_b,
                        'cond_B':     COND_B,
                        'r':          r,
                        'p':          p,
                        'n_vox_A':    v1.size,
                        'n_vox_B':    v2.size,
                    })

            if pair_rs:
                zs       = np.array([fisher_z(r) for r in pair_rs])
                mean_r   = inv_fisher_z(np.mean(zs))
                std_r    = np.std(pair_rs, ddof=1)
                median_r = np.median(pair_rs)
                n_pairs  = len(pair_rs)
            else:
                mean_r = std_r = median_r = np.nan
                n_pairs = 0

            summary_rows.append({
                'subject':    subj,
                'hemisphere': hemi,
                'mean_r':     mean_r,
                'median_r':   median_r,
                'std_r':      std_r,
                'n_pairs':    n_pairs,
            })
            pairs_rows += pair_meta

    df_summary = pd.DataFrame(summary_rows)
    df_pairs   = pd.DataFrame(pairs_rows)
    df_summary.to_csv(OUT_DIR / f"results_summary_{COND_A}_vs_{COND_B}.csv", index=False)
    df_pairs.to_csv(OUT_DIR / f"results_pairs_{COND_A}_vs_{COND_B}.csv", index=False)
    print(f"  Wrote results_summary_{COND_A}_vs_{COND_B}.csv  ({len(df_summary)} rows)")

print("\nDone.")
