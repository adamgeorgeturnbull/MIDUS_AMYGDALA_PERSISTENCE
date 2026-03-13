#!/usr/bin/env python3
"""
run_cross_corr_aCompCor.py

Identical to run_cross_corr.py but uses aCompCor GLM output
(24 motion + 6 aCompCor regressors) for preprocessing comparison.

Input:
    /scratch/groups/fvlin/MIDUS/M3/voxelwise_betas_aCompCor/<subid>/
        <subid>_voxelwise_amygdala_betas.csv

Output:
    /scratch/groups/fvlin/MIDUS/M3/voxelwise_betas_aCompCor_summary/
        results_summary_<COND_A>_vs_<COND_B>.csv
        results_pairs_<COND_A>_vs_<COND_B>.csv
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import math

# ========== USER SETTINGS ==========
BASE_DIR = Path("/scratch/groups/fvlin/MIDUS/M3/voxelwise_betas_aCompCor")
OUT_DIR  = Path("/scratch/groups/fvlin/MIDUS/M3/voxelwise_betas_aCompCor_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Condition names as they appear in the CSV 'condition' column
COND_A = "pos_image"   # positive images
COND_B = "pos_face"    # faces following positive images

# Compute bilateral (L+R concatenated) in addition to per-hemisphere
COMPUTE_BILATERAL = True

# Minimum number of valid voxels required to compute a correlation
MIN_VOXELS = 10
# ===================================

def fisher_z(r):
    """Fisher z-transform: arctanh(r)."""
    return 0.5 * np.log((1 + r) / (1 - r))

def inv_fisher_z(z):
    """Inverse Fisher z-transform: tanh(z)."""
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

summary_rows = []
pairs_rows = []

subjects = sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])
print(f"Processing {len(subjects)} subjects")

for subj in subjects:
    subj_dir = BASE_DIR / subj

    # Load per-run voxelwise betas (exclude the concatenated allruns file)
    csv_files = [f for f in subj_dir.glob("*voxelwise_amygdala_betas.csv") if "allruns" not in f.name]
    if not csv_files:
        print(f"WARNING: no voxelwise CSV for {subj}, skipping")
        continue

    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]

    # Voxel beta columns (beta_0, beta_1, ...)
    voxel_cols = [c for c in df.columns if c.startswith("beta_")]

    # Build lookup: (run, condition, hemisphere) -> 1D array of voxel betas
    data_lookup = {}
    for _, row in df.iterrows():
        key = (row['run'], row['condition'], row['hemisphere'])
        vals = row[voxel_cols].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]  # drop NaN voxels (padding from unequal hemi sizes)
        data_lookup[key] = vals

    # Identify available runs for each condition
    runs_A = sorted({run for (run, cond, hemi) in data_lookup.keys() if cond == COND_A})
    runs_B = sorted({run for (run, cond, hemi) in data_lookup.keys() if cond == COND_B})

    # Process each hemisphere (and bilateral if requested)
    hemis = sorted({hemi for (_, _, hemi) in data_lookup.keys()})
    if COMPUTE_BILATERAL:
        hemis = hemis + ["BI"]

    for hemi in hemis:
        pair_rs = []
        pair_meta = []

        def get_vec(run, cond, hemi_local):
            """Retrieve voxel vector, concatenating L+R for bilateral."""
            if hemi_local != "BI":
                return data_lookup.get((run, cond, hemi_local), np.array([]))
            else:
                left  = data_lookup.get((run, cond, "L"), np.array([]))
                right = data_lookup.get((run, cond, "R"), np.array([]))
                if left.size == 0 and right.size == 0:
                    return np.array([])
                return np.concatenate([left, right])

        # Compute all cross-run pairs (different runs only)
        for run_a in runs_A:
            for run_b in runs_B:
                if run_a == run_b:
                    continue  # skip same-run pairs (not cross-run)
                vec_a = get_vec(run_a, COND_A, hemi)
                vec_b = get_vec(run_b, COND_B, hemi)

                if vec_a.size < MIN_VOXELS or vec_b.size < MIN_VOXELS:
                    continue

                # Trim to matching length if voxel counts differ slightly
                nmin = min(vec_a.size, vec_b.size)
                if nmin == 0:
                    continue
                v1 = vec_a[:nmin]
                v2 = vec_b[:nmin]

                # Skip if either vector is constant (pearsonr undefined)
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
                    'n_vox_B':    v2.size
                })

        # Average pairwise correlations in Fisher z-space
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
            'n_pairs':    n_pairs
        })
        pairs_rows += pair_meta

# Write outputs
df_summary = pd.DataFrame(summary_rows)
df_pairs   = pd.DataFrame(pairs_rows)

if df_summary.empty:
    print("ERROR: No data collected — check that extract_amygdala_aCompCor.sh produced .csv files in:")
    print(f"  {BASE_DIR}/<subid>/<subid>_voxelwise_amygdala_betas.csv")
    print("Inspect the extract_amygdala_aCompCor SLURM log files for errors.")
    sys.exit(1)

df_summary.to_csv(OUT_DIR / f"results_summary_{COND_A}_vs_{COND_B}.csv", index=False)
df_pairs.to_csv(OUT_DIR / f"results_pairs_{COND_A}_vs_{COND_B}.csv", index=False)

print("Done. Wrote:")
print(f"  {OUT_DIR / f'results_summary_{COND_A}_vs_{COND_B}.csv'}")
print(f"  {OUT_DIR / f'results_pairs_{COND_A}_vs_{COND_B}.csv'}")
