#!/usr/bin/env python3
"""
run_cross_corr_vmPFC.py

Compute cross-run voxelwise vmPFC persistence using pairwise spatial
correlations between image conditions across different runs.

Parallel to run_cross_corr.py but for vmPFC spherical ROIs (ant_vmPFC,
post_vmPFC), using output from extract_vmPFC.sh.

Persistence is operationalized as the mean spatial correlation between
vmPFC activation patterns for the same condition in different runs.
High cross-run correlation indicates stable, condition-specific spatial
patterning within the vmPFC — the same measure as amygdala persistence
but for the vmPFC, providing a comparison ROI.

Conditions:
    neg_image, neu_image, pos_image (each run separately)

Method:
    For each subject, seed (ant_vmPFC / post_vmPFC), and condition:
    1. Extract voxelwise beta vectors per run
    2. Compute Pearson correlations between all cross-run pairs (i != j)
    3. Average in Fisher z-space to get a single persistence estimate

Input:
    /scratch/groups/fvlin/MIDUS/voxelwise_vmPFC_betas/<subid>/
        <subid>_voxelwise_vmPFC_betas.csv
    Columns: subject, run, condition, seed, nvox_resampled, beta_0..N

Output:
    results_summary_vmPFC_persistence.csv  — one row per subject x seed x condition
        subject, seed, condition, mean_r, median_r, std_r, n_pairs
    results_pairs_vmPFC_persistence.csv    — all pairwise cross-run correlations

Run this locally (or as a short non-array Sherlock job) after extract_vmPFC.sh completes.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ========== USER SETTINGS ==========
BASE_DIR = Path("/scratch/groups/fvlin/MIDUS/voxelwise_vmPFC_betas")
OUT_DIR  = Path("/scratch/groups/fvlin/MIDUS/vmPFC_persistence_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS  = ['neg_image', 'neu_image', 'pos_image']
SEEDS       = ['ant_vmPFC', 'post_vmPFC']
MIN_VOXELS  = 10
# ===================================

def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

def inv_fisher_z(z):
    return np.tanh(z)

summary_rows = []
pairs_rows   = []

subjects = sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])
print(f"Processing {len(subjects)} subjects")

for subj in subjects:
    subj_dir = BASE_DIR / subj
    csv_files = list(subj_dir.glob("*_voxelwise_vmPFC_betas.csv"))
    if not csv_files:
        print(f"WARNING: no vmPFC CSV for {subj}, skipping")
        continue

    df = pd.read_csv(csv_files[0])
    df.columns = [c.strip() for c in df.columns]
    voxel_cols = [c for c in df.columns if c.startswith("beta_")]

    # Build lookup: (run, condition, seed) -> 1D array
    data_lookup = {}
    for _, row in df.iterrows():
        key = (row['run'], row['condition'], row['seed'])
        vals = row[voxel_cols].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        data_lookup[key] = vals

    for seed in SEEDS:
        for cond in CONDITIONS:
            # Get all runs that have data for this seed/condition
            runs_avail = sorted({run for (run, c, s) in data_lookup.keys()
                                  if c == cond and s == seed})
            if len(runs_avail) < 2:
                print(f"  {subj} {seed} {cond}: <2 runs, skipping")
                continue

            pair_rs  = []
            pair_meta = []

            for i, run_a in enumerate(runs_avail):
                for run_b in runs_avail[i+1:]:
                    vec_a = data_lookup.get((run_a, cond, seed), np.array([]))
                    vec_b = data_lookup.get((run_b, cond, seed), np.array([]))

                    if vec_a.size < MIN_VOXELS or vec_b.size < MIN_VOXELS:
                        continue

                    # Trim to matching length if sizes differ
                    nmin = min(vec_a.size, vec_b.size)
                    v1, v2 = vec_a[:nmin], vec_b[:nmin]

                    if np.std(v1) == 0 or np.std(v2) == 0:
                        continue

                    r, p = pearsonr(v1, v2)
                    pair_rs.append(r)
                    pair_meta.append({
                        'subject':   subj,
                        'seed':      seed,
                        'condition': cond,
                        'run_A':     run_a,
                        'run_B':     run_b,
                        'r':         r,
                        'p':         p,
                        'n_vox':     nmin,
                    })

            if pair_rs:
                zs     = np.array([fisher_z(r) for r in pair_rs])
                mean_r = inv_fisher_z(np.mean(zs))
                std_r  = np.std(pair_rs, ddof=1)
                med_r  = np.median(pair_rs)
                n_p    = len(pair_rs)
            else:
                mean_r = std_r = med_r = np.nan
                n_p = 0

            summary_rows.append({
                'subject':   subj,
                'seed':      seed,
                'condition': cond,
                'mean_r':    mean_r,
                'median_r':  med_r,
                'std_r':     std_r,
                'n_pairs':   n_p,
            })
            pairs_rows += pair_meta

df_summary = pd.DataFrame(summary_rows)
df_pairs   = pd.DataFrame(pairs_rows)

# Also write a wide-format summary (one row per subject, columns = seed_condition_mean_r)
# to facilitate merging with behavioral data
df_wide = df_summary.pivot_table(
    index='subject', columns=['seed', 'condition'], values='mean_r'
)
df_wide.columns = [f"{s}_{c}_mean_r" for s, c in df_wide.columns]
df_wide = df_wide.reset_index()

summary_file = OUT_DIR / "results_summary_vmPFC_persistence.csv"
pairs_file   = OUT_DIR / "results_pairs_vmPFC_persistence.csv"
wide_file    = OUT_DIR / "results_wide_vmPFC_persistence.csv"

df_summary.to_csv(summary_file, index=False)
df_pairs.to_csv(pairs_file, index=False)
df_wide.to_csv(wide_file, index=False)

print("Done. Wrote:")
print(f"  {summary_file}  ({len(df_summary)} rows)")
print(f"  {pairs_file}    ({len(df_pairs)} rows)")
print(f"  {wide_file}     ({len(df_wide)} subjects)")
