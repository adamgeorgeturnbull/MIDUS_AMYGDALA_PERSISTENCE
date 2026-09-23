#!/usr/bin/env python3
"""vmPFC image-to-following-face spatial persistence across distinct runs.

Each valence uses six directional pairs for three complete runs: image in A
versus face following that valence in B, A != B. Average correlations in Fisher
z space, retaining the existing +/-0.9999 clipping convention. No new GLM fits.

Legacy wide column names (<seed>_<valence>_image_mean_r) are retained for the
existing merge/analysis consumers; they now denote IMAGE-TO-FACE persistence.
Summary/pair exports explicitly record both conditions and missing/invalid pairs.
Default outputs are separate from the previous image-to-image results.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path('/scratch/groups/fvlin/MIDUS/M3_stc_rerun')
SEEDS = ('ant_vmPFC', 'post_vmPFC')
VALENCES = ('neg', 'neu', 'pos')
RUNS = ('run-01', 'run-02', 'run-03')
MIN_VOXELS = 10


def calculate(df, subject):
    """Pure calculation; accepts synthetic frames without filesystem access."""
    required = {'run', 'condition', 'seed', 'nvox_resampled'}
    if not required.issubset(df.columns):
        raise ValueError('Missing required voxel metadata')
    if df.duplicated(['run', 'condition', 'seed']).any():
        raise ValueError('Duplicate run/condition/seed rows')
    cols = [c for c in df.columns if c.startswith('beta_')]
    if cols != [f'beta_{i}' for i in range(len(cols))]:
        raise ValueError('Voxel columns are not in consecutive spatial order')
    lookup = {}
    for _, row in df.iterrows():
        n = float(row['nvox_resampled'])
        if not np.isfinite(n) or n != int(n) or not 0 <= n <= len(cols):
            raise ValueError('Invalid voxel count')
        n = int(n)
        values = row[cols].to_numpy(dtype=float)
        if not np.isnan(values[n:]).all():
            raise ValueError('Unexpected values beyond declared voxel count')
        # Preserve spatial positions: do not drop internal NaNs or truncate vectors.
        lookup[(row['run'], row['condition'], row['seed'])] = values[:n]
    summaries, pairs = [], []
    for seed in SEEDS:
        for valence in VALENCES:
            image_cond, face_cond = f'{valence}_image', f'{valence}_face'
            rs = []
            for a in RUNS:
                for b in RUNS:
                    if a == b:
                        continue
                    x = lookup.get((a, image_cond, seed))
                    y = lookup.get((b, face_cond, seed))
                    status, r, p = 'ok', np.nan, np.nan
                    if x is None or y is None:
                        status = 'missing_map'
                    elif len(x) != len(y):
                        status = 'voxel_count_mismatch'
                    elif len(x) < MIN_VOXELS:
                        status = 'too_few_voxels'
                    elif not (np.isfinite(x).all() and np.isfinite(y).all()):
                        status = 'nonfinite_voxels'
                    elif np.std(x) == 0 or np.std(y) == 0:
                        status = 'constant_pattern'
                    else:
                        r, p = pearsonr(x, y)
                        if not np.isfinite(r):
                            status = 'nonfinite_correlation'
                        else:
                            rs.append(r)
                    pairs.append(dict(subject=subject, seed=seed, condition=image_cond,
                                      run_A=a, cond_A=image_cond, run_B=b, cond_B=face_cond,
                                      r=r, p=p, n_vox_A=0 if x is None else len(x),
                                      n_vox_B=0 if y is None else len(y), status=status))
            summaries.append(dict(subject=subject, seed=seed, condition=image_cond,
                                  cond_A=image_cond, cond_B=face_cond,
                                  mean_r=float(np.tanh(np.mean(np.arctanh(np.clip(rs, -.9999, .9999))))) if rs else np.nan,
                                  median_r=float(np.median(rs)) if rs else np.nan,
                                  std_r=float(np.std(rs, ddof=1)) if len(rs) > 1 else np.nan,
                                  n_pairs=len(rs), expected_pairs=6,
                                  status='complete' if len(rs) == 6 else 'review_required'))
    return summaries, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'voxelwise_vmPFC_betas_image_face')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'vmPFC_persistence_image_face_summary')
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError('Input directory is absent')
    summaries, pairs = [], []
    input_status = []
    for directory in sorted(args.input_dir.iterdir()):
        if not directory.is_dir():
            continue
        files = list(directory.glob('*_voxelwise_vmPFC_betas.csv'))
        if not files:
            input_status.append(dict(subject=directory.name, status='missing_voxel_csv'))
            # Preserve absent subjects in summaries so retained-sample review can
            # distinguish missing extraction from a valid six-pair estimate.
            empty = pd.DataFrame(columns=['run', 'condition', 'seed', 'nvox_resampled'])
            s, p = calculate(empty, directory.name)
            summaries.extend(s)
            pairs.extend(p)
            continue
        if len(files) > 1:
            raise ValueError('Multiple voxel CSVs in an input directory; selection requires review')
        input_status.append(dict(subject=directory.name, status='one_voxel_csv'))
        s, p = calculate(pd.read_csv(files[0]), directory.name)
        summaries.extend(s)
        pairs.extend(p)
    if not summaries:
        raise ValueError('No input summaries produced')
    summary, pair = pd.DataFrame(summaries), pd.DataFrame(pairs)
    wide = summary.pivot(index='subject', columns=['seed', 'condition'], values='mean_r')
    wide.columns = [f'{s}_{c}_mean_r' for s, c in wide.columns]
    # Never overwrite a previous correction run without a separate explicit archive.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output_dir/'results_summary_vmPFC_persistence.csv', index=False)
    pair.to_csv(args.output_dir/'results_pairs_vmPFC_persistence.csv', index=False)
    wide.reset_index().to_csv(args.output_dir/'results_wide_vmPFC_persistence.csv', index=False)
    inventory = pd.DataFrame(input_status)
    inventory.to_csv(args.output_dir/'input_status_vmPFC_persistence.csv', index=False)
    print('Input directory status counts:', inventory['status'].value_counts().to_dict())
    print('Summary rows:', len(summary))
    print('Pair status counts:', pair['status'].value_counts().to_dict())
    print('Summary status counts:', summary['status'].value_counts().to_dict())
    print('Private participant-level files saved on Sherlock. Review retained-sample completeness before promotion.')


if __name__ == '__main__':
    main()
