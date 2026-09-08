#!/usr/bin/env python3
# compare_mr1_glm_inputs.py
#
# Diagnostic: compare historical and reproduced MR1 GLM inputs/outputs.
# All output is aggregate only — no participant IDs, relative paths,
# participant-specific filenames, or participant-level values are printed.
#
# Families compared:
#   1. Clean event tables: sub-*/run-*_clean_events.tsv (sep=tab)
#      Row alignment: composite key (onset, duration, trial_type)
#      These three columns form a unique identifier for each event.
#   2. Design matrices:  sub-*/run-*_design_matrix.csv (sep=comma)
#      Row alignment: positional (rows represent sequential TRs; order preserved)
#
# Read-only: never writes, renames, or deletes data.
# Dependencies: Python standard library, NumPy, pandas.
#
# Usage: python3 compare_mr1_glm_inputs.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

HIST_ROOT  = Path('/scratch/groups/fvlin/MIDUS/MR1/GLM_output')
REPRO_ROOT = Path('/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/GLM_output')


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_files(root, pattern):
    """
    Return {relative_path_key: absolute_Path} for every file matching pattern.
    Keys contain participant identifiers and run labels and are NEVER printed.
    """
    result = {}
    for p in sorted(root.glob(pattern)):
        key = p.relative_to(root)   # internal match key only — never printed
        result[key] = p
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-file comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_pair(h_path, r_path, align_keys=None, sep=','):
    """
    Compare one matched file pair.

    align_keys=None   → positional row comparison (design matrices: scan order
                        is meaningful, so rows are compared position-for-position).
    align_keys=[...]  → composite-key alignment (event tables: rows are sorted
                        and matched on the specified columns).

    Catches all per-file exceptions; returns generic error category strings.
    Never prints participant identifiers, file paths, or participant-level values.

    Result keys:
        same_cols, same_row_count,
        cats_match, cat_mismatch,
        nums_exact, nums_close,
        max_abs_diff, error
    """
    out = dict(
        same_cols=False, same_row_count=False,
        cats_match=False, cat_mismatch=False,
        nums_exact=False, nums_close=False,
        max_abs_diff=np.nan, error=None,
    )

    # ── Read ──────────────────────────────────────────────────────────────────
    try:
        h_df = pd.read_csv(h_path, sep=sep)
    except Exception:
        out['error'] = 'read_error'; return out
    try:
        r_df = pd.read_csv(r_path, sep=sep)
    except Exception:
        out['error'] = 'read_error'; return out

    # ── Column identity ───────────────────────────────────────────────────────
    if list(h_df.columns) != list(r_df.columns):
        out['error'] = 'schema_mismatch'; return out
    out['same_cols'] = True

    # ── Row count ─────────────────────────────────────────────────────────────
    out['same_row_count'] = (len(h_df) == len(r_df))

    # ── Row alignment ─────────────────────────────────────────────────────────
    if align_keys:
        # Composite-key alignment for event tables
        missing = [k for k in align_keys if k not in h_df.columns]
        if missing:
            out['error'] = 'key_columns_missing'; return out

        # Uniqueness check — values are never printed
        if h_df[align_keys].duplicated().any() or r_df[align_keys].duplicated().any():
            out['error'] = 'key_not_unique'; return out

        try:
            h_al = h_df.set_index(align_keys).sort_index()
            r_al = r_df.set_index(align_keys).sort_index()
            if not h_al.index.equals(r_al.index):
                out['error'] = 'alignment_mismatch'; return out
        except Exception:
            out['error'] = 'alignment_error'; return out
    else:
        # Positional alignment for design matrices
        if not out['same_row_count']:
            out['error'] = 'row_count_mismatch'; return out
        h_al = h_df
        r_al = r_df

    # ── Categorical (nonnumeric) column comparison ─────────────────────────────
    cat_cols = [c for c in h_al.columns if not pd.api.types.is_numeric_dtype(h_al[c])]
    if cat_cols:
        cats_ok = all(
            (h_al[c].fillna('__nan__') == r_al[c].fillna('__nan__')).all()
            for c in cat_cols
        )
        out['cats_match']   = cats_ok
        out['cat_mismatch'] = not cats_ok
    else:
        out['cats_match']   = True    # no nonnumeric columns after alignment
        out['cat_mismatch'] = False

    # ── Numeric column comparison ─────────────────────────────────────────────
    num_cols = [c for c in h_al.columns if pd.api.types.is_numeric_dtype(h_al[c])]
    if num_cols:
        try:
            h_num = h_al[num_cols].to_numpy(dtype=float)
            r_num = r_al[num_cols].to_numpy(dtype=float)

            # Exact match — NaN in the same positions is considered equal
            out['nums_exact'] = bool(np.array_equal(h_num, r_num, equal_nan=True))

            # Tolerance match
            out['nums_close'] = bool(
                np.allclose(h_num, r_num, rtol=1e-7, atol=1e-10, equal_nan=True)
            )

            # Max absolute difference with explicit infinity handling.
            # Matching NaN pairs and matching same-sign infinity pairs are equal.
            # Any finite/nonfinite mismatch or opposite-sign infinity → np.inf.
            equal_nf = (
                (np.isnan(h_num)    & np.isnan(r_num))    |
                ((h_num == np.inf)  & (r_num == np.inf))  |
                ((h_num == -np.inf) & (r_num == -np.inf))
            )
            mismatch_nf = (~np.isfinite(h_num) | ~np.isfinite(r_num)) & ~equal_nf
            if mismatch_nf.any():
                out['max_abs_diff'] = np.inf
            else:
                fb = np.isfinite(h_num) & np.isfinite(r_num)
                out['max_abs_diff'] = (
                    float(np.max(np.abs(h_num[fb] - r_num[fb]))) if fb.any() else 0.0
                )
        except Exception:
            out['error'] = 'numeric_comparison_error'
    else:
        # No numeric columns remaining after alignment (e.g. all content was
        # in the alignment keys for event tables): treat as exact match.
        out['nums_exact']   = True
        out['nums_close']   = True
        out['max_abs_diff'] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Family-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compare_family(label, h_dict, r_dict, align_keys=None, sep=','):
    """
    Aggregate comparison over a family of matched files.
    Prints aggregate results only — no participant identifiers or paths.
    Returns a summary dict for the final interpretation.
    """
    h_keys    = set(h_dict.keys())
    r_keys    = set(r_dict.keys())
    matched   = h_keys & r_keys
    hist_only  = len(h_keys  - r_keys)
    repro_only = len(r_keys  - h_keys)

    cnt = dict(same_cols=0, same_row_count=0,
               cats_match=0, nums_exact=0, nums_close=0)
    n_cat_mismatch = 0
    error_counts   = {}
    per_file_diffs = []    # one max_abs_diff per comparable file (may include np.inf)
    n_with_diff    = 0

    for key in matched:
        res = compare_pair(h_dict[key], r_dict[key], align_keys=align_keys, sep=sep)
        if res['error']:
            error_counts[res['error']] = error_counts.get(res['error'], 0) + 1
            continue
        for k in cnt:
            if res[k]:
                cnt[k] += 1
        if res['cat_mismatch']:
            n_cat_mismatch += 1
        d = res['max_abs_diff']
        if not np.isnan(d):
            per_file_diffs.append(d)
            if d > 0.0:
                n_with_diff += 1

    n_matched    = len(matched)
    n_errors     = sum(error_counts.values())
    n_comparable = n_matched - n_errors

    # Distribution stats over per-file max absolute differences
    finite_diffs = [d for d in per_file_diffs if np.isfinite(d)]
    n_inf_diffs  = sum(1 for d in per_file_diffs if d == np.inf)
    overall_max  = max(per_file_diffs) if per_file_diffs else 0.0
    median_diff  = float(np.median(finite_diffs)) if finite_diffs else 0.0
    p95_diff     = float(np.percentile(finite_diffs, 95)) if finite_diffs else 0.0

    w        = 46    # label column width
    sep_line = '─' * 62
    print(f'\n{sep_line}')
    print(f'  {label}')
    print(f'{sep_line}')
    print(f'  {"Historical files:":{w}} {len(h_keys)}')
    print(f'  {"Reproduction files:":{w}} {len(r_keys)}')
    print(f'  {"Matched pairs:":{w}} {n_matched}')
    print(f'  {"Historical only:":{w}} {hist_only}')
    print(f'  {"Reproduction only:":{w}} {repro_only}')
    if error_counts:
        cats = ', '.join(f'{k}={v}' for k, v in sorted(error_counts.items()))
        print(f'  {"Comparison errors:":{w}} {n_errors}  ({cats})')
    print(f'  {"Comparable pairs:":{w}} {n_comparable}')
    if n_comparable > 0:
        print(f'  {"Identical column schemas:":{w}} {cnt["same_cols"]} / {n_comparable}')
        print(f'  {"Identical row counts:":{w}} {cnt["same_row_count"]} / {n_comparable}')
        print(f'  {"Categorical values match:":{w}} {cnt["cats_match"]} / {n_comparable}')
        print(f'  {"Categorical mismatches:":{w}} {n_cat_mismatch} / {n_comparable}')
        print(f'  {"Numeric exact (NaN=equal):":{w}} {cnt["nums_exact"]} / {n_comparable}')
        print(f'  {"Numeric allclose(rtol=1e-7, atol=1e-10):":{w}} {cnt["nums_close"]} / {n_comparable}')
        print(f'  {"Files with any numeric difference:":{w}} {n_with_diff} / {n_comparable}')
        if n_inf_diffs > 0:
            print(f'  {"Files with infinite numeric difference:":{w}} {n_inf_diffs} / {n_comparable}')
        print(f'  {"Max abs numeric diff (overall):":{w}} {overall_max:.6e}')
        print(f'  {"Median per-file max abs diff:":{w}} {median_diff:.6e}')
        print(f'  {"95th pct per-file max abs diff:":{w}} {p95_diff:.6e}')

    return dict(
        label=label,
        n_hist=len(h_keys), n_repro=len(r_keys),
        n_matched=n_matched, n_comparable=n_comparable,
        n_errors=n_errors, hist_only=hist_only, repro_only=repro_only,
        **cnt,
        n_cat_mismatch=n_cat_mismatch,
        n_with_diff=n_with_diff, overall_max=overall_max,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Global failure: both roots must exist
    missing_roots = []
    if not HIST_ROOT.exists():
        missing_roots.append(f'Historical root not found: {HIST_ROOT}')
    if not REPRO_ROOT.exists():
        missing_roots.append(f'Reproduction root not found: {REPRO_ROOT}')
    if missing_roots:
        for m in missing_roots:
            print(f'ERROR: {m}', file=sys.stderr)
        sys.exit(1)

    bar = '=' * 62
    print(bar)
    print('MR1 GLM Input/Output Comparison')
    print(f'Historical:   {HIST_ROOT}')
    print(f'Reproduction: {REPRO_ROOT}')
    print('All output is aggregate only — no participant identifiers.')
    print(bar)

    results = []

    # ── Family 1: Clean event tables ──────────────────────────────────────────
    # One TSV per (participant, run).  Columns: onset, duration, trial_type.
    # Alignment: composite key (onset, duration, trial_type).  After alignment
    # these three columns form the index; any additional columns are compared
    # as values.  For the standard three-column format (onset, duration,
    # trial_type only), all content is in the key and the comparison verifies
    # that both files contain the same set of (onset, duration, trial_type)
    # tuples with no duplicates.
    evt_pat = 'sub-*/run-*_clean_events.tsv'
    h_evt   = find_files(HIST_ROOT,  evt_pat)
    r_evt   = find_files(REPRO_ROOT, evt_pat)
    results.append(compare_family(
        'Clean Event Tables (run-*_clean_events.tsv)',
        h_evt, r_evt,
        align_keys=['onset', 'duration', 'trial_type'],
        sep='\t',
    ))

    # ── Family 2: Design matrices ─────────────────────────────────────────────
    # One CSV per (participant, run).  Columns: one per predictor or drift term.
    # Rows represent sequential TRs (scan order preserved); compared positionally.
    # All columns are expected to be numeric.
    dm_pat = 'sub-*/run-*_design_matrix.csv'
    h_dm   = find_files(HIST_ROOT,  dm_pat)
    r_dm   = find_files(REPRO_ROOT, dm_pat)
    results.append(compare_family(
        'Design Matrices (run-*_design_matrix.csv)',
        h_dm, r_dm,
        align_keys=None,
        sep=',',
    ))

    # Global failure: no files found in either root across all families
    if sum(r['n_hist'] + r['n_repro'] for r in results) == 0:
        print('\nERROR: no GLM output files found in either root.', file=sys.stderr)
        sys.exit(1)

    # ── Final interpretation ──────────────────────────────────────────────────
    print(f'\n{bar}')
    print('Final Interpretation')
    print(bar)

    evt_res = results[0]
    dm_res  = results[1]

    evt_exact = (evt_res['n_comparable'] > 0 and
                 evt_res['nums_exact'] == evt_res['n_comparable'] and
                 evt_res['cats_match'] == evt_res['n_comparable'])
    dm_exact  = (dm_res['n_comparable'] > 0 and
                 dm_res['nums_exact'] == dm_res['n_comparable'] and
                 dm_res['cats_match'] == dm_res['n_comparable'])
    evt_close = (evt_res['n_comparable'] > 0 and
                 evt_res['nums_close'] == evt_res['n_comparable'])
    dm_close  = (dm_res['n_comparable'] > 0 and
                 dm_res['nums_close'] == dm_res['n_comparable'])

    any_count_mismatch = any(r['hist_only'] > 0 or r['repro_only'] > 0 for r in results)
    any_errors         = any(r['n_errors'] > 0 for r in results)
    no_structural      = not any_count_mismatch and not any_errors

    if no_structural and evt_exact and dm_exact:
        print('  RESULT: GLM inputs/designs reproduce exactly.')
    elif no_structural and evt_close and dm_close:
        print('  RESULT: GLM inputs/designs are numerically equivalent within tolerance '
              '(allclose rtol=1e-7, atol=1e-10).')
        print(f'          Event tables max diff:    {evt_res["overall_max"]:.6e}')
        print(f'          Design matrices max diff: {dm_res["overall_max"]:.6e}')
    elif any_errors or any_count_mismatch:
        print('  RESULT: Structural comparison failure — review family details above.')
        if any_count_mismatch:
            print('          File count mismatch in one or more families.')
        if any_errors:
            print('          Comparison errors in one or more families.')
    else:
        # Roots present, file counts match, no errors, but content differs
        if not (evt_exact or evt_close):
            print('  RESULT: Event tables differ — review event table details above.')
        if not (dm_exact or dm_close):
            print('  RESULT: Design matrices differ — review design matrix details above.')
        if (evt_exact or evt_close) and (dm_exact or dm_close):
            print('  RESULT: GLM inputs/designs are within tolerance.')


if __name__ == '__main__':
    main()
