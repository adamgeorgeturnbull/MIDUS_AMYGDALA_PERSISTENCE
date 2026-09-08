#!/usr/bin/env python3
# compare_mr1_reproduction_outputs.py
#
# Compare retained historical MR1 outputs against the MR1 reproduction outputs.
# All console output is aggregate only — no participant identifiers,
# participant-specific filenames, or participant-level values are ever printed.
#
# Expected counts (from reproduction pipeline design):
#   - 123 voxelwise amygdala CSV files (one per participant)
#   - 122 LSS FC CSV files (one per participant; reason for missing file under investigation)
#   - 368 FD summary rows (122 x 3 runs + 1 x 2 runs = 368 participant x run records)
#
# Read-only: never writes, renames, or deletes any data.
# Dependencies: Python standard library, NumPy, pandas.
#
# Usage:  python3 compare_mr1_reproduction_outputs.py

import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HIST_ROOT  = Path('/scratch/groups/fvlin/MIDUS/MR1')
REPRO_ROOT = Path('/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828')

# ─────────────────────────────────────────────────────────────────────────────
# File utilities
# ─────────────────────────────────────────────────────────────────────────────

def _file_md5(path):
    """Return MD5 hex digest of a file without loading it fully into memory."""
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def find_per_subject_files(root, pattern):
    """
    Return {relative_path_key: absolute_Path} for every file matching pattern
    under root.  Keys contain subject identifiers and are NEVER printed.
    """
    result = {}
    for p in sorted(root.glob(pattern)):
        key = p.relative_to(root)   # internal match key only — never printed
        result[key] = p
    return result


def infer_align_keys(files_dict, candidates):
    """
    Inspect the first readable file in files_dict and return every candidate
    column name that is present, preserving candidate order.  Alignment keys are
    semantic identifiers — they are returned regardless of the dtype pandas
    inferred for that column (e.g. 'run' written as '01','02','03' may be read
    as integer, but it is still a valid and required alignment key).  Falls back
    to returning candidates unchanged if no file is readable, so compare_pair
    can report key_columns_missing cleanly.
    """
    for path in files_dict.values():
        try:
            df = pd.read_csv(path, nrows=2)
            return [k for k in candidates if k in df.columns]
        except Exception:
            continue
    return list(candidates)   # fallback


# ─────────────────────────────────────────────────────────────────────────────
# Per-file comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_pair(h_path, r_path, align_keys):
    """
    Compare one matched file pair.  Returns a result dict.
    Catches all per-file exceptions and reports generic error categories.
    Never prints participant identifiers, filenames, or file paths.

    Result keys:
        same_cols, same_row_count, byte_identical,
        cats_match, nums_exact, nums_close,
        max_abs_diff, error
    """
    out = dict(same_cols=False, same_row_count=False, byte_identical=False,
               cats_match=False, nums_exact=False, nums_close=False,
               max_abs_diff=np.nan, error=None)

    # ── MD5 check: record byte_identical but always continue with full validation.
    # Schema, row count, key uniqueness, and alignment must still be verified.
    try:
        out['byte_identical'] = (_file_md5(h_path) == _file_md5(r_path))
    except Exception:
        pass  # byte_identical remains False; continue with content comparison

    # ── Read ──────────────────────────────────────────────────────────────────
    try:
        h_df = pd.read_csv(h_path)
    except Exception:
        out['error'] = 'read_error'; return out
    try:
        r_df = pd.read_csv(r_path)
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
        missing_keys = [k for k in align_keys if k not in h_df.columns]
        if missing_keys:
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
        # No alignment keys: compare by position (suitable for single-row files)
        if len(h_df) != len(r_df):
            out['error'] = 'alignment_mismatch'; return out
        h_al = h_df
        r_al = r_df

    # ── Categorical (nonnumeric, post-index) column comparison ────────────────
    cat_cols = [c for c in h_al.columns
                if not pd.api.types.is_numeric_dtype(h_al[c])]
    if cat_cols:
        out['cats_match'] = all(
            (h_al[c].fillna('__nan__') == r_al[c].fillna('__nan__')).all()
            for c in cat_cols
        )
    else:
        out['cats_match'] = True   # no nonnumeric columns to compare

    # ── Numeric column comparison ─────────────────────────────────────────────
    num_cols = [c for c in h_al.columns
                if pd.api.types.is_numeric_dtype(h_al[c])]
    if num_cols:
        try:
            h_num = h_al[num_cols].to_numpy(dtype=float)
            r_num = r_al[num_cols].to_numpy(dtype=float)

            # Exact match — NaN in the same positions is considered equal
            # (handles structural NaN padding in voxelwise beta columns)
            out['nums_exact'] = bool(np.array_equal(h_num, r_num, equal_nan=True))

            # Tolerance match
            out['nums_close'] = bool(
                np.allclose(h_num, r_num, rtol=1e-7, atol=1e-10, equal_nan=True)
            )

            # Max absolute difference with explicit infinity handling.
            # Matching NaN pairs and matching same-sign infinity pairs are
            # treated as equal (zero contribution).  Any other finite/nonfinite
            # mismatch — one side finite and the other nonfinite, or infinities
            # of opposite sign — contributes np.inf to max_abs_diff.
            equal_nonfinite = (
                (np.isnan(h_num)       & np.isnan(r_num))        |  # NaN == NaN
                ((h_num == np.inf)     & (r_num == np.inf))      |  # +inf == +inf
                ((h_num == -np.inf)    & (r_num == -np.inf))        # -inf == -inf
            )
            mismatch_nonfinite = (
                (~np.isfinite(h_num) | ~np.isfinite(r_num)) & ~equal_nonfinite
            )
            if mismatch_nonfinite.any():
                out['max_abs_diff'] = np.inf
            else:
                finite_both = np.isfinite(h_num) & np.isfinite(r_num)
                if finite_both.any():
                    out['max_abs_diff'] = float(
                        np.max(np.abs(h_num[finite_both] - r_num[finite_both]))
                    )
                else:
                    out['max_abs_diff'] = 0.0
        except Exception:
            out['error'] = 'numeric_comparison_error'
    else:
        out['nums_exact'] = out['nums_close'] = True
        out['max_abs_diff'] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Family-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compare_family(label, h_dict, r_dict, align_keys):
    """
    Compare all matched file pairs in one output family.
    Prints aggregate results only — no participant identifiers or file paths.
    Returns a summary dict for the final interpretation.
    """
    h_keys    = set(h_dict.keys())
    r_keys    = set(r_dict.keys())
    matched   = h_keys & r_keys
    hist_only = len(h_keys - r_keys)
    repro_only = len(r_keys - h_keys)

    cnt = dict(same_cols=0, same_row_count=0, byte_identical=0,
               cats_match=0, nums_exact=0, nums_close=0)
    error_counts = {}
    max_diff     = 0.0
    n_with_diff  = 0

    for key in matched:
        res = compare_pair(h_dict[key], r_dict[key], align_keys)
        if res['error']:
            error_counts[res['error']] = error_counts.get(res['error'], 0) + 1
            continue
        for k in cnt:
            if res[k]:
                cnt[k] += 1
        d = res['max_abs_diff']
        if not np.isnan(d):
            max_diff = max(max_diff, d)
            if d > 0.0:
                n_with_diff += 1

    n_matched    = len(matched)
    n_errors     = sum(error_counts.values())
    n_comparable = n_matched - n_errors

    w = 42  # label width for alignment
    sep = '─' * 58
    print(f'\n{sep}')
    print(f'  {label}')
    print(f'{sep}')
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
        print(f'  {"Identical columns:":{w}} {cnt["same_cols"]} / {n_comparable}')
        print(f'  {"Identical row counts:":{w}} {cnt["same_row_count"]} / {n_comparable}')
        print(f'  {"Byte-identical:":{w}} {cnt["byte_identical"]} / {n_comparable}')
        print(f'  {"Categorical values match:":{w}} {cnt["cats_match"]} / {n_comparable}')
        print(f'  {"Numeric exact (NaN=equal):":{w}} {cnt["nums_exact"]} / {n_comparable}')
        print(f'  {"Numeric allclose(rtol=1e-7, atol=1e-10):":{w}} {cnt["nums_close"]} / {n_comparable}')
        print(f'  {"Files with any numeric difference:":{w}} {n_with_diff} / {n_comparable}')
        print(f'  {"Max absolute numeric difference:":{w}} {max_diff:.6e}')

    return dict(
        label=label,
        n_hist=len(h_keys), n_repro=len(r_keys),
        n_matched=n_matched, n_comparable=n_comparable,
        n_errors=n_errors,
        hist_only=hist_only, repro_only=repro_only,
        **cnt,
        n_with_diff=n_with_diff, max_diff=max_diff,
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

    bar = '=' * 58
    print(bar)
    print('MR1 Reproduction vs Historical Output Comparison')
    print(f'Historical:   {HIST_ROOT}')
    print(f'Reproduction: {REPRO_ROOT}')
    print('All output is aggregate only — no participant identifiers.')
    print(bar)

    results = []

    # ── Family 1: Voxelwise amygdala betas ────────────────────────────────────
    # One CSV per participant.  Each file has rows: run × condition × hemisphere.
    # Alignment keys: subject, run, condition, hemisphere
    #   (all nonnumeric; together they form a unique row key within each file)
    vox_pat = 'voxelwise_betas/sub-*/*_voxelwise_amygdala_betas.csv'
    h_vox   = find_per_subject_files(HIST_ROOT,  vox_pat)
    r_vox   = find_per_subject_files(REPRO_ROOT, vox_pat)
    vox_keys = infer_align_keys(
        h_vox or r_vox,
        ['subject', 'run', 'condition', 'hemisphere'],
    )
    results.append(compare_family(
        'Voxelwise Amygdala Betas', h_vox, r_vox, vox_keys,
    ))

    # ── Family 2: LSS FC per-subject CSVs ────────────────────────────────────
    # One CSV per participant, exactly one data row.
    # Alignment key: nonnumeric identifier column(s), inspected dynamically.
    # Expected column: subid (string, e.g. sub-123).  All other columns are
    # numeric Fisher-z FC values.  After subid becomes the index, no categorical
    # columns remain, so cats_match is trivially True for correct files.
    lss_pat  = 'BetaSeries_LSS_output/sub-*/*_betaSeries_LSS_ROI_all_conditions.csv'
    h_lss    = find_per_subject_files(HIST_ROOT,  lss_pat)
    r_lss    = find_per_subject_files(REPRO_ROOT, lss_pat)
    lss_keys = infer_align_keys(h_lss or r_lss, ['subid', 'subject'])
    results.append(compare_family(
        'LSS FC Beta-Series ROI', h_lss, r_lss, lss_keys,
    ))

    # ── Family 3: FD summary (single file each) ───────────────────────────────
    # One row per participant × run.
    # Alignment keys: subject, run (semantic identifiers; pandas may infer run
    # as numeric, but infer_align_keys returns it regardless of dtype).
    # Remaining columns (mean_fd, n_spikes, pct_spikes, flagged) are all numeric.
    h_fd_path = HIST_ROOT  / 'fd_qc' / 'fd_summary.csv'
    r_fd_path = REPRO_ROOT / 'fd_qc' / 'fd_summary.csv'
    _fd_key   = Path('fd_qc/fd_summary.csv')
    h_fd      = {_fd_key: h_fd_path} if h_fd_path.exists() else {}
    r_fd      = {_fd_key: r_fd_path} if r_fd_path.exists() else {}
    fd_keys   = infer_align_keys(h_fd or r_fd, ['subject', 'run'])
    results.append(compare_family(
        'FD Summary (fd_summary.csv)', h_fd, r_fd, fd_keys,
    ))

    # ── Final interpretation ──────────────────────────────────────────────────
    print(f'\n{bar}')
    print('Final Interpretation')
    print(bar)

    # Determine outcome across all families
    all_exact = all(
        r['n_comparable'] > 0
        and r['nums_exact'] == r['n_comparable']
        and r['cats_match'] == r['n_comparable']
        for r in results
    )
    all_close = all(
        r['n_comparable'] > 0 and r['nums_close'] == r['n_comparable']
        for r in results
    )
    any_count_mismatch = any(
        r['hist_only'] > 0 or r['repro_only'] > 0
        for r in results
    )
    any_errors    = any(r['n_errors'] > 0 for r in results)
    any_diff      = any(r['n_with_diff'] > 0 for r in results)
    overall_max   = max(r['max_diff'] for r in results)

    if all_exact and not any_count_mismatch and not any_errors:
        print('  RESULT: Exact reproduction — all matched outputs are byte-identical')
        print('          or have identical content.')
    elif all_close and not any_count_mismatch and not any_errors:
        print('  RESULT: Numerically equivalent within tolerance')
        print('          (allclose rtol=1e-7, atol=1e-10, equal_nan=True).')
        if any_diff:
            print(f'          Maximum absolute numeric difference: {overall_max:.6e}')
    else:
        print('  RESULT: Differences detected — review per-family details above.')
        if any_count_mismatch:
            print('          File count mismatch in one or more families.')
        if any_errors:
            print('          Comparison errors in one or more families.')
        if any_diff:
            print(f'          Numeric differences present; max diff: {overall_max:.6e}')
        if not any_count_mismatch and not any_errors and not any_diff:
            print('          Categorical value mismatch in one or more families.')


if __name__ == '__main__':
    main()
