#!/usr/bin/env python3
"""
summarize_run_completeness.py  (MR1 / MIDUS Refresher)

Build a participant-level task-run completeness file for the MR1 reproduction.
n_pairs=6 alone does not detect shortened acquisitions; this script captures
the raw volume count from each run's fMRIPrep confound file (which runGLM.sh
already validated to match the BOLD volume count), so no BOLD images are loaded.

Input root:
    /scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/derivatives

Files discovered:
    sub-*/func/sub-*_task-EmotionRegulation_run-0[123]_desc-confounds_timeseries.tsv
    Expected: 368 files across 123 participants (122 with 3 runs, 1 with 2 runs)

Output:
    /scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/run_completeness.csv
    Columns: MIDUSID, run1_n_volumes, run2_n_volumes, run3_n_volumes,
             n_task_runs, n_complete_231_runs, all_three_runs_231
    Missing runs are blank (NaN) in their run-specific volume column.

Read-only: never writes, renames, or deletes existing files.
Console output is aggregate only — no participant identifiers are printed.
"""

import sys
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

DERIV_ROOT = Path("/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/derivatives")
OUT_FILE   = Path("/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/run_completeness.csv")

EXPECTED_FILES        = 368
EXPECTED_PARTICIPANTS = 123
EXPECTED_RUN_DIST     = {3: 122, 2: 1}
N_VOLUMES_COMPLETE    = 231

GLOB_PATTERN = (
    "sub-*/func/"
    "sub-*_task-EmotionRegulation_run-0[123]_desc-confounds_timeseries.tsv"
)

# Full filename regex; must fullmatch the basename only
FNAME_RE = re.compile(
    r'^(sub-\d+)_task-EmotionRegulation_run-(0[123])'
    r'_desc-confounds_timeseries\.tsv$'
)

# ── Root validation ───────────────────────────────────────────────────────────
if not DERIV_ROOT.is_dir():
    print(f"ERROR: derivatives root not found: {DERIV_ROOT}", file=sys.stderr)
    sys.exit(1)

# ── File discovery ────────────────────────────────────────────────────────────
all_files = sorted(DERIV_ROOT.glob(GLOB_PATTERN))
if len(all_files) != EXPECTED_FILES:
    print(
        f"ERROR: found {len(all_files)} confound files, "
        f"expected {EXPECTED_FILES}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"Found {len(all_files)} confound files")

# ── Validate filenames, read volumes, check for duplicates ───────────────────
# records: {(subid_str, run_str): n_volumes}  — keys never printed
records = {}

for fpath in all_files:
    m = FNAME_RE.fullmatch(fpath.name)
    if m is None:
        print(
            "ERROR: a discovered filename does not match the expected pattern",
            file=sys.stderr,
        )
        sys.exit(1)

    subid_str = m.group(1)   # e.g. 'sub-12345'
    run_str   = m.group(2)   # '01', '02', or '03'
    key       = (subid_str, run_str)

    if key in records:
        print(
            "ERROR: duplicate subject/run combination detected",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read first column only for volume count — avoids loading wide confound matrix
    try:
        df_conf = pd.read_csv(fpath, sep='\t', usecols=[0])
    except Exception:
        print(
            f"ERROR: confound file is not readable (run-{run_str})",
            file=sys.stderr,
        )
        sys.exit(1)

    n_volumes = len(df_conf)
    if n_volumes == 0:
        print(
            f"ERROR: confound file is empty (run-{run_str})",
            file=sys.stderr,
        )
        sys.exit(1)

    records[key] = n_volumes

# ── Validate participant count ─────────────────────────────────────────────────
participants = sorted({subid for (subid, _) in records})
if len(participants) != EXPECTED_PARTICIPANTS:
    print(
        f"ERROR: found {len(participants)} participants, "
        f"expected {EXPECTED_PARTICIPANTS}",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Validate run-count distribution ───────────────────────────────────────────
subj_run_count = Counter(subid for (subid, _) in records)
run_count_dist = dict(Counter(subj_run_count.values()))
if run_count_dist != EXPECTED_RUN_DIST:
    print(
        f"ERROR: run-count distribution {run_count_dist} "
        f"does not match expected {EXPECTED_RUN_DIST}",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Build output rows ─────────────────────────────────────────────────────────
rows = []
for subid_str in participants:
    midusid = int(subid_str.replace('sub-', ''))

    v1 = records.get((subid_str, '01'), None)
    v2 = records.get((subid_str, '02'), None)
    v3 = records.get((subid_str, '03'), None)

    n_task_runs = sum(1 for v in (v1, v2, v3) if v is not None)
    n_complete  = sum(1 for v in (v1, v2, v3) if v == N_VOLUMES_COMPLETE)
    all_three   = int(
        v1 is not None and v2 is not None and v3 is not None
        and v1 == N_VOLUMES_COMPLETE
        and v2 == N_VOLUMES_COMPLETE
        and v3 == N_VOLUMES_COMPLETE
    )

    rows.append({
        'MIDUSID':             midusid,
        'run1_n_volumes':      v1 if v1 is not None else np.nan,
        'run2_n_volumes':      v2 if v2 is not None else np.nan,
        'run3_n_volumes':      v3 if v3 is not None else np.nan,
        'n_task_runs':         n_task_runs,
        'n_complete_231_runs': n_complete,
        'all_three_runs_231':  all_three,
    })

df_out = pd.DataFrame(rows, columns=[
    'MIDUSID', 'run1_n_volumes', 'run2_n_volumes', 'run3_n_volumes',
    'n_task_runs', 'n_complete_231_runs', 'all_three_runs_231',
])

# ── Aggregate console summary ─────────────────────────────────────────────────
print(f"Participants: {len(df_out)}")

all_vol_counts = list(records.values())
vol_dist = Counter(all_vol_counts)
print("n_volumes distribution across all runs:")
for nvol, cnt in sorted(vol_dist.items()):
    print(f"  {nvol} volumes: {cnt} run(s)")

run_dist_summary = Counter(df_out['n_task_runs'].tolist())
print("n_task_runs distribution:")
for nruns, cnt in sorted(run_dist_summary.items()):
    print(f"  {nruns} run(s): {cnt} participant(s)")

n_all_231 = int(df_out['all_three_runs_231'].sum())
print(f"all_three_runs_231 = 1: {n_all_231} participant(s)")

# ── Write output ───────────────────────────────────────────────────────────────
df_out.to_csv(OUT_FILE, index=False)
print(f"Saved: {OUT_FILE}")
