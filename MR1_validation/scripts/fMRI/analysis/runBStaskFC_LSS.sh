#!/bin/bash
# runBStaskFC_LSS.sh  (MR1 / MIDUS Refresher)
#
# SLURM array job: ROI-level beta-series task-based FC using Least Squares Separate (LSS).
#
# PRIMARY ROI-level FC method. Replaces runBStaskFC_LSA.sh (see archive/).
# Output format is identical to LSA version for downstream compatibility
# with analysis scripts 04_fc_affect.py, 05_fc_persistence.py etc.
#
# LSS method (Mumford et al., 2012):
#   For each trial i in condition C:
#     - Regressor 1: trial_i (trial of interest, modeled individually)
#     - Regressor 2: all OTHER trials of condition C, collapsed ('C_others')
#     - Remaining conditions: one collapsed regressor each ('other_cond_all')
#     - Plus 24 motion parameters
#   Extract mean beta within each ROI for trial_i → ROI beta-series value.
#   Correlate seed beta-series with target beta-series per condition.
#
# ROIs:
#   Seeds:   L/R amygdala (Harvard-Oxford 50% threshold, 2mm)
#   Targets: ant_vmPFC 10mm sphere at [-2, 46, -10]
#            post_vmPFC 10mm sphere at [0, 26, -12]
#            (Tashjian et al., 2021, Trends in Cognitive Sciences)
#
# Output per subject:
#   <out_dir>/<subid>/<subid>_betaSeries_LSS_ROI_all_conditions.csv
#     One row per subject.
#     Columns: subid,
#              l_amyg-ant_vmPFC_neg,  l_amyg-ant_vmPFC_neu,  l_amyg-ant_vmPFC_pos,
#              l_amyg-ant_vmPFC_neg_vs_neu,  l_amyg-ant_vmPFC_neg_vs_pos,
#              l_amyg-post_vmPFC_neg, ... (same for all 4 pairs)
#     Values: Fisher z-transformed correlation, averaged across runs.
#
#SBATCH -J betaSeries_LSS_MR1
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/betaSeries_LSS_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/betaSeries_LSS_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-123%20

set -euo pipefail

module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

SUBJECT_LIST=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/MR1_completed_subject_list.txt
bids_root_dir=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/MR_P5_ImagingSession
derivatives_dir=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/derivatives
out_dir=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/BetaSeries_LSS_output
mkdir -p "$out_dir"

export bids_root_dir=$bids_root_dir
export derivatives_dir=$derivatives_dir
export out_dir=$out_dir

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

# Get subject and validate format
subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
if [[ ! "$subid" =~ ^sub-[0-9]+$ ]]; then
    echo "ABORT: subject at position ${SLURM_ARRAY_TASK_ID} does not match expected pattern" >&2
    exit 1
fi
export subid=$subid

python3 << 'EOF'
import os, sys
from pathlib import Path

# Nilearn import and version check — abort before any further work on failure
try:
    import nilearn
except ImportError:
    sys.exit("ABORT: nilearn is not importable")
if nilearn.__version__ != '0.12.0':
    sys.exit(f"ABORT: nilearn {nilearn.__version__} != required 0.12.0")

import nibabel as nib
import pandas as pd
import numpy as np
from nilearn import image, datasets
from nilearn.maskers import NiftiMasker, NiftiSpheresMasker
from nilearn.glm.first_level import FirstLevelModel
from scipy.stats import pearsonr

array_task_id   = os.environ.get('SLURM_ARRAY_TASK_ID', '?')
subid           = os.environ['subid']
bids_root_dir   = Path(os.environ['bids_root_dir'])
derivatives_dir = Path(os.environ['derivatives_dir'])
out_dir         = Path(os.environ['out_dir'])
sub_out_dir     = out_dir / subid
sub_out_dir.mkdir(parents=True, exist_ok=True)

print(f"Array task {array_task_id}: starting LSS ROI beta-series FC")
sys.stdout.flush()

# -------------------------------------------------------
# Define ROI maskers
# -------------------------------------------------------
atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
labels = atlas.labels
atlas_img = atlas.filename
left_idx  = labels.index('Left Amygdala')
right_idx = labels.index('Right Amygdala')

def get_atlas_mask(idx):
    return image.math_img("img == {}".format(idx), img=atlas_img)

ant_vmPFC_coords  = [(-2, 46, -10)]
post_vmPFC_coords = [(0, 26, -12)]

maskers = {
    'l_amyg':     NiftiMasker(mask_img=get_atlas_mask(left_idx),  standardize=True),
    'r_amyg':     NiftiMasker(mask_img=get_atlas_mask(right_idx), standardize=True),
    'ant_vmPFC':  NiftiSpheresMasker(seeds=ant_vmPFC_coords,  radius=10, standardize=True),
    'post_vmPFC': NiftiSpheresMasker(seeds=post_vmPFC_coords, radius=10, standardize=True),
}

roi_pairs = [
    ('l_amyg', 'ant_vmPFC'),
    ('l_amyg', 'post_vmPFC'),
    ('r_amyg', 'ant_vmPFC'),
    ('r_amyg', 'post_vmPFC'),
]

# -------------------------------------------------------
# Run parameters
# -------------------------------------------------------
N_DUMMIES  = 4
TR         = 2.0
conditions = ['neg', 'neu', 'pos']

motion_columns = [
    'trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
    'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
    'trans_z','trans_z_derivative1','trans_z_power2','trans_z_derivative1_power2',
    'rot_x','rot_x_derivative1','rot_x_power2','rot_x_derivative1_power2',
    'rot_y','rot_y_derivative1','rot_y_power2','rot_y_derivative1_power2',
    'rot_z','rot_z_derivative1','rot_z_power2','rot_z_derivative1_power2'
]

def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))

def build_lss_events(events_all, trial_of_interest, cond_of_interest, all_conditions):
    """
    Build LSS event table for one trial.
      trial_of_interest -> its own unique regressor
      other trials of same condition -> '<cond>_others'
      other conditions -> '<cond>_all' (one per condition)
    """
    rows = []
    for _, ev in events_all.iterrows():
        t = ev['trial_type']
        if t == trial_of_interest:
            rows.append({'onset': ev['onset'], 'duration': ev['duration'],
                         'trial_type': trial_of_interest})
        elif t.startswith(cond_of_interest + '_'):
            rows.append({'onset': ev['onset'], 'duration': ev['duration'],
                         'trial_type': f'{cond_of_interest}_others'})
        else:
            cond = t.rsplit('_', 1)[0]
            rows.append({'onset': ev['onset'], 'duration': ev['duration'],
                         'trial_type': f'{cond}_all'})
    return pd.DataFrame(rows)

# -------------------------------------------------------
# Pre-loop: count available BOLD runs; require at least two
# -------------------------------------------------------
available_bolds = []
for r in ['01', '02', '03']:
    bf = (derivatives_dir / subid / 'func' /
          f"{subid}_task-EmotionRegulation_run-{r}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    if bf.exists():
        available_bolds.append(r)

if len(available_bolds) < 2:
    sys.exit(f"ABORT: fewer than 2 BOLD runs found (array task {array_task_id})")

# -------------------------------------------------------
# Main loop: per run
# -------------------------------------------------------
completed_runs   = 0
results_all_runs = []

for run in ['01', '02', '03']:
    bold_file      = (derivatives_dir / subid / 'func' /
                      f"{subid}_task-EmotionRegulation_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    confounds_file = (derivatives_dir / subid / 'func' /
                      f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv")
    events_file    = (bids_root_dir / subid / 'func' /
                      f"{subid}_task-EmotionRegulation_run-{run}_events.tsv")

    # Missing BOLD: skip (allowed; at-least-2 already enforced above)
    if not bold_file.exists():
        print(f"  run {run}: BOLD absent, skipping")
        continue

    # BOLD exists: confounds and events must both be present
    if not confounds_file.exists():
        sys.exit(f"ABORT: run {run} BOLD exists but confounds file is absent")
    if not events_file.exists():
        sys.exit(f"ABORT: run {run} BOLD exists but events file is absent")

    print(f"Array task {array_task_id}: run {run}", flush=True)

    # ── BOLD validation ────────────────────────────────────────────────────────
    bold_nib = nib.load(str(bold_file))
    if bold_nib.ndim != 4:
        sys.exit(f"ABORT: run {run} BOLD is not 4D (shape {bold_nib.shape})")
    n_bold_vols = bold_nib.shape[3]
    if n_bold_vols <= 4:
        sys.exit(f"ABORT: run {run} BOLD has {n_bold_vols} volume(s), must be > 4")

    # ── Confound loading and alignment ─────────────────────────────────────────
    confounds = pd.read_csv(confounds_file, sep='\t')
    if len(confounds) != n_bold_vols:
        sys.exit(f"ABORT: run {run} confound rows ({len(confounds)}) != BOLD volumes ({n_bold_vols})")

    missing_motion = [c for c in motion_columns if c not in confounds.columns]
    if missing_motion:
        sys.exit(f"ABORT: run {run} missing {len(missing_motion)} motion column(s): {missing_motion}")

    # ── Events loading and validation ──────────────────────────────────────────
    events = pd.read_csv(events_file, sep='\t')

    required_event_cols = ['onset', 'duration', 'database', 'valence']
    missing_event_cols  = [c for c in required_event_cols if c not in events.columns]
    if missing_event_cols:
        sys.exit(f"ABORT: run {run} events missing column(s): {missing_event_cols}")

    for col in ['onset', 'duration']:
        raw         = events[col].copy()
        events[col] = pd.to_numeric(raw, errors='coerce')
        bad_conv    = raw[raw.notna() & events[col].isna()]
        if len(bad_conv) > 0:
            sys.exit(f"ABORT: run {run} events '{col}' has {len(bad_conv)} non-numeric value(s)")
        if events[col].isna().any():
            sys.exit(f"ABORT: run {run} events '{col}' has {events[col].isna().sum()} missing value(s)")

    # ── Valence normalization (MR1 events use full words) ──────────────────────
    val_map = {'negative': 'neg', 'neutral': 'neu', 'positive': 'pos'}
    events['valence'] = events['valence'].map(val_map).fillna(events['valence'])

    # ── Timing alignment: subtract dummy duration ──────────────────────────────
    # MR1 confirmed: N_DUMMIES = 4 (dummies in-file; BIDS metadata reports zero
    # discarded volumes; no onset_trimmed column present in MR1 events files).
    events['onset'] = events['onset'] - N_DUMMIES * TR
    if (events['onset'] < 0).any():
        sys.exit(f"ABORT: run {run} has {(events['onset'] < 0).sum()} negative onset(s) after dummy removal")

    # ── Require all three conditions with at least three trials each ───────────
    for cond in conditions:
        n_trials = int((events['valence'] == cond).sum())
        if n_trials == 0:
            sys.exit(f"ABORT: run {run} has no trials for condition '{cond}'")
        if n_trials < 3:
            sys.exit(f"ABORT: run {run} has only {n_trials} trial(s) for '{cond}', need >= 3")

    # ── Build trial-wise events (duration overridden to 6.0 s) ────────────────
    clean_events_list = []
    for val in conditions:
        df_val = events[events['valence'] == val][['onset', 'duration']].copy()
        df_val['duration'] = 6.0
        df_val = df_val.reset_index(drop=True)
        df_val['trial_type'] = [f"{val}_{i+1:03d}" for i in range(len(df_val))]
        clean_events_list.append(df_val)
    events_all = pd.concat(clean_events_list, ignore_index=True).sort_values('onset').reset_index(drop=True)

    trials_by_cond = {
        cond: [t for t in events_all['trial_type'] if t.startswith(cond + '_')]
        for cond in conditions
    }
    print(f"  run {run}: " + ", ".join(f"{c}={len(v)}" for c, v in trials_by_cond.items()), flush=True)

    # ── BOLD trimming ─────────────────────────────────────────────────────────
    bold_img = image.index_img(bold_file, slice(N_DUMMIES, None))

    # ── Confound trimming and post-trim validation ─────────────────────────────
    motion_regressors = confounds[motion_columns].iloc[N_DUMMIES:].reset_index(drop=True)
    n_trimmed_bold    = bold_img.shape[3]
    if len(motion_regressors) != n_trimmed_bold:
        sys.exit(
            f"ABORT: run {run} trimmed confound rows ({len(motion_regressors)}) "
            f"!= trimmed BOLD volumes ({n_trimmed_bold})"
        )
    if motion_regressors.isnull().any().any() or not np.isfinite(motion_regressors.values).all():
        sys.exit(f"ABORT: run {run} trimmed motion matrix contains missing or non-finite values")

    # ── Fit maskers once on this run's BOLD (sets space/shape) ────────────────
    for masker in maskers.values():
        masker.fit(bold_img)

    # ── Accumulate per-trial ROI betas ────────────────────────────────────────
    # Structure: beta_series[roi][cond] = [beta_trial_1, beta_trial_2, ...]
    beta_series = {roi: {cond: [] for cond in conditions} for roi in maskers}

    for cond in conditions:
        for trial_i in trials_by_cond[cond]:
            events_lss = build_lss_events(events_all, trial_i, cond, conditions)

            glm_lss = FirstLevelModel(
                t_r=TR,
                slice_time_ref=0.5,
                hrf_model='glover',
                drift_model='cosine',
                high_pass=1/128,
                standardize=False,
                noise_model='ar1',
                minimize_memory=True
            )
            glm_lss.fit(run_imgs=bold_img, events=events_lss, confounds=motion_regressors)

            beta_img = glm_lss.compute_contrast(trial_i, output_type='effect_size')

            for roi, masker in maskers.items():
                roi_vals  = masker.transform(beta_img)
                beta_val  = float(np.mean(roi_vals))
                if not np.isfinite(beta_val):
                    sys.exit(f"ABORT: run {run} trial {trial_i} ROI {roi} produced non-finite beta")
                beta_series[roi][cond].append(beta_val)

    # Convert to arrays
    for roi in maskers:
        for cond in conditions:
            beta_series[roi][cond] = np.array(beta_series[roi][cond])

    # ── Compute Fisher z-transformed correlations per pair/condition ───────────
    results_run = {'run': run}

    for seed, target in roi_pairs:
        pair_name = f'{seed}-{target}'
        z_vals    = {}

        for cond in conditions:
            s_ts = beta_series[seed][cond]
            t_ts = beta_series[target][cond]
            if len(s_ts) < 3 or len(t_ts) < 3:
                sys.exit(f"ABORT: run {run} {pair_name} {cond} has fewer than 3 beta values")
            r, _ = pearsonr(s_ts, t_ts)
            z     = fisher_z(r)
            if not np.isfinite(z):
                sys.exit(f"ABORT: run {run} {pair_name} {cond} Fisher-z is non-finite (r={r:.4f})")
            z_vals[cond]                    = z
            results_run[f'{pair_name}_{cond}'] = z

        # Contrast maps
        neg_vs_neu = z_vals['neg'] - z_vals['neu']
        neg_vs_pos = z_vals['neg'] - z_vals['pos']
        if not np.isfinite(neg_vs_neu):
            sys.exit(f"ABORT: run {run} {pair_name}_neg_vs_neu is non-finite")
        if not np.isfinite(neg_vs_pos):
            sys.exit(f"ABORT: run {run} {pair_name}_neg_vs_pos is non-finite")
        results_run[f'{pair_name}_neg_vs_neu'] = neg_vs_neu
        results_run[f'{pair_name}_neg_vs_pos'] = neg_vs_pos

    results_all_runs.append(results_run)
    completed_runs += 1
    print(f"  run {run} complete", flush=True)

# -------------------------------------------------------
# Post-loop: verify completed run count matches available runs
# -------------------------------------------------------
if completed_runs != len(available_bolds):
    sys.exit(
        f"ABORT: expected {len(available_bolds)} completed runs, "
        f"got {completed_runs} (array task {array_task_id})"
    )

# -------------------------------------------------------
# Average across runs and save
# -------------------------------------------------------
results_df   = pd.DataFrame(results_all_runs)
results_mean = results_df.drop(columns='run').mean().to_frame().T
results_mean['subid'] = subid
results_mean = results_mean[['subid'] + [c for c in results_mean.columns if c != 'subid']]

# Validate final output: exactly one row, all numeric values finite
if len(results_mean) != 1:
    sys.exit(f"ABORT: expected 1 output row, got {len(results_mean)}")
for col in [c for c in results_mean.columns if c != 'subid']:
    val = results_mean[col].iloc[0]
    if not np.isfinite(val):
        sys.exit(f"ABORT: final output column '{col}' is non-finite")

out_file = sub_out_dir / f"{subid}_betaSeries_LSS_ROI_all_conditions.csv"
results_mean.to_csv(out_file, index=False)
print(f"Array task {array_task_id}: completed {completed_runs} run(s), output saved")
EOF
