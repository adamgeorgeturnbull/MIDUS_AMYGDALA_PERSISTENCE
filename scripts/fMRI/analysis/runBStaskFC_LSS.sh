#!/bin/bash
# runBStaskFC_LSS.sh
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
#SBATCH -J betaSeries_LSS
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/betaSeries_LSS_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/betaSeries_LSS_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-158%20

set -euo pipefail

module purge
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312

# Verify nilearn is available in the current environment
python3 -c "import nilearn; print(f'nilearn {nilearn.__version__} available')" || {
  echo "ERROR: nilearn is not available in the current Python environment"
  exit 1
}

BIDS_ROOT_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_ImagingSession"
DERIVATIVES_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/derivatives"
OUT_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/BetaSeries_LSS_output"
SUBJECT_LIST="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_subject_list.txt"

# Validate top-level inputs
if [ ! -f "$SUBJECT_LIST" ]; then
  echo "ERROR: subject list not found: $SUBJECT_LIST"; exit 1
fi
if [ ! -d "$BIDS_ROOT_DIR" ]; then
  echo "ERROR: BIDS root not found: $BIDS_ROOT_DIR"; exit 1
fi
if [ ! -d "$DERIVATIVES_DIR" ]; then
  echo "ERROR: derivatives directory not found: $DERIVATIVES_DIR"; exit 1
fi

# Get subject
SUBJ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
if [ -z "$SUBJ" ]; then
  echo "ERROR: empty subject ID at line ${SLURM_ARRAY_TASK_ID} of $SUBJECT_LIST"; exit 1
fi
if ! [[ "$SUBJ" =~ ^sub-[0-9]+$ ]]; then
  echo "ERROR: subject ID '${SUBJ}' does not match sub-[0-9]+"; exit 1
fi

mkdir -p "$OUT_DIR"

export BIDS_ROOT_DIR DERIVATIVES_DIR OUT_DIR SUBJ

echo "[$(date)] LSS beta-series FC: $SUBJ"

python3 << 'EOF'
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn import image, datasets
from nilearn.maskers import NiftiMasker, NiftiSpheresMasker
from nilearn.glm.first_level import FirstLevelModel
from scipy.stats import pearsonr

subid           = os.environ['SUBJ']
bids_root_dir   = Path(os.environ['BIDS_ROOT_DIR'])
derivatives_dir = Path(os.environ['DERIVATIVES_DIR'])
out_dir         = Path(os.environ['OUT_DIR'])
sub_out_dir     = out_dir / subid
sub_out_dir.mkdir(parents=True, exist_ok=True)

print(f"Running LSS ROI beta-series FC: {subid}")
sys.stdout.flush()

# -------------------------------------------------------
# Define ROI maskers
# -------------------------------------------------------
atlas      = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
labels     = atlas.labels
atlas_img  = atlas.filename
left_idx   = labels.index('Left Amygdala')
right_idx  = labels.index('Right Amygdala')

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
runs       = ['01', '02', '03']
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
# Main loop: per run
# -------------------------------------------------------
results_all_runs = []

for run in runs:
    bold_file      = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = derivatives_dir / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_desc-confounds_timeseries.tsv"
    events_file    = bids_root_dir   / subid / 'func' / f"{subid}_task-EmotionRegulation_run-{run}_events.tsv"

    missing = [str(p) for p in [bold_file, confounds_file, events_file] if not p.exists()]
    if missing:
        print(f"  WARNING: run {run} missing file(s), skipping: {missing}", flush=True)
        continue

    bold_img          = image.index_img(bold_file, slice(4, None))
    confounds         = pd.read_csv(confounds_file, sep='\t')
    motion_regressors = confounds[motion_columns].iloc[4:]

    # Build trial-wise events
    events = pd.read_csv(events_file, sep='\t')
    available_conds = [v for v in conditions if v in events['valence'].values]
    clean_events_list = []
    for val in available_conds:
        df_val = events[events['valence'] == val][['onset_trimmed', 'duration']].copy()
        df_val = df_val.rename(columns={'onset_trimmed': 'onset'})
        df_val['duration'] = 6.0
        df_val = df_val.reset_index(drop=True)
        df_val['trial_type'] = [f"{val}_{i+1:03d}" for i in range(len(df_val))]
        clean_events_list.append(df_val)
    if not clean_events_list:
        print(f"  WARNING: run {run} events file has no valid valence values — skipping run", flush=True)
        continue
    events_all = pd.concat(clean_events_list, ignore_index=True).sort_values('onset').reset_index(drop=True)

    trials_by_cond = {
        cond: [t for t in events_all['trial_type'] if t.startswith(cond + '_')]
        for cond in available_conds
    }
    print(f"  Run {run}: " + ", ".join(f"{c}={len(v)}" for c, v in trials_by_cond.items()), flush=True)

    # Fit maskers once on this run's BOLD (sets space/shape)
    for masker in maskers.values():
        masker.fit(bold_img)

    # Accumulate per-trial ROI betas for this run
    # Structure: beta_series[roi][cond] = [beta_trial_1, beta_trial_2, ...]
    beta_series = {roi: {cond: [] for cond in available_conds} for roi in maskers}

    for cond in available_conds:
        for trial_i in trials_by_cond[cond]:
            events_lss = build_lss_events(events_all, trial_i, cond, available_conds)

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
                roi_vals = masker.transform(beta_img)
                beta_series[roi][cond].append(float(np.mean(roi_vals)))

    # Convert to arrays
    for roi in maskers:
        for cond in available_conds:
            beta_series[roi][cond] = np.array(beta_series[roi][cond])

    # -------------------------------------------------------
    # Compute Fisher z-transformed correlations per pair/condition
    # -------------------------------------------------------
    results_run = {'run': run}

    for seed, target in roi_pairs:
        pair_name = f'{seed}-{target}'
        z_vals = {}

        for cond in available_conds:
            s_ts = beta_series[seed][cond]
            t_ts = beta_series[target][cond]
            if len(s_ts) >= 3 and len(t_ts) >= 3:
                r, _ = pearsonr(s_ts, t_ts)
                z_vals[cond] = fisher_z(r)
            else:
                z_vals[cond] = np.nan
            results_run[f'{pair_name}_{cond}'] = z_vals[cond]

        # Contrast maps
        if 'neg' in z_vals and 'neu' in z_vals and not (np.isnan(z_vals['neg']) or np.isnan(z_vals['neu'])):
            results_run[f'{pair_name}_neg_vs_neu'] = z_vals['neg'] - z_vals['neu']
        else:
            results_run[f'{pair_name}_neg_vs_neu'] = np.nan

        if 'neg' in z_vals and 'pos' in z_vals and not (np.isnan(z_vals['neg']) or np.isnan(z_vals['pos'])):
            results_run[f'{pair_name}_neg_vs_pos'] = z_vals['neg'] - z_vals['pos']
        else:
            results_run[f'{pair_name}_neg_vs_pos'] = np.nan

    results_all_runs.append(results_run)
    print(f"  Run {run} complete", flush=True)

# -------------------------------------------------------
# Average across runs and save
# -------------------------------------------------------
if not results_all_runs:
    print(f"ERROR: no runs completed for {subid}")
    sys.exit(1)

results_df   = pd.DataFrame(results_all_runs)
results_mean = results_df.drop(columns='run').mean().to_frame().T
results_mean['subid'] = subid
results_mean = results_mean[['subid'] + [c for c in results_mean.columns if c != 'subid']]

out_file = sub_out_dir / f"{subid}_betaSeries_LSS_ROI_all_conditions.csv"
results_mean.to_csv(out_file, index=False)
print(f"Saved: {out_file}")
print(results_mean.to_string(index=False))
EOF
