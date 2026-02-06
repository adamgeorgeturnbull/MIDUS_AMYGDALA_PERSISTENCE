#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
getMotion.py

Extract framewise displacement (FD) motion metrics from fMRIPrep confounds
files and generate a run-level motion summary with QC flags.

Reads fMRIPrep *_desc-confounds_timeseries.tsv files for each subject and run,
computes mean FD, spike counts, and proportion of high-motion volumes, then
flags runs that exceed QC thresholds.

QC Thresholds:
    - Mean FD > 0.5 mm
    - Proportion of spikes (FD > 0.9 mm) > 20% of TRs
    A run is flagged (fail_motion_qc = True) if EITHER threshold is exceeded.

Input:
    fMRIPrep derivatives directory containing:
    sub-*/func/*_desc-confounds_timeseries.tsv

Output:
    motion_summary.csv - One row per subject x run with columns:
        subject, run, mean_fd, n_spikes, n_trs_used, prop_spikes, fail_motion_qc

@author: aturnbu2
"""

import os, glob
import pandas as pd

# fMRIPrep derivatives directory containing confound timeseries files
confounds_dir = "/scratch/groups/fvlin/MIDUS/derivatives"
output_dir = "/scratch/groups/fvlin/MIDUS"

# QC thresholds
MEAN_FD_THRESH = 0.5     # mean FD (mm) above which a run is flagged
SPIKE_FD_THRESH = 0.9    # single-volume FD (mm) threshold defining a "spike"
PROP_SPIKES_THRESH = 0.2  # max proportion of spike volumes before flagging

# Find all confound files across subjects and runs
confound_files = glob.glob(os.path.join(confounds_dir, "sub-*", "func", "*_desc-confounds_timeseries.tsv"))

rows = []
for f in confound_files:
    sub = f.split("/")[-3]              # e.g. sub-10036
    run = os.path.basename(f).split("_")[2]  # e.g. run-01
    df = pd.read_csv(f, sep="\t")

    if "framewise_displacement" in df.columns:
        # fMRIPrep writes "n/a" for the first TR; coerce to NaN
        fd = pd.to_numeric(df["framewise_displacement"], errors="coerce")
        mean_fd = fd.mean(skipna=True)
        n_spikes = (fd > SPIKE_FD_THRESH).sum()
        n_trs = fd.notna().sum()
        prop_spikes = n_spikes / n_trs if n_trs > 0 else None
    else:
        mean_fd, n_spikes, n_trs, prop_spikes = None, None, None, None

    # Flag run if either threshold exceeded
    fail_motion_qc = False
    if mean_fd is not None and prop_spikes is not None:
        if mean_fd > MEAN_FD_THRESH or prop_spikes > PROP_SPIKES_THRESH:
            fail_motion_qc = True

    rows.append({
        "subject": sub,
        "run": run,
        "mean_fd": mean_fd,
        "n_spikes": n_spikes,
        "n_trs_used": n_trs,
        "prop_spikes": prop_spikes,
        "fail_motion_qc": fail_motion_qc
    })

summary = pd.DataFrame(rows)
output_file = os.path.join(output_dir, "motion_summary.csv")
summary.to_csv(output_file, index=False)
print(f"Wrote {output_file} with {len(summary)} rows")
