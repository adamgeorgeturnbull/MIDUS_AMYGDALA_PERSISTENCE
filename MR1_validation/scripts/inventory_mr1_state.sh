#!/bin/bash
# inventory_mr1_state.sh
#
# Run this ON SHERLOCK (login node, no SLURM needed) to establish the current
# state of the MIDUS MR1 (Refresher) data and any preprocessing/analysis output.
# It only reads/lists — it changes nothing.
#
# Usage:
#   bash inventory_mr1_state.sh > mr1_state_report.txt 2>&1
#   # then scp mr1_state_report.txt back to your laptop
#
# The report tells us where the pipeline currently stands so we know which
# steps still need to run.

MR1=/scratch/groups/fvlin/MIDUS/MR1
BIDS=$MR1/MR_P5_ImagingSession
DERIV=$MR1/derivatives

echo "=========================================================="
echo " MR1 STATE REPORT — $(date)"
echo " Root: $MR1"
echo "=========================================================="

echo
echo "##### 1. Top-level MR1 directory #####"
if [ -d "$MR1" ]; then
    ls -la "$MR1"
else
    echo "MISSING: $MR1 does not exist"
fi

echo
echo "##### 2. Subject-list files #####"
for f in MR1_subject_list.txt missing_list.txt recon_error_list.txt; do
    if [ -f "$MR1/$f" ]; then
        echo "-- $f : $(wc -l < "$MR1/$f") lines"
        head -3 "$MR1/$f"
    else
        echo "-- $f : MISSING"
    fi
done

echo
echo "##### 3. BIDS raw imaging ($BIDS) #####"
if [ -d "$BIDS" ]; then
    echo "Subject dirs (sub-*): $(find "$BIDS" -maxdepth 1 -type d -name 'sub-*' | wc -l)"
    echo "Example subject dirs:"; find "$BIDS" -maxdepth 1 -type d -name 'sub-*' | sort | head -5
    echo
    echo "EmotionRegulation BOLD runs (raw): $(find "$BIDS" -path '*func*task-EmotionRegulation*bold.nii.gz' | wc -l)"
    echo "events.tsv files:                 $(find "$BIDS" -path '*func*task-EmotionRegulation*events.tsv' | wc -l)"
    echo "T1w anat files:                   $(find "$BIDS" -path '*anat*T1w.nii.gz' | wc -l)"
    echo "N4-corrected T1w (recon prep):    $(find "$BIDS" -path '*anat*T1w_N4Corrected.nii.gz' | wc -l)"
    echo
    echo "Per-subject run counts (subject : n_bold_runs):"
    for s in $(find "$BIDS" -maxdepth 1 -type d -name 'sub-*' | sort); do
        n=$(find "$s" -path '*task-EmotionRegulation*bold.nii.gz' | wc -l)
        echo "  $(basename "$s") : $n"
    done
    echo
    echo "One example events.tsv (header + first rows):"
    ev=$(find "$BIDS" -path '*func*task-EmotionRegulation*events.tsv' | sort | head -1)
    if [ -n "$ev" ]; then echo "  $ev"; head -5 "$ev"; fi
else
    echo "MISSING: $BIDS does not exist"
fi

echo
echo "##### 4. fMRIPrep derivatives ($DERIV) #####"
if [ -d "$DERIV" ]; then
    echo "Subject dirs (sub-*): $(find "$DERIV" -maxdepth 1 -type d -name 'sub-*' | wc -l)"
    echo "Preproc BOLD in MNI152NLin2009cAsym:"
    echo "  $(find "$DERIV" -path '*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' | wc -l) files"
    echo "Confounds timeseries: $(find "$DERIV" -name '*desc-confounds_timeseries.tsv' | wc -l) files"
    echo "fMRIPrep HTML reports: $(find "$DERIV" -maxdepth 1 -name '*.html' | wc -l)"
    echo
    echo "Per-subject preproc-bold counts (subject : n_runs_preprocessed):"
    for s in $(find "$DERIV" -maxdepth 1 -type d -name 'sub-*' | sort); do
        n=$(find "$s" -path '*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' | wc -l)
        echo "  $(basename "$s") : $n"
    done
else
    echo "MISSING: $DERIV does not exist (fMRIPrep likely not run yet)"
fi

echo
echo "##### 5. Downstream analysis output #####"
for d in fd_qc GLM_output voxelwise_betas vmPFC_betas betaSeries roi_activations log working freesurfer_subjects; do
    p="$MR1/$d"
    if [ -d "$p" ]; then
        echo "-- $d/ : EXISTS — $(find "$p" -maxdepth 1 | wc -l) entries"
    else
        echo "-- $d/ : absent"
    fi
done

echo
echo "##### 6. Recent log activity (evidence of what ran) #####"
if [ -d "$MR1/log" ]; then
    echo "Newest 10 log files:"
    ls -lt "$MR1/log" 2>/dev/null | head -11
else
    echo "No log directory."
fi

echo
echo "##### 7. Scripts present on Sherlock #####"
echo "Home preprocessing dir:"
ls -la /home/users/aturnbu2/MIDUS_ap/MR1/scripts/preprocessing/preprocessing/ 2>/dev/null || echo "  (path not found — adjust if needed)"

echo
echo "=========================================================="
echo " END OF REPORT"
echo "=========================================================="
