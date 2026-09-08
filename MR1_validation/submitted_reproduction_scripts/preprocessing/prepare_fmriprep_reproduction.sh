#!/bin/bash
# Prepare and validate the clean MR1 fMRIPrep reproduction workspace.
#
# Run on Sherlock only after the Oak-to-Scratch rsync completes. The script
# prints aggregate counts only; participant identifiers are written to the
# private Sherlock subject-list file but are never printed.

set -euo pipefail

REPRO_ROOT="/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828"
BIDS_ROOT="${REPRO_ROOT}/MR_P5_ImagingSession"
SUBJECT_LIST="${REPRO_ROOT}/MR1_task_subject_list.txt"
FS_LICENSE="/home/users/aturnbu2/freesurfer_license.txt"
CONTAINER="/home/groups/fvlin/simg/fmriprep-24.1.0.simg"
FSAVERAGE="${REPRO_ROOT}/derivatives/sourcedata/freesurfer/fsaverage"
EXPECTED_SUBJECTS=124
EXPECTED_BOLD_RUNS=371
EXPECTED_JSON_RUNS=371

if [ ! -d "$BIDS_ROOT" ]; then
  echo "ERROR: BIDS root not found: $BIDS_ROOT"
  exit 1
fi

if [ ! -r "$FS_LICENSE" ]; then
  echo "ERROR: FreeSurfer license missing or unreadable: $FS_LICENSE"
  exit 1
fi

if [ ! -r "$CONTAINER" ]; then
  echo "ERROR: fMRIPrep container missing or unreadable: $CONTAINER"
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required for the aggregate BIDS metadata checks"
  exit 1
fi

mkdir -p "${REPRO_ROOT}/derivatives"
mkdir -p "${REPRO_ROOT}/working"
mkdir -p "${REPRO_ROOT}/log"

# Require one complete, pre-seeded fsaverage template before launching the
# concurrent array. This prevents tasks from racing to initialize it.
if [ -d "$FSAVERAGE" ]; then
  N_MISSING_FS_LABELS=0
  for label in \
    lh.BA1_exvivo.label \
    lh.BA1_exvivo.thresh.label \
    rh.BA1_exvivo.label \
    rh.BA1_exvivo.thresh.label; do
    if [ ! -s "${FSAVERAGE}/label/${label}" ]; then
      N_MISSING_FS_LABELS=$((N_MISSING_FS_LABELS + 1))
    fi
  done
  if [ "$N_MISSING_FS_LABELS" -ne 0 ]; then
    echo "ERROR: shared fsaverage is incomplete ($N_MISSING_FS_LABELS BA1 labels missing)"
    echo "Run seed_freesurfer_template_reproduction.sh before fMRIPrep"
    exit 1
  fi
  echo "  Shared fsaverage: complete"
else
  echo "ERROR: shared fsaverage has not been seeded"
  echo "Run seed_freesurfer_template_reproduction.sh before fMRIPrep"
  exit 1
fi

TMP_LIST="${SUBJECT_LIST}.tmp.$$"
trap 'rm -f "$TMP_LIST"' EXIT

# Select only participants with at least one EmotionRegulation BOLD run. The
# three raw subject directories without task BOLD data are intentionally not
# submitted to the task-specific fMRIPrep array.
find "$BIDS_ROOT" -type f \
  -name 'sub-*_task-EmotionRegulation_run-*_bold.nii.gz' \
  -printf '%h\n' \
  | sed 's#/func$##' \
  | xargs -r -n 1 basename \
  | sort -u > "$TMP_LIST"

N_SUBJECTS=$(wc -l < "$TMP_LIST")
N_BOLD=$(find "$BIDS_ROOT" -type f \
  -name 'sub-*_task-EmotionRegulation_run-*_bold.nii.gz' | wc -l)
N_JSON=$(find "$BIDS_ROOT" -type f \
  -name 'sub-*_task-EmotionRegulation_run-*_bold.json' | wc -l)

echo "MR1 fMRIPrep reproduction preflight"
echo "  Task participants: $N_SUBJECTS"
echo "  Task BOLD runs:    $N_BOLD"
echo "  Task JSON files:   $N_JSON"

if [ "$N_SUBJECTS" -ne "$EXPECTED_SUBJECTS" ]; then
  echo "ERROR: expected $EXPECTED_SUBJECTS task participants, found $N_SUBJECTS"
  exit 1
fi

if [ "$N_BOLD" -ne "$EXPECTED_BOLD_RUNS" ]; then
  echo "ERROR: expected $EXPECTED_BOLD_RUNS task BOLD runs, found $N_BOLD"
  exit 1
fi

if [ "$N_JSON" -ne "$EXPECTED_JSON_RUNS" ]; then
  echo "ERROR: expected $EXPECTED_JSON_RUNS task JSON files, found $N_JSON"
  exit 1
fi

# All task runs must share the validated MR1 timing definition. TaskName is
# intentionally excluded because its run suffix differs across runs.
N_TIMING_PATTERNS=$(find "$BIDS_ROOT" -type f \
  -name 'sub-*_task-EmotionRegulation_run-*_bold.json' -print0 \
  | xargs -0 jq -c '{RepetitionTime,SliceTiming}' \
  | sort -u \
  | wc -l)

if [ "$N_TIMING_PATTERNS" -ne 1 ]; then
  echo "ERROR: expected one shared TR/SliceTiming definition, found $N_TIMING_PATTERNS"
  exit 1
fi

EXAMPLE_JSON=$(find "$BIDS_ROOT" -type f \
  -name 'sub-*_task-EmotionRegulation_run-*_bold.json' -print -quit)
TR=$(jq -r '.RepetitionTime' "$EXAMPLE_JSON")
N_SLICE_TIMES=$(jq -r '.SliceTiming | length' "$EXAMPLE_JSON")

echo "  RepetitionTime:    $TR"
echo "  SliceTiming count: $N_SLICE_TIMES"

if [ "$TR" != "2" ] && [ "$TR" != "2.0" ]; then
  echo "ERROR: expected RepetitionTime 2 seconds, found $TR"
  exit 1
fi

if [ "$N_SLICE_TIMES" -ne 40 ]; then
  echo "ERROR: expected 40 SliceTiming values, found $N_SLICE_TIMES"
  exit 1
fi

INVALID_IDS=$(awk '!/^sub-[0-9]+$/ {n++} END {print n+0}' "$TMP_LIST")
if [ "$INVALID_IDS" -ne 0 ]; then
  echo "ERROR: subject list contains $INVALID_IDS invalid identifier(s)"
  exit 1
fi

N_TWO_RUNS=0
N_THREE_RUNS=0
N_OTHER_RUNS=0

while IFS= read -r SUBJ; do
  N_RUNS=$(find "${BIDS_ROOT}/${SUBJ}/func" -maxdepth 1 -type f \
    -name "${SUBJ}_task-EmotionRegulation_run-*_bold.nii.gz" | wc -l)
  case "$N_RUNS" in
    2) N_TWO_RUNS=$((N_TWO_RUNS + 1)) ;;
    3) N_THREE_RUNS=$((N_THREE_RUNS + 1)) ;;
    *) N_OTHER_RUNS=$((N_OTHER_RUNS + 1)) ;;
  esac
done < "$TMP_LIST"

echo "  Participants with 2 runs: $N_TWO_RUNS"
echo "  Participants with 3 runs: $N_THREE_RUNS"
echo "  Participants with another run count: $N_OTHER_RUNS"

if [ "$N_TWO_RUNS" -ne 1 ] || [ "$N_THREE_RUNS" -ne 123 ] || [ "$N_OTHER_RUNS" -ne 0 ]; then
  echo "ERROR: unexpected task run-count distribution"
  exit 1
fi

mv "$TMP_LIST" "$SUBJECT_LIST"
trap - EXIT

echo "  Subject list written: $SUBJECT_LIST"
echo "Preflight passed. No participant identifiers were printed."
