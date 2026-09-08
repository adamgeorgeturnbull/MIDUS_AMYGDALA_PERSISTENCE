#!/bin/bash
# Clean MR1 fMRIPrep reproduction.
#
# Scientific decision: use the valid BIDS SliceTiming metadata in the 371 MR1
# EmotionRegulation JSON sidecars. Do not run a separate manual STC step and do
# not pass --ignore slicetiming. fMRIPrep therefore performs its standard BIDS
# slice-timing correction.

#SBATCH --job-name=fmriprep_MR1_repro
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/fmriprep_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/log/fmriprep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=1-124%20

set -euo pipefail

REPRO_ROOT="/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828"
BIDS_ROOT="${REPRO_ROOT}/MR_P5_ImagingSession"
OUT_DIR="${REPRO_ROOT}/derivatives"
WORK_ROOT="${REPRO_ROOT}/working"
SUBJECT_LIST="${REPRO_ROOT}/MR1_task_subject_list.txt"
FS_LICENSE="/home/users/aturnbu2/freesurfer_license.txt"
CONTAINER="/home/groups/fvlin/simg/fmriprep-24.1.0.simg"
EXPECTED_SUBJECTS=124
NTHREADS=4

if [ ! -s "$SUBJECT_LIST" ]; then
  echo "ERROR: subject list missing or empty: $SUBJECT_LIST"
  exit 1
fi

N_SUBJECTS=$(wc -l < "$SUBJECT_LIST")
if [ "$N_SUBJECTS" -ne "$EXPECTED_SUBJECTS" ]; then
  echo "ERROR: expected $EXPECTED_SUBJECTS subject-list entries, found $N_SUBJECTS"
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -lt 1 ] || [ "$SLURM_ARRAY_TASK_ID" -gt "$EXPECTED_SUBJECTS" ]; then
  echo "ERROR: array index outside valid range 1-${EXPECTED_SUBJECTS}"
  exit 1
fi

SUBJ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
if ! [[ "$SUBJ" =~ ^sub-[0-9]+$ ]]; then
  echo "ERROR: invalid or empty subject identifier at array index ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

if [ ! -d "${BIDS_ROOT}/${SUBJ}" ]; then
  echo "ERROR: participant BIDS directory is missing"
  exit 1
fi

N_BOLD=$(find "${BIDS_ROOT}/${SUBJ}/func" -maxdepth 1 -type f \
  -name "${SUBJ}_task-EmotionRegulation_run-*_bold.nii.gz" | wc -l)
if [ "$N_BOLD" -ne 2 ] && [ "$N_BOLD" -ne 3 ]; then
  echo "ERROR: participant has an unexpected number of EmotionRegulation BOLD runs: $N_BOLD"
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

mkdir -p "$OUT_DIR"
SUBJ_WORK="${WORK_ROOT}/${SUBJ}"
mkdir -p "$SUBJ_WORK"

echo "[$(date)] Starting fMRIPrep 24.1.0 for one MR1 task participant"
echo "  Array index: $SLURM_ARRAY_TASK_ID of $EXPECTED_SUBJECTS"
echo "  EmotionRegulation runs: $N_BOLD"
echo "  Slice timing: BIDS metadata handled by fMRIPrep"

unset PYTHONPATH

singularity run --cleanenv \
  --bind "${BIDS_ROOT}":/data:ro \
  --bind "${OUT_DIR}":/out \
  --bind "${SUBJ_WORK}":/working \
  --bind "${FS_LICENSE}":/freesurfer_license.txt \
  "$CONTAINER" \
  /data /out participant \
  --task-id EmotionRegulation \
  --skip-bids-validation \
  --participant-label "$SUBJ" \
  --fs-license-file /freesurfer_license.txt \
  --nthreads "$NTHREADS" \
  --stop-on-first-crash \
  --work-dir /working

echo "[$(date)] fMRIPrep completed successfully for array index $SLURM_ARRAY_TASK_ID"
