#!/bin/bash
# slurm_fmriprep_parallel_pre_fs.sh
#
# SLURM array job: fMRIPrep with pre-computed FreeSurfer surfaces.
#
# Variant of slurm_fmriprep_parallel.sh for subjects where fMRIPrep's built-in
# FreeSurfer recon-all failed. These subjects had recon-all run separately
# (see slurm_recon_all_parallel.sh), and the completed FreeSurfer output is
# passed to fMRIPrep via --fs-subjects-dir to skip the surface reconstruction.
#
# Subject list: recon_error_list.txt (subjects with pre-run FreeSurfer)
# Container: fmriprep-24.1.0.simg
#
#SBATCH -J fMRIprep_MIDUS
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/fmriprep_pre_fs_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/fmriprep_pre_fs_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-10

set -euo pipefail

SUBJECT_LIST="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/recon_error_list.txt"
BIDS_ROOT="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_ImagingSession"
OUT_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/derivatives"
WORK_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/working_pre_fs"
FS_SUBJECTS_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/freesurfer_subjects"
FS_LICENSE="/home/users/aturnbu2/freesurfer_license.txt"
CONTAINER="/home/groups/fvlin/simg/fmriprep-24.1.0.simg"
NTHREADS=4

# ── Validate subject list ─────────────────────────────────────────────────────
if [ ! -f "$SUBJECT_LIST" ] || [ ! -s "$SUBJECT_LIST" ]; then
  echo "ERROR: subject list missing or empty: $SUBJECT_LIST"
  exit 1
fi

N_SUBJECTS=$(wc -l < "$SUBJECT_LIST")
if [ "$N_SUBJECTS" -ne 10 ]; then
  echo "ERROR: recon_error_list.txt contains ${N_SUBJECTS} entries; expected exactly 10"
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -lt 1 ] || [ "$SLURM_ARRAY_TASK_ID" -gt 10 ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is outside the valid range 1-10"
  exit 1
fi

SUBJ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")

if [ -z "$SUBJ" ]; then
  echo "ERROR: empty subject ID at line ${SLURM_ARRAY_TASK_ID} of $SUBJECT_LIST"
  exit 1
fi

if ! [[ "$SUBJ" =~ ^sub-[0-9]+$ ]]; then
  echo "ERROR: subject ID '${SUBJ}' does not match sub-[0-9]+"
  exit 1
fi

# ── Validate BIDS directory ───────────────────────────────────────────────────
if [ ! -d "${BIDS_ROOT}/${SUBJ}" ]; then
  echo "ERROR: BIDS directory not found for ${SUBJ}: ${BIDS_ROOT}/${SUBJ}"
  exit 1
fi

# ── Validate FreeSurfer license and container ─────────────────────────────────
if [ ! -f "$FS_LICENSE" ] || [ ! -r "$FS_LICENSE" ]; then
  echo "ERROR: FreeSurfer license missing or unreadable: $FS_LICENSE"
  exit 1
fi

if [ ! -f "$CONTAINER" ] || [ ! -r "$CONTAINER" ]; then
  echo "ERROR: Singularity container missing or unreadable: $CONTAINER"
  exit 1
fi

# ── Validate completed FreeSurfer reconstruction ──────────────────────────────
FS_SUBJ_DIR="${FS_SUBJECTS_DIR}/${SUBJ}"

if [ ! -d "$FS_SUBJ_DIR" ]; then
  echo "ERROR: FreeSurfer subject directory not found: $FS_SUBJ_DIR"
  exit 1
fi

RECON_LOG="${FS_SUBJ_DIR}/scripts/recon-all.log"
if [ ! -f "$RECON_LOG" ]; then
  echo "ERROR: recon-all.log not found: $RECON_LOG"
  exit 1
fi

if ! grep -q "finished without error" "$RECON_LOG"; then
  echo "ERROR: recon-all.log does not contain 'finished without error': $RECON_LOG"
  exit 1
fi

for REQUIRED_FILE in \
    "${FS_SUBJ_DIR}/mri/aseg.mgz" \
    "${FS_SUBJ_DIR}/surf/lh.white" \
    "${FS_SUBJ_DIR}/surf/rh.white"; do
  if [ ! -f "$REQUIRED_FILE" ] || [ ! -s "$REQUIRED_FILE" ]; then
    echo "ERROR: required FreeSurfer output missing or empty: $REQUIRED_FILE"
    exit 1
  fi
done

# ── Create output and working directories if needed ───────────────────────────
mkdir -p "$OUT_DIR"
mkdir -p "$WORK_DIR"

# ── Report ────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting fMRIPrep (pre-computed FreeSurfer) for: $SUBJ"
echo "  BIDS root:        $BIDS_ROOT"
echo "  Output dir:       $OUT_DIR"
echo "  Working dir:      $WORK_DIR"
echo "  FreeSurfer dir:   $FS_SUBJ_DIR"
echo "  License:          $FS_LICENSE"
echo "  Container:        $CONTAINER"

# ── Run fMRIPrep ──────────────────────────────────────────────────────────────
export FS_LICENSE

unset PYTHONPATH

singularity run --cleanenv \
    --bind "${BIDS_ROOT}":/data:ro \
    --bind "${OUT_DIR}":/out \
    --bind "${WORK_DIR}":/working \
    --bind "${FS_LICENSE}":/freesurfer_license.txt \
    --bind "${FS_SUBJECTS_DIR}":/fs_subjects \
    "$CONTAINER" \
    /data /out participant \
    --task-id EmotionRegulation \
    --skip-bids-validation \
    --ignore slicetiming \
    --participant-label "$SUBJ" \
    --fs-license-file /freesurfer_license.txt \
    --fs-subjects-dir /fs_subjects \
    --nthreads $NTHREADS \
    --stop-on-first-crash \
    --work-dir /working
