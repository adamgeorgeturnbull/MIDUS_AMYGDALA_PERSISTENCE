#!/bin/bash
# slurm_recon_all_parallel.sh
#
# SLURM array job: FreeSurfer recon-all for subjects that failed during
# fMRIPrep's built-in FreeSurfer processing.
#
# Runs recon-all on N4-corrected T1w images and saves output to a shared
# FreeSurfer subjects directory. The resulting surfaces are then supplied to
# fMRIPrep via --fs-subjects-dir (see slurm_fmriprep_parallel_pre_fs.sh).
#
# Subject list: recon_error_list.txt (subjects needing reprocessing)
# Dependencies: FreeSurfer 7.4.1
#
#SBATCH -J reconall_MIDUS
#SBATCH --output=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/reconall_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/M3_stc_rerun/log/reconall_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-10

set -euo pipefail

SUBJECT_LIST="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/recon_error_list.txt"
FS_LICENSE="/home/users/aturnbu2/freesurfer_license.txt"
SUBJECTS_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/freesurfer_subjects"
BASE_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_ImagingSession"

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

# ── Validate T1 image ─────────────────────────────────────────────────────────
T1_IMAGE="${BASE_DIR}/${SUBJ}/anat/${SUBJ}_T1w_N4Corrected.nii.gz"

if [ ! -f "$T1_IMAGE" ] || [ ! -s "$T1_IMAGE" ]; then
  echo "ERROR: T1 image missing or empty: $T1_IMAGE"
  exit 1
fi

# ── Validate FreeSurfer license ───────────────────────────────────────────────
if [ ! -f "$FS_LICENSE" ] || [ ! -r "$FS_LICENSE" ]; then
  echo "ERROR: FreeSurfer license missing or unreadable: $FS_LICENSE"
  exit 1
fi

# ── Create FreeSurfer output directory if needed ──────────────────────────────
mkdir -p "$SUBJECTS_DIR"

# ── Report ────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting recon-all"
echo "  Subject:    $SUBJ"
echo "  T1 input:   $T1_IMAGE"
echo "  Output dir: $SUBJECTS_DIR"

# ── Load FreeSurfer and override environment (module overwrites SUBJECTS_DIR) ──
ml biology
ml freesurfer/7.4.1

export SUBJECTS_DIR="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/freesurfer_subjects"
export FS_LICENSE="/home/users/aturnbu2/freesurfer_license.txt"

if [ "$SUBJECTS_DIR" != "/scratch/groups/fvlin/MIDUS/M3_stc_rerun/freesurfer_subjects" ]; then
  echo "ERROR: SUBJECTS_DIR is '${SUBJECTS_DIR}' after module load; expected rerun path"
  exit 1
fi
if [ "$FS_LICENSE" != "/home/users/aturnbu2/freesurfer_license.txt" ]; then
  echo "ERROR: FS_LICENSE is '${FS_LICENSE}' after module load; expected rerun path"
  exit 1
fi
echo "  SUBJECTS_DIR: $SUBJECTS_DIR"
echo "  FS_LICENSE:   $FS_LICENSE"

# ── Run recon-all ─────────────────────────────────────────────────────────────
recon-all -i "$T1_IMAGE" -subjid "$SUBJ" -all -openmp "$SLURM_CPUS_PER_TASK"
