#!/bin/bash
# slurm_M3_stc_parallel.sh
#
# SLURM array job: Slice timing correction for all MIDUS M3 subjects.
# Runs the same axis-swap + slicetimer pipeline as M3_slice_time_correction.sh
# in parallel across 160 subjects using SLURM job arrays.
#
# Each array task processes one subject's EmotionRegulation runs (up to 3).
# Temp files are written to a per-task scratch directory and cleaned up after.
# Corrected files overwrite originals in-place.
#
# Dependencies: FSL 6.0.7.10
# Input: M3_subject_list.txt, M3_slice_order.txt
#
#SBATCH -J stc_MIDUS
#SBATCH --output=/scratch/groups/fvlin/MIDUS/log/stc_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/log/stc_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-160

# Create temp working directory
TMPDIR="/scratch/users/aturnbu2/tmp_stc_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"

# Load FSL
ml contribs
ml poldrack

# Variables
TR=2.0
SLICE_ORDER="--ocustom=/home/users/aturnbu2/M3_slice_order.txt"
SUBJECT_LIST="/scratch/groups/fvlin/MIDUS/M3_subject_list.txt"
BASE_DIR="/scratch/groups/fvlin/MIDUS/M3/M3_ImagingSession"

# Get subject ID for this array task
SUBJ_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
FUNC_DIR="${BASE_DIR}/${SUBJ_ID}/func"

echo "[$(date)] Starting STC for subject: $SUBJ_ID"

# Exit if no func directory
if [ ! -d "$FUNC_DIR" ]; then
  echo "ERROR: Functional directory not found for $SUBJ_ID"
  exit 1
fi

# Loop through all EmotionRegulation runs
shopt -s nullglob
for INPUT_FILE in "$FUNC_DIR"/${SUBJ_ID}_task-EmotionRegulation_run-*_bold.nii.gz; do
  echo "  Processing file: $INPUT_FILE"

  OUT_FILE="$INPUT_FILE"
  BASENAME=$(basename "$INPUT_FILE" .nii.gz)

  # Define temp files with absolute paths
  TEMP_INPUT="${TMPDIR}/${BASENAME}_input.nii.gz"
  TEMP_SWAPPED="${TMPDIR}/${BASENAME}_swapped.nii.gz"
  TEMP_STC="${TMPDIR}/${BASENAME}_stc.nii.gz"
  TEMP_OUTPUT="${TMPDIR}/${BASENAME}_output.nii.gz"  

  ml fsl/6.0.7.10

  # Step 0: Reorient to standard
  fslreorient2std "$INPUT_FILE" "$TEMP_INPUT"

  # Step 1: Swap axes to make slice dim = dim3
  fslswapdim "$TEMP_INPUT" z y x "$TEMP_SWAPPED"

  # Step 2: Slice timing correction
  slicetimer -i "$TEMP_SWAPPED" -o "$TEMP_STC" -r $TR $SLICE_ORDER

  # Step 3: Swap back
  fslswapdim "$TEMP_STC" z y x "$TEMP_OUTPUT"

  # Step 4: Restore geometry
  fslcpgeom "$INPUT_FILE" "$TEMP_OUTPUT"

  ml fsl  

  # Step 5: Overwrite original file
  mv "$TEMP_OUTPUT" "$OUT_FILE"

  echo "  Done with: $INPUT_FILE"
done

# Cleanup
rm -rf "$TMPDIR"
