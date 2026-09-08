#!/bin/bash
# Seed a complete FreeSurfer fsaverage template for the MR1 reproduction.
#
# Concurrent fMRIPrep array tasks can race while initializing the shared
# derivatives/sourcedata/freesurfer/fsaverage directory. This script validates
# the template, archives an incomplete copy, and seeds fsaverage once from the
# exact fMRIPrep 24.1.0 container before the array is resumed.

set -euo pipefail

REPRO_ROOT="/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828"
FS_OUTPUT_ROOT="${REPRO_ROOT}/derivatives/sourcedata/freesurfer"
FSAVERAGE="${FS_OUTPUT_ROOT}/fsaverage"
ARCHIVE="${FS_OUTPUT_ROOT}/fsaverage_incomplete_array_race_41148496"
CONTAINER="/home/groups/fvlin/simg/fmriprep-24.1.0.simg"

REQUIRED_LABELS=(
  lh.BA1_exvivo.label
  lh.BA1_exvivo.thresh.label
  rh.BA1_exvivo.label
  rh.BA1_exvivo.thresh.label
)

if [ ! -r "$CONTAINER" ]; then
  echo "ERROR: fMRIPrep container missing or unreadable: $CONTAINER"
  exit 1
fi

if ! command -v singularity >/dev/null 2>&1; then
  echo "ERROR: singularity command not found"
  exit 1
fi

mkdir -p "$FS_OUTPUT_ROOT"

missing_count() {
  local root="$1"
  local n=0
  local label
  for label in "${REQUIRED_LABELS[@]}"; do
    if [ ! -s "${root}/label/${label}" ]; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

if [ -d "$FSAVERAGE" ]; then
  N_MISSING=$(missing_count "$FSAVERAGE")
  if [ "$N_MISSING" -eq 0 ]; then
    echo "fsaverage is already complete; no changes needed"
    exit 0
  fi

  echo "Incomplete fsaverage detected: $N_MISSING required BA1 label(s) missing"
  if [ -e "$ARCHIVE" ]; then
    echo "ERROR: archive target already exists: $ARCHIVE"
    exit 1
  fi
  mv "$FSAVERAGE" "$ARCHIVE"
  echo "Archived incomplete fsaverage without deleting it"
fi

singularity exec \
  --bind "${FS_OUTPUT_ROOT}":/fs_output \
  "$CONTAINER" \
  bash -c 'cp -a "$FREESURFER_HOME/subjects/fsaverage" /fs_output/fsaverage'

N_MISSING=$(missing_count "$FSAVERAGE")
if [ "$N_MISSING" -ne 0 ]; then
  echo "ERROR: seeded fsaverage is missing $N_MISSING required BA1 label(s)"
  exit 1
fi

echo "Seeded complete fsaverage from fMRIPrep 24.1.0 container"
echo "  Required BA1 labels present: ${#REQUIRED_LABELS[@]}"
echo "  Incomplete template retained at: $ARCHIVE"
