#!/bin/bash
# slurm_fmriprep_parallel.sh
#
# SLURM array job: Run fMRIPrep (v24.1.0) on MIDUS M3 EmotionRegulation data.
#
# Runs fMRIPrep via Singularity for each subject in the subject list.
# Processes the EmotionRegulation task only, skips BIDS validation (non-standard
# dataset), and ignores slice timing (handled separately by STC scripts).
#
# fMRIPrep performs: motion correction, susceptibility distortion correction,
# coregistration, normalization to MNI152NLin2009cAsym, and confound estimation.
#
# Subject list: missing_list.txt (subjects still needing processing)
# Container: fmriprep-24.1.0.simg
# Output space: MNI152NLin2009cAsym
#
#SBATCH -J fMRIprep_MIDUS
#SBATCH --output=/scratch/groups/fvlin/MIDUS/log/fmriprep_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/log/fmriprep_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-2

# User inputs:
bids_root_dir=/scratch/groups/fvlin/MIDUS/M3/M3_ImagingSession
out_dir=/scratch/groups/fvlin/MIDUS/derivatives
subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/missing_list.txt)
nthreads=4

#export TEMPLATEFLOW_HOME=$HOME/.cache/templateflow
export FS_LICENSE=/home/users/aturnbu2/freesurfer_license.txt

work_dir=/scratch/groups/fvlin/MIDUS/working

unset PYTHONPATH

#   --skull-strip-t1w skip \
set -e
singularity run --cleanenv \
    --bind ${bids_root_dir}:/data:ro \
    --bind ${out_dir}:/out \
    --bind ${work_dir}:/working \
    --bind ${FS_LICENSE}:/freesurfer_license.txt \
    /home/groups/vhenders/simg/fmriprep-24.1.0.simg \
    /data /out participant \
    --task-id EmotionRegulation \
    --skip-bids-validation \
    --ignore slicetiming \
    --participant-label $subid \
    --fs-license-file $FS_LICENSE \
    --nthreads $nthreads \
    --stop-on-first-crash \
    --work-dir $work_dir
