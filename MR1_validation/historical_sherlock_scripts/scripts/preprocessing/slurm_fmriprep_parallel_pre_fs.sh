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
# TODO: Update --array=1-N to match number of subjects in recon_error_list.txt.
#
#SBATCH -J fMRIprep_MIDUS_MR1
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1/log/fmriprep_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1/log/fmriprep_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-N  # TODO: replace N with subject count

# User inputs:
bids_root_dir=/scratch/groups/fvlin/MIDUS/MR1/MR_P5_ImagingSession
out_dir=/scratch/groups/fvlin/MIDUS/MR1/derivatives
fs_subjects_dir=/scratch/groups/fvlin/MIDUS/MR1/freesurfer_subjects
subid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/MR1/recon_error_list.txt)
nthreads=4

#export TEMPLATEFLOW_HOME=$HOME/.cache/templateflow
export FS_LICENSE=/home/users/aturnbu2/freesurfer_license.txt

work_dir=/scratch/groups/fvlin/MIDUS/MR1/working

unset PYTHONPATH

#   --skull-strip-t1w skip \
set -e
singularity run --cleanenv \
    --bind ${bids_root_dir}:/data:ro \
    --bind ${out_dir}:/out \
    --bind ${work_dir}:/working \
    --bind ${FS_LICENSE}:/freesurfer_license.txt \
    --bind ${fs_subjects_dir}:/fs_subjects \
    /home/groups/vhenders/simg/fmriprep-24.1.0.simg \
    /data /out participant \
    --task-id EmotionRegulation \
    --skip-bids-validation \
    --participant-label $subid \
    --fs-license-file $FS_LICENSE \
    --fs-subjects-dir /fs_subjects \
    --nthreads $nthreads \
    --stop-on-first-crash \
    --work-dir $work_dir
