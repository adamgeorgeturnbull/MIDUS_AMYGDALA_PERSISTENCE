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
# TODO: Update --array=1-N to match number of subjects in recon_error_list.txt.
#
#SBATCH -J reconall_MIDUS_MR1
#SBATCH --output=/scratch/groups/fvlin/MIDUS/MR1/log/reconall_%A_%a.log
#SBATCH --error=/scratch/groups/fvlin/MIDUS/MR1/log/reconall_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-user=aturnbu2@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-N  # TODO: replace N with subject count

# Load FreeSurfer
ml biology
ml freesurfer/7.4.1

# Set FreeSurfer SUBJECTS_DIR
export SUBJECTS_DIR=/scratch/groups/fvlin/MIDUS/MR1/freesurfer_subjects

# Get the subject ID for this task
SUBJ=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/groups/fvlin/MIDUS/MR1/recon_error_list.txt)

# Path to N4-corrected T1w image
T1_IMAGE=/scratch/groups/fvlin/MIDUS/MR1/MR_P5_ImagingSession/${SUBJ}/anat/${SUBJ}_T1w_N4Corrected.nii.gz

# Run recon-all
recon-all -i $T1_IMAGE -subjid $SUBJ -all -openmp $SLURM_CPUS_PER_TASK
