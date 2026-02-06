#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fixOrientation.py

Fix NIfTI orientation for MIDUS M3 EmotionRegulation BOLD data.

Some MIDUS M3 scans were acquired or converted with swapped spatial axes
(axes 0 and 2 transposed), resulting in incorrect orientation metadata. This
script corrects the issue by:

1. Swapping data axes 0 and 2 (transposes the first and third spatial dims)
2. Swapping the corresponding voxel dimensions (pixdim[1] <-> pixdim[3])
3. Reorienting the image to standard RAS+ orientation using nibabel

The corrected images are saved in-place (overwriting the originals). This must
be run BEFORE fMRIPrep or any other preprocessing.

WARNING: subjects = subjects[:1] limits this to a single subject for testing.
         Remove or adjust this line to process all subjects.

Input:
    BIDS-formatted EmotionRegulation BOLD NIfTI files (*_bold.nii.gz)
    Subject list file (one sub-XXXXX per line)

Output:
    Corrected NIfTI files saved in-place (same filenames)

@author: aturnbu2
"""

import os
import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, inv_ornt_aff, apply_orientation

# BIDS root containing subject directories
bids_root = "/scratch/groups/fvlin/MIDUS/M3/M3_ImagingSession"
subject_list_file = "/scratch/groups/fvlin/MIDUS/M3_subject_list.txt"

# Load subject list
with open(subject_list_file) as f:
    subjects = [line.strip() for line in f if line.strip()]

# --- LIMIT TO ONE SUBJECT FOR TESTING ---
subjects = subjects[:1]

for subj in subjects:
    func_dir = os.path.join(bids_root, subj, "func")
    if not os.path.isdir(func_dir):
        print(f"No func dir for {subj}")
        continue

    for fname in os.listdir(func_dir):
        if fname.endswith("_bold.nii.gz") and "EmotionRegulation" in fname:
            filepath = os.path.join(func_dir, fname)
            print(f"Fixing: {filepath}")

            try:
                img = nib.load(filepath)
                data = img.get_fdata()
                affine = img.affine.copy()
                header = img.header.copy()

                # Step 1: Swap axes 0 and 2 to correct the transposed spatial dims
                data_swapped = np.swapaxes(data, 0, 2)
                shape_swapped = data_swapped.shape

                # Step 2: Swap corresponding voxel dimensions in the header
                pixdim = header['pixdim'].copy()
                pixdim[1], pixdim[3] = pixdim[3], pixdim[1]
                header['pixdim'] = pixdim

                # Step 3: Reorient to RAS+ standard orientation
                ornt = io_orientation(affine)
                target_ornt = axcodes2ornt(('R', 'A', 'S'))
                transform = nib.orientations.ornt_transform(ornt, target_ornt)
                data_reoriented = apply_orientation(data_swapped, transform)

                # Update affine matrix to match the reoriented data
                new_affine = affine @ inv_ornt_aff(transform, shape_swapped)

                # Save corrected image (overwrites original)
                new_img = nib.Nifti1Image(data_reoriented, new_affine)
                new_img.header['pixdim'][1:4] = pixdim[1:4]
                new_img.to_filename(filepath)

                print(f"  Saved: {fname}")
                print(f"  New shape: {data_reoriented.shape}")
                print(f"  New pixdim: {new_img.header['pixdim'][1:4]}")

            except Exception as e:
                print(f"Error with {fname}: {e}")

print("Done.")









