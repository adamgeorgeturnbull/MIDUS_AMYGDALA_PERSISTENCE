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

Input:
    BIDS-formatted EmotionRegulation BOLD NIfTI files (*_bold.nii.gz)
    Subject list file (one sub-XXXXX per line)

Output:
    Corrected NIfTI files saved in-place (same filenames)

@author: aturnbu2
"""

import glob
import os
import re
import sys
import tempfile

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, io_orientation

BIDS_ROOT    = "/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_ImagingSession"
SUBJECT_LIST = "/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_subject_list.txt"
ID_PATTERN   = re.compile(r"^sub-[0-9]+$")

# ── Load and validate subject list ────────────────────────────────────────────
if not os.path.isfile(SUBJECT_LIST):
    sys.exit(f"ERROR: subject list not found: {SUBJECT_LIST}")

with open(SUBJECT_LIST) as f:
    subjects = [line.strip() for line in f if line.strip()]

if not subjects:
    sys.exit(f"ERROR: subject list is empty: {SUBJECT_LIST}")

for subj in subjects:
    if not ID_PATTERN.match(subj):
        sys.exit(
            f"ERROR: invalid subject ID '{subj}' in subject list "
            "(expected sub-[0-9]+)"
        )

print(f"Subject list: {len(subjects)} subjects")

# ── Validate subject directories and BOLD file counts ─────────────────────────
missing_dirs = [
    subj for subj in subjects
    if not os.path.isdir(os.path.join(BIDS_ROOT, subj, "func"))
]
if missing_dirs:
    sys.exit(
        f"ERROR: func directory missing for {len(missing_dirs)} subject(s): "
        f"{missing_dirs}"
    )

no_bold = []
for subj in subjects:
    pattern = os.path.join(
        BIDS_ROOT, subj, "func",
        f"{subj}_task-EmotionRegulation_run-*_bold.nii.gz",
    )
    n = len(glob.glob(pattern))
    print(f"  {subj}: {n} EmotionRegulation BOLD run(s)")
    if n == 0:
        no_bold.append(subj)

if no_bold:
    sys.exit(
        f"ERROR: no EmotionRegulation BOLD files found for: {no_bold}"
    )

# ── Process subjects ──────────────────────────────────────────────────────────
subjects_done = 0
files_done    = 0

for subj in subjects:
    func_dir   = os.path.join(BIDS_ROOT, subj, "func")
    bold_files = sorted(glob.glob(
        os.path.join(func_dir, f"{subj}_task-EmotionRegulation_run-*_bold.nii.gz")
    ))

    for filepath in bold_files:
        fname = os.path.basename(filepath)
        print(f"Fixing: {filepath}")

        img    = nib.load(filepath)
        data   = img.get_fdata()
        affine = img.affine.copy()
        header = img.header.copy()

        # Step 1: Swap axes 0 and 2 to correct the transposed spatial dims
        data_swapped  = np.swapaxes(data, 0, 2)
        shape_swapped = data_swapped.shape

        # Step 2: Swap corresponding voxel dimensions in the header
        pixdim = header["pixdim"].copy()
        pixdim[1], pixdim[3] = pixdim[3], pixdim[1]
        header["pixdim"] = pixdim

        # Step 3: Reorient to RAS+ standard orientation
        ornt            = io_orientation(affine)
        target_ornt     = axcodes2ornt(("R", "A", "S"))
        transform       = nib.orientations.ornt_transform(ornt, target_ornt)
        data_reoriented = apply_orientation(data_swapped, transform)

        # Update affine matrix to match the reoriented data
        new_affine = affine @ inv_ornt_aff(transform, shape_swapped)

        new_img = nib.Nifti1Image(data_reoriented, new_affine)
        new_img.header["pixdim"][1:4] = pixdim[1:4]

        # Write to a temp file, verify it is readable and 4-dimensional,
        # then atomically replace the original.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=func_dir, suffix=".nii.gz", prefix=".fix_orient_tmp_"
        )
        os.close(tmp_fd)
        try:
            new_img.to_filename(tmp_path)

            check = nib.load(tmp_path)
            if check.ndim != 4:
                raise RuntimeError(
                    f"Expected 4-dimensional output, got {check.ndim}D "
                    f"(shape {check.shape})"
                )

            os.replace(tmp_path, filepath)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        print(f"  Saved:      {fname}")
        print(f"  New shape:  {data_reoriented.shape}")
        print(f"  New pixdim: {new_img.header['pixdim'][1:4]}")
        files_done += 1

    subjects_done += 1

print(f"\nDone. {subjects_done} subject(s), {files_done} file(s) corrected.")
