#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extractSliceTiming.py

Extract slice acquisition order from a BIDS JSON sidecar and save as an
FSL-compatible slice order text file.

BIDS JSON files contain a "SliceTiming" field with the acquisition time (in
seconds) for each slice. FSL's slicetimer requires a text file listing the
acquisition order (0-based integer indices sorted by ascending time). This
script converts between the two formats.

Note: This was run once on a single representative subject (sub-10036) because
all runs in this dataset share the same slice timing parameters. The resulting
slice order file is then used for all subjects in FSL preprocessing.

Input:
    BIDS JSON sidecar for one functional run (any subject, same protocol)

Output:
    M3_slice_order.txt - One integer per line, giving FSL slice acquisition order

@author: aturnbu2
"""

import json
import numpy as np

# Path to a representative BIDS JSON sidecar (same timing for all subjects)
json_path = "/scratch/groups/fvlin/MIDUS/M3/M3_ImagingSession/sub-10036/func/sub-10036_task-EmotionRegulation_run-01_bold.json"

# Output path for FSL slice order file
output_path = "/home/users/aturnbu2/M3_slice_order.txt"

# Load JSON sidecar
with open(json_path, "r") as f:
    metadata = json.load(f)

# Extract SliceTiming field (list of acquisition times in seconds, one per slice)
slice_times = metadata["SliceTiming"]

# Convert times to acquisition order: argsort returns indices that would sort
# the times, giving a 0-based ordering from earliest to latest acquired slice
order = np.argsort(slice_times)

# Save as text file (one integer per line) for FSL slicetimer --ocustom
np.savetxt(output_path, order, fmt="%d")

print(f"Saved slice order file: {output_path}")
