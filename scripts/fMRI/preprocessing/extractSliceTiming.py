#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extractSliceTiming.py

Extract slice acquisition order from a BIDS JSON sidecar and save as an
FSL-compatible slice order text file.

BIDS JSON files contain a "SliceTiming" field with the acquisition time (in
seconds) for each slice. FSL's slicetimer --ocustom requires a text file
listing the acquisition order as 1-based integer indices (first slice = 1,
last slice = N). This script converts between the two formats.

Note: This was run once on a representative input because
all runs in this dataset share the same slice timing parameters. The resulting
slice order file is then used for all subjects in FSL preprocessing.

Input:
    BIDS JSON sidecar for one functional run (any subject, same protocol)

Output:
    M3_slice_order.txt - One integer per line, giving FSL slice acquisition order

@author: aturnbu2
"""

import json
import math
import sys

import numpy as np

# Path to a representative BIDS JSON sidecar (same timing for all subjects)
import argparse
parser = argparse.ArgumentParser(description="Extract the validated 1-based M3 slice order")
parser.add_argument("json_path", help="Path to a representative BIDS JSON sidecar")
parser.add_argument("--output", default="/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_slice_order.txt")
args = parser.parse_args()
json_path = args.json_path

# Output path for FSL slice order file
output_path = args.output

# Load JSON sidecar
with open(json_path, "r") as f:
    metadata = json.load(f)

# Extract SliceTiming field (list of acquisition times in seconds, one per slice)
slice_times = metadata["SliceTiming"]

N_SLICES_EXPECTED = 40

if not isinstance(slice_times, list):
    sys.exit(f"ERROR: SliceTiming is not a list (got {type(slice_times).__name__})")

if len(slice_times) != N_SLICES_EXPECTED:
    sys.exit(
        f"ERROR: SliceTiming has {len(slice_times)} entries; "
        f"expected {N_SLICES_EXPECTED}"
    )

for i, t in enumerate(slice_times):
    if isinstance(t, bool):
        sys.exit(f"ERROR: SliceTiming[{i}] is a boolean, expected a number")
    if not isinstance(t, (int, float)):
        sys.exit(f"ERROR: SliceTiming[{i}] = {t!r} is not numeric")
    if not math.isfinite(t):
        sys.exit(f"ERROR: SliceTiming[{i}] = {t} is not finite")

if len(set(slice_times)) != N_SLICES_EXPECTED:
    sys.exit("ERROR: SliceTiming contains duplicate values")

# Convert times to acquisition order: argsort gives the rank of each slice by
# acquisition time; +1 converts to 1-based indices required by FSL slicetimer.
order = np.argsort(slice_times) + 1

# Validate: result must match the expected interleaved pattern
# (odd slices acquired first: 1, 3, 5, ..., 39, then even: 2, 4, 6, ..., 40)
expected_order = (
    list(range(1, N_SLICES_EXPECTED + 1, 2))   # 1, 3, 5, ..., 39
    + list(range(2, N_SLICES_EXPECTED + 1, 2))  # 2, 4, 6, ..., 40
)
if order.tolist() != expected_order:
    sys.exit(
        f"ERROR: slice acquisition order does not match expected interleaved pattern.\n"
        f"  Expected: {expected_order}\n"
        f"  Got:      {order.tolist()}"
    )
print(f"Validated: {N_SLICES_EXPECTED}-slice interleaved order (odd-first) confirmed.")

# Save as text file (one integer per line) for FSL slicetimer --ocustom
np.savetxt(output_path, order, fmt="%d")

print(f"Saved slice order file: {output_path}")
