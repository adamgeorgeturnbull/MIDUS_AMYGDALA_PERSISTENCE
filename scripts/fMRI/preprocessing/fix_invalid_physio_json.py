#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_invalid_physio_json.py

Validate all JSON files under the M3 BIDS imaging session directory and repair
a known source-data defect in *_physio.json files where a comma is missing
between the SamplingFrequency and StartTime fields, e.g.:

    "SamplingFrequency": 1000.0
    "StartTime": -2.083,

Any JSON file that is invalid for any other reason, or that matches the defect
pattern more than once, is left untouched and reported.

Run from any directory:
    python3 scripts/fMRI/preprocessing/fix_invalid_physio_json.py
"""

import json
import os
import re
import sys
import tempfile

BIDS_ROOT = "/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_ImagingSession"

# Matches a numeric SamplingFrequency value immediately followed (on the next
# non-blank line) by "StartTime" with no intervening comma.
DEFECT_RE = re.compile(
    r'("SamplingFrequency"\s*:\s*-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)'
    r'(\s*\n\s*)'
    r'("StartTime"\s*:)',
)

def find_json_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".json"):
                yield os.path.join(dirpath, fname)

def repair(path, content):
    """Return repaired content string, or None if repair is not applicable."""
    matches = DEFECT_RE.findall(content)
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        return None
    repaired = DEFECT_RE.sub(r'\1,\2\3', content)
    return repaired

def main():
    if not os.path.isdir(BIDS_ROOT):
        sys.exit(f"ERROR: BIDS root not found: {BIDS_ROOT}")

    n_checked  = 0
    n_repaired = 0
    n_valid    = 0
    unrecognized = []

    for path in sorted(find_json_files(BIDS_ROOT)):
        n_checked += 1
        fname = os.path.basename(path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError as exc:
            print(f"UNRECOGNIZED INVALID (encoding error — {exc}): {path}")
            unrecognized.append(path)
            continue

        try:
            json.loads(content)
            n_valid += 1
            continue
        except json.JSONDecodeError:
            pass

        # File is invalid — only attempt repair on *_physio.json files
        if not fname.endswith("_physio.json"):
            print(f"UNRECOGNIZED INVALID (not a physio file): {path}")
            unrecognized.append(path)
            continue

        repaired = repair(path, content)

        if repaired is None:
            matches = DEFECT_RE.findall(content)
            if len(matches) > 1:
                print(f"UNRECOGNIZED INVALID (defect pattern found {len(matches)} times): {path}")
            else:
                print(f"UNRECOGNIZED INVALID (no known defect pattern found): {path}")
            unrecognized.append(path)
            continue

        try:
            json.loads(repaired)
        except json.JSONDecodeError as exc:
            print(f"UNRECOGNIZED INVALID (repair did not produce valid JSON: {exc}): {path}")
            unrecognized.append(path)
            continue

        # Write atomically
        dirpath = os.path.dirname(path)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dirpath, prefix=".physio_json_fix_tmp_")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(repaired)
            os.replace(tmp_path, path)
        except Exception as exc:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            print(f"ERROR: failed to write repaired file {path}: {exc}")
            unrecognized.append(path)
            continue

        print(f"REPAIRED: {path}")
        n_repaired += 1

    print(f"\nSummary")
    print(f"  JSON files checked:        {n_checked}")
    print(f"  Valid (unchanged):         {n_valid}")
    print(f"  Repaired:                  {n_repaired}")
    print(f"  Unrecognized invalid:      {len(unrecognized)}")

    if unrecognized:
        print("\nUnrecognized invalid files:")
        for p in unrecognized:
            print(f"  {p}")
        sys.exit(1)

if __name__ == "__main__":
    main()
