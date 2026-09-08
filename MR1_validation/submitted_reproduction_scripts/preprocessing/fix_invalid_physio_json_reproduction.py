#!/usr/bin/env python3
"""Repair the known MR1 physio-JSON comma defect in the Scratch reproduction.

The Oak source is never modified. The default is a read-only dry run; pass
``--apply`` to atomically update only recognized ``*_physio.json`` files in the
clean MR1 reproduction directory. Output is aggregate-only and never prints
participant identifiers or participant-level paths.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path


BIDS_ROOT = Path(
    "/scratch/groups/fvlin/MIDUS/MR1_reproduction_20260828/"
    "MR_P5_ImagingSession"
)
EXPECTED_REPAIRS = 493

DEFECT_RE = re.compile(
    r'("SamplingFrequency"\s*:\s*-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)'
    r'(\s*\n\s*)'
    r'("StartTime"\s*:)',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically apply the validated repairs (default: dry run)",
    )
    return parser.parse_args()


def repaired_content(content):
    """Return repaired valid JSON text, or None for an unrecognized defect."""
    matches = list(DEFECT_RE.finditer(content))
    if len(matches) != 1:
        return None
    repaired = DEFECT_RE.sub(r"\1,\2\3", content)
    try:
        json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return repaired


def atomic_write(path, content):
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".physio_json_fix_tmp_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def validate_all_json():
    n_total = 0
    n_invalid = 0
    for path in BIDS_ROOT.rglob("*.json"):
        n_total += 1
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
        except (UnicodeDecodeError, json.JSONDecodeError):
            n_invalid += 1
    return n_total, n_invalid


def main():
    args = parse_args()

    if not BIDS_ROOT.is_dir():
        sys.exit(f"ERROR: MR1 reproduction BIDS root not found: {BIDS_ROOT}")

    n_total = 0
    n_valid = 0
    repair_plan = []
    n_unrecognized = 0

    for path in sorted(BIDS_ROOT.rglob("*.json")):
        n_total += 1
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            n_unrecognized += 1
            continue

        try:
            json.loads(content)
            n_valid += 1
            continue
        except json.JSONDecodeError:
            pass

        if not path.name.endswith("_physio.json"):
            n_unrecognized += 1
            continue

        repaired = repaired_content(content)
        if repaired is None:
            n_unrecognized += 1
            continue

        repair_plan.append((path, repaired))

    print("MR1 physio JSON validation")
    print(f"  JSON files checked:        {n_total}")
    print(f"  Valid before repair:       {n_valid}")
    print(f"  Recognized comma defects:  {len(repair_plan)}")
    print(f"  Unrecognized invalid:      {n_unrecognized}")

    if n_unrecognized:
        sys.exit("ERROR: unrecognized invalid JSON file(s); no changes made")

    if len(repair_plan) not in (0, EXPECTED_REPAIRS):
        sys.exit(
            f"ERROR: expected either 0 (already repaired) or {EXPECTED_REPAIRS} "
            f"recognized defects, found {len(repair_plan)}; no changes made"
        )

    if not repair_plan:
        print("  Status: already valid; no repairs needed")
        return

    if not args.apply:
        print("  DRY RUN: no files changed; rerun with --apply to repair")
        return

    for path, content in repair_plan:
        atomic_write(path, content)

    checked_after, invalid_after = validate_all_json()
    print(f"  Repaired atomically:       {len(repair_plan)}")
    print(f"  JSON files rechecked:      {checked_after}")
    print(f"  Invalid after repair:      {invalid_after}")

    if invalid_after:
        sys.exit("ERROR: invalid JSON remains after repair")

    print("Repair complete. Oak source was not modified.")


if __name__ == "__main__":
    main()
