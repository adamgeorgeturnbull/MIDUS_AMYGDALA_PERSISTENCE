#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_subject_list.py

Build and verify M3_subject_list.txt from the BIDS participants.tsv.

Steps:
  1. Read participant IDs from participants.tsv (first column, skip header).
     Validate format (sub-[0-9]+) and detect duplicates.
  2. List all sub-* subdirectories present on disk in the BIDS root.
     IDs in TSV but absent from disk are noted as behavioral-only and excluded.
     IDs on disk but absent from TSV are treated as an error.
  3. For every subject, count EmotionRegulation BOLD runs.
     Exit if any subject has zero runs. Warn if any have fewer than three
     but retain them so downstream task-completeness QC can handle them.
  4. Write the verified list atomically to OUTPUT_PATH.
  5. Print final subject count and required SLURM --array range.

Run from any directory:
    python make_subject_list.py
"""

import glob
import os
import re
import sys
import tempfile

BIDS_ROOT   = "/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_ImagingSession"
TSV_PATH    = os.path.join(BIDS_ROOT, "participants.tsv")
OUTPUT_PATH = "/scratch/groups/fvlin/MIDUS/M3_stc_rerun/M3_subject_list.txt"

ID_PATTERN = re.compile(r"^sub-[0-9]+$")

# ── Invalidate any pre-existing subject list ──────────────────────────────────
if os.path.exists(OUTPUT_PATH):
    try:
        os.remove(OUTPUT_PATH)
        print(f"Invalidated pre-existing subject list: {OUTPUT_PATH}")
    except OSError as exc:
        sys.exit(f"ERROR: could not remove pre-existing subject list: {exc}")

# ── Read TSV ──────────────────────────────────────────────────────────────────
if not os.path.isfile(TSV_PATH):
    sys.exit(f"ERROR: participants.tsv not found at {TSV_PATH}")

with open(TSV_PATH) as f:
    lines = [ln.rstrip("\n") for ln in f if ln.strip()]

if not lines:
    sys.exit("ERROR: participants.tsv is empty")

header = lines[0].split("\t")
if not header[0].strip().lower().startswith("participant"):
    sys.exit(f"ERROR: first column is '{header[0]}', expected 'participant_id'")

# Preserve order to detect duplicates; do not collapse to a set yet.
tsv_id_list = []
for lineno, line in enumerate(lines[1:], start=2):
    pid = line.split("\t")[0].strip()
    if not pid:
        continue
    if not pid.startswith("sub-"):
        pid = "sub-" + pid
    if not ID_PATTERN.match(pid):
        sys.exit(
            f"ERROR: line {lineno} of participants.tsv has invalid ID '{pid}' "
            "(expected sub-[0-9]+)"
        )
    tsv_id_list.append(pid)

seen, duplicates = {}, []
for pid in tsv_id_list:
    if pid in seen:
        duplicates.append(pid)
    seen[pid] = True

if duplicates:
    sys.exit(
        f"ERROR: duplicate participant IDs in participants.tsv: "
        f"{sorted(set(duplicates))}"
    )

tsv_ids = set(tsv_id_list)
print(f"participants.tsv : {len(tsv_ids)} unique entries")

# ── List subfolders on disk ───────────────────────────────────────────────────
if not os.path.isdir(BIDS_ROOT):
    sys.exit(f"ERROR: BIDS root not found: {BIDS_ROOT}")

disk_ids = {
    d for d in os.listdir(BIDS_ROOT)
    if d.startswith("sub-") and os.path.isdir(os.path.join(BIDS_ROOT, d))
}

invalid_disk = sorted(d for d in disk_ids if not ID_PATTERN.match(d))
if invalid_disk:
    sys.exit(f"ERROR: non-standard sub-* folders on disk: {invalid_disk}")

print(f"Subfolders on disk: {len(disk_ids)}")

# ── Cross-check TSV vs disk ───────────────────────────────────────────────────
in_tsv_not_disk = sorted(tsv_ids - disk_ids)
on_disk_not_tsv = sorted(disk_ids - tsv_ids)

# Subjects in TSV but absent from disk completed the neuroscience behavioral
# session (PANAS) but have no valid MRI scan. Expected and excluded silently.
if in_tsv_not_disk:
    print(
        f"NOTE: {len(in_tsv_not_disk)} TSV participant(s) have no BIDS directory "
        "(behavioral-only, no imaging data — excluded from subject list)."
    )

# Subjects on disk but absent from TSV are genuinely unexpected — exit.
if on_disk_not_tsv:
    print(f"ERROR: {len(on_disk_not_tsv)} sub-* director(y/ies) on disk are absent from participants.tsv:")
    for pid in on_disk_not_tsv:
        print(f"  {pid}")
    sys.exit(1)

# ── Count EmotionRegulation BOLD runs per subject ─────────────────────────────
verified = sorted(disk_ids)
no_runs  = []
few_runs = []

for pid in verified:
    pattern = os.path.join(
        BIDS_ROOT, pid, "func",
        f"{pid}_task-EmotionRegulation_run-*_bold.nii.gz",
    )
    n = len(glob.glob(pattern))
    if n == 0:
        no_runs.append(pid)
    elif n < 3:
        few_runs.append((pid, n))

if few_runs:
    print(
        f"\nWARNING: {len(few_runs)} subject(s) have fewer than 3 "
        "EmotionRegulation runs (retained — downstream QC will decide exclusion):"
    )
    for pid, n in few_runs:
        print(f"  {pid}: {n} run(s)")

if no_runs:
    print(
        f"\nNOTE: {len(no_runs)} subject(s) have a BIDS directory but no "
        "EmotionRegulation BOLD files (anatomical-only scan — excluded from subject list):"
    )
    for pid in no_runs:
        print(f"  {pid}")
    verified = [pid for pid in verified if pid not in no_runs]

# ── Write atomically ──────────────────────────────────────────────────────────
out_dir = os.path.dirname(OUTPUT_PATH)
os.makedirs(out_dir, exist_ok=True)

tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".subject_list_tmp_")
try:
    with os.fdopen(tmp_fd, "w") as f:
        for pid in verified:
            f.write(pid + "\n")
    os.replace(tmp_path, OUTPUT_PATH)
except Exception as exc:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    sys.exit(f"ERROR: failed to write subject list: {exc}")

n = len(verified)
print(f"\nWrote {n} subjects to: {OUTPUT_PATH}")
print(f"SLURM --array range:  1-{n}")
