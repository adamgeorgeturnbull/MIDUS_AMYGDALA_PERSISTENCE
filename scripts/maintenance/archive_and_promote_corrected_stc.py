#!/usr/bin/env python3
"""
archive_and_promote_corrected_stc.py

Archive old bad-STC fMRI products and promote corrected-STC summary files
into their canonical data/fMRI/ locations.

Dry-run (default — safe, no changes):
    python3 scripts/maintenance/archive_and_promote_corrected_stc.py

Apply (archive old files, copy corrected files into canonical locations):
    python3 scripts/maintenance/archive_and_promote_corrected_stc.py --apply

Optional arguments:
    --project-root PATH   Repository root (default: two levels above this script)
    --import-root  PATH   Staging dir   (default: <project-root>/data/fMRI/corrected_stc_import)

This script never deletes files.  In --apply mode it copies old canonical files
into the archive directory before copying corrected staged files into their
canonical locations.  Staged copies are left intact.
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ARCHIVE_REASON = (
    "Invalid zero-based FSL custom slice-order file in prior preprocessing."
)

ARCHIVE_DIR_NAME  = "bad_stc_0based_slice_order"
MANIFEST_FILENAME = "MANIFEST.csv"
README_FILENAME   = "README.txt"

README_TEXT = """\
BAD-STC ARCHIVE — DO NOT USE FOR ANALYSIS
==========================================
Files in this directory were generated using an incorrect (zero-based) FSL
custom slice-order file during preprocessing of the MIDUS Amygdala Persistence
fMRI dataset.  The error caused all slice-timing correction steps to use
one-based indices shifted down by one position, producing systematically
incorrect timing corrections throughout the preprocessing pipeline.

These files are preserved here for audit purposes only.
They must NOT be used for any analysis.

Corrected files (derived from the M3_stc_rerun pipeline) are in the
parent data/fMRI/ directory.

See MANIFEST.csv in this directory for the full archival record.
"""

# Files and directories to archive, relative to data/fMRI/.
# Directories have no trailing slash.  Order is for display only.
ARCHIVE_ALLOWLIST = [
    "fd_summary.csv",
    "all_subjects_betaSeries_LSS_all_conditions_M2ID.csv",
    "all_subjects_betaSeries_all_conditions_M2ID.csv",
    "betaSeries_neg_vs_neu.csv",
    "betaSeries_neg_vs_neu_threat_safety.csv",
    "negative_persistence_cross_run.csv",
    "positive_persistence_cross_run.csv",
    "negative_persistence_concat.csv",
    "vmPFC_persistence_wide.csv",
    "vmPFC_persistence_wide_aCompCor.csv",
    "all_subjects_roi_activations.csv",
    "all_subjects_roi_activations_aCompCor.csv",
    "condition_fd_wide.csv",
    "fmri_qc_processed.csv",
    "aCompCor_persistence_summary",   # directory
    "GroupSeedFC_output",             # directory
    # group_*.nii.gz and group_*.png at data/fMRI/ root are discovered at runtime
]

# (staged path relative to import_root, canonical path relative to data/fMRI/)
PROMOTIONS = [
    (
        "fd_qc/fd_summary.csv",
        "fd_summary.csv",
    ),
    (
        "BetaSeries_LSS_output/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv",
        "all_subjects_betaSeries_LSS_all_conditions_M2ID.csv",
    ),
    (
        "voxelwise_betas_summary/results_summary_neg_image_vs_neg_face.csv",
        "negative_persistence_cross_run.csv",
    ),
    (
        "voxelwise_betas_summary/results_summary_pos_image_vs_pos_face.csv",
        "positive_persistence_cross_run.csv",
    ),
    (
        "vmPFC_persistence_summary/results_wide_vmPFC_persistence.csv",
        "vmPFC_persistence_wide.csv",
    ),
    (
        "ROI_activations_output/all_subjects_roi_activations.csv",
        "all_subjects_roi_activations.csv",
    ),
    (
        "condition_fd_wide.csv",
        "condition_fd_wide.csv",
    ),
]

# Canonical-relative paths that receive corrected replacements via PROMOTIONS.
# These are archived by copy so the original stays available until promotion overwrites it.
# All other allowlisted items are archived by move (os.replace / os.rename) so they no
# longer remain under their old canonical paths after archiving.
PROMOTION_CANONICAL_RELS = frozenset(canonical_rel for _, canonical_rel in PROMOTIONS)

# Validation spec per staged file.
# unique_key: tuple of column names whose combination must be unique, or None.
# required_cols: columns that must be present (in addition to id_col).
# run_col: name of column whose integer value must be in 1–3.
# hemisphere_check: if True, each subject must have exactly one L, R, and BI row.
VALIDATIONS = [
    {
        "staged_rel":    "fd_qc/fd_summary.csv",
        "label":         "FD summary",
        "expected_rows": 469,
        "id_col":        "subject",
        "unique_key":    ("subject", "run"),
        "required_cols": ["subject", "run", "mean_fd"],
        "run_col":       "run",
    },
    {
        "staged_rel":    "BetaSeries_LSS_output/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv",
        "label":         "LSS combined",
        "expected_rows": 156,
        "id_col":        "M2ID",
        "unique_key":    ("M2ID",),
    },
    {
        "staged_rel":    "voxelwise_betas_summary/results_summary_neg_image_vs_neg_face.csv",
        "label":         "negative persistence",
        "expected_rows": 468,
        "id_col":        "subject",
        "unique_key":    ("subject", "hemisphere"),
        "required_cols": ["subject", "hemisphere", "mean_r", "n_pairs"],
        "hemisphere_check": True,
    },
    {
        "staged_rel":    "voxelwise_betas_summary/results_summary_pos_image_vs_pos_face.csv",
        "label":         "positive persistence",
        "expected_rows": 468,
        "id_col":        "subject",
        "unique_key":    ("subject", "hemisphere"),
        "required_cols": ["subject", "hemisphere", "mean_r", "n_pairs"],
        "hemisphere_check": True,
    },
    {
        "staged_rel":    "vmPFC_persistence_summary/results_wide_vmPFC_persistence.csv",
        "label":         "vmPFC wide persistence",
        "expected_rows": 151,
        "id_col":        "subject",
        "unique_key":    ("subject",),
    },
    {
        "staged_rel":    "ROI_activations_output/all_subjects_roi_activations.csv",
        "label":         "ROI activations",
        "expected_rows": 148,
        "id_col":        "M2ID",
        "unique_key":    ("M2ID",),
    },
    {
        "staged_rel":    "condition_fd_wide.csv",
        "label":         "condition FD",
        "expected_rows": 156,
        "id_col":        "M2ID",
        "unique_key":    ("M2ID",),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_content_manifest(dirpath):
    """Return {relative_path_str: sha256} for every file under dirpath."""
    manifest = {}
    for p in sorted(dirpath.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(dirpath))
            manifest[rel] = sha256_file(p)
    return manifest


def atomic_copy_verified(src, dst, src_cksum=None):
    """
    Copy src to dst atomically via a temp sibling file, verifying SHA-256.
    Returns the verified checksum.  Removes the temp file on any failure.
    """
    if src_cksum is None:
        src_cksum = sha256_file(src)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=dst.parent, prefix=dst.name + "._", suffix=".tmp"
    )
    tmp_path = Path(tmp_path_str)
    try:
        os.close(tmp_fd)
        shutil.copy2(str(src), str(tmp_path))
        tmp_cksum = sha256_file(tmp_path)
        if tmp_cksum != src_cksum:
            raise ValueError(
                f"SHA-256 mismatch for {src.name} after copy to temp: "
                f"src={src_cksum[:16]}  tmp={tmp_cksum[:16]}"
            )
        os.replace(str(tmp_path), str(dst))
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return src_cksum


def atomic_copytree(src, dst):
    """
    Copy directory src to dst atomically via a temp sibling directory.
    Verifies SHA-256 of every file before the atomic rename.
    Cleans up any prior interrupted temp directory (*.._installing).
    """
    tmp_dir = dst.parent / (dst.name + "._installing")
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir))
    shutil.copytree(str(src), str(tmp_dir))
    src_manifest = dir_content_manifest(src)
    tmp_manifest  = dir_content_manifest(tmp_dir)
    if src_manifest != tmp_manifest:
        shutil.rmtree(str(tmp_dir))
        raise ValueError(
            f"Content mismatch after copying directory {src.name}: "
            f"{len(src_manifest)} src file(s) vs {len(tmp_manifest)} copied file(s)"
        )
    os.rename(str(tmp_dir), str(dst))


def read_csv(path):
    """Return (fieldnames, rows) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)          # fieldnames populated as a side effect
        return list(reader.fieldnames or []), rows


def validate_staged_file(path, spec):
    """Return list of error strings.  Empty list means the file passes."""
    errors = []

    try:
        fieldnames, rows = read_csv(path)
    except Exception as exc:
        return [f"cannot read CSV: {exc}"]

    n_rows     = len(rows)
    expected   = spec["expected_rows"]
    id_col     = spec["id_col"]
    unique_key = spec.get("unique_key")
    req_cols   = spec.get("required_cols", [])
    run_col    = spec.get("run_col")
    hemi_check = spec.get("hemisphere_check", False)

    if n_rows != expected:
        errors.append(f"expected {expected} data rows, found {n_rows}")

    # Required columns (always includes id_col)
    all_required = [id_col] + [c for c in req_cols if c != id_col]
    missing_req  = [c for c in all_required if c not in fieldnames]
    if missing_req:
        errors.append(
            f"required column(s) not found: {missing_req}  (columns: {fieldnames})"
        )
        return errors  # can't continue without required columns

    # Blank identifier values — universal check on id_col for every staged file
    blanks = [
        i + 1 for i, row in enumerate(rows)
        if not str(row.get(id_col, "")).strip()
    ]
    if blanks:
        sample = blanks[:5]
        suffix = f" (+{len(blanks) - 5} more)" if len(blanks) > 5 else ""
        errors.append(f"blank '{id_col}' in data row(s): {sample}{suffix}")

    # Run column: values must be integers in 1–3
    if run_col:
        bad_runs = []
        for i, row in enumerate(rows):
            val     = str(row.get(run_col, "")).strip()
            stripped = val[4:] if val.startswith("run-") else val
            try:
                n = int(stripped)
                if not (1 <= n <= 3):
                    bad_runs.append((i + 1, val))
            except ValueError:
                bad_runs.append((i + 1, val))
        if bad_runs:
            sample = bad_runs[:5]
            errors.append(f"run values outside 1–3 in row(s): {sample}")

    # Unique key check
    if unique_key:
        missing_cols = [k for k in unique_key if k not in fieldnames]
        if missing_cols:
            errors.append(
                f"unique-key column(s) not found: {missing_cols}  "
                f"(columns: {fieldnames})"
            )
        else:
            seen = set()
            dups = []
            for row in rows:
                key = tuple(row.get(k, "") for k in unique_key)
                if key in seen:
                    dups.append(key)
                seen.add(key)
            if dups:
                sample = dups[:5]
                suffix = f" (+{len(dups) - 5} more)" if len(dups) > 5 else ""
                errors.append(
                    f"duplicate key {unique_key}: {sample}{suffix}"
                )

    # Hemisphere check: each subject must have exactly one L, R, and BI row
    if hemi_check:
        EXPECTED_HEMIS = {"L", "R", "BI"}
        hemi_by_subj   = {}
        for row in rows:
            subj = str(row.get("subject", "")).strip()
            hemi = str(row.get("hemisphere", "")).strip()
            if subj:
                hemi_by_subj.setdefault(subj, []).append(hemi)
        bad_subjects = []
        for subj, hemis in hemi_by_subj.items():
            if set(hemis) != EXPECTED_HEMIS or len(hemis) != 3:
                bad_subjects.append((subj, sorted(hemis)))
        if bad_subjects:
            sample = bad_subjects[:3]
            suffix = (
                f" (+{len(bad_subjects) - 3} more)" if len(bad_subjects) > 3 else ""
            )
            errors.append(
                f"subjects with wrong hemisphere set (expected L, R, BI): "
                f"{sample}{suffix}"
            )

    return errors


def append_manifest(manifest_path, entries):
    """Append rows to the manifest CSV, writing a header if the file is new."""
    fieldnames = ["timestamp", "action", "source", "destination", "checksum", "reason"]
    need_header = (
        not manifest_path.exists() or manifest_path.stat().st_size == 0
    )
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerows(entries)


# ─────────────────────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────────────────────

def compute_already_promoted(import_root, fmri_dir):
    """
    Return the set of canonical Paths whose content already matches their
    staged source.  Used to skip both archiving and re-promoting such files.
    """
    promoted = set()
    for staged_rel, canonical_rel in PROMOTIONS:
        staged    = import_root / staged_rel
        canonical = fmri_dir   / canonical_rel
        if canonical.is_file() and staged.is_file():
            if sha256_file(canonical) == sha256_file(staged):
                promoted.add(canonical)
    return promoted


def collect_archive_targets(fmri_dir, archive_dir, import_root):
    """
    Build the list of archive target dicts from the static allowlist plus the
    runtime group_* glob.  Never includes the archive dir or import dir.
    """
    protected = {archive_dir, import_root, fmri_dir / "corrected_stc_import"}

    rels_seen = set()
    rels      = []
    for rel in ARCHIVE_ALLOWLIST:
        rels_seen.add(rel)
        rels.append(rel)

    # Discover root-level group_* files dynamically
    for pattern in ("group_*.nii.gz", "group_*.png"):
        for p in sorted(fmri_dir.glob(pattern)):
            if p.name not in rels_seen:
                rels_seen.add(p.name)
                rels.append(p.name)

    targets = []
    for rel in rels:
        src = fmri_dir / rel
        if src in protected or src == archive_dir:
            continue
        targets.append({
            "rel":    rel,
            "src":    src,
            "exists": src.exists(),
            "is_dir": src.is_dir() if src.exists() else False,
        })
    return targets


def plan_archive(archive_dir, targets, already_promoted):
    """
    Determine the archive action for each target.
    Returns (actions, errors).  Errors block --apply.
    """
    actions = []
    errors  = []

    for t in targets:
        rel = t["rel"]
        src = t["src"]
        dst = archive_dir / rel

        if not t["exists"]:
            # An archive-only item missing from the canonical path may have been
            # relocated to the archive on a prior run.  Report it as already archived.
            if rel not in PROMOTION_CANONICAL_RELS and dst.exists():
                if dst.is_dir():
                    actions.append({
                        "rel": rel, "src": src, "dst": dst,
                        "action": "already_archived_dir", "is_dir": True,
                        "note": "already moved to archive on prior run",
                    })
                else:
                    actions.append({
                        "rel": rel, "src": src, "dst": dst,
                        "action": "already_archived", "is_dir": False,
                        "checksum": sha256_file(dst),
                        "note": "already moved to archive on prior run",
                    })
            else:
                actions.append({
                    "rel": rel, "src": src, "dst": dst,
                    "action": "skip_missing", "is_dir": False,
                    "note": "not present in data/fMRI",
                })
            continue

        if t["is_dir"]:
            tmp_dir = archive_dir / (rel + "._installing")
            if dst.exists():
                # Compare directory contents rather than trusting existence alone
                try:
                    src_manifest = dir_content_manifest(src)
                    dst_manifest = dir_content_manifest(dst)
                    content_match = (src_manifest == dst_manifest)
                except Exception as exc:
                    errors.append(
                        f"Cannot compare directory '{rel}/': {exc}"
                    )
                    actions.append({
                        "rel": rel, "src": src, "dst": dst,
                        "action": "conflict_dir", "is_dir": True,
                        "note": "cannot verify archive directory contents",
                    })
                    continue
                if content_match:
                    actions.append({
                        "rel": rel, "src": src, "dst": dst,
                        "action": "already_archived_dir", "is_dir": True,
                        "note": "archive directory exists and contents match — skipped",
                    })
                else:
                    errors.append(
                        f"CONFLICT: archive directory '{rel}/' exists but its contents "
                        f"differ from the source.  Manual inspection required.\n"
                        f"  Archive: {dst}\n"
                        f"  Source : {src}"
                    )
                    actions.append({
                        "rel": rel, "src": src, "dst": dst,
                        "action": "conflict_dir", "is_dir": True,
                        "note": "archive directory contents differ from source",
                    })
            else:
                if rel in PROMOTION_CANONICAL_RELS:
                    note = "copy directory tree to archive via atomic rename"
                    if tmp_dir.exists():
                        note += " (prior ._installing dir will be cleaned up)"
                    actions.append({
                        "rel": rel, "src": src, "dst": dst,
                        "action": "archive_dir", "is_dir": True,
                        "note": note,
                    })
                else:
                    actions.append({
                        "rel": rel, "src": src, "dst": dst,
                        "action": "archive_dir_move", "is_dir": True,
                        "note": "move directory to archive (os.rename) — preserves data, removes from canonical path",
                    })
            continue

        # File: if canonical already holds the corrected version, do not archive it
        if src in already_promoted:
            actions.append({
                "rel": rel, "src": src, "dst": dst,
                "action": "skip_already_promoted", "is_dir": False,
                "note": "canonical already matches corrected staged file",
            })
            continue

        src_cksum = sha256_file(src)

        if dst.exists():
            dst_cksum = sha256_file(dst)
            if dst_cksum == src_cksum:
                actions.append({
                    "rel": rel, "src": src, "dst": dst,
                    "action": "already_archived", "is_dir": False,
                    "checksum": src_cksum,
                    "note": "archive copy already matches canonical — idempotent",
                })
            else:
                errors.append(
                    f"CONFLICT: archive already has '{rel}' with a different checksum.\n"
                    f"  Archive  ({dst_cksum[:16]}): {dst}\n"
                    f"  Current  ({src_cksum[:16]}): {src}\n"
                    f"  Refusing to overwrite.  Manual inspection required."
                )
                actions.append({
                    "rel": rel, "src": src, "dst": dst,
                    "action": "conflict", "is_dir": False,
                    "checksum": src_cksum,
                    "note": "checksum mismatch with existing archive copy",
                })
        else:
            if rel in PROMOTION_CANONICAL_RELS:
                actions.append({
                    "rel": rel, "src": src, "dst": dst,
                    "action": "archive", "is_dir": False,
                    "checksum": src_cksum,
                    "note": "copy to archive",
                })
            else:
                actions.append({
                    "rel": rel, "src": src, "dst": dst,
                    "action": "archive_move", "is_dir": False,
                    "checksum": src_cksum,
                    "note": "move to archive (os.replace) — preserves data, removes from canonical path",
                })

    return actions, errors


def plan_promotions(import_root, fmri_dir, already_promoted):
    """
    Determine the promotion action for each staged file.
    Returns a list of action dicts.
    Assumes all staged files have already been validated to exist.
    """
    actions = []

    for staged_rel, canonical_rel in PROMOTIONS:
        staged    = import_root / staged_rel
        canonical = fmri_dir   / canonical_rel

        staged_cksum = sha256_file(staged)

        if canonical in already_promoted:
            actions.append({
                "staged_rel": staged_rel, "canon_rel": canonical_rel,
                "staged": staged, "canonical": canonical,
                "action": "already_promoted",
                "checksum": staged_cksum,
                "note": "canonical already matches staged source",
            })
        else:
            note = "overwrites old version" if canonical.exists() else "new file"
            actions.append({
                "staged_rel": staged_rel, "canon_rel": canonical_rel,
                "staged": staged, "canonical": canonical,
                "action": "promote",
                "checksum": staged_cksum,
                "note": note,
            })

    return actions


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_plan(archive_actions, promote_actions, archive_dir, dry_run):
    mode = "DRY-RUN" if dry_run else "APPLY"
    print(f"\n{'='*62}")
    print(f"  {mode} — archive_and_promote_corrected_stc.py")
    print(f"{'='*62}")

    print("\n── ARCHIVE PLAN ───────────────────────────────────────────")
    for a in archive_actions:
        act = a["action"]
        rel = a["rel"]
        if act == "skip_missing":
            print(f"  [NOT PRESENT]       {rel}")
        elif act == "skip_already_promoted":
            print(f"  [SKIP — PROMOTED]   {rel}  (already the corrected version)")
        elif act == "already_archived":
            print(f"  [ALREADY ARCHIVED]  {rel}  ({a['checksum'][:12]}...)")
        elif act == "already_archived_dir":
            print(f"  [ALREADY ARCHIVED]  {rel}/  (directory, contents verified)")
        elif act == "conflict":
            print(f"  [CONFLICT]          {rel}  ← checksum mismatch in existing archive")
        elif act == "conflict_dir":
            print(f"  [CONFLICT DIR]      {rel}/  ← contents differ from source")
        elif act == "archive":
            print(f"  [ARCHIVE]           {rel}  ({a['checksum'][:12]}...)")
        elif act == "archive_dir":
            print(f"  [ARCHIVE DIR]       {rel}/")
        elif act == "archive_move":
            print(f"  [MOVE TO ARCHIVE]   {rel}  ({a['checksum'][:12]}...)  ← removed from canonical")
        elif act == "archive_dir_move":
            print(f"  [MOVE DIR TO ARCH]  {rel}/  ← removed from canonical")

    print(f"  → {archive_dir}")

    print("\n── PROMOTION PLAN ─────────────────────────────────────────")
    for a in promote_actions:
        act = a["action"]
        if act == "already_promoted":
            print(f"  [ALREADY PROMOTED]  {a['canon_rel']}")
        else:
            print(f"  [PROMOTE]           {a['staged_rel']}")
            print(f"                    → {a['canon_rel']}  "
                  f"({a['checksum'][:12]}...)  [{a['note']}]")

    if dry_run:
        print(
            "\n  Dry-run complete — no changes made.\n"
            "  Re-run with --apply to execute the actions above."
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

def apply_archive(archive_actions, archive_dir, manifest_path):
    ts      = datetime.now(timezone.utc).isoformat()
    entries = []

    for a in archive_actions:
        if a["action"] not in ("archive", "archive_dir", "archive_move", "archive_dir_move"):
            continue   # skip, already_archived, already_promoted, conflict (blocked earlier)

        src, dst = a["src"], a["dst"]
        dst.parent.mkdir(parents=True, exist_ok=True)

        if a["action"] == "archive_dir":
            atomic_copytree(src, dst)
            print(f"  Archived dir : {src.name}/  →  {dst}")
            entries.append({
                "timestamp":   ts,
                "action":      "archive_dir",
                "source":      str(src),
                "destination": str(dst),
                "checksum":    "",
                "reason":      ARCHIVE_REASON,
            })
        elif a["action"] == "archive_dir_move":
            os.rename(str(src), str(dst))
            print(f"  Preserved dir: {src.name}/  →  {dst}  (moved)")
            entries.append({
                "timestamp":   ts,
                "action":      "archive_dir_move",
                "source":      str(src),
                "destination": str(dst),
                "checksum":    "",
                "reason":      ARCHIVE_REASON,
            })
        elif a["action"] == "archive":
            atomic_copy_verified(src, dst, src_cksum=a["checksum"])
            print(f"  Archived     : {src.name}  →  archive  ({a['checksum'][:12]}...)")
            entries.append({
                "timestamp":   ts,
                "action":      "archive",
                "source":      str(src),
                "destination": str(dst),
                "checksum":    a["checksum"],
                "reason":      ARCHIVE_REASON,
            })
        elif a["action"] == "archive_move":
            os.replace(str(src), str(dst))
            print(f"  Preserved    : {src.name}  →  archive  ({a['checksum'][:12]}...)  (moved)")
            entries.append({
                "timestamp":   ts,
                "action":      "archive_move",
                "source":      str(src),
                "destination": str(dst),
                "checksum":    a["checksum"],
                "reason":      ARCHIVE_REASON,
            })

    if entries:
        append_manifest(manifest_path, entries)


def apply_promotions(promote_actions, manifest_path):
    ts            = datetime.now(timezone.utc).isoformat()
    entries       = []
    verify_errors = []

    for a in promote_actions:
        if a["action"] == "already_promoted":
            print(f"  Already promoted : {a['canon_rel']}")
            continue

        staged, canonical = a["staged"], a["canonical"]
        canonical.parent.mkdir(parents=True, exist_ok=True)
        try:
            post_cksum = atomic_copy_verified(staged, canonical, src_cksum=a["checksum"])
        except Exception as exc:
            verify_errors.append(str(exc))
            print(f"  ERROR: {exc}")
            entries.append({
                "timestamp":   ts,
                "action":      "promote_failed",
                "source":      str(staged),
                "destination": str(canonical),
                "checksum":    a["checksum"],
                "reason":      ARCHIVE_REASON,
            })
            continue

        print(f"  Promoted     : {canonical.name}  ({post_cksum[:12]}...)")
        entries.append({
            "timestamp":   ts,
            "action":      "promote",
            "source":      str(staged),
            "destination": str(canonical),
            "checksum":    post_cksum,
            "reason":      ARCHIVE_REASON,
        })

    if entries:
        append_manifest(manifest_path, entries)

    return verify_errors


def write_readme(archive_dir):
    readme = archive_dir / README_FILENAME
    if not readme.exists():
        readme.write_text(README_TEXT, encoding="utf-8")
        print(f"  Wrote README : {readme}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Archive bad-STC fMRI products and promote corrected-STC summary "
            "files into canonical data/fMRI/ locations.  Default is dry-run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Dry-run:  python3 scripts/maintenance/archive_and_promote_corrected_stc.py\n"
            "Apply:    python3 scripts/maintenance/archive_and_promote_corrected_stc.py --apply"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute archive and promotion (default: dry-run).",
    )
    parser.add_argument(
        "--project-root",
        metavar="PATH",
        default=None,
        help="Repository root (default: two levels above this script).",
    )
    parser.add_argument(
        "--import-root",
        metavar="PATH",
        default=None,
        help="Staging directory (default: <project-root>/data/fMRI/corrected_stc_import).",
    )
    args    = parser.parse_args()
    dry_run = not args.apply

    # ── Resolve and validate paths ───────────────────────────────────────────
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parent.parent.parent
    )
    fmri_dir    = project_root / "data" / "fMRI"
    import_root = (
        Path(args.import_root).resolve()
        if args.import_root
        else fmri_dir / "corrected_stc_import"
    )
    archive_dir   = fmri_dir / ARCHIVE_DIR_NAME
    manifest_path = archive_dir / MANIFEST_FILENAME

    print(f"\nProject root : {project_root}")
    print(f"data/fMRI    : {fmri_dir}")
    print(f"Import root  : {import_root}")
    print(f"Archive dir  : {archive_dir}")

    if not fmri_dir.is_dir():
        print(f"\nERROR: data/fMRI not found: {fmri_dir}")
        sys.exit(1)
    if not import_root.is_dir():
        print(f"\nERROR: staging directory not found: {import_root}")
        sys.exit(1)

    # ── Staged-file validation ───────────────────────────────────────────────
    print("\n── STAGED FILE VALIDATION ─────────────────────────────────")
    val_errors = []
    for spec in VALIDATIONS:
        path = import_root / spec["staged_rel"]
        if not path.exists():
            val_errors.append(f"MISSING: {spec['staged_rel']}")
            print(f"  MISSING  [{spec['label']}]  {spec['staged_rel']}")
            continue
        errs = validate_staged_file(path, spec)
        if errs:
            for e in errs:
                val_errors.append(f"{spec['staged_rel']}: {e}")
            print(f"  ERROR    [{spec['label']}]  {'; '.join(errs)}")
        else:
            key_desc = (
                f"key={spec['unique_key']}"
                if spec.get("unique_key") else "no uniqueness check"
            )
            print(
                f"  OK       [{spec['label']}]  "
                f"{spec['expected_rows']} rows  id={spec['id_col']}  {key_desc}"
            )

    if val_errors:
        print(f"\nAborting: {len(val_errors)} validation error(s).")
        sys.exit(1)

    print("  All staged files valid.")

    # ── Build plan ───────────────────────────────────────────────────────────
    already_promoted = compute_already_promoted(import_root, fmri_dir)
    targets          = collect_archive_targets(fmri_dir, archive_dir, import_root)
    archive_actions, archive_errors = plan_archive(archive_dir, targets, already_promoted)
    promote_actions                 = plan_promotions(import_root, fmri_dir, already_promoted)

    print_plan(archive_actions, promote_actions, archive_dir, dry_run)

    if archive_errors:
        print("ERRORS — the following conflict(s) block --apply:")
        for e in archive_errors:
            print(f"\n  {e}")
        if dry_run:
            print("\n  (shown in dry-run; would abort --apply)")
        else:
            sys.exit(1)
        return

    if dry_run:
        return

    # ── Apply ────────────────────────────────────────────────────────────────
    print("── APPLYING ───────────────────────────────────────────────")
    archive_dir.mkdir(parents=True, exist_ok=True)
    write_readme(archive_dir)

    apply_archive(archive_actions, archive_dir, manifest_path)
    verify_errors = apply_promotions(promote_actions, manifest_path)

    if verify_errors:
        print("\nVERIFICATION ERRORS (post-copy checksum mismatch):")
        for e in verify_errors:
            print(f"  {e}")
        sys.exit(1)

    print(f"\nDone.  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
