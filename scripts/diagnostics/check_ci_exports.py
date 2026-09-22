"""Read aggregate result tables only; check CI exports and pre-rerun values.

Run after the user executes the CI reruns. Does not calculate any CI.
Reports file/column names and counts, never participant data or raw logs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

M3_DIRS = ["00a_diary_panas", "00b_task_conditions", "00c_vmPFC_convergence",
           "00d_erq_context", "00e_motion_check", "01_affect_age",
           "02_persistence_affect", "03_persistence_age", "03b_persistence_age_mediation",
           "04_fc_affect", "04ex_fc_affect_antpost", "05_fc_persistence",
           "06_persistence_affect_moderation", "07_fc_affect_moderation",
           "sensitivity_novel_sample"]
MR1_DIRS = ["02_persistence_affect", "03_persistence_age", "04_fc_affect", "05_fc_persistence"]


def compare_previous(old, new):
    """An output-only change must preserve previous rows and columns."""
    pd.testing.assert_frame_equal(new[old.columns].reset_index(drop=True),
                                  old.reset_index(drop=True), check_dtype=False,
                                  check_exact=False, rtol=1e-8, atol=1e-10)


def describe_changes(old, new):
    """Summarize aggregate differences without printing cell contents."""
    missing = old.columns.difference(new.columns).tolist()
    if missing:
        return ["missing legacy columns: " + ", ".join(missing)]
    if len(old) != len(new):
        return [f"row count changed: {len(old)} -> {len(new)}; compare row identities before values"]
    old = old.reset_index(drop=True)
    new = new[old.columns].reset_index(drop=True)
    lines = []
    # Detect exact row permutations; never pair reordered rows silently.
    old_hash = pd.util.hash_pandas_object(old, index=False).to_numpy()
    new_hash = pd.util.hash_pandas_object(new, index=False).to_numpy()
    if not np.array_equal(old_hash, new_hash) and np.array_equal(np.sort(old_hash), np.sort(new_hash)):
        return ["same legacy rows in a different order; no cell values changed"]
    for column in old.columns:
        a, b = old[column], new[column]
        try:
            pd.testing.assert_series_equal(a, b, check_dtype=False,
                                           check_exact=False, rtol=1e-8, atol=1e-10)
            continue
        except AssertionError:
            pass
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            changed = ~np.isclose(av, bv, rtol=1e-8, atol=1e-10, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            maximum = np.max(np.abs(av[finite] - bv[finite])) if finite.any() else float("nan")
            line = (f"{column}: {changed.sum()} changed rows; max absolute difference={maximum:.8g}; "
                    f"finite/missing status changes={(np.isfinite(av) != np.isfinite(bv)).sum()}")
            if column in {"p", "pvalue", "p_value"} or column.startswith("p_") or column.endswith("_p"):
                line += f"; crossings of p < .05={np.sum(finite & ((av < .05) != (bv < .05)))}"
            lines.append(line)
        else:
            equal = (a.eq(b) | (a.isna() & b.isna())).fillna(False)
            lines.append(f"{column}: {(~equal).sum()} changed labels/values; verify row alignment")
    return lines


def describe_ci_issues(frame):
    lines = []
    statuses = {"ok", "nonconverged", "invalid_variance", "invalid_interval",
                "unavailable", "undefined_correlation", "invalid_sample"}
    for column in frame.columns:
        if column.endswith("ci_status"):
            for status, count in frame[column].fillna("missing").value_counts().items():
                if status != "ok":
                    label = status if status in statuses or status == "missing" else "unrecognized"
                    lines.append(f"{column}: {label} in {count} rows")
    return lines or ["check required fields, finite bounds, ordering, and confidence level"]


def check_frame(frame):
    if frame.empty:
        raise ValueError("empty output")
    if "indirect_ab" in frame:
        bounds = ["boot_ci_low", "boot_ci_high"]
        if not np.isfinite(frame[bounds].to_numpy(dtype=float)).all():
            raise ValueError("missing bootstrap bounds")
        if not (frame[bounds[0]] <= frame[bounds[1]]).all():
            raise ValueError("reversed bootstrap bounds")
        if not (frame["ci_level"] == 95).all():
            raise ValueError("unexpected bootstrap confidence level")
        return
    prefixes = ["interaction_", "predictor_", "moderator_"] if "beta_interaction" in frame else [""]
    for prefix in prefixes:
        names = [prefix + suffix for suffix in
                 ["ci_low", "ci_high", "ci_level", "ci_sidedness", "ci_method", "ci_status"]]
        if not all(c in frame for c in names):
            raise ValueError("missing CI fields: " + prefix)
        if not frame[prefix + "ci_status"].eq("ok").all():
            raise ValueError("CI status requires review: " + prefix)
        if not np.isfinite(frame[names[:2]].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite bounds: " + prefix)
        if not (frame[names[0]] <= frame[names[1]]).all():
            raise ValueError("reversed bounds: " + prefix)
        if not frame[prefix + "ci_level"].eq(95).all() or not frame[prefix + "ci_sidedness"].eq("two-sided").all():
            raise ValueError("unexpected CI level/sidedness: " + prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True,
                        help="Snapshot root containing the two results/tables trees")
    parser.add_argument("--details", action="store_true",
                        help="Summarize changed columns and CI statuses; no cell contents")
    args = parser.parse_args()
    if not (args.before / "results/tables").is_dir() or not (args.before / "MR1_validation/results/tables").is_dir():
        parser.error("Snapshot must contain both original results/tables trees")
    checked = changed = failed = new_files = 0
    for base, dirs in [(Path("results/tables"), M3_DIRS),
                       (Path("MR1_validation/results/tables"), MR1_DIRS)]:
        for name in dirs:
            folder = base / name
            files = sorted(folder.rglob("*.csv")) if folder.is_dir() else []
            if not files:
                print(f"MISSING: {folder}")
                failed += 1
            # Also detect deleted result files from the pre-rerun snapshot.
            old_folder = args.before / folder
            if old_folder.is_dir():
                for old in old_folder.rglob("*.csv"):
                    current = folder / old.relative_to(old_folder)
                    if not current.is_file():
                        print(f"REMOVED: {current}")
                        failed += 1
            for path in files:
                try:
                    frame = pd.read_csv(path)
                except (ValueError, pd.errors.EmptyDataError):
                    print(f"UNREADABLE OUTPUT: {path}")
                    failed += 1
                    continue
                try:
                    check_frame(frame)
                    checked += 1
                except (ValueError, KeyError, pd.errors.EmptyDataError):
                    print(f"CI REVIEW REQUIRED: {path}")
                    failed += 1
                    if args.details:
                        for line in describe_ci_issues(frame):
                            print("  " + line)
                old_path = args.before / path
                if not old_path.is_file():
                    print(f"NEW OUTPUT (no baseline comparison): {path}")
                    new_files += 1
                    continue
                try:
                    old_frame = pd.read_csv(old_path)
                    compare_previous(old_frame, frame)
                except (AssertionError, KeyError, ValueError):
                    print(f"EXISTING VALUES CHANGED: {path}")
                    changed += 1
                    if args.details:
                        # Re-read here so an unreadable baseline cannot leave a stale frame.
                        try:
                            for line in describe_changes(pd.read_csv(old_path), frame):
                                print("  " + line)
                        except (ValueError, KeyError, TypeError):
                            print("  baseline cannot be compared; inspect it locally")
    print(f"CI files passed: {checked}; CI/missing-file issues: {failed}; "
          f"changed existing outputs: {changed}; new files: {new_files}")
    if failed or changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
