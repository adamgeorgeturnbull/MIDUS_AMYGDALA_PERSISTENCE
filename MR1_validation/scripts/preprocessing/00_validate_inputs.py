#!/usr/bin/env python3
"""
00_validate_inputs.py (MR1)

Privacy-safe validation report for MR1 input files.

Run this BEFORE the full preprocessing pipeline to verify that all required
input files are present, have the expected structure, and pass critical
integrity checks. Exits with a non-zero code on any hard failure.

Stop conditions (EXIT 1):
  - Any required file is missing (P2, P5, or MKER1_variables1.tsv)
  - Duplicate MIDUSID in participant-level files (P5, MKER1)
  - MIDUSID ≠ MRID disagreement in P5 (where both non-missing)
  - Unexpected SAMPLMAJ codes in MKER1 (expected: 21 only)
  - Required column absent from a file

Warnings (EXIT 0):
  - No MKER1 ∩ P5 overlap (unexpected — log as warning)
  - Rows with missing MIDUSID in P2 after MRID fill

Note on P2: the daily diary is long-format (multiple rows per participant),
so repeated MIDUSID values are expected and are NOT an error.

Note on MKER1_variables1.tsv: required for this MR1 reproduction because it
supplies the planned Refresher demographic fallback for P5 missing values.

Note on MKER1_variables2.tsv: not consumed by the current preprocessing or
analysis pipeline and is therefore outside this validator's scope.

Privacy: prints only aggregate counts — never prints participant IDs or rows.

Inputs validated:
  data/raw/MR1_P2_variables.csv
  data/raw/MR1_P5_variables.csv
  data/raw/MKER1_variables1.tsv   (required — Refresher demographic fallback)

Run from MR1_validation/ directory.
"""

import os
import sys

import pandas as pd

RAW_DIR = "data/raw"
P2_FILE   = os.path.join(RAW_DIR, "MR1_P2_variables.csv")
P5_FILE   = os.path.join(RAW_DIR, "MR1_P5_variables.csv")
MKER1_FILE = os.path.join(RAW_DIR, "MKER1_variables1.tsv")


def _resolve_path(path):
    """Return actual path, trying .tsv variant if .csv not found (and vice versa)."""
    if os.path.isfile(path):
        return path
    alt = path.replace(".csv", ".tsv") if path.endswith(".csv") else path.replace(".tsv", ".csv")
    return alt if os.path.isfile(alt) else path


def _read_raw(path):
    """Read CSV or TSV using the separator that matches the file extension."""
    actual = _resolve_path(path)
    sep = "\t" if actual.endswith(".tsv") else ","
    return pd.read_csv(actual, encoding="utf-8-sig", sep=sep)

MKER1_EXPECTED_SAMPLMAJ = {21}

P2_REQUIRED_COLS = {
    "MIDUSID", "MRID",
    "RA2DDAY", "RA2DIMON", "RA2DIYEAR",
} | {f"RA2DC{i}" for i in range(1, 28)}

P5_REQUIRED_COLS = {
    "MIDUSID", "MRID",
    "RA1PRSEX", "RA1PB1", "RA1PF1", "RA1PF7A", "RA1PBYEAR",
    "RA5PAGE", "RA5PDATE_YR", "RA5PDATE_MO",
    "RA5SER", "RA5SES", "RA5SPGP", "RA5SPGN",
}

MKER1_REQUIRED_COLS = {
    "MRID", "SAMPLMAJ",
    "RAACRSEX", "RAACB1", "RAACF1", "RAACF7A", "RAACBYEAR",
    "RAACRAGE", "RAACIDATE_YR", "RAACIDATE_MO",
}

ERRORS   = []
WARNINGS = []


def error(msg):
    ERRORS.append(msg)
    print(f"  [FAIL]    {msg}")


def warn(msg):
    WARNINGS.append(msg)
    print(f"  [WARN]    {msg}")


def ok(msg):
    print(f"  [OK]      {msg}")


def _to_numeric_id(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.round().astype("Int64")


def check_file_exists(path, label, required=True):
    actual = _resolve_path(path)
    if os.path.isfile(actual):
        size_kb = os.path.getsize(actual) // 1024
        ext_note = f" [.tsv]" if actual.endswith(".tsv") else ""
        ok(f"{label} found ({size_kb} KB){ext_note}")
        return True
    if required:
        error(f"{label} NOT FOUND: {path} (also tried .tsv/.csv variant)")
    else:
        warn(f"{label} not found (optional): {path}")
    return False


def check_required_cols(df, required_cols, label):
    missing = required_cols - set(df.columns)
    if missing:
        error(f"{label}: missing required columns: {sorted(missing)}")
        return False
    ok(f"{label}: required columns present")
    return True


def validate_p2(path):
    print(f"\n--- P2 Daily Diary ({path}) ---")
    df = _read_raw(path)
    df.columns = df.columns.str.strip()
    print(f"  Rows loaded: {len(df)}")

    if not check_required_cols(df, P2_REQUIRED_COLS, "P2"):
        return

    df["MIDUSID"] = _to_numeric_id(df["MIDUSID"])
    df["MRID"]    = _to_numeric_id(df["MRID"])

    # Fill MIDUSID from MRID where missing (non-day-1 rows)
    df["MIDUSID"] = df["MIDUSID"].fillna(df["MRID"])

    n_missing = df["MIDUSID"].isna().sum()
    if n_missing > 0:
        warn(f"P2: {n_missing} rows still have missing MIDUSID after MRID fill")
    else:
        ok(f"P2: MIDUSID fully populated after MRID fill")

    n_unique = df["MIDUSID"].dropna().nunique()
    ok(f"P2: {n_unique} unique participants, {len(df)} total rows")

    # Agreement check
    both = df[df["MIDUSID"].notna() & df["MRID"].notna()]
    disagree = (both["MIDUSID"] != both["MRID"]).sum()
    if disagree > 0:
        error(f"P2: {disagree} rows have MIDUSID ≠ MRID (data integrity violation)")
    else:
        ok(f"P2: MIDUSID == MRID agreement check PASSED ({len(both)} rows checked)")

    return df["MIDUSID"].dropna().unique()


def validate_p5(path):
    print(f"\n--- P5 Neuroscience ({path}) ---")
    df = _read_raw(path)
    df.columns = df.columns.str.strip()
    print(f"  Rows loaded: {len(df)}")

    if not check_required_cols(df, P5_REQUIRED_COLS, "P5"):
        return None

    df["MIDUSID"] = _to_numeric_id(df["MIDUSID"])
    df["MRID"]    = _to_numeric_id(df["MRID"])

    n_missing = df["MIDUSID"].isna().sum()
    if n_missing > 0:
        warn(f"P5: {n_missing} rows with missing MIDUSID (will be dropped in pipeline)")
    df = df[df["MIDUSID"].notna()].copy()
    ok(f"P5: {len(df)} participants after dropping missing MIDUSID")

    # Uniqueness
    n_dup = df["MIDUSID"].duplicated().sum()
    if n_dup > 0:
        error(f"P5: {n_dup} duplicate MIDUSID values (expected one row per participant)")
    else:
        ok("P5: MIDUSID uniqueness check PASSED")

    # Agreement
    both = df[df["MIDUSID"].notna() & df["MRID"].notna()]
    disagree = (both["MIDUSID"] != both["MRID"]).sum()
    if disagree > 0:
        error(f"P5: {disagree} participants have MIDUSID ≠ MRID")
    else:
        ok(f"P5: MIDUSID == MRID agreement check PASSED ({len(both)} participants)")

    # Key variable presence
    for col in ["RA1PRSEX", "RA1PB1", "RA1PF1", "RA1PF7A", "RA1PBYEAR"]:
        n_valid = df[col].notna().sum() if col in df.columns else 0
        note = "present" if col in df.columns else "NOT IN FILE"
        ok(f"P5: {col}: {n_valid} non-missing  [{note}]") if col in df.columns \
            else warn(f"P5: {col} not in file (demographics fallback needed)")

    return df["MIDUSID"].dropna()


def validate_mker1(path, p5_ids):
    print(f"\n--- MKER1 ({path}) ---")
    df = _read_raw(path)
    df.columns = df.columns.str.strip()
    print(f"  Rows loaded: {len(df)}")

    if not check_required_cols(df, MKER1_REQUIRED_COLS, "MKER1"):
        return

    df["MRID"] = _to_numeric_id(df["MRID"])
    df["MIDUSID"] = df["MRID"].copy()

    n_missing = df["MIDUSID"].isna().sum()
    if n_missing > 0:
        warn(f"MKER1: {n_missing} rows with missing MRID (will be dropped)")
    df = df[df["MIDUSID"].notna()].copy()

    # Uniqueness
    n_dup = df["MIDUSID"].duplicated().sum()
    if n_dup > 0:
        error(f"MKER1: {n_dup} duplicate MIDUSID values")
    else:
        ok("MKER1: MIDUSID uniqueness check PASSED")

    # SAMPLMAJ
    if "SAMPLMAJ" in df.columns:
        observed = set(df["SAMPLMAJ"].dropna().astype(int).unique())
        unexpected = observed - MKER1_EXPECTED_SAMPLMAJ
        if unexpected:
            error(f"MKER1: unexpected SAMPLMAJ codes: {unexpected} (expected {MKER1_EXPECTED_SAMPLMAJ})")
        else:
            ok(f"MKER1: SAMPLMAJ codes: {sorted(observed)} — all expected")

    # MKER1-specific demographics columns
    for col in ["RAACRSEX", "RAACB1", "RAACF1", "RAACF7A", "RAACBYEAR",
                "RAACRAGE", "RAACIDATE_YR", "RAACIDATE_MO"]:
        n_valid = df[col].notna().sum() if col in df.columns else 0
        note = f"{n_valid} non-missing" if col in df.columns else "NOT IN FILE"
        ok(f"MKER1: {col}: {note}") if col in df.columns \
            else warn(f"MKER1: {col} not in file")

    # Overlap with P5
    if p5_ids is not None:
        mk_ids  = set(df["MIDUSID"].dropna())
        p5_set  = set(p5_ids)
        overlap = len(mk_ids & p5_set)
        ok(f"MKER1 ∩ P5: {overlap} participants")
        if overlap == 0:
            warn("MKER1: no overlap with P5 — verify IDs are on the same scale")


def main():
    print("=" * 60)
    print("00_validate_inputs.py (MR1)")
    print("Privacy-safe — prints aggregate counts only")
    print("=" * 60)

    # ── File existence ────────────────────────────────────────────────────────
    print("\n--- File presence ---")
    p2_ok    = check_file_exists(P2_FILE,   "MR1_P2_variables.csv",  required=True)
    p5_ok    = check_file_exists(P5_FILE,   "MR1_P5_variables.csv",  required=True)
    mker1_ok = check_file_exists(MKER1_FILE, "MKER1_variables1.tsv", required=True)

    if not (p2_ok and p5_ok and mker1_ok):
        print("\nRequired files missing — cannot continue validation.")
        sys.exit(1)

    # ── Per-file validation ───────────────────────────────────────────────────
    validate_p2(P2_FILE)
    p5_ids = validate_p5(P5_FILE)
    validate_mker1(MKER1_FILE, p5_ids)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Validation summary: {len(ERRORS)} error(s), {len(WARNINGS)} warning(s)")

    if ERRORS:
        print("\nERRORS (pipeline will abort):")
        for e in ERRORS:
            print(f"  - {e}")
        sys.exit(1)

    if WARNINGS:
        print("\nWarnings (pipeline may proceed but review advised):")
        for w in WARNINGS:
            print(f"  - {w}")

    print("\nAll checks PASSED. Proceed to 01_harmonize_ids.py.")


if __name__ == "__main__":
    main()
