#!/usr/bin/env python3
"""
01_harmonize_ids.py (MR1)

Harmonize participant IDs across MR1 datasets and integrate the MKE Refresher 1
(MKER1) aggregate supplement.

Inputs:
  data/raw/MR1_P2_variables.csv     daily diary, long format (multiple rows per
                                    participant; repeated MIDUSID is expected)
  data/raw/MR1_P5_variables.csv     neuroscience + survey, one row per participant
  data/raw/MKER1_variables1.tsv     MKE Refresher 1 aggregate demographics
                                    (required — supplies Refresher demographic
                                    fallback for P5 missing values)

Note on MKER1_variables2.tsv: not consumed by the current preprocessing or
analysis pipeline and is therefore outside this script's scope.

Outputs:
  data/processed/mr1p2_ids.csv
  data/processed/mr1p5_ids.csv
  data/processed/mker1_ids.csv
  logs/01_harmonize_ids_log_<timestamp>.txt

Privacy: prints only aggregate counts — never prints participant IDs or rows.

Stop conditions (abort with sys.exit):
  - MKER1_variables1.tsv missing (required)
  - Required column absent from a file
  - Duplicate MIDUSID in any participant-level file (P5, MKER1)
  - MIDUSID ≠ MRID for any non-missing pair (data integrity violation)
  - Unexpected SAMPLMAJ codes in MKER1 (expected: 21)

Warnings (logged, do not abort):
  - No overlap between MKER1 and P5 (unexpected — verify IDs are on same scale)
  - Rows with missing MIDUSID in P2 after MRID fill

Run from MR1_validation/ directory.
"""

import os
import sys
from datetime import datetime

import pandas as pd


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


def _require_columns(df, required, label, log_fn):
    """Exit with a clear aggregate message if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        log_fn(f"  ERROR: {label} is missing required column(s): {sorted(missing)}")
        log_fn(f"  Available columns: {list(df.columns)}")
        sys.exit(1)


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_DIR = "logs"

P2_FILE = os.path.join(RAW_DIR, "MR1_P2_variables.csv")
P5_FILE = os.path.join(RAW_DIR, "MR1_P5_variables.csv")
MKER1_FILE = os.path.join(RAW_DIR, "MKER1_variables1.tsv")

MKER1_EXPECTED_SAMPLMAJ = {21}


def write_log(message, log_file):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")


def _to_int_id(series):
    """Convert an ID series to nullable Int64, avoiding scientific notation."""
    s = pd.to_numeric(series, errors="coerce")
    # Round to remove floating-point artifacts from CSV encoding
    s = s.round().astype("Int64")
    return s


def harmonize_p2(path, log_fn):
    """Load P2 daily diary. Long format — no uniqueness check."""
    log_fn(f"\n{'=' * 60}")
    log_fn(f"Dataset: P2 daily diary  ({path})")

    df = _read_raw(path)
    df.columns = df.columns.str.strip()
    log_fn(f"  Rows loaded: {len(df)}")

    _require_columns(df, ["MIDUSID", "MRID"], "P2", log_fn)

    # Both MIDUSID and MRID are present; MIDUSID is only on day-1 rows
    if "MRID" in df.columns:
        df["MRID"] = _to_int_id(df["MRID"])
    if "MIDUSID" in df.columns:
        df["MIDUSID"] = _to_int_id(df["MIDUSID"])
        # Fill MIDUSID from MRID where missing (non-day-1 rows)
        if "MRID" in df.columns:
            df["MIDUSID"] = df["MIDUSID"].fillna(df["MRID"])
    elif "MRID" in df.columns:
        df["MIDUSID"] = df["MRID"]
    else:
        raise ValueError("P2 file has neither MIDUSID nor MRID column")

    # Assert agreement: where both are non-missing they must match
    if "MRID" in df.columns:
        both = df[df["MIDUSID"].notna() & df["MRID"].notna()]
        disagree = (both["MIDUSID"] != both["MRID"]).sum()
        log_fn(f"  Rows where MIDUSID and MRID both non-missing: {len(both)}")
        if disagree > 0:
            sys.exit(
                f"FATAL: {disagree} rows have MIDUSID ≠ MRID in P2. "
                "Investigate before proceeding."
            )
        log_fn(f"  MIDUSID == MRID agreement check: PASSED (0 mismatches)")

    n_unique = df["MIDUSID"].dropna().nunique()
    n_missing = df["MIDUSID"].isna().sum()
    log_fn(f"  Unique participants: {n_unique}")
    log_fn(f"  Rows with missing MIDUSID after fill: {n_missing}")
    if n_missing > 0:
        log_fn(f"  WARNING: {n_missing} rows still have missing MIDUSID — check fill logic")

    return df


def harmonize_p5(path, log_fn):
    """Load P5 neuroscience survey. One row per participant — enforce uniqueness."""
    log_fn(f"\n{'=' * 60}")
    log_fn(f"Dataset: P5 neuroscience  ({path})")

    df = _read_raw(path)
    df.columns = df.columns.str.strip()
    log_fn(f"  Rows loaded: {len(df)}")

    _require_columns(df, ["MIDUSID", "MRID"], "P5", log_fn)

    df["MIDUSID"] = _to_int_id(df["MIDUSID"])
    if "MRID" in df.columns:
        df["MRID"] = _to_int_id(df["MRID"])

    n_missing = df["MIDUSID"].isna().sum()
    log_fn(f"  Missing MIDUSID: {n_missing}")
    if n_missing > 0:
        log_fn(f"  WARNING: {n_missing} rows with missing MIDUSID will be dropped")

    df = df[df["MIDUSID"].notna()].copy()

    # Assert no duplicate MIDUSID
    n_dup = df["MIDUSID"].duplicated().sum()
    if n_dup > 0:
        sys.exit(
            f"FATAL: {n_dup} duplicate MIDUSID values in P5. "
            "Expected one row per participant. Investigate before proceeding."
        )
    log_fn(f"  Duplicate MIDUSID check: PASSED (0 duplicates)")

    # Assert MIDUSID == MRID agreement
    if "MRID" in df.columns:
        both = df[df["MIDUSID"].notna() & df["MRID"].notna()]
        disagree = (both["MIDUSID"] != both["MRID"]).sum()
        if disagree > 0:
            sys.exit(
                f"FATAL: {disagree} participants have MIDUSID ≠ MRID in P5. "
                "Investigate before proceeding."
            )
        log_fn(f"  MIDUSID == MRID agreement check: PASSED (0 mismatches)")

    log_fn(f"  Participants after cleaning: {len(df)}")
    return df


def harmonize_mker1(path, log_fn):
    """
    Load MKER1 aggregate demographics. Create canonical MIDUSID from MRID.

    MKER1 does not contain a MIDUSID column; MIDUSID is created from MRID.
    SAMPLMAJ must equal 21 (Milwaukee Refresher 1).
    """
    log_fn(f"\n{'=' * 60}")
    log_fn(f"Dataset: MKER1 demographics  ({path})")

    if not os.path.isfile(_resolve_path(path)):
        log_fn(f"  ERROR: MKER1 file not found: {path}")
        sys.exit(1)

    df = _read_raw(path)
    df.columns = df.columns.str.strip()
    log_fn(f"  Rows loaded: {len(df)}")

    _require_columns(df, [
        "MRID", "SAMPLMAJ",
        "RAACRSEX", "RAACB1", "RAACF1", "RAACF7A", "RAACBYEAR",
        "RAACRAGE", "RAACIDATE_YR", "RAACIDATE_MO",
    ], "MKER1", log_fn)

    df["MRID"] = _to_int_id(df["MRID"])

    # Create canonical MIDUSID from MRID
    df["MIDUSID"] = df["MRID"].copy()

    n_missing = df["MIDUSID"].isna().sum()
    log_fn(f"  Missing MRID/MIDUSID: {n_missing}")
    if n_missing > 0:
        log_fn(f"  WARNING: {n_missing} rows with missing MRID will be dropped")

    df = df[df["MIDUSID"].notna()].copy()

    # Assert no duplicate MIDUSID
    n_dup = df["MIDUSID"].duplicated().sum()
    if n_dup > 0:
        sys.exit(
            f"FATAL: {n_dup} duplicate MIDUSID values in MKER1. "
            "Expected one row per participant. Investigate before proceeding."
        )
    log_fn(f"  Duplicate MIDUSID check: PASSED (0 duplicates)")

    # Validate SAMPLMAJ (guaranteed present by _require_columns)
    observed = set(df["SAMPLMAJ"].dropna().astype(int).unique())
    unexpected = observed - MKER1_EXPECTED_SAMPLMAJ
    if unexpected:
        sys.exit(
            f"FATAL: Unexpected SAMPLMAJ codes in MKER1: {unexpected}. "
            f"Expected only {MKER1_EXPECTED_SAMPLMAJ}. Investigate before proceeding."
        )
    log_fn(f"  SAMPLMAJ codes present: {sorted(observed)} — all expected")

    log_fn(f"  MKER1 participants after cleaning: {len(df)}")
    return df


def report_overlaps(p5_ids, p2_ids, mker1_df, log_fn):
    """Report aggregate overlap counts. Never prints individual IDs."""
    log_fn(f"\n{'=' * 60}")
    log_fn("Overlap diagnostics (aggregate counts only)")

    p5_set = set(p5_ids)
    p2_set = set(p2_ids)

    log_fn(f"  P5 unique participants:        {len(p5_set)}")
    log_fn(f"  P2 unique participants:        {len(p2_set)}")
    log_fn(f"  P2 ∩ P5:                       {len(p5_set & p2_set)}")
    log_fn(f"  P2 only (not in P5):           {len(p2_set - p5_set)}")
    log_fn(f"  P5 only (not in P2):           {len(p5_set - p2_set)}")

    mk_set = set(mker1_df["MIDUSID"].dropna())
    log_fn(f"  MKER1 unique participants:     {len(mk_set)}")
    log_fn(f"  MKER1 ∩ P5:                    {len(mk_set & p5_set)}")
    log_fn(f"  MKER1 ∩ P2:                    {len(mk_set & p2_set)}")
    log_fn(f"  MKER1 ∩ P2 ∩ P5:               {len(mk_set & p2_set & p5_set)}")
    log_fn(f"  MKER1 only (not in P5):        {len(mk_set - p5_set)}")

    if len(mk_set & p5_set) == 0:
        log_fn(
            "  WARNING: No overlap between MKER1 and P5. "
            "This is unexpected — verify that IDs are on the same scale."
        )


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"01_harmonize_ids_log_{ts}.txt")

    def log_fn(msg):
        write_log(msg, log_file)

    log_fn("MR1 ID Harmonization")
    log_fn(f"Timestamp: {ts}")

    # ── P2 ───────────────────────────────────────────────────────────────────
    df_p2 = harmonize_p2(P2_FILE, log_fn)
    df_p2.to_csv(os.path.join(PROCESSED_DIR, "mr1p2_ids.csv"), index=False)
    log_fn(f"  Saved: data/processed/mr1p2_ids.csv")

    # ── P5 ───────────────────────────────────────────────────────────────────
    df_p5 = harmonize_p5(P5_FILE, log_fn)
    df_p5.to_csv(os.path.join(PROCESSED_DIR, "mr1p5_ids.csv"), index=False)
    log_fn(f"  Saved: data/processed/mr1p5_ids.csv")

    # ── MKER1 ────────────────────────────────────────────────────────────────
    df_mker1 = harmonize_mker1(MKER1_FILE, log_fn)
    df_mker1.to_csv(os.path.join(PROCESSED_DIR, "mker1_ids.csv"), index=False)
    log_fn(f"  Saved: data/processed/mker1_ids.csv")

    # ── Overlap report ───────────────────────────────────────────────────────
    p5_ids = df_p5["MIDUSID"].dropna()
    p2_ids = df_p2["MIDUSID"].dropna().unique()
    report_overlaps(p5_ids, p2_ids, df_mker1, log_fn)

    log_fn(f"\n{'=' * 60}")
    log_fn("01_harmonize_ids.py completed successfully.")


if __name__ == "__main__":
    main()
