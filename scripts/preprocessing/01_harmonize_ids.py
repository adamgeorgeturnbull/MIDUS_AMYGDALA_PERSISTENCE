#!/usr/bin/env python3
"""
01_harmonize_ids.py

Harmonize participant IDs across MIDUS datasets, check uniqueness and data types,
and save processed versions for later merging.

Inputs:
- data/raw/M3P2_variables.csv (Project 2: daily diary affect, long format)
- data/raw/M3P5_variables.csv (Project 5: neuroscience + survey demographics)
- data/raw/MKE2_variables.csv (Milwaukee sample: demographic supplement)

Outputs:
- data/processed/m3p2_ids.csv
- data/processed/m3p5_ids.csv
- data/processed/mke2_ids.csv
- logs/01_harmonize_ids_log_<timestamp>.txt

Run from project root directory.
"""

import os
from datetime import datetime

import pandas as pd

# ============================================================================
# Paths
# ============================================================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_DIR = "logs"

# Dataset configurations
DATASETS = {
    "project2_daily_diary": {
        "path": os.path.join(RAW_DIR, "M3P2_variables.csv"),
        "check_duplicates": False,  # Long format: multiple rows per participant
        "output": "m3p2_ids.csv"
    },
    "project5_neuroscience": {
        "path": os.path.join(RAW_DIR, "M3P5_variables.csv"),
        "check_duplicates": True,  # Wide format: one row per participant
        "output": "m3p5_ids.csv"
    },
    "milwaukee_sample": {
        "path": os.path.join(RAW_DIR, "MKE2_variables.csv"),
        "check_duplicates": True,  # Wide format: one row per participant
        "output": "mke2_ids.csv"
    }
}

# ============================================================================
# Helper Functions
# ============================================================================
def write_log(message, log_file):
    """Print message and append to log file."""
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")


def harmonize_ids(df, key="M2ID", check_duplicates=True, log_file=None):
    """
    Check ID column, coerce to numeric, and validate.

    Args:
        df: DataFrame to harmonize
        key: Name of ID column (default: M2ID)
        check_duplicates: Whether to check for duplicate IDs
        log_file: Path to log file for messages

    Returns:
        DataFrame with harmonized IDs
    """
    log_fn = lambda msg: write_log(msg, log_file) if log_file else print(msg)

    log_fn(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure key exists
    if key not in df.columns:
        raise ValueError(f"Key column '{key}' not found in dataset")

    # Log original dtype
    log_fn(f"{key} dtype before conversion: {df[key].dtype}")

    # Coerce ID to numeric
    df[key] = pd.to_numeric(df[key], errors="coerce")

    # Check for missing IDs
    missing = df[key].isna().sum()
    log_fn(f"Missing {key}: {missing}")

    # Check for duplicate IDs if requested
    if check_duplicates:
        duplicates = df[key].duplicated().sum()
        log_fn(f"Duplicated {key}: {duplicates}")
    else:
        log_fn("Duplicate check skipped (expected long format)")

    return df


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    # Ensure output directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create log file
    log_file = os.path.join(
        LOG_DIR, f"01_harmonize_ids_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    # Process each dataset
    for name, info in DATASETS.items():
        write_log(f"\nProcessing dataset: {name}", log_file)

        path = info["path"]
        output_path = os.path.join(PROCESSED_DIR, info["output"])
        check_dups = info["check_duplicates"]

        # Check if file exists
        if not os.path.isfile(path):
            write_log(f"WARNING: File not found: {path}", log_file)
            continue

        # Load and harmonize dataset
        df = pd.read_csv(path)
        df_clean = harmonize_ids(
            df, key="M2ID", check_duplicates=check_dups, log_file=log_file
        )

        # Save harmonized dataset
        df_clean.to_csv(output_path, index=False)
        write_log(f"Saved harmonized dataset to {output_path}", log_file)

    write_log("\nAll datasets harmonized successfully.", log_file)


if __name__ == "__main__":
    main()

