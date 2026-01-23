#!/usr/bin/env python3
"""
01_harmonize_ids.py

Purpose:
Harmonize participant IDs across MIDUS datasets, check uniqueness and data types,
and save processed versions for later merging.

Datasets:
- MIDUS Project 5: neuroscience + survey demographics
- MIDUS Project 2: daily diary affect (long format)
- MIDUS Milwaukee sample: demographic supplement

Run this script from the project root directory.
"""

import pandas as pd
import os
from datetime import datetime

# =========================
# Paths (relative to project root)
# =========================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
LOG_DIR = "logs"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(
    LOG_DIR, f"01_harmonize_ids_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

def write_log(message):
    """Print message and append to log file."""
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# =========================
# Datasets to harmonize
# =========================
datasets = {
    "project2_daily_diary": {
        "path": os.path.join(RAW_DIR, "M3P2_variables.csv"),
        "check_duplicates": False,
        "output": "m3p2_ids.csv"
    },
    "project5_neuroscience": {
        "path": os.path.join(RAW_DIR, "M3P5_variables.csv"),
        "check_duplicates": True,
        "output": "m3p5_ids.csv"
    },
    "milwaukee_sample": {
        "path": os.path.join(RAW_DIR, "MKE2_variables.csv"),
        "check_duplicates": True,
        "output": "mke2_ids.csv"
    }
}

# =========================
# Harmonization function
# =========================
def harmonize_ids(df, key="M2ID", check_duplicates=True):
    """
    Check ID column, coerce to numeric, log missing values,
    and optionally check for duplicate IDs.
    """
    write_log(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure key exists
    if key not in df.columns:
        raise ValueError(f"Key column '{key}' not found in dataset")

    # Log original dtype
    write_log(f"{key} dtype before conversion: {df[key].dtype}")

    # Coerce ID to numeric
    df[key] = pd.to_numeric(df[key], errors="coerce")

    # Missing IDs
    missing = df[key].isna().sum()
    write_log(f"Missing {key}: {missing}")

    # Duplicate IDs
    if check_duplicates:
        duplicates = df[key].duplicated().sum()
        write_log(f"Duplicated {key}: {duplicates}")
    else:
        write_log("Duplicate check skipped (expected long format)")

    return df

# =========================
# Process datasets
# =========================
for name, info in datasets.items():
    write_log(f"\nProcessing dataset: {name}")

    path = info["path"]
    output_path = os.path.join(PROCESSED_DIR, info["output"])
    check_dups = info["check_duplicates"]

    if not os.path.isfile(path):
        write_log(f"WARNING: File not found: {path}")
        continue

    df = pd.read_csv(path)
    df_clean = harmonize_ids(df, key="M2ID", check_duplicates=check_dups)

    df_clean.to_csv(output_path, index=False)
    write_log(f"Saved harmonized dataset to {output_path}")

write_log("\nAll datasets harmonized successfully.")

