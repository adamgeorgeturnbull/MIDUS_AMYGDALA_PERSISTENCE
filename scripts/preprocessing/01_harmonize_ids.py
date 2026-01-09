#!/usr/bin/env python3
"""
01_harmonize_ids.py
Purpose: Harmonize participant IDs across MIDUS datasets (daily diary and neuroscience+demographics),
         check uniqueness and data types, and save processed versions for later merging.
Run this from the master folder.
"""

import pandas as pd
import os
from datetime import datetime

# =========================
# Paths (relative to master folder)
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
    "daily_diary": {
        "path": os.path.join(RAW_DIR, "M3P2_variables.csv"),
        "check_duplicates": False
    },
    "neuroscience": {
        "path": os.path.join(RAW_DIR, "M3P5_variables_and_demos.csv"),
        "check_duplicates": True
    }
}

processed_files = {
    "daily_diary": os.path.join(PROCESSED_DIR, "m3p2_ids.csv"),
    "neuroscience": os.path.join(PROCESSED_DIR, "m3p5_ids.csv")
}

# =========================
# Harmonization function
# =========================
def harmonize_ids(df, key="M2ID", check_duplicates=True):
    """Check ID column, ensure numeric, check duplicates (if applicable) and missing values."""
    write_log(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Ensure key exists
    if key not in df.columns:
        raise ValueError(f"Key column '{key}' not found in dataset!")

    # Check data type of key
    write_log(f"{key} dtype before conversion: {df[key].dtype}")

    # Convert to numeric if possible
    df[key] = pd.to_numeric(df[key], errors="coerce")

    # Check for missing IDs
    missing = df[key].isna().sum()
    write_log(f"Missing {key}: {missing}")

    # Check for duplicates if requested
    if check_duplicates:
        duplicates = df[key].duplicated().sum()
        write_log(f"Duplicated {key}: {duplicates}")
    else:
        write_log(f"Duplicate check skipped for this dataset (expected long format).")

    return df

# =========================
# Process each dataset
# =========================
for name, info in datasets.items():
    write_log(f"\nProcessing dataset: {name}")
    path = info["path"]
    check_dups = info["check_duplicates"]

    if not os.path.isfile(path):
        write_log(f"WARNING: File not found: {path}")
        continue

    df = pd.read_csv(path)
    df_clean = harmonize_ids(df, key="M2ID", check_duplicates=check_dups)
    df_clean.to_csv(processed_files[name], index=False)
    write_log(f"Saved harmonized dataset to {processed_files[name]}")

write_log("\nAll datasets harmonized successfully.")
