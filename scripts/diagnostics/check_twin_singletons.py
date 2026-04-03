#!/usr/bin/env python3
"""
check_twin_singletons.py

For each analytic sample, count how many twin_pair_ dummy columns are
singletons (sum == 1, meaning only one twin from the pair is present).
Singletons act as person-specific indicators that absorb that individual
from OLS regressions, biasing estimates.

Run from project root directory.
"""

import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/processed/midus_with_fmri.csv")

df = pd.read_csv(DATA_FILE)
twin_cols = [c for c in df.columns if c.startswith("twin_pair_")]
print(f"Total twin_pair_ columns in dataset: {len(twin_cols)}")

has_affect = df[["PA_score", "NA_score"]].notna().any(axis=1)

samples = {
    "01 neuro subsample      (N~137, C5PAGE + diary)":
        df[df["C5PAGE"].notna() & has_affect],

    "02 conservative diary+fMRI (N~80, qc_cons + diary)":
        df[(df["qc_conservative"] == 1) & has_affect],

    "03/04/05 conservative fMRI (N~128, qc_cons)":
        df[df["qc_conservative"] == 1],

    "02 PANAS sensitivity    (N~128, has persistence + PANAS)":
        df[df["qc_conservative"] == 1],   # PANAS uses same fMRI sample
}

print()
for label, sample in samples.items():
    present = [c for c in twin_cols if c in sample.columns]
    singletons = [c for c in present if sample[c].sum() == 1]
    pairs      = [c for c in present if sample[c].sum() >= 2]
    zeros      = [c for c in present if sample[c].sum() == 0]

    print(f"{label}")
    print(f"  N = {len(sample)}")
    print(f"  twin_pair_ dummies with sum >= 2 (proper pairs): {len(pairs)}")
    print(f"  twin_pair_ dummies with sum == 1 (singletons):   {len(singletons)}")
    print(f"  twin_pair_ dummies with sum == 0 (absent):       {len(zeros)}")
    if singletons:
        print(f"  Singleton columns: {singletons}")
    print()
