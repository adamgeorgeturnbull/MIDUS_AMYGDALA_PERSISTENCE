#!/usr/bin/env python3
"""
07_sample_descriptives.py

Sample descriptives (N, age, sex) for the five analytic samples used in the
MIDUS Amygdala Persistence paper.

Samples
-------
1. daily_diary          – Daily diary completers with ≥1 affect measure (N~1,174)
                          Source: midus_merged_clean.csv
                          Age col: C2PAGE (MIDUS II age)

2. neuro_behavioral     – Neuroscience subsample (PANAS administered), no fMRI
                          QC applied (N~231)
                          Source: midus_merged_clean.csv
                          Age col: C5PAGE

3. neuro_diary_behavioral – Diary ∩ Neuroscience overlap, no fMRI QC (N~137)
                          Used for Analysis 01 neuro subsample and PANAS
                          sensitivity (Analyses 06/07).
                          Source: midus_merged_clean.csv
                          Age col: C5PAGE

4. neuro_fmri_conservative – Conservative fMRI QC (N~134); no diary requirement.
                          Used for Analysis 03 (age → persistence).
                          Source: midus_with_fmri.csv
                          Age col: C5PAGE

5. primary              – Diary + conservative fMRI QC + FC data (N~80).
                          Primary analytic sample for Analyses 02, 04, 05, 06, 07.
                          Source: midus_with_fmri.csv
                          Age col: C5PAGE

Outputs
-------
results/tables/sample_descriptives.csv

Run from project root directory.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from analysis_utils import (
    PROCESSED_DIR, MASTER_FILE, FC_FILE,
    load_master, prepare_persistence_vars, get_samples,
)

RESULTS_DIR = Path("results/tables")
OUTPUT_FILE = RESULTS_DIR / "sample_descriptives.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def describe_sample(df, age_col, label):
    """Return a dict of N, age mean/SD/range, %Female for df."""
    age = df[age_col].dropna()
    n   = len(df)
    n_sex  = df["sex"].notna().sum()
    # sex coded 1=Male, 2=Female in MIDUS
    pct_female = (df["sex"] == 2).sum() / n_sex * 100 if n_sex > 0 else float("nan")

    return {
        "sample":    label,
        "N":         n,
        "N_age":     len(age),
        "age_mean":  round(age.mean(), 1),
        "age_SD":    round(age.std(),  1),
        "age_min":   int(age.min()),
        "age_max":   int(age.max()),
        "pct_female": round(pct_female, 1),
    }


def has_diary(df):
    """Boolean mask: participant has ≥1 non-missing daily affect measure."""
    return df[["PA_score", "NA_score"]].notna().any(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load behavioral data (midus_merged_clean) ─────────────────────────────
    beh_file = PROCESSED_DIR / "midus_merged_clean.csv"
    beh = pd.read_csv(beh_file)
    beh["M2ID"] = beh["M2ID"].astype(str)
    print(f"Loaded behavioral data: {beh.shape[0]} rows")

    rows = []

    # 1. Daily diary (≥1 affect measure; C2PAGE for age)
    mask_diary = has_diary(beh)
    diary = beh[mask_diary].copy()
    rows.append(describe_sample(diary, "C2PAGE", "1_daily_diary"))
    print(f"  1. Daily diary: N={len(diary)}")

    # 2. Neuroscience behavioral (C5PAGE present; no fMRI QC)
    mask_neuro = beh["C5PAGE"].notna()
    neuro = beh[mask_neuro].copy()
    rows.append(describe_sample(neuro, "C5PAGE", "2_neuro_behavioral"))
    print(f"  2. Neuro behavioral: N={len(neuro)}")

    # 3. Diary ∩ Neuroscience behavioral overlap (no fMRI QC)
    overlap_beh = beh[mask_diary & mask_neuro].copy()
    rows.append(describe_sample(overlap_beh, "C5PAGE", "3_neuro_diary_behavioral"))
    print(f"  3. Neuro+Diary behavioral: N={len(overlap_beh)}")

    # ── Load fMRI master data ─────────────────────────────────────────────────
    # load_master(fc=True) merges the LSS beta-series FC file, which is needed
    # to define the N=80 primary sample (FC data availability).
    df_fc = load_master(fc=True)
    prepare_persistence_vars(df_fc)

    # qc_conservative flag defined in midus_with_fmri.csv:
    #   visual QC pass on all 3 runs + mean FD < 0.5 mm + ≥6 valid image–face pairs
    qc_cons = df_fc.get("qc_conservative", pd.Series(0, index=df_fc.index)) == 1

    # 4. Conservative fMRI sample — no diary requirement (Analysis 03)
    #    Use get_samples with require_diary=False
    _, cons_fmri = get_samples(df_fc, require_diary=False)
    rows.append(describe_sample(cons_fmri, "C5PAGE", "4_neuro_fmri_conservative"))
    print(f"  4. Conservative fMRI (no diary): N={len(cons_fmri)}")

    # 5. Primary analytic sample — diary + conservative fMRI + FC (N~80)
    #    get_samples with check_fc_col requires FC data present
    PRIMARY_FC = "l_amyg-ant_vmPFC_neg_vs_neu"
    _, primary = get_samples(df_fc, check_fc_col=PRIMARY_FC)
    rows.append(describe_sample(primary, "C5PAGE", "5_primary"))
    print(f"  5. Primary (diary+fMRI+FC): N={len(primary)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = pd.DataFrame(rows).set_index("sample")
    out.to_csv(OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(out.to_string())


if __name__ == "__main__":
    main()
