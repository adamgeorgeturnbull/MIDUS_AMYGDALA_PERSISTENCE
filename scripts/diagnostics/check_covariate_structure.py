#!/usr/bin/env python3
"""
check_covariate_structure.py

Reports the covariate structure (race dummies, twin dummies, diary covariates)
for each primary analytic sample, including which are retained vs dropped by
the zero-variance and singleton-twin filters in OLS and MLM.

Run from project root directory.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from analysis_utils import (
    load_master, get_samples, get_covariates, prepare_persistence_vars,
)

DIARY_COVS = ["time_P2_P5", "n_days_complete"]


def report_covariates(name, df, base_covs, outcome="PA_score", predictor="neg_persist_crossrun_mean_z_L", add_diary_covs=True):
    print(f"\n{'─' * 70}")
    print(f"  Sample: {name}  (N={len(df)})")
    print(f"{'─' * 70}")

    if add_diary_covs:
        diary_covs = [c for c in DIARY_COVS if c in df.columns and c not in base_covs]
    else:
        diary_covs = []
    covs = base_covs + diary_covs

    # Exclude twin dummies from dropna cols — they are sparse and would drop
    # participants who don't belong to any twin pair (all zeros across dummies)
    non_twin_covs = [c for c in covs if not c.startswith("twin_pair_")]
    cols = list(dict.fromkeys([outcome, predictor] + non_twin_covs))  # deduplicate preserving order
    cols = [c for c in cols if c in df.columns]
    data = df[cols].dropna()
    # Add twin dummies back for reporting (using full df restricted to dropna rows)
    twin_cols_present = [c for c in covs if c.startswith("twin_pair_") and c in df.columns]
    data_with_twins = df.loc[data.index, twin_cols_present] if twin_cols_present else pd.DataFrame(index=data.index)
    print(f"  N after dropna on [{outcome}, {predictor}, covariates]: {len(data)}")

    race_cols = [c for c in covs if c.startswith("race_")]
    other_cols = [c for c in covs if not c.startswith("race_") and not c.startswith("twin_pair_")]

    print(f"\n  Other covariates: {other_cols}")

    print(f"\n  Race dummies ({len(race_cols)} total):")
    for c in race_cols:
        if c not in data.columns:
            print(f"    {c}: NOT IN DATA")
            continue
        n = int(data[c].sum())
        std = data[c].std()
        retained_ols = std > 0
        retained_mlm = std > 0  # same filter for MLM
        status = "retained" if retained_ols else "DROPPED (zero variance)"
        print(f"    {c}: n={n}, std={std:.3f}  -> {status}")

    print(f"\n  Twin pair dummies ({len(twin_cols_present)} total):")
    n_pairs = sum(1 for c in twin_cols_present if data_with_twins[c].sum() >= 2)
    n_singletons = sum(1 for c in twin_cols_present if data_with_twins[c].sum() == 1)
    n_absent = sum(1 for c in twin_cols_present if data_with_twins[c].sum() == 0)
    print(f"    Proper pairs (sum>=2, retained in OLS):  {n_pairs}")
    print(f"    Singletons  (sum==1, dropped from OLS):  {n_singletons}")
    print(f"    Absent      (sum==0, dropped from OLS):  {n_absent}")
    print(f"    MLM: ALL twin dummies excluded (random effect used instead)")


def main():
    print("=" * 70)
    print("Covariate structure across primary analytic samples")
    print("=" * 70)

    # --- Full diary sample (Analysis 01) ---
    df_diary = pd.read_csv("data/processed/midus_merged_clean.csv")
    df_diary["M2ID"] = df_diary["M2ID"].astype(str)
    has_affect = df_diary[["PA_score", "NA_score"]].notna().any(axis=1)
    diary_full = df_diary[has_affect].copy()
    race_cols = [c for c in diary_full.columns if c.startswith("race_")]
    twin_cols = [c for c in diary_full.columns if c.startswith("twin_pair_")]
    covs_01_full = ["sex"] + race_cols + twin_cols + ["n_days_complete"]
    covs_01_full = [c for c in covs_01_full if c in diary_full.columns]
    report_covariates("01 Full diary (age→affect)", diary_full, covs_01_full,
                      outcome="PA_score", predictor="C2PAGE", add_diary_covs=False)

    # --- Neuroscience subsample (Analysis 01 neuro) ---
    neuro = df_diary[df_diary["C5PAGE"].notna() & has_affect].copy()
    covs_01_neuro = ["sex"] + race_cols + twin_cols + ["n_days_complete", "time_P2_P5"]
    covs_01_neuro = [c for c in covs_01_neuro if c in neuro.columns]
    report_covariates("01 Neuro subsample (age→affect)", neuro, covs_01_neuro,
                      outcome="PA_score", predictor="C5PAGE", add_diary_covs=False)

    # --- Conservative diary+fMRI (Analyses 02, 04, 06, 07) ---
    df = load_master()
    prepare_persistence_vars(df)
    _, cons = get_samples(df)
    base_covs = get_covariates(cons)
    report_covariates("02/06 Conservative diary+fMRI (N~80)", cons, base_covs,
                      outcome="PA_score", predictor="neg_persist_crossrun_mean_z_L")

    # --- Conservative fMRI only (Analysis 03) ---
    _, cons3 = get_samples(df, require_diary=False)
    base_covs3 = get_covariates(cons3)
    report_covariates("03 Conservative fMRI only (N~128)", cons3, base_covs3,
                      outcome="neg_persist_crossrun_mean_z_L", predictor="C5PAGE",
                      add_diary_covs=False)

    # --- Conservative diary+fMRI with FC (Analyses 04, 05, 07) ---
    df_fc = load_master(fc=True)
    prepare_persistence_vars(df_fc)
    _, cons_fc = get_samples(df_fc, check_fc_col="l_amyg-ant_vmPFC_neg_vs_neu")
    base_covs_fc = get_covariates(cons_fc)
    report_covariates("04/07 Conservative diary+fMRI+FC (N~80)", cons_fc, base_covs_fc,
                      outcome="PA_score", predictor="l_amyg-ant_vmPFC_neg_vs_neu")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
