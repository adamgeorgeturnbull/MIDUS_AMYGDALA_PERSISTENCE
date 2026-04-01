#!/usr/bin/env python3
"""
00d_erq_context.py

Contextual analyses supporting interpretation of ERQ moderation findings
(analyses 06 and 07).

Sections:
  1. Reappraisal ↔ Suppression
     Describes the relationship between the two ERQ subscales.

  2. ERQ → Affect
     Do reappraisal and suppression predict trait affect?
     - Diary outcomes (PA_score, NA_score): participants with diary data
     - PANAS outcomes (C5SPGP, C5SPGN, C5SPGN_log): participants with PANAS

  3. Age → ERQ
     Does age predict reappraisal and suppression use?
     Age excluded from covariates (it is the predictor).

Sample  : full behavioral sample (all participants with relevant data in
          midus_merged_clean.csv); no fMRI QC restriction applied since no
          neuroimaging measures are involved.
Tests   : two-tailed throughout (descriptive context)

Outputs:
  results/tables/00d_erq_context/reappraisal_suppression/
  results/tables/00d_erq_context/erq_affect_diary/
  results/tables/00d_erq_context/erq_affect_panas/
  results/tables/00d_erq_context/age_erq/

Run from project root directory.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS_DIR, get_covariates, run_analysis_set, save_results,
)

DATA_FILE = Path("data/processed/midus_merged_clean.csv")
BASE_DIR  = RESULTS_DIR / "00d_erq_context"

ERQ_VARS       = ["C5SER", "C5SES"]
DIARY_OUTCOMES = ["PA_score", "NA_score", "NA_score_log"]
PANAS_OUTCOMES = ["C5SPGP", "C5SPGN", "C5SPGN_log"]


def main():
    print("=" * 70)
    print("Analysis 00d: ERQ Context")
    print("  (reappraisal/suppression ↔ affect and age)")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    df["M2ID"] = df["M2ID"].astype(str)
    print(f"  Full behavioral sample N = {len(df)}")

    # Sub-samples defined by data availability (no fMRI QC restriction)
    has_erq   = df[ERQ_VARS].notna().all(axis=1)
    has_diary = df[DIARY_OUTCOMES].notna().any(axis=1)
    has_panas = df[PANAS_OUTCOMES[:2]].notna().any(axis=1)
    has_age   = df["C5PAGE"].notna()

    erq_sample   = df[has_erq].copy()
    diary_sample = df[has_erq & has_diary].copy()
    panas_sample = df[has_erq & has_panas].copy()
    age_sample   = df[has_erq & has_age].copy()

    print(f"  N with ERQ data          = {len(erq_sample)}")
    print(f"  N with ERQ + diary       = {len(diary_sample)}")
    print(f"  N with ERQ + PANAS       = {len(panas_sample)}")

    # -------------------------------------------------------------------------
    # 1. Reappraisal ↔ Suppression
    # -------------------------------------------------------------------------
    print("\n--- Section 1: Reappraisal ↔ Suppression ---")
    covs_neuro = get_covariates(erq_sample)

    corr, ols, mlm = run_analysis_set(
        erq_sample,
        predictors=["C5SER"],
        outcomes=["C5SES"],
        base_covariates=covs_neuro,
    )
    save_results(corr, ols, mlm, BASE_DIR / "reappraisal_suppression",
                 label="00d Reappraisal → Suppression  [full sample, two-tailed]",
                 predictors=["C5SER"], outcomes=["C5SES"],
                 covariates=covs_neuro, n=len(erq_sample))

    # -------------------------------------------------------------------------
    # 2a. ERQ → Diary Affect
    # -------------------------------------------------------------------------
    print("\n--- Section 2a: ERQ → Diary Affect ---")
    covs_diary = get_covariates(diary_sample)

    corr, ols, mlm = run_analysis_set(
        diary_sample,
        predictors=ERQ_VARS,
        outcomes=DIARY_OUTCOMES,
        base_covariates=covs_diary,
    )
    save_results(corr, ols, mlm, BASE_DIR / "erq_affect_diary",
                 label="00d ERQ → Diary Affect  [ERQ+diary sample, two-tailed]",
                 predictors=ERQ_VARS, outcomes=DIARY_OUTCOMES,
                 covariates=covs_diary, n=len(diary_sample))

    # -------------------------------------------------------------------------
    # 2b. ERQ → PANAS Affect
    # -------------------------------------------------------------------------
    print("\n--- Section 2b: ERQ → PANAS Affect ---")
    covs_panas = get_covariates(panas_sample)

    corr, ols, mlm = run_analysis_set(
        panas_sample,
        predictors=ERQ_VARS,
        outcomes=PANAS_OUTCOMES,
        base_covariates=covs_panas,
    )
    save_results(corr, ols, mlm, BASE_DIR / "erq_affect_panas",
                 label="00d ERQ → PANAS Affect  [ERQ+PANAS sample, two-tailed]",
                 predictors=ERQ_VARS, outcomes=PANAS_OUTCOMES,
                 covariates=covs_panas, n=len(panas_sample))

    # -------------------------------------------------------------------------
    # 3. Age → ERQ (C5PAGE excluded from covariates)
    # -------------------------------------------------------------------------
    print("\n--- Section 3: Age → ERQ ---")
    covs_age = [c for c in get_covariates(age_sample) if c != "C5PAGE"]

    corr, ols, mlm = run_analysis_set(
        age_sample,
        predictors=["C5PAGE"],
        outcomes=ERQ_VARS,
        base_covariates=covs_age,
    )
    save_results(corr, ols, mlm, BASE_DIR / "age_erq",
                 label="00d Age → ERQ  [full sample, two-tailed]",
                 predictors=["C5PAGE"], outcomes=ERQ_VARS,
                 covariates=covs_age, n=len(age_sample))


if __name__ == "__main__":
    main()
