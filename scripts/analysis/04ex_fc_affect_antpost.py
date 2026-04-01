#!/usr/bin/env python3
"""
04ex_fc_affect_antpost.py

EXPLORATORY follow-up to Analysis 04.

Tests whether the anterior-minus-posterior vmPFC connectivity contrast
(neg − neu condition) predicts trait affect. This contrast was not
pre-registered and is run as an exploratory analysis after the primary
anterior/posterior ROI results. Interpret with caution.

The ant−post contrast captures safety-biased connectivity: relatively
greater coupling of the amygdala with anterior (extinction/safety) vs
posterior (threat) vmPFC during negative stimuli.

Predictors : l_amyg-ant_minus_post_vmPFC_neg_vs_neu  (computed; left)
             r_amyg-ant_minus_post_vmPFC_neg_vs_neu  (right, if available)
Outcomes   : PA_score, NA_score  (daily diary)
             C5SPGP, C5SPGN, C5SPGN_log  (PANAS, full fMRI sample)
Methods    : Pearson correlation, OLS regression, MLM
Tests      : One-tailed (directional hypothesis: safety bias → higher PA, lower NA)

Outputs:
  results/tables/04ex_fc_affect_antpost/diary/
  results/tables/04ex_fc_affect_antpost/panas/

Run from project root directory.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS_DIR, load_master, get_samples, get_covariates,
    run_analysis_set, save_results,
)

OUT_DIR = RESULTS_DIR / "04ex_fc_affect_antpost"
OUTCOMES_DIARY = ["PA_score", "NA_score"]
OUTCOMES_PANAS = ["C5SPGP", "C5SPGN", "C5SPGN_log"]

# Directional hypothesis: safety-biased connectivity (ant > post) →
# higher PA and lower NA
EXPECTED_DIRECTIONS = {
    "PA_score":   +1,
    "NA_score":   -1,
    "C5SPGP":     +1,
    "C5SPGN":     -1,
    "C5SPGN_log": -1,
}


def main():
    print("=" * 70)
    print("Analysis 04ex: FC ant−post contrast → Affect  (Exploratory)")
    print("=" * 70)

    df = load_master(fc=True)

    # Compute ant−post contrasts (neg−neu)
    df["l_amyg-ant_minus_post_vmPFC_neg_vs_neu"] = (
        df["l_amyg-ant_vmPFC_neg_vs_neu"] - df["l_amyg-post_vmPFC_neg_vs_neu"]
    )
    if "r_amyg-ant_vmPFC_neg_vs_neu" in df.columns and "r_amyg-post_vmPFC_neg_vs_neu" in df.columns:
        df["r_amyg-ant_minus_post_vmPFC_neg_vs_neu"] = (
            df["r_amyg-ant_vmPFC_neg_vs_neu"] - df["r_amyg-post_vmPFC_neg_vs_neu"]
        )

    full, cons = get_samples(df, check_fc_col="l_amyg-ant_vmPFC_neg_vs_neu")
    print(f"  Conservative N (diary+fMRI) = {len(cons)}")
    base_covs = get_covariates(cons)

    diary_preds = [p for p in [
        "l_amyg-ant_minus_post_vmPFC_neg_vs_neu",
        "r_amyg-ant_minus_post_vmPFC_neg_vs_neu",
    ] if p in cons.columns]

    # -------------------------------------------------------------------------
    # Diary affect (conservative sample, one-tailed)
    # -------------------------------------------------------------------------
    print("\n--- Diary affect ---")
    corr, ols, mlm = run_analysis_set(
        cons, diary_preds, OUTCOMES_DIARY, base_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, mlm, OUT_DIR / "diary",
                 label="04ex FC ant−post (neg−neu) → Diary Affect  [conservative, one-tailed]",
                 predictors=diary_preds, outcomes=OUTCOMES_DIARY,
                 covariates=base_covs, n=len(cons),
                 one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -------------------------------------------------------------------------
    # PANAS (full conservative fMRI sample, two-tailed)
    # -------------------------------------------------------------------------
    print("\n--- PANAS ---")
    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1
    has_qc      = df.get("qc_conservative",    pd.Series(0, index=df.index)) == 1
    fc_col      = "l_amyg-ant_vmPFC_neg_vs_neu"
    panas_cons  = df[has_persist & has_qc & df[fc_col].notna()].copy()
    panas_cons["l_amyg-ant_minus_post_vmPFC_neg_vs_neu"] = (
        panas_cons["l_amyg-ant_vmPFC_neg_vs_neu"] - panas_cons["l_amyg-post_vmPFC_neg_vs_neu"]
    )
    if "r_amyg-ant_vmPFC_neg_vs_neu" in panas_cons.columns:
        panas_cons["r_amyg-ant_minus_post_vmPFC_neg_vs_neu"] = (
            panas_cons["r_amyg-ant_vmPFC_neg_vs_neu"] - panas_cons["r_amyg-post_vmPFC_neg_vs_neu"]
        )
    panas_covs = get_covariates(panas_cons)
    print(f"  PANAS N (conservative fMRI) = {len(panas_cons)}")

    panas_preds = [p for p in [
        "l_amyg-ant_minus_post_vmPFC_neg_vs_neu",
        "r_amyg-ant_minus_post_vmPFC_neg_vs_neu",
    ] if p in panas_cons.columns]

    corr, ols, mlm = run_analysis_set(
        panas_cons, panas_preds, OUTCOMES_PANAS, panas_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, mlm, OUT_DIR / "panas",
                 label="04ex FC ant−post (neg−neu) → PANAS  [conservative fMRI, one-tailed]",
                 predictors=panas_preds, outcomes=OUTCOMES_PANAS,
                 covariates=panas_covs, n=len(panas_cons),
                 one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
