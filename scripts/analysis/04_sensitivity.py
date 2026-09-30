#!/usr/bin/env python3
"""
04_sensitivity.py (M3)

Sensitivity analyses for Analysis 04: FC → Affect.

Sections (anterior and posterior vmPFC connections in each):
  1. Right-amygdala neg−neu FC → diary affect (two-tailed)
     → sensitivity_right_amygdala/
  2. Left-amygdala pos−neu FC → diary affect (two-tailed)
     → sensitivity_pos_vs_neu/
  3. Left-amygdala neg−neu FC → PANAS (one-tailed)
     → sensitivity_panas/
  4. Left-amygdala negative-only FC → diary affect (one-tailed)
     → sensitivity_neg_condition/

Diary and PANAS outcomes each include PA, raw NA, and log NA.
Directional tests expect positive PA and negative NA associations; raw and
log NA use the same direction and covariates.

Methods: Pearson correlation, OLS, and family-random-intercept MLM.

Adjusted covariates:
  C5PAGE, sex, and race_2 – race_6 (White is the reference).
  Diary models additionally include time_P2_P5 and n_days_complete.
  PANAS models omit diary covariates and do not require diary participation.
  OLS includes available twin-pair indicators; MLM uses family grouping instead.

QC criteria (qc_conservative == 1):
  - visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (n_pairs == 6)

Interpret sensitivities in relation to significant primary findings; otherwise
report for transparency.

Each output directory contains: correlations.csv, regressions.csv, mlm.csv, _methods.txt

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

BASE_DIR = RESULTS_DIR / "04_fc_affect"
OUTCOMES_DIARY = ["PA_score", "NA_score", "NA_score_log"]

EXPECTED_DIRECTIONS = {
    "PA_score":    +1,
    "NA_score":    -1,
    "NA_score_log": -1,
    "C5SPGP":      +1,
    "C5SPGN":      -1,
    "C5SPGN_log":  -1,
}


def main():
    print("=" * 70)
    print("Analysis 04 — Sensitivity Analyses")
    print("=" * 70)

    df = load_master(fc=True)

    # Compute pos−neu contrasts from individual condition columns
    if "l_amyg-ant_vmPFC_pos" in df.columns and "l_amyg-ant_vmPFC_neu" in df.columns:
        df["l_amyg-ant_vmPFC_pos_vs_neu"]  = df["l_amyg-ant_vmPFC_pos"]  - df["l_amyg-ant_vmPFC_neu"]
        df["l_amyg-post_vmPFC_pos_vs_neu"] = df["l_amyg-post_vmPFC_pos"] - df["l_amyg-post_vmPFC_neu"]

    full, cons = get_samples(df, check_fc_col="l_amyg-ant_vmPFC_neg_vs_neu")
    print(f"  Final N (diary+fMRI) = {len(cons)}")
    base_covs = get_covariates(cons)

    # -------------------------------------------------------------------------
    # 1. Right amygdala neg−neu (hemisphere specificity)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Right Amygdala (neg−neu) ---")
    right_preds = [p for p in [
        "r_amyg-ant_vmPFC_neg_vs_neu",
        "r_amyg-post_vmPFC_neg_vs_neu",
    ] if p in cons.columns]

    if right_preds:
        corr, ols, mlm = run_analysis_set(
            cons, right_preds, OUTCOMES_DIARY, base_covs,
        )
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_right_amygdala",
                     label="04 FC-R (neg−neu) → Affect  [final, two-tailed]",
                     predictors=right_preds, outcomes=OUTCOMES_DIARY,
                     covariates=base_covs, n=len(cons))
    else:
        print("  No right amygdala neg−neu variables found — skipping.")

    # -------------------------------------------------------------------------
    # 2. pos−neu contrast (condition specificity)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: pos−neu contrast ---")
    pos_neu_preds = [p for p in [
        "l_amyg-ant_vmPFC_pos_vs_neu",
        "l_amyg-post_vmPFC_pos_vs_neu",
    ] if p in cons.columns]

    if pos_neu_preds:
        corr, ols, mlm = run_analysis_set(
            cons, pos_neu_preds, OUTCOMES_DIARY, base_covs,
        )
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_pos_vs_neu",
                     label="04 FC (pos−neu) → Affect  [final, two-tailed]",
                     predictors=pos_neu_preds, outcomes=OUTCOMES_DIARY,
                     covariates=base_covs, n=len(cons))
    else:
        print("  pos−neu FC variables not found — skipping.")
        print("  (pos and neu condition columns required from runBStaskFC_LSS.sh)")

    # -------------------------------------------------------------------------
    # 3. PANAS (convergent validity)
    #    Uses full final fMRI sample (no diary requirement)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: PANAS ---")
    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1
    has_qc      = df.get("qc_conservative",    pd.Series(0, index=df.index)) == 1
    fc_col      = "l_amyg-ant_vmPFC_neg_vs_neu"
    panas_cons  = df[has_persist & has_qc & df[fc_col].notna()].copy()
    panas_covs = get_covariates(panas_cons)
    print(f"  PANAS N (final fMRI) = {len(panas_cons)}")

    panas_preds = ["l_amyg-ant_vmPFC_neg_vs_neu", "l_amyg-post_vmPFC_neg_vs_neu"]
    panas_outcomes = ["C5SPGP", "C5SPGN", "C5SPGN_log"]
    corr, ols, mlm = run_analysis_set(
        panas_cons, panas_preds, panas_outcomes, panas_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, mlm, BASE_DIR / "sensitivity_panas",
                 label="04 FC (neg−neu) → PANAS  [final fMRI, one-tailed]",
                 predictors=panas_preds, outcomes=panas_outcomes,
                 covariates=panas_covs, n=len(panas_cons),
                 one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -------------------------------------------------------------------------
    # 4. Neg condition only (robustness — original primary operationalisation)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Neg condition only ---")
    neg_preds = [p for p in [
        "l_amyg-ant_vmPFC_neg",
        "l_amyg-post_vmPFC_neg",
    ] if p in cons.columns]

    if neg_preds:
        corr, ols, mlm = run_analysis_set(
            cons, neg_preds, OUTCOMES_DIARY, base_covs,
            one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
        )
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_neg_condition",
                     label="04 FC (neg only) → Affect  [final, one-tailed]",
                     predictors=neg_preds, outcomes=OUTCOMES_DIARY,
                     covariates=base_covs, n=len(cons),
                     one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)
    else:
        print("  Neg condition FC variables not found — skipping.")


if __name__ == "__main__":
    main()
