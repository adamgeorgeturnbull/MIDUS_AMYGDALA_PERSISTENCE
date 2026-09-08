#!/usr/bin/env python3
"""
02_sensitivity.py (MR1)

Sensitivity analyses for Analysis 02: Persistence → Affect.

Sections:
  1. Right hemisphere negative persistence → sensitivity_right_hemisphere/
     Hemispheric specificity check; two-tailed inference.
  2. PANAS (convergent validity)           → sensitivity_panas/
     One-tailed inference; same directional hypotheses as primary.
     Does not require diary participation.

Both sections use the conservative QC criterion (qc_conservative == 1):
  - historical visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (n_pairs == 6)
  - all three task runs present with exactly 231 volumes

Methods: Pearson correlation and participant-level OLS (available MR1 P5
analytic inputs do not contain a usable family or other grouping identifier).

Each output directory contains: correlations.csv, regressions.csv, _methods.txt

Do not condition execution on whether the primary analysis is significant.

Run from MR1_validation/ directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS_DIR, load_master, get_samples, get_covariates,
    prepare_persistence_vars, run_analysis_set, save_results,
)

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = RESULTS_DIR / "02_persistence_affect"

RIGHT_PREDICTORS = ["neg_persist_crossrun_mean_z_R"]
DIARY_OUTCOMES   = ["PA_score", "NA_score"]

PANAS_PREDICTORS = ["neg_persist_crossrun_mean_z_L"]
PANAS_OUTCOMES   = ["RA5SPGP", "RA5SPGN", "RA5SPGN_log"]
PANAS_EXPECTED   = {"RA5SPGP": -1, "RA5SPGN": +1, "RA5SPGN_log": +1}


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 02 (MR1) — Sensitivity Analyses")
    print("=" * 70)

    df = load_master(fc=False)
    prepare_persistence_vars(df)
    _, cons = get_samples(df)
    print(f"  Conservative N (diary+fMRI) = {len(cons)}")
    base_covs = get_covariates(cons)

    # ── 1. Right hemisphere (two-tailed hemispheric specificity check) ─────────
    print("\n--- Sensitivity: Right Hemisphere ---")
    corr, ols = run_analysis_set(
        cons, RIGHT_PREDICTORS, DIARY_OUTCOMES, base_covs,
    )
    save_results(corr, ols, BASE_DIR / "sensitivity_right_hemisphere",
                 label="02 Persistence (R) → Affect  [conservative, two-tailed]",
                 predictors=RIGHT_PREDICTORS, outcomes=DIARY_OUTCOMES,
                 covariates=base_covs, n=len(cons))

    # ── 2. PANAS (one-tailed convergent validity) ──────────────────────────────
    print("\n--- Sensitivity: PANAS ---")
    _, panas_cons = get_samples(df, require_diary=False)
    panas_covs = get_covariates(panas_cons)
    print(f"  PANAS N (conservative fMRI) = {len(panas_cons)}")
    corr, ols = run_analysis_set(
        panas_cons, PANAS_PREDICTORS, PANAS_OUTCOMES, panas_covs,
        one_tailed=True, expected_directions=PANAS_EXPECTED,
    )
    save_results(corr, ols, BASE_DIR / "sensitivity_panas",
                 label="02 Persistence (L) → PANAS  [conservative fMRI, one-tailed]",
                 predictors=PANAS_PREDICTORS, outcomes=PANAS_OUTCOMES,
                 covariates=panas_covs, n=len(panas_cons),
                 one_tailed=True, expected_directions=PANAS_EXPECTED)


if __name__ == "__main__":
    main()
