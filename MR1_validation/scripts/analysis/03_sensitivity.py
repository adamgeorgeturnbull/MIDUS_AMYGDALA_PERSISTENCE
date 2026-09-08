#!/usr/bin/env python3
"""
03_sensitivity.py (MR1)

Sensitivity analyses for Analysis 03: Age → Persistence.

Sections:
  1. Age → right-amygdala negative persistence  → sensitivity_right_hemisphere/
     Evaluates hemispheric generalization; two-tailed inference.
  2. Age → bilateral positive persistence        → sensitivity_positive_persistence/
     Evaluates valence specificity; two-tailed inference.

These are predefined sensitivity/extension analyses.  They are not substitutes
for the direct left-negative-persistence age target (Analysis 03 primary), nor
are they selected based on provisional MR1 significance.

Inference:
  Both sections use two-tailed inference, matching the corrected M3 sensitivity
  logic.  Nominal p-values are reported without multiplicity adjustment.
  Significance in one hemisphere and not the other is not evidence that the
  hemispheric coefficients differ; a direct coefficient comparison would be
  required to support that claim.

Corrected vmPFC persistence and concatenated persistence are not produced by
the MR1 pipeline and are therefore not analyzed here.

Participant-level OLS is used because the available MR1 P5 analytic inputs do
not contain a usable family or other grouping identifier.

Adjusted OLS covariates:
  sex, race_2 – race_6  (RA5PAGE is the predictor; no diary covariates apply)

Conservative criterion (qc_conservative == 1):
  - historical visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (n_pairs == 6)
  - all three task runs present with exactly 231 volumes

No diary participation is required.

Required persistence variables are validated by run_analysis_set; the script
aborts if any are absent.  No section is gated on the Analysis 03 primary result.

Each output directory contains: correlations.csv, regressions.csv, _methods.txt

Privacy: aggregate console output only — never prints participant IDs or rows.

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
BASE_DIR = RESULTS_DIR / "03_persistence_age"

PREDICTORS = ["RA5PAGE"]

RIGHT_NEG_OUTCOMES = [
    "neg_persist_crossrun_mean_z_R",
]

POSITIVE_OUTCOMES = [
    "pos_persist_crossrun_mean_z_L",
    "pos_persist_crossrun_mean_z_R",
]


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 03 (MR1) — Sensitivity Analyses")
    print("=" * 70)

    df = load_master(fc=False)
    prepare_persistence_vars(df)
    _, cons = get_samples(df, require_diary=False)
    print(f"  Conservative N = {len(cons)}")

    # RA5PAGE is the predictor — exclude it from covariates.
    # No diary covariates apply (outcomes are neuroimaging measures).
    base_covs = [c for c in get_covariates(cons) if c not in PREDICTORS]

    # ── 1. Right hemisphere negative persistence (two-tailed) ─────────────────
    print("\n--- Sensitivity: Right Hemisphere ---")
    corr, ols = run_analysis_set(cons, PREDICTORS, RIGHT_NEG_OUTCOMES, base_covs)
    save_results(corr, ols, BASE_DIR / "sensitivity_right_hemisphere",
                 label="03 Age → Persistence (R)  [conservative, two-tailed]",
                 predictors=PREDICTORS, outcomes=RIGHT_NEG_OUTCOMES,
                 covariates=base_covs, n=len(cons))

    # ── 2. Positive persistence, bilateral (two-tailed) ────────────────────────
    print("\n--- Sensitivity: Positive Persistence ---")
    corr, ols = run_analysis_set(cons, PREDICTORS, POSITIVE_OUTCOMES, base_covs)
    save_results(corr, ols, BASE_DIR / "sensitivity_positive_persistence",
                 label="03 Age → Positive Persistence  [conservative, two-tailed]",
                 predictors=PREDICTORS, outcomes=POSITIVE_OUTCOMES,
                 covariates=base_covs, n=len(cons))


if __name__ == "__main__":
    main()
