#!/usr/bin/env python3
"""
03_persistence_age.py

Extension: age-related differences in amygdala persistence to negative images.

Primary analysis (conservative fMRI sample):
  Predictor  : C5PAGE (age at neuroscience visit)
  Outcome    : neg_persist_crossrun_mean_z_L  (Fisher z, left hemisphere)
  Methods    : Pearson correlation, OLS regression, MLM (random intercept for family)
  Tests      : One-tailed (older age → lower persistence; prespecified direction)

Adjusted-model covariates:
  sex and race_2 through race_6; White (race == 1) is the reference category.
  C5PAGE is the predictor and is excluded from covariates by the fitting helpers.
  No diary covariates apply. Covariates with no complete-case variation are omitted.
  OLS additionally uses available twin-pair indicators, excluding singletons;
  MLM uses a family random intercept instead of twin-pair indicators.
  Pearson correlations are unadjusted.

Conservative criterion (qc_conservative == 1):
  - historical visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (six left-amygdala negative cross-run pairs)

Negative-persistence availability is required; diary participation is not.
Each method applies its own model-variable complete-case restriction.
The conservative result is primary. Results before conservative QC restriction
are archived in full_sample/.

Sensitivity analyses (right hemisphere and alternative persistence measures)
are in 03_sensitivity.py.

Inputs:
  data/processed/midus_with_fmri.csv (from preprocessing scripts 08-09)

Outputs: results/tables/03_persistence_age/
  correlations.csv, regressions.csv, mlm.csv, _methods.txt
  Corresponding broader-sample outputs are saved under full_sample/.

Run from project root directory.
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
OUT_DIR = RESULTS_DIR / "03_persistence_age"

PREDICTORS = ["C5PAGE"]
OUTCOMES   = ["neg_persist_crossrun_mean_z_L"]

# Older age → reduced negative persistence
EXPECTED_DIRECTIONS = {"neg_persist_crossrun_mean_z_L": -1}


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 03: Age → Persistence  (Confirmatory Extension)")
    print("=" * 70)

    df = load_master()
    prepare_persistence_vars(df)
    full, cons = get_samples(df, require_diary=False)
    print(f"  Full N = {len(full)} | Conservative N = {len(cons)}")

    # No diary covariates — purely neuroscience measures
    base_covs = get_covariates(cons)

    # -- Primary: conservative sample --
    corr, ols, mlm = run_analysis_set(cons, PREDICTORS, OUTCOMES, base_covs,
                                      one_tailed=True,
                                      expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr, ols, mlm, OUT_DIR,
                 label="03 Age → Persistence  [conservative, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs,
                 n=len(cons), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -- Full sample archive --
    base_covs_full = get_covariates(full)
    corr_f, ols_f, mlm_f = run_analysis_set(full, PREDICTORS, OUTCOMES, base_covs_full,
                                             one_tailed=True,
                                             expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr_f, ols_f, mlm_f, OUT_DIR / "full_sample",
                 label="03 Age → Persistence  [full sample, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs_full,
                 n=len(full), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
