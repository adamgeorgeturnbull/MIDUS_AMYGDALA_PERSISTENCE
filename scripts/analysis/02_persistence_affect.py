#!/usr/bin/env python3
"""
02_persistence_affect.py

Primary analysis: left amygdala negative persistence and daily life affect.

Primary analysis (conservative fMRI sample):
  Predictors : neg_persist_crossrun_mean_z_L  (Fisher z, left hemisphere)
  Outcomes   : PA_score, NA_score  (daily diary)
  Methods    : Pearson correlation, OLS regression, MLM (random intercept for family)
  Tests      : One-tailed (confirmatory replication, pre-registered directional hypothesis)

Full sample results saved to full_sample/.
Sensitivity analyses (right hemisphere, PANAS, other persistence) in 02_sensitivity.py.

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
OUT_DIR = RESULTS_DIR / "02_persistence_affect"

PREDICTORS = ["neg_persist_crossrun_mean_z_L"]
OUTCOMES   = ["PA_score", "NA_score", "NA_score_log"]

# Greater negative persistence → lower PA, higher NA
EXPECTED_DIRECTIONS = {"PA_score": -1, "NA_score": +1, "NA_score_log": +1}


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 02: Persistence → Affect  (Replication)")
    print("=" * 70)

    df = load_master()
    prepare_persistence_vars(df)
    full, cons = get_samples(df)
    print(f"  Full N = {len(full)} | Conservative N = {len(cons)}")

    base_covs = get_covariates(cons)

    # -- Primary: conservative sample --
    corr, ols, mlm = run_analysis_set(cons, PREDICTORS, OUTCOMES, base_covs,
                                      one_tailed=True,
                                      expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr, ols, mlm, OUT_DIR,
                 label="02 Persistence → Affect  [conservative]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs,
                 n=len(cons), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -- Full sample archive --
    base_covs_full = get_covariates(full)
    corr_f, ols_f, mlm_f = run_analysis_set(full, PREDICTORS, OUTCOMES, base_covs_full,
                                             one_tailed=True,
                                             expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr_f, ols_f, mlm_f, OUT_DIR / "full_sample",
                 label="02 Persistence → Affect  [full sample]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs_full,
                 n=len(full), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
