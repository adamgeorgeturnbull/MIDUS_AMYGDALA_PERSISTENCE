#!/usr/bin/env python3
"""
05_fc_persistence.py

Primary analysis: left amygdala–vmPFC functional connectivity (neg − neu contrast)
and amygdala negative affect persistence.

Directional hypothesis: greater amygdala–vmPFC connectivity specifically during
negative relative to neutral stimuli → lower cross-run persistence of negative
affect responses, reflecting vmPFC regulatory dampening of sustained amygdala
threat processing. Tested for:
  - Anterior vmPFC (safety/extinction signalling)
  - Posterior vmPFC (threat regulation)

Primary analysis (conservative fMRI sample, no diary required):
  Predictors : l_amyg-ant_vmPFC_neg_vs_neu, l_amyg-post_vmPFC_neg_vs_neu
               (Fisher z, LSS)
  Outcome    : neg_persist_crossrun_mean_z_L  (Fisher z, left hemisphere)
  Methods    : Pearson correlation, OLS regression, MLM (random intercept for family)
  Tests      : One-tailed (pre-registered directional hypothesis)

Full sample results saved to full_sample/.
Sensitivity analyses in 05_sensitivity.py.
Exploratory ant−post contrast in 05ex_fc_persistence_antpost.py.

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
OUT_DIR = RESULTS_DIR / "05_fc_persistence"

PREDICTORS = [
    "l_amyg-ant_vmPFC_neg_vs_neu",
    "l_amyg-post_vmPFC_neg_vs_neu",
]
OUTCOMES = ["neg_persist_crossrun_mean_z_L"]

# Pre-registered directional hypothesis: higher differential FC → lower persistence
EXPECTED_DIRECTIONS = {
    "neg_persist_crossrun_mean_z_L": -1,
}


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 05: FC (neg−neu) → Persistence  (Primary, one-tailed)")
    print("=" * 70)

    df = load_master(fc=True)
    prepare_persistence_vars(df)
    full, cons = get_samples(df, check_fc_col=PREDICTORS[0], require_diary=False)
    print(f"  Full N = {len(full)} | Conservative N = {len(cons)}")

    # No diary covariates — purely neuroimaging measures
    base_covs = get_covariates(cons)

    # -- Primary: conservative sample, one-tailed --
    corr, ols, mlm = run_analysis_set(
        cons, PREDICTORS, OUTCOMES, base_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, mlm, OUT_DIR,
                 label="05 FC (neg−neu) → Persistence  [conservative, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs,
                 n=len(cons), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -- Full sample archive --
    base_covs_full = get_covariates(full)
    corr_f, ols_f, mlm_f = run_analysis_set(
        full, PREDICTORS, OUTCOMES, base_covs_full,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr_f, ols_f, mlm_f, OUT_DIR / "full_sample",
                 label="05 FC (neg−neu) → Persistence  [full sample, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs_full,
                 n=len(full), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
