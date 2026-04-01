#!/usr/bin/env python3
"""
04_fc_affect.py

Primary analysis: left amygdala–vmPFC functional connectivity (neg − neu contrast)
and daily life affect.

Directional hypothesis: greater amygdala–vmPFC connectivity specifically during
negative relative to neutral stimuli → higher PA and lower NA in daily life,
reflecting vmPFC regulatory modulation of amygdala threat responses. Tested for:
  - Anterior vmPFC (safety/extinction signalling)
  - Posterior vmPFC (threat regulation)
  - Ant minus post difference (safety-biased connectivity)

Primary analysis (conservative fMRI sample):
  Predictors : l_amyg-ant_vmPFC_neg_vs_neu, l_amyg-post_vmPFC_neg_vs_neu,
               l_amyg-ant_minus_post_vmPFC_neg_vs_neu  (Fisher z; last is computed)
  Outcomes   : PA_score, NA_score, NA_score_log  (daily diary)
  Methods    : Pearson correlation, OLS regression, MLM (random intercept for family)
  Tests      : One-tailed (pre-registered directional hypothesis)

Full sample results saved to full_sample/.
Sensitivity analyses in 04_sensitivity.py.

Run from project root directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS_DIR, load_master, get_samples, get_covariates,
    run_analysis_set, save_results,
)

# ============================================================================
# Configuration
# ============================================================================
OUT_DIR = RESULTS_DIR / "04_fc_affect"

PREDICTORS = [
    "l_amyg-ant_vmPFC_neg_vs_neu",
    "l_amyg-post_vmPFC_neg_vs_neu",
]
OUTCOMES = ["PA_score", "NA_score", "NA_score_log"]

# Pre-registered directional hypotheses (apply to all three predictors)
EXPECTED_DIRECTIONS = {
    "PA_score":     +1,   # higher differential FC → higher PA
    "NA_score":     -1,   # higher differential FC → lower NA
    "NA_score_log": -1,
}


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 04: FC (neg − neu contrast) → Affect  (Primary, one-tailed)")
    print("=" * 70)

    df = load_master(fc=True)
    full, cons = get_samples(df, check_fc_col=PREDICTORS[0])
    print(f"  Full N = {len(full)} | Conservative N = {len(cons)}")

    base_covs = get_covariates(cons)

    # -- Primary: conservative sample, one-tailed --
    corr, ols, mlm = run_analysis_set(
        cons, PREDICTORS, OUTCOMES, base_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, mlm, OUT_DIR,
                 label="04 FC (neg−neu) → Affect  [conservative, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs,
                 n=len(cons), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -- Full sample archive --
    base_covs_full = get_covariates(full)
    corr_f, ols_f, mlm_f = run_analysis_set(
        full, PREDICTORS, OUTCOMES, base_covs_full,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr_f, ols_f, mlm_f, OUT_DIR / "full_sample",
                 label="04 FC (neg−neu) → Affect  [full sample, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs_full,
                 n=len(full), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
