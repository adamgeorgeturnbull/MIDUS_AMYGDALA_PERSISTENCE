#!/usr/bin/env python3
"""
04_fc_affect.py

Primary analysis: left amygdala–vmPFC functional connectivity (neg − neu contrast)
and daily life affect.

Directional hypothesis: greater left-amygdala–vmPFC connectivity during negative
relative to neutral stimuli → higher PA and lower NA in daily life.
Both anterior and posterior vmPFC connections are tested.

Primary analysis (final fMRI + diary + FC sample):
  Predictors : l_amyg-ant_vmPFC_neg_vs_neu, l_amyg-post_vmPFC_neg_vs_neu
  Outcomes   : PA_score, NA_score, NA_score_log (daily diary)
  Methods    : Pearson correlation, OLS regression, MLM (family random intercept)
  Tests      : One-tailed (higher FC → higher PA, lower raw/log NA)

Adjusted-model covariates:
  C5PAGE, sex, race_2 through race_6 (White is the reference category),
  time_P2_P5 and n_days_complete. Zero-variance covariates are omitted.
  OLS additionally uses available twin-pair indicators, excluding singletons;
  MLM uses a family random intercept instead. Correlations are unadjusted.

Sample selection requires negative-persistence availability, diary affect,
nonmissing anterior negative-minus-neutral FC, and qc_conservative == 1.
The QC flag combines visual QC, mean FD < 0.5 mm and six negative-persistence
cross-run pairs. Each method then applies model-specific complete-case selection.

Full-sample results before QC restriction are archived in full_sample/.
Sensitivity analyses are in 04_sensitivity.py. The anterior-minus-posterior
contrast is a separate exploratory analysis in 04ex_fc_affect_antpost.py;
it is not a predictor in this script.

Inputs:
  data/processed/midus_with_fmri.csv and the FC file selected by load_master(fc=True).

Outputs: results/tables/04_fc_affect/
  correlations.csv, regressions.csv, mlm.csv, _methods.txt
  Corresponding broader-sample outputs are saved under full_sample/.

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

# Directional hypotheses apply to both regional FC predictors
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
    print(f"  Full N = {len(full)} | Final N = {len(cons)}")

    base_covs = get_covariates(cons)

    # -- Primary: final sample, one-tailed --
    corr, ols, mlm = run_analysis_set(
        cons, PREDICTORS, OUTCOMES, base_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, mlm, OUT_DIR,
                 label="04 FC (neg−neu) → Affect  [final, one-tailed]",
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
