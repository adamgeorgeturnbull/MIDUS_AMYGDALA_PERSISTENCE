#!/usr/bin/env python3
"""
04_fc_affect.py (MR1)

Primary analysis: left amygdala–vmPFC FC (negative > neutral contrast) and
daily life affect.

Directional hypothesis: greater amygdala–vmPFC connectivity during negative
relative to neutral stimuli → higher PA and lower NA in daily life, reflecting
vmPFC regulatory modulation of amygdala threat responses.

Direct R4 replication target:
  Predictor : l_amyg-ant_vmPFC_neg_vs_neu  (anterior vmPFC)
  Outcome   : NA_score
  Direction : negative (higher differential FC → lower NA)
  Note      : the posterior-vmPFC predictor and remaining outcomes are part of
              the prespecified analysis family but may not be substituted for the
              anterior-vmPFC / NA_score result when judging direct replication.

Analysis family:
  Predictors : l_amyg-ant_vmPFC_neg_vs_neu, l_amyg-post_vmPFC_neg_vs_neu
  Outcomes   : PA_score, NA_score, NA_score_log  (daily diary)
  Directions : PA_score = +1, NA_score = -1, NA_score_log = -1
  Methods    : Pearson correlation and participant-level OLS
               (available MR1 P5 inputs do not contain a usable family or other
               grouping identifier; MLM is therefore not applicable)
  Tests      : One-tailed (pre-registered directional hypotheses)

Adjusted OLS covariates:
  RA5PAGE, sex, race_2 – race_6  (base; returned by get_covariates)
  time_P2_P5, n_days_complete    (diary-specific; added automatically by
                                  run_analysis_set for diary outcomes)

Sample: conservative fMRI + diary + FC sample (get_samples with
check_fc_col=PREDICTORS[0] and require_diary=True).

Conservative criterion (qc_conservative == 1):
  - historical visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (n_pairs == 6)
  - all three task runs present with exactly 231 volumes

FC availability: l_amyg-ant_vmPFC_neg_vs_neu must be nonmissing; validated by
get_samples (check_fc_col). Both FC predictors must be present in the DataFrame;
validated by run_analysis_set.

The conservative result is primary. Full-sample results (less restrictive QC)
are saved to full_sample/ as an archived secondary analysis.

Inputs:
  data/processed/mr1_with_fmri.csv  (merged with FC; scripts 08-09)

Outputs: results/tables/04_fc_affect/
  correlations.csv
  regressions.csv
  _methods.txt
  full_sample/correlations.csv
  full_sample/regressions.csv
  full_sample/_methods.txt

Privacy: prints only aggregate sample sizes — never prints participant IDs or rows.

Run from MR1_validation/ directory.
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

# Pre-registered directional hypotheses
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
    print("Analysis 04 (MR1): FC (neg−neu) → Affect  (Primary, one-tailed)")
    print("=" * 70)

    df = load_master(fc=True)
    full, cons = get_samples(df, check_fc_col=PREDICTORS[0], require_diary=True)
    print(f"  Full N = {len(full)} | Conservative N = {len(cons)}")

    base_covs = get_covariates(cons)

    # -- Primary: conservative sample, one-tailed --
    corr, ols = run_analysis_set(
        cons, PREDICTORS, OUTCOMES, base_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, OUT_DIR,
                 label="04 FC (neg−neu) → Affect  [conservative, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs,
                 n=len(cons), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -- Full sample archive --
    base_covs_full = get_covariates(full)
    corr_f, ols_f = run_analysis_set(
        full, PREDICTORS, OUTCOMES, base_covs_full,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr_f, ols_f, OUT_DIR / "full_sample",
                 label="04 FC (neg−neu) → Affect  [full sample archive, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs_full,
                 n=len(full), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
