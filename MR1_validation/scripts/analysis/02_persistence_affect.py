#!/usr/bin/env python3
"""
02_persistence_affect.py (MR1)

Primary analysis: left amygdala negative persistence and daily life affect.
Replication of Puccetti et al. (2021) and the M3 primary finding.

Primary analysis (conservative fMRI + diary sample):
  Predictor  : neg_persist_crossrun_mean_z_L  (Fisher z, left hemisphere)
  Outcomes   : PA_score, NA_score, NA_score_log  (daily diary)
  Methods    : Pearson correlation and participant-level OLS
               (available MR1 P5 inputs contain no usable family or other
               grouping identifier; MLM is therefore not applicable)
  Tests      : One-tailed (confirmatory replication, pre-registered directional
               hypothesis: greater negative persistence → lower PA, higher NA)

Adjusted OLS covariates:
  RA5PAGE, sex, race_2 – race_6  (base; returned by get_covariates)
  time_P2_P5, n_days_complete    (diary-specific; added automatically by
                                  run_analysis_set for diary outcomes)

Conservative criterion (qc_conservative == 1):
  - historical visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (n_pairs == 6)
  - all three task runs present with exactly 231 volumes (all_three_runs_231)
  The conservative result is primary.

Full-sample results (less restrictive QC) are saved to full_sample/ as an
archived secondary result.

Sensitivity analyses (right hemisphere, PANAS, other persistence) in
02_sensitivity.py.

Inputs:
  data/processed/mr1_with_fmri.csv  (from scripts 08-09)

Outputs: results/tables/02_persistence_affect/
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
    print("Analysis 02 (MR1): Persistence → Affect  (Replication)")
    print("=" * 70)

    df = load_master(fc=False)
    prepare_persistence_vars(df)

    if PREDICTORS[0] not in df.columns:
        print(f"ERROR: {PREDICTORS[0]} not found. Run scripts 01-09 first.")
        sys.exit(1)

    full, cons = get_samples(df, require_diary=True)
    print(f"  Full N = {len(full)} | Conservative N = {len(cons)}")

    base_covs = get_covariates(cons)

    # -- Primary: conservative sample --
    corr, ols = run_analysis_set(cons, PREDICTORS, OUTCOMES, base_covs,
                                 one_tailed=True,
                                 expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr, ols, OUT_DIR,
                 label="02 Persistence → Affect  [conservative]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs,
                 n=len(cons), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -- Full sample archive --
    base_covs_full = get_covariates(full)
    corr_f, ols_f = run_analysis_set(full, PREDICTORS, OUTCOMES, base_covs_full,
                                     one_tailed=True,
                                     expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr_f, ols_f, OUT_DIR / "full_sample",
                 label="02 Persistence → Affect  [full sample archive]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs_full,
                 n=len(full), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
