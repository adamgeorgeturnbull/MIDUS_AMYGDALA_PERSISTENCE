#!/usr/bin/env python3
"""
03_persistence_age.py (MR1)

Confirmatory extension: age-related differences in left amygdala negative
persistence in MR1. Prespecified direction: older age → lower persistence.

Primary analysis (conservative fMRI sample):
  Predictor  : RA5PAGE (age at neuroscience visit, P5)
  Outcome    : neg_persist_crossrun_mean_z_L  (Fisher z, left hemisphere)
  Methods    : Pearson correlation and participant-level OLS
               (available MR1 P5 inputs do not contain a usable family or
               other grouping identifier; MLM is therefore not applicable)
  Tests      : One-tailed (older age → lower persistence; prespecified)

Adjusted OLS covariates:
  sex, race_2 – race_6  (RA5PAGE is the predictor; no diary covariates apply)

Conservative criterion (qc_conservative == 1):
  - historical visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (n_pairs == 6)
  - all three task runs present with exactly 231 volumes

No diary participation is required; the conservative fMRI sample is used.

The conservative result is primary. Full-sample results (less restrictive QC)
are saved to full_sample/ as an archived secondary analysis.

Inputs:
  data/processed/mr1_with_fmri.csv  (from scripts 08-09)

Outputs: results/tables/03_persistence_age/
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
OUT_DIR = RESULTS_DIR / "03_persistence_age"

PREDICTORS = ["RA5PAGE"]
OUTCOMES   = ["neg_persist_crossrun_mean_z_L"]

# Older age → reduced negative persistence
EXPECTED_DIRECTIONS = {"neg_persist_crossrun_mean_z_L": -1}


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 03 (MR1): Age → Persistence  (Confirmatory Extension)")
    print("=" * 70)

    df = load_master(fc=False)
    prepare_persistence_vars(df)

    if OUTCOMES[0] not in df.columns:
        print(f"ERROR: {OUTCOMES[0]} not found. Run fMRI pipeline + script 09 first.")
        sys.exit(1)

    full, cons = get_samples(df, require_diary=False)
    print(f"  Full N = {len(full)} | Conservative N = {len(cons)}")

    # RA5PAGE is the predictor — exclude it from covariates.
    # No diary covariates apply (persistence is a neuroscience-only outcome).
    base_covs = [c for c in get_covariates(cons) if c not in PREDICTORS]

    # -- Primary: conservative sample --
    corr, ols = run_analysis_set(cons, PREDICTORS, OUTCOMES, base_covs,
                                 one_tailed=True,
                                 expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr, ols, OUT_DIR,
                 label="03 Age → Persistence  [conservative, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs,
                 n=len(cons), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -- Full sample archive --
    base_covs_full = [c for c in get_covariates(full) if c not in PREDICTORS]
    corr_f, ols_f = run_analysis_set(full, PREDICTORS, OUTCOMES, base_covs_full,
                                     one_tailed=True,
                                     expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr_f, ols_f, OUT_DIR / "full_sample",
                 label="03 Age → Persistence  [full sample archive, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs_full,
                 n=len(full), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
