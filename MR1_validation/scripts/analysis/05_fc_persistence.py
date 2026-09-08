#!/usr/bin/env python3
"""
05_fc_persistence.py (MR1)

Specificity/extension analysis: left amygdala–vmPFC FC (neg − neu contrast)
and cross-run negative affect persistence.

This corresponds to the corrected M3 Analysis 05. The corrected M3 result was
null (no significant FC–persistence relationship). Consequently:
  - An MR1 null result must not be interpreted as evidence of equivalence;
  - An MR1 significant result would be interpreted as an MR1-specific effect,
    not as replication of a positive M3 finding.
  - This script is labelled a specificity/extension check, not a direct
    positive replication.

Directional hypothesis (pre-specified, matching M3): greater amygdala–vmPFC
connectivity specifically during negative relative to neutral stimuli → lower
cross-run persistence of negative affect responses.

Analysis specification:
  Predictors : l_amyg-ant_vmPFC_neg_vs_neu, l_amyg-post_vmPFC_neg_vs_neu
  Outcome    : neg_persist_crossrun_mean_z_L  (Fisher z, left hemisphere)
  Direction  : higher differential FC → lower persistence (−1)
  Methods    : Pearson correlation and participant-level OLS
               (available MR1 P5 inputs do not contain a usable family or other
               grouping identifier; MLM is therefore not applicable)
  Tests      : One-tailed (pre-registered directional hypothesis)

Adjusted OLS covariates:
  RA5PAGE, sex, race_2 – race_6
  No diary covariates apply (outcome is a neuroimaging measure).

No diary participation is required; the conservative fMRI + FC sample is used.

Conservative criterion (qc_conservative == 1):
  - historical visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (n_pairs == 6)
  - all three task runs present with exactly 231 volumes

FC availability: l_amyg-ant_vmPFC_neg_vs_neu must be nonmissing (check_fc_col).
Persistence availability: neg_persist_crossrun_mean_z_L validated by
run_analysis_set; prepare_persistence_vars computes it from raw r columns.

The conservative result is the main specification. Full-sample results (less
restrictive QC) are saved to full_sample/ as an archived secondary analysis.

Inputs:
  data/processed/mr1_with_fmri.csv  (merged with FC; scripts 08-09)

Outputs: results/tables/05_fc_persistence/
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
OUT_DIR = RESULTS_DIR / "05_fc_persistence"

PREDICTORS = [
    "l_amyg-ant_vmPFC_neg_vs_neu",
    "l_amyg-post_vmPFC_neg_vs_neu",
]
OUTCOMES = ["neg_persist_crossrun_mean_z_L"]

# Pre-registered direction: higher differential FC → lower persistence
EXPECTED_DIRECTIONS = {
    "neg_persist_crossrun_mean_z_L": -1,
}


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 05 (MR1): FC (neg−neu) → Persistence  (Specificity, one-tailed)")
    print("=" * 70)

    df = load_master(fc=True)
    prepare_persistence_vars(df)
    full, cons = get_samples(df, check_fc_col=PREDICTORS[0], require_diary=False)
    print(f"  Full N = {len(full)} | Conservative N = {len(cons)}")

    # No diary covariates — outcome is a neuroimaging measure
    base_covs = get_covariates(cons)

    # -- Primary: conservative sample, one-tailed --
    corr, ols = run_analysis_set(
        cons, PREDICTORS, OUTCOMES, base_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, OUT_DIR,
                 label="05 FC (neg−neu) → Persistence  [conservative, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs,
                 n=len(cons), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -- Full sample archive --
    base_covs_full = get_covariates(full)
    corr_f, ols_f = run_analysis_set(
        full, PREDICTORS, OUTCOMES, base_covs_full,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr_f, ols_f, OUT_DIR / "full_sample",
                 label="05 FC (neg−neu) → Persistence  [full sample archive, one-tailed]",
                 predictors=PREDICTORS, outcomes=OUTCOMES, covariates=base_covs_full,
                 n=len(full), one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
