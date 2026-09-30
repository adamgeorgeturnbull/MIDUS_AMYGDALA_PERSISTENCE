#!/usr/bin/env python3
"""
03_sensitivity.py (M3)

Sensitivity analyses for Analysis 03: Age → Persistence.

Sections:
  1. Age → right-amygdala negative persistence → sensitivity_right_hemisphere/
  2. Age → bilateral positive persistence     → sensitivity_positive_persistence/
  3. Age → anterior/posterior vmPFC persistence → sensitivity_vmPFC/

Methods: Pearson correlation, OLS, and family-random-intercept MLM.
All sections use two-tailed tests.

Adjusted covariates:
  sex and race dummy variables (race_2 – race_6; White is the reference).
  C5PAGE is the predictor and is excluded from the covariate set by the helpers.
  OLS includes available twin-pair indicators; MLM uses family grouping instead.
  No diary covariates apply.

QC criteria (qc_conservative == 1):
  - visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (n_pairs == 6)

Negative persistence must be available; diary participation is not required.
Positive and vmPFC persistence sections use available columns and are skipped
if their respective variables are absent. Model helpers omit incomplete cases.

Each output directory contains: correlations.csv, regressions.csv, mlm.csv,
_methods.txt.

Interpret sensitivities in relation to significant primary findings; otherwise
report for transparency.

Run from project root directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS_DIR, load_master, get_samples, get_covariates,
    prepare_persistence_vars, run_analysis_set, save_results,
)

BASE_DIR   = RESULTS_DIR / "03_persistence_age"
PREDICTORS = ["C5PAGE"]


def main():
    print("=" * 70)
    print("Analysis 03 — Sensitivity Analyses")
    print("=" * 70)

    df = load_master()
    prepare_persistence_vars(df)
    _, cons = get_samples(df, require_diary=False)
    print(f"  Final N = {len(cons)}")
    # No diary covariates — purely neuroscience measures
    base_covs = get_covariates(cons)

    # -------------------------------------------------------------------------
    # 1. Right hemisphere
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Right Hemisphere ---")
    corr, ols, mlm = run_analysis_set(
        cons, PREDICTORS, ["neg_persist_crossrun_mean_z_R"], base_covs,
    )
    save_results(corr, ols, mlm, BASE_DIR / "sensitivity_right_hemisphere",
                 label="03 Age → Persistence (R)  [final]",
                 predictors=PREDICTORS, outcomes=["neg_persist_crossrun_mean_z_R"],
                 covariates=base_covs, n=len(cons))

    # -------------------------------------------------------------------------
    # 2. Positive cross-run persistence
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Positive Persistence ---")
    pos_vars = [v for v in [
        "pos_persist_crossrun_mean_z_L",
        "pos_persist_crossrun_mean_z_R",
    ] if v in cons.columns]

    if pos_vars:
        corr, ols, mlm = run_analysis_set(cons, PREDICTORS, pos_vars, base_covs)
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_positive_persistence",
                     label="03 Age → Positive Persistence  [final]",
                     predictors=PREDICTORS, outcomes=pos_vars,
                     covariates=base_covs, n=len(cons))
    else:
        print("  No positive persistence variables found — skipping.")

    # -------------------------------------------------------------------------
    # 3. vmPFC persistence (comparison ROI — available after Sherlock extraction)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: vmPFC Persistence ---")
    vmPFC_vars = [c for c in cons.columns
                  if "vmPFC" in c and "neg_image" in c and c.endswith("_mean_z")]

    if vmPFC_vars:
        corr, ols, mlm = run_analysis_set(cons, PREDICTORS, vmPFC_vars, base_covs)
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_vmPFC",
                     label="03 Age → vmPFC Persistence  [final]",
                     predictors=PREDICTORS, outcomes=vmPFC_vars,
                     covariates=base_covs, n=len(cons))
    else:
        print("  No vmPFC persistence variables found — skipping.")


if __name__ == "__main__":
    main()
