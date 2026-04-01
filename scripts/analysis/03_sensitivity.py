#!/usr/bin/env python3
"""
03_sensitivity.py

Sensitivity analyses for Analysis 03: Age → Persistence.

Run ONLY if primary findings in 03_persistence_age.py show meaningful signal.

Sections:
  1. Right hemisphere persistence     → sensitivity_right_hemisphere/
  2. Positive cross-run persistence   → sensitivity_positive_persistence/
  3. Concatenated negative persistence → sensitivity_concat_persistence/
  4. vmPFC persistence (if available) → sensitivity_vmPFC/

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
    print(f"  Conservative N = {len(cons)}")
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
                 label="03 Age → Persistence (R)  [conservative]",
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
                     label="03 Age → Positive Persistence  [conservative]",
                     predictors=PREDICTORS, outcomes=pos_vars,
                     covariates=base_covs, n=len(cons))
    else:
        print("  No positive persistence variables found — skipping.")

    # -------------------------------------------------------------------------
    # 3. Concatenated negative persistence
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Concatenated Negative Persistence ---")
    concat_vars = [v for v in [
        "neg_persist_concat_z_L",
        "neg_persist_concat_z_R",
    ] if v in cons.columns]

    if concat_vars:
        corr, ols, mlm = run_analysis_set(cons, PREDICTORS, concat_vars, base_covs)
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_concat_persistence",
                     label="03 Age → Concatenated Persistence  [conservative]",
                     predictors=PREDICTORS, outcomes=concat_vars,
                     covariates=base_covs, n=len(cons))
    else:
        print("  No concatenated persistence variables found — skipping.")

    # -------------------------------------------------------------------------
    # 4. vmPFC persistence (comparison ROI — available after Sherlock extraction)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: vmPFC Persistence ---")
    vmPFC_vars = [c for c in cons.columns
                  if "vmPFC" in c and "neg_image" in c and c.endswith("_mean_z")]

    if vmPFC_vars:
        corr, ols, mlm = run_analysis_set(cons, PREDICTORS, vmPFC_vars, base_covs)
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_vmPFC",
                     label="03 Age → vmPFC Persistence  [conservative]",
                     predictors=PREDICTORS, outcomes=vmPFC_vars,
                     covariates=base_covs, n=len(cons))
    else:
        print("  No vmPFC persistence variables found — skipping.")


if __name__ == "__main__":
    main()
