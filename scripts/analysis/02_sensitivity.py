#!/usr/bin/env python3
"""
02_sensitivity.py

Sensitivity analyses for Analysis 02: Persistence → Affect.

Run ONLY if primary findings in 02_persistence_affect.py show meaningful signal.

Sections:
  1. Right hemisphere persistence          → sensitivity_right_hemisphere/
  2. PANAS (convergent validity)           → sensitivity_panas/
       Includes C5SPGP, C5SPGN, C5SPGN_log — all one-tailed
  3. vmPFC persistence (two-tailed)        → sensitivity_vmpfc_persistence/
  4. ROI activations (two-tailed)          → sensitivity_roi_activations/
  5. Other persistence operationalisations → sensitivity_other_persistence/
     (positive cross-run, concatenated negative)

Run from project root directory.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS_DIR, load_master, get_samples, get_covariates,
    prepare_persistence_vars, run_analysis_set, save_results,
)

BASE_DIR = RESULTS_DIR / "02_persistence_affect"
OUTCOMES_DIARY = ["PA_score", "NA_score"]

# PANAS directions mirror the primary affect directions (same construct)
PANAS_EXPECTED = {"C5SPGP": -1, "C5SPGN": +1, "C5SPGN_log": +1}  # more persistence → lower PA, higher NA


def main():
    print("=" * 70)
    print("Analysis 02 — Sensitivity Analyses")
    print("=" * 70)

    df = load_master()
    prepare_persistence_vars(df)
    _, cons = get_samples(df)
    print(f"  Conservative N (diary+fMRI) = {len(cons)}")
    base_covs = get_covariates(cons)

    # -------------------------------------------------------------------------
    # 1. Right hemisphere persistence
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Right Hemisphere ---")
    corr, ols, mlm = run_analysis_set(
        cons, ["neg_persist_crossrun_mean_z_R"], OUTCOMES_DIARY, base_covs,
    )
    save_results(corr, ols, mlm, BASE_DIR / "sensitivity_right_hemisphere",
                 label="02 Persistence (R) → Affect  [conservative]",
                 predictors=["neg_persist_crossrun_mean_z_R"],
                 outcomes=OUTCOMES_DIARY, covariates=base_covs, n=len(cons))

    # -------------------------------------------------------------------------
    # 2. PANAS (convergent validity — run only after primary is significant)
    #    Includes log-transformed PANAS NA; all one-tailed (same construct)
    #    Uses full conservative fMRI sample (no diary requirement)
    # -------------------------------------------------------------------------
    panas_outcomes = ["C5SPGP", "C5SPGN", "C5SPGN_log"]
    print("\n--- Sensitivity: PANAS ---")
    # Build PANAS sample and covariates together: conservative fMRI, no diary needed
    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1
    has_qc = df.get("qc_conservative", pd.Series(0, index=df.index)) == 1
    panas_cons = df[has_persist & has_qc].copy()
    prepare_persistence_vars(panas_cons)
    panas_covs = get_covariates(panas_cons)  # no diary covariates
    print(f"  PANAS N (conservative fMRI) = {len(panas_cons)}")
    corr, ols, mlm = run_analysis_set(
        panas_cons, ["neg_persist_crossrun_mean_z_L"], panas_outcomes, panas_covs,
        one_tailed=True, expected_directions=PANAS_EXPECTED,
    )
    save_results(corr, ols, mlm, BASE_DIR / "sensitivity_panas",
                 label="02 Persistence (L) → PANAS  [conservative fMRI, one-tailed]",
                 predictors=["neg_persist_crossrun_mean_z_L"],
                 outcomes=panas_outcomes, covariates=panas_covs, n=len(panas_cons),
                 one_tailed=True, expected_directions=PANAS_EXPECTED)

    # -------------------------------------------------------------------------
    # 3. vmPFC persistence (two-tailed — direction not pre-specified)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: vmPFC Persistence ---")
    vmpfc_persist_vars = [v for v in [
        "ant_vmPFC_neg_image_mean_r",
        "post_vmPFC_neg_image_mean_r",
    ] if v in cons.columns]

    if vmpfc_persist_vars:
        corr, ols, mlm = run_analysis_set(
            cons, vmpfc_persist_vars, OUTCOMES_DIARY, base_covs,
        )
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_vmpfc_persistence",
                     label="02 vmPFC Persistence → Affect  [conservative, two-tailed]",
                     predictors=vmpfc_persist_vars, outcomes=OUTCOMES_DIARY,
                     covariates=base_covs, n=len(cons))
    else:
        print("  vmPFC persistence variables not found — skipping.")
        print("  (run run_cross_corr_vmPFC.py and 09_merge_fmri_data.py first)")

    # -------------------------------------------------------------------------
    # 4. ROI activations (two-tailed — opposing directions for amyg vs vmPFC)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: ROI Activations ---")
    roi_act_vars = [v for v in [
        "l_amyg_neg_image",
        "ant_vmPFC_neg_image",
        "post_vmPFC_neg_image",
    ] if v in cons.columns]

    if roi_act_vars:
        corr, ols, mlm = run_analysis_set(
            cons, roi_act_vars, OUTCOMES_DIARY, base_covs,
        )
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_roi_activations",
                     label="02 ROI Activations → Affect  [conservative, two-tailed]",
                     predictors=roi_act_vars, outcomes=OUTCOMES_DIARY,
                     covariates=base_covs, n=len(cons))
    else:
        print("  ROI activation variables not found — skipping.")
        print("  (run extract_roi_activations.sh and combineROIActivations.py first)")

    # -------------------------------------------------------------------------
    # 5. Other persistence operationalisations
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Other Persistence ---")
    other_persist = [v for v in [
        "pos_persist_crossrun_mean_z_L",
        "neg_persist_concat_z_L",
    ] if v in cons.columns]

    if other_persist:
        corr, ols, mlm = run_analysis_set(
            cons, other_persist, OUTCOMES_DIARY, base_covs,
        )
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_other_persistence",
                     label="02 Other Persistence → Affect  [conservative]",
                     predictors=other_persist, outcomes=OUTCOMES_DIARY,
                     covariates=base_covs, n=len(cons))
    else:
        print("  No alternative persistence variables found — skipping.")


if __name__ == "__main__":
    main()
