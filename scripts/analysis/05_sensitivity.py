#!/usr/bin/env python3
"""
05_sensitivity.py

Sensitivity analyses for Analysis 05: FC (neg−neu) → Persistence.

Run ONLY if primary findings in 05_fc_persistence.py show meaningful signal.

Sections:
  1. Right amygdala FC (neg−neu)      → sensitivity_right_amygdala/
  2. pos−neu contrast (specificity)   → sensitivity_pos_vs_neu/
  3. Right hemisphere persistence     → sensitivity_right_hemisphere/
  4. Other persistence operationalisations → sensitivity_other_persistence/
  5. Neg condition only (robustness)  → sensitivity_neg_condition/

Run from project root directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS_DIR, load_master, get_samples, get_covariates,
    prepare_persistence_vars, run_analysis_set, save_results,
)

BASE_DIR = RESULTS_DIR / "05_fc_persistence"
PRIMARY_OUTCOME  = ["neg_persist_crossrun_mean_z_L"]
PRIMARY_PREDS    = ["l_amyg-ant_vmPFC_neg_vs_neu", "l_amyg-post_vmPFC_neg_vs_neu"]
EXPECTED_DIRECTIONS = {"neg_persist_crossrun_mean_z_L": -1}


def run_negative_condition(cons, base_covs):
    # -------------------------------------------------------------------------
    # 5. Neg condition only (robustness — original operationalisation)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Neg condition only ---")
    neg_preds = [p for p in [
        "l_amyg-ant_vmPFC_neg",
        "l_amyg-post_vmPFC_neg",
    ] if p in cons.columns]

    if neg_preds:
        corr, ols, mlm = run_analysis_set(
            cons, neg_preds, PRIMARY_OUTCOME, base_covs,
            one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
            mlm_maxiter_by_predictor={"l_amyg-ant_vmPFC_neg": 2000},
        )
        target = (mlm.loc[mlm["predictor"].eq("l_amyg-ant_vmPFC_neg")]
                  if "predictor" in mlm else mlm.iloc[:0])
        if len(target) != 1 or target.iloc[0]["ci_status"] != "ok":
            raise RuntimeError("Targeted connectivity fit still requires review; outputs not replaced.")
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_neg_condition",
                     label="05 FC (neg only) → Persistence  [conservative, one-tailed]",
                     predictors=neg_preds, outcomes=PRIMARY_OUTCOME,
                     covariates=base_covs, n=len(cons),
                     one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)
        (BASE_DIR / "sensitivity_neg_condition" / "fit_notes.md").write_text(
            "Anterior negative-condition connectivity MLM: maxiter=2000; "
            "model, sample, covariates, REML, optimizer order, and directional test unchanged. "
            "Optimizers continue in the existing order until a converged fit with a valid "
            "target-coefficient CI is obtained; no selection by p value. "
            "Iteration limit approved after separate optimizer diagnostics. "
            "Posterior model retains its original iteration setting.\n"
        )
    else:
        print("  Neg condition FC variables not found — skipping.")


def main(negative_condition_only=False):
    print("=" * 70)
    print("Analysis 05 — Sensitivity Analyses")
    print("=" * 70)

    df = load_master(fc=True)
    prepare_persistence_vars(df)

    # Compute pos−neu contrasts
    if "l_amyg-ant_vmPFC_pos" in df.columns and "l_amyg-ant_vmPFC_neu" in df.columns:
        df["l_amyg-ant_vmPFC_pos_vs_neu"]  = df["l_amyg-ant_vmPFC_pos"]  - df["l_amyg-ant_vmPFC_neu"]
        df["l_amyg-post_vmPFC_pos_vs_neu"] = df["l_amyg-post_vmPFC_pos"] - df["l_amyg-post_vmPFC_neu"]

    full, cons = get_samples(df, check_fc_col="l_amyg-ant_vmPFC_neg_vs_neu",
                             require_diary=False)
    print(f"  Conservative N = {len(cons)}")
    base_covs = get_covariates(cons)
    if negative_condition_only:
        run_negative_condition(cons, base_covs)
        return

    # -------------------------------------------------------------------------
    # 1. Right amygdala neg−neu (hemisphere specificity)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Right Amygdala (neg−neu) ---")
    right_preds = [p for p in [
        "r_amyg-ant_vmPFC_neg_vs_neu",
        "r_amyg-post_vmPFC_neg_vs_neu",
    ] if p in cons.columns]

    if right_preds:
        corr, ols, mlm = run_analysis_set(
            cons, right_preds, PRIMARY_OUTCOME, base_covs,
        )
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_right_amygdala",
                     label="05 FC-R (neg−neu) → Persistence  [conservative, two-tailed]",
                     predictors=right_preds, outcomes=PRIMARY_OUTCOME,
                     covariates=base_covs, n=len(cons))
    else:
        print("  No right amygdala neg−neu variables found — skipping.")

    # -------------------------------------------------------------------------
    # 2. pos−neu contrast (condition specificity)
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: pos−neu contrast ---")
    pos_neu_preds = [p for p in [
        "l_amyg-ant_vmPFC_pos_vs_neu",
        "l_amyg-post_vmPFC_pos_vs_neu",
    ] if p in cons.columns]

    if pos_neu_preds:
        corr, ols, mlm = run_analysis_set(
            cons, pos_neu_preds, PRIMARY_OUTCOME, base_covs,
        )
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_pos_vs_neu",
                     label="05 FC (pos−neu) → Persistence  [conservative, two-tailed]",
                     predictors=pos_neu_preds, outcomes=PRIMARY_OUTCOME,
                     covariates=base_covs, n=len(cons))
    else:
        print("  pos−neu FC variables not found — skipping.")

    # -------------------------------------------------------------------------
    # 3. Right hemisphere persistence
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Right Hemisphere Persistence ---")
    corr, ols, mlm = run_analysis_set(
        cons, PRIMARY_PREDS, ["neg_persist_crossrun_mean_z_R"], base_covs,
    )
    save_results(corr, ols, mlm, BASE_DIR / "sensitivity_right_hemisphere",
                 label="05 FC (L, neg−neu) → Persistence (R)  [conservative, two-tailed]",
                 predictors=PRIMARY_PREDS, outcomes=["neg_persist_crossrun_mean_z_R"],
                 covariates=base_covs, n=len(cons))

    # -------------------------------------------------------------------------
    # 4. Other persistence operationalisations
    # -------------------------------------------------------------------------
    print("\n--- Sensitivity: Other Persistence ---")
    other_persist = [v for v in [
        "pos_persist_crossrun_mean_z_L",
        "neg_persist_concat_z_L",
    ] if v in cons.columns]

    if other_persist:
        corr, ols, mlm = run_analysis_set(
            cons, PRIMARY_PREDS, other_persist, base_covs,
        )
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_other_persistence",
                     label="05 FC (L, neg−neu) → Other Persistence  [conservative, two-tailed]",
                     predictors=PRIMARY_PREDS, outcomes=other_persist,
                     covariates=base_covs, n=len(cons))
    else:
        print("  No alternative persistence variables found — skipping.")

    run_negative_condition(cons, base_covs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-condition-only", action="store_true")
    main(negative_condition_only=parser.parse_args().negative_condition_only)
