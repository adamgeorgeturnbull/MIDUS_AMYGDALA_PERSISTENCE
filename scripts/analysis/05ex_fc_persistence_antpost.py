#!/usr/bin/env python3
"""
05ex_fc_persistence_antpost.py

EXPLORATORY follow-up to Analysis 05.

Tests whether the anterior-minus-posterior vmPFC connectivity contrast
(neg − neu) predicts amygdala persistence. Not pre-registered.

Directional hypothesis: safety-biased connectivity (ant > post) →
lower persistence (higher FC → less sustained threat processing).

Predictors : l_amyg-ant_minus_post_vmPFC_neg_vs_neu  (computed; left)
             r_amyg-ant_minus_post_vmPFC_neg_vs_neu  (right, if available)
Outcome    : neg_persist_crossrun_mean_z_L
Methods    : Pearson correlation, OLS regression, MLM
Tests      : One-tailed (directional hypothesis)

Output:
  results/tables/05ex_fc_persistence_antpost/

Run from project root directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS_DIR, load_master, get_samples, get_covariates,
    prepare_persistence_vars, run_analysis_set, save_results,
)

OUT_DIR    = RESULTS_DIR / "05ex_fc_persistence_antpost"
OUTCOMES   = ["neg_persist_crossrun_mean_z_L"]
EXPECTED_DIRECTIONS = {"neg_persist_crossrun_mean_z_L": -1}


def main():
    print("=" * 70)
    print("Analysis 05ex: FC ant−post (neg−neu) → Persistence  (Exploratory)")
    print("=" * 70)

    df = load_master(fc=True)
    prepare_persistence_vars(df)

    df["l_amyg-ant_minus_post_vmPFC_neg_vs_neu"] = (
        df["l_amyg-ant_vmPFC_neg_vs_neu"] - df["l_amyg-post_vmPFC_neg_vs_neu"]
    )
    if "r_amyg-ant_vmPFC_neg_vs_neu" in df.columns and "r_amyg-post_vmPFC_neg_vs_neu" in df.columns:
        df["r_amyg-ant_minus_post_vmPFC_neg_vs_neu"] = (
            df["r_amyg-ant_vmPFC_neg_vs_neu"] - df["r_amyg-post_vmPFC_neg_vs_neu"]
        )

    full, cons = get_samples(df, check_fc_col="l_amyg-ant_vmPFC_neg_vs_neu",
                             require_diary=False)
    print(f"  Conservative N = {len(cons)}")
    base_covs = get_covariates(cons)

    preds = [p for p in [
        "l_amyg-ant_minus_post_vmPFC_neg_vs_neu",
        "r_amyg-ant_minus_post_vmPFC_neg_vs_neu",
    ] if p in cons.columns]

    corr, ols, mlm = run_analysis_set(
        cons, preds, OUTCOMES, base_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, mlm, OUT_DIR,
                 label="05ex FC ant−post (neg−neu) → Persistence  [conservative, one-tailed]",
                 predictors=preds, outcomes=OUTCOMES,
                 covariates=base_covs, n=len(cons),
                 one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
