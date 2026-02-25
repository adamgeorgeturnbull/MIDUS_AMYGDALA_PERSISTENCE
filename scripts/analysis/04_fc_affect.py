#!/usr/bin/env python3
"""
04_fc_affect.py

Condition-level analysis of amygdala-vmPFC functional connectivity and affect.

Tests whether amygdala-vmPFC connectivity relates to daily-life affect when
examining each condition (negative, neutral, positive) separately, as well as
the neg > neu and neg > pos contrasts. This approach uses a 3-condition GLM
(neg, neu, pos) and tests condition-specific associations rather than relying
solely on the neg > neu contrast.

Key analysis decisions:
- Two-tailed tests (FC hypotheses are exploratory)
- Results tiered: primary (L amyg neg, PA/NA), secondary (R amyg neg),
  sensitivity (bilateral, other conditions, contrasts, PANAS, log-transforms)

FC measures are ROI-level beta-series correlations (Fisher z-transformed)
between amygdala seeds (L, R) and vmPFC targets (anterior = safety signaling,
posterior = threat signaling), based on Tashjian et al. (2021, TICS).

Analysis plan:
1. Zero-order correlations between FC and affect
2. OLS regressions: affect ~ FC + covariates
   - Base covariates: C5PAGE, sex, race dummies, twin dummies
   - Diary-only covariates: time_P2_P5, n_days_complete (PA/NA outcomes only)

Input:
  - data/fMRI/betaSeries_all_conditions.csv
    (per-condition Fisher z correlations from runBStaskFC.sh v2)
  - data/processed/midus_with_fmri.csv

Output:
  - results/tables/04_fc_affect_correlations_full.csv
  - results/tables/04_fc_affect_correlations_conservative.csv
  - results/tables/04_fc_affect_regressions_full.csv
  - results/tables/04_fc_affect_regressions_conservative.csv

Run from project root directory.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier_utils import get_fc_tier, get_affect_tier, combine_tiers, print_by_tier

# ============================================================================
# Paths and Constants
# ============================================================================
PROCESSED_DIR = Path("data/processed")
FMRI_DIR = Path("data/fMRI")
RESULTS_DIR = Path("results/tables")

MASTER_FILE = PROCESSED_DIR / "midus_with_fmri.csv"
FC_FILE = FMRI_DIR / "betaSeries_all_conditions.csv"

MIN_N_CORR = 10
MIN_N_REG = 20

AFFECT_VARS = {
    "PA_score": "Positive Affect (Diary)",
    "NA_score": "Negative Affect (Diary)",
    "NA_score_log": "Negative Affect [log] (Diary)",
    "C5SPGP": "Positive Affect (PANAS)",
    "C5SPGN": "Negative Affect (PANAS)",
    "C5SPGN_log": "Negative Affect [log] (PANAS)",
}

ROI_PAIRS = [
    ("l_amyg", "ant_vmPFC"),
    ("l_amyg", "post_vmPFC"),
    ("r_amyg", "ant_vmPFC"),
    ("r_amyg", "post_vmPFC"),
]

CONDITIONS = ["neg", "neu", "pos", "neg_vs_neu", "neg_vs_pos"]

DIARY_AFFECT_VARS = {"PA_score", "NA_score", "NA_score_log"}


# ============================================================================
# Helper Functions
# ============================================================================
def parse_fc_var(fc_var):
    """Parse condition and seed_target from FC variable name."""
    if "safety_vs_threat" in fc_var:
        condition = fc_var.rsplit("_", 1)[-1]
        seed_target = fc_var.rsplit("_", 1)[0]
    elif "_vs_" in fc_var:
        condition = "_".join(fc_var.rsplit("_", 3)[-3:])
        seed_target = fc_var.rsplit("_", 3)[0]
    else:
        condition = fc_var.rsplit("_", 1)[-1]
        seed_target = fc_var.rsplit("_", 1)[0]
    return condition, seed_target


def compute_correlations(df, fc_vars, affect_vars, min_n=MIN_N_CORR):
    """Compute bivariate Pearson correlations between FC and affect vars."""
    results = []
    for fc_var in fc_vars:
        for affect_var in affect_vars:
            data = df[[fc_var, affect_var]].dropna()
            n = len(data)
            if n < min_n:
                continue
            r, p = stats.pearsonr(data[fc_var], data[affect_var])
            condition, seed_target = parse_fc_var(fc_var)

            results.append({
                "fc_var": fc_var,
                "condition": condition,
                "seed_target": seed_target,
                "affect_var": affect_var,
                "n": n,
                "r": r,
                "p": p,
                "tier": combine_tiers(
                    get_fc_tier(fc_var, condition=condition, seed_target=seed_target),
                    get_affect_tier(affect_var),
                ),
            })
    return pd.DataFrame(results)


def run_regression(df, fc_var, affect_var, covariates):
    """
    Run OLS regression: affect ~ FC + covariates.

    Returns dictionary with results, or None if insufficient data.
    """
    vars_needed = [affect_var, fc_var] + covariates
    data = df[vars_needed].dropna()

    if len(data) < MIN_N_REG:
        return None

    y = data[affect_var]
    X = data[[fc_var] + covariates]
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"    Error fitting model for {affect_var} ~ {fc_var}: {e}")
        return None

    condition, seed_target = parse_fc_var(fc_var)
    fc_idx = 1  # First column after intercept

    return {
        "fc_var": fc_var,
        "condition": condition,
        "seed_target": seed_target,
        "affect_var": affect_var,
        "n": int(model.nobs),
        "beta_fc": model.params.iloc[fc_idx],
        "se_fc": model.bse.iloc[fc_idx],
        "t_fc": model.tvalues.iloc[fc_idx],
        "p_fc": model.pvalues.iloc[fc_idx],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "tier": combine_tiers(
            get_fc_tier(fc_var, condition=condition, seed_target=seed_target),
            get_affect_tier(affect_var),
        ),
    }


def run_all_regressions(df, fc_vars, affect_vars, base_covariates,
                        diary_covariates):
    """Run all FC x affect regressions.

    Diary-specific covariates only included for daily diary outcomes.
    """
    results = []
    for fc_var in fc_vars:
        for affect_var in affect_vars:
            if affect_var in DIARY_AFFECT_VARS:
                covariates = base_covariates + diary_covariates
            else:
                covariates = base_covariates
            result = run_regression(df, fc_var, affect_var, covariates)
            if result is not None:
                results.append(result)
    return pd.DataFrame(results)


def run_sample_analysis(df, sample_name, fc_vars, affect_vars,
                        base_covariates, diary_covariates):
    """Run correlation + regression analysis for a given sample."""
    print(f"\n{'=' * 70}")
    print(f"Condition-Level FC Analysis: {sample_name}")
    print(f"{'=' * 70}")
    print(f"  N = {len(df)}")

    # ========================================================================
    # Zero-Order Correlations
    # ========================================================================
    print("\nComputing zero-order correlations...")
    corr_results = compute_correlations(df, fc_vars, list(affect_vars.keys()))

    if len(corr_results) > 0:
        print(f"\n  Computed {len(corr_results)} correlations")

        def _fmt_corr(row):
            sig = ("***" if row["p"] < 0.001 else "**" if row["p"] < 0.01
                   else "*" if row["p"] < 0.05 else "")
            return (f"{row['fc_var']:45s} x {row['affect_var']:15s}: "
                    f"r = {row['r']:7.3f}, p = {row['p']:.4f}{sig:3s}, "
                    f"n = {int(row['n'])}")

        print_by_tier(corr_results, _fmt_corr, p_col="p")
    else:
        print("  No correlations computed (insufficient data)")

    corr_file = RESULTS_DIR / f"04_fc_affect_correlations_{sample_name}.csv"
    corr_results.to_csv(corr_file, index=False)
    print(f"\n  Saved to {corr_file}")

    # ========================================================================
    # OLS Regressions with Covariates
    # ========================================================================
    print("\nRunning OLS regressions (controlling for covariates)...")
    reg_results = run_all_regressions(df, fc_vars, list(affect_vars.keys()),
                                      base_covariates, diary_covariates)

    if len(reg_results) > 0:
        print(f"\n  Computed {len(reg_results)} regressions")

        def _fmt_reg(row):
            sig = ("***" if row["p_fc"] < 0.001 else "**" if row["p_fc"] < 0.01
                   else "*" if row["p_fc"] < 0.05 else "")
            return (f"{row['fc_var']:45s} -> {row['affect_var']:15s}: "
                    f"b = {row['beta_fc']:7.3f}, p = {row['p_fc']:.4f}{sig:3s}, "
                    f"n = {int(row['n'])}")

        print_by_tier(reg_results, _fmt_reg, p_col="p_fc")
    else:
        print("  No regressions computed (insufficient data)")

    reg_file = RESULTS_DIR / f"04_fc_affect_regressions_{sample_name}.csv"
    reg_results.to_csv(reg_file, index=False)
    print(f"\n  Saved to {reg_file}")

    return corr_results, reg_results


# ============================================================================
# Main
# ============================================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Analysis 04: Condition-Level FC and Affect")
    print("=" * 70)

    # Load master behavioral data
    print(f"\nLoading behavioral data from {MASTER_FILE}...")
    master = pd.read_csv(MASTER_FILE)
    print(f"  Loaded {len(master)} participants")

    # Load condition-level FC data
    print(f"Loading condition-level FC from {FC_FILE}...")
    fc = pd.read_csv(FC_FILE)
    print(f"  Loaded {len(fc)} participants, {len(fc.columns) - 1} FC columns")

    # Ensure M2ID types match for merge
    master["M2ID"] = master["M2ID"].astype(str)
    fc["M2ID"] = fc["M2ID"].astype(str)

    # Merge
    df = master.merge(fc, on="M2ID", how="inner")
    print(f"  Merged: {len(df)} participants with both FC and behavioral data")

    # Build FC variable list from available columns
    fc_vars = []
    for seed, target in ROI_PAIRS:
        pair = f"{seed}-{target}"
        for cond in CONDITIONS:
            col = f"{pair}_{cond}"
            if col in df.columns:
                fc_vars.append(col)

    # Also add safety-vs-threat derived variables per condition
    for cond in CONDITIONS:
        l_ant = f"l_amyg-ant_vmPFC_{cond}"
        l_post = f"l_amyg-post_vmPFC_{cond}"
        r_ant = f"r_amyg-ant_vmPFC_{cond}"
        r_post = f"r_amyg-post_vmPFC_{cond}"
        if l_ant in df.columns and l_post in df.columns:
            derived = f"l_amyg_safety_vs_threat_{cond}"
            df[derived] = df[l_ant] - df[l_post]
            fc_vars.append(derived)
        if r_ant in df.columns and r_post in df.columns:
            derived = f"r_amyg_safety_vs_threat_{cond}"
            df[derived] = df[r_ant] - df[r_post]
            fc_vars.append(derived)

    print(f"\n  FC variables: {len(fc_vars)}")
    print(f"  Affect variables: {len(AFFECT_VARS)}")

    # Covariates
    race_dummies = [col for col in df.columns if col.startswith("race_")]
    twin_dummies = [col for col in df.columns if col.startswith("twin_pair_")]

    base_covariates = [
        "C5PAGE",
        "sex",
    ] + race_dummies + twin_dummies

    diary_covariates = [
        "time_P2_P5",
        "n_days_complete",
    ]

    print(f"\nCovariates (for regressions):")
    print(f"  Base (all models): C5PAGE, sex, {len(race_dummies)} race, {len(twin_dummies)} twin dummies")
    print(f"  Diary-only (PA/NA outcomes): time_P2_P5, n_days_complete")

    # Define samples
    has_fc = df[fc_vars[0]].notna() if fc_vars else pd.Series(False, index=df.index)
    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1

    full_sample = df[has_fc & has_persist].copy()
    conservative_sample = full_sample[full_sample.get("qc_conservative", 0) == 1].copy()

    # Run analyses
    full_corr, full_reg = run_sample_analysis(
        full_sample, "full", fc_vars, AFFECT_VARS,
        base_covariates, diary_covariates
    )
    cons_corr, cons_reg = run_sample_analysis(
        conservative_sample, "conservative", fc_vars, AFFECT_VARS,
        base_covariates, diary_covariates
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    for label, corr_results, reg_results in [
        ("Full", full_corr, full_reg),
        ("Conservative", cons_corr, cons_reg),
    ]:
        print(f"\n{label}:")
        if len(corr_results) > 0:
            for tier in ["primary", "secondary", "sensitivity"]:
                tier_res = corr_results[corr_results["tier"] == tier]
                if len(tier_res) == 0:
                    continue
                n_sig = (tier_res["p"] < 0.05).sum()
                print(f"  Correlations {tier.upper()}: {n_sig}/{len(tier_res)} significant")
        else:
            print("  No correlations")
        if len(reg_results) > 0:
            for tier in ["primary", "secondary", "sensitivity"]:
                tier_res = reg_results[reg_results["tier"] == tier]
                if len(tier_res) == 0:
                    continue
                n_sig = (tier_res["p_fc"] < 0.05).sum()
                print(f"  Regressions  {tier.upper()}: {n_sig}/{len(tier_res)} significant")
        else:
            print("  No regressions")


if __name__ == "__main__":
    main()
