#!/usr/bin/env python3
"""
05_fc_affect.py

Test associations between amygdala-vmPFC task-based functional connectivity
(beta-series, neg > neu) and daily life affect.

FC measures are ROI-level beta-series correlations (Fisher z-transformed
neg - neu contrast) between amygdala seeds (L, R) and vmPFC targets
(anterior = safety signaling, posterior = threat signaling), based on
Tashjian et al. (2021, TICS).

FC variables (6):
  - L amygdala -> anterior vmPFC (safety)
  - L amygdala -> posterior vmPFC (threat)
  - R amygdala -> anterior vmPFC (safety)
  - R amygdala -> posterior vmPFC (threat)
  - L amygdala: safety - threat (relative connectivity)
  - R amygdala: safety - threat (relative connectivity)

Note: FC values are already Fisher z-transformed from runBStaskFC.sh
(fisher_z(r_neg) - fisher_z(r_neu)). No additional transform needed.

Analysis plan:
1. Zero-order correlations between FC and affect
2. Multiple linear regressions controlling for:
   - Age (C5PAGE)
   - Gender (sex)
   - Race (dummy-coded)
   - Twin status (dummy-coded)
   - Time between visits (time_P2_P5)
   - Number of diary interviews completed (n_days_complete)

Two versions:
- Full sample: All participants with FC + affect data
- Conservative sample: Full + mean FD < 0.5 AND all 3 runs pass QC

Inputs:
- data/processed/midus_with_fmri.csv

Outputs:
- results/tables/05_fc_affect_correlations_full.csv
- results/tables/05_fc_affect_regressions_full.csv
- results/tables/05_fc_affect_correlations_conservative.csv
- results/tables/05_fc_affect_regressions_conservative.csv

Run from project root directory.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ============================================================================
# Paths and Constants
# ============================================================================
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")

DATA_FILE = PROCESSED_DIR / "midus_with_fmri.csv"

# Minimum sample size for analyses
MIN_N_CORR = 10
MIN_N_REG = 20


# ============================================================================
# Helper Functions
# ============================================================================
def get_full_sample(df):
    """
    Define full analysis sample: participants with FC data.

    Per-model dropna() handles missing affect outcomes, so we do NOT
    pre-filter on affect availability. This lets PANAS analyses include
    participants who have MRI data but no daily diary data.
    """
    has_fc = df["has_beta_series"] == 1

    sample = df[has_fc].copy()

    print(f"\nFull sample: {len(sample)} participants")
    print(f"  - With daily diary affect: {sample['PA_score'].notna().sum()}")

    return sample


def get_conservative_sample(df):
    """
    Define conservative analysis sample with strict quality criteria.
    """
    sample = get_full_sample(df)

    qc_pass = sample["qc_conservative"] == 1
    conservative = sample[qc_pass].copy()

    print(f"\nConservative sample: {len(conservative)} participants")
    print(f"  - Excluded for failed QC: {(~qc_pass).sum()}")
    print(f"  - With daily diary affect: {conservative['PA_score'].notna().sum()}")

    if "all_runs_pass" in sample.columns and "fd_pass" in sample.columns:
        excluded = sample[~qc_pass]
        runs_fail = (excluded["all_runs_pass"] == 0).sum()
        fd_fail = (excluded["fd_pass"] == 0).sum()
        both_fail = ((excluded["all_runs_pass"] == 0) & (excluded["fd_pass"] == 0)).sum()
        print(f"  - Failed run QC only: {runs_fail - both_fail}")
        print(f"  - Failed FD criterion only: {fd_fail - both_fail}")
        print(f"  - Failed both: {both_fail}")

    return conservative


def compute_correlations(df, fc_vars, affect_vars):
    """
    Compute zero-order correlations between FC and affect measures.
    """
    results = []

    for fc_var in fc_vars:
        for affect_var in affect_vars:
            data = df[[fc_var, affect_var]].dropna()
            n = len(data)

            if n < MIN_N_CORR:
                continue

            r, p = stats.pearsonr(data[fc_var], data[affect_var])

            results.append({
                "fc_var": fc_var,
                "affect_var": affect_var,
                "n": n,
                "r": r,
                "p": p,
            })

    return pd.DataFrame(results)


def run_regression(df, fc_var, affect_var, covariates):
    """
    Run OLS regression: affect ~ FC + covariates.
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

    fc_idx = 1  # First column after intercept

    result = {
        "fc_var": fc_var,
        "affect_var": affect_var,
        "n": int(model.nobs),
        "beta_fc": model.params.iloc[fc_idx],
        "se_fc": model.bse.iloc[fc_idx],
        "t_fc": model.tvalues.iloc[fc_idx],
        "p_fc": model.pvalues.iloc[fc_idx],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
    }

    return result


def run_all_regressions(df, fc_vars, affect_vars, base_covariates,
                        diary_covariates, diary_affect_vars):
    """
    Run all FC x affect regressions.

    Diary-specific covariates only included for daily diary outcomes.
    """
    results = []

    for fc_var in fc_vars:
        for affect_var in affect_vars:
            if affect_var in diary_affect_vars:
                covariates = base_covariates + diary_covariates
            else:
                covariates = base_covariates
            result = run_regression(df, fc_var, affect_var, covariates)
            if result is not None:
                results.append(result)

    return pd.DataFrame(results)


def run_sample_analysis(sample, sample_name, fc_vars, affect_vars,
                        base_covariates, diary_covariates, diary_affect_vars):
    """
    Run complete analysis (correlations + regressions) for a given sample.
    """
    print("\n" + "=" * 80)
    print(f"Analysis: {sample_name}")
    print("=" * 80)

    # ========================================================================
    # Zero-Order Correlations
    # ========================================================================
    print("\nComputing zero-order correlations...")

    corr_results = compute_correlations(sample, fc_vars, affect_vars)

    if len(corr_results) > 0:
        print(f"\n  Computed {len(corr_results)} correlations")
        print("\nCorrelation results:")
        for _, row in corr_results.iterrows():
            sig_marker = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            print(f"  {row['fc_var']:45s} x {row['affect_var']:15s}: "
                  f"r = {row['r']:6.3f}, p = {row['p']:.4f}{sig_marker:3s}, n = {int(row['n'])}")
    else:
        print("\n  No correlations computed (insufficient data)")

    corr_file = RESULTS_DIR / f"05_fc_affect_correlations_{sample_name}.csv"
    corr_results.to_csv(corr_file, index=False)
    print(f"\n  Correlation results saved to {corr_file}")

    # ========================================================================
    # OLS Regressions with Covariates
    # ========================================================================
    print("\nRunning OLS regressions (controlling for covariates)...")

    reg_results = run_all_regressions(sample, fc_vars, affect_vars,
                                      base_covariates, diary_covariates, diary_affect_vars)

    if len(reg_results) > 0:
        print(f"\n  Computed {len(reg_results)} regressions")
        print("\nRegression results (FC effect):")
        for _, row in reg_results.iterrows():
            sig_marker = "***" if row["p_fc"] < 0.001 else "**" if row["p_fc"] < 0.01 else "*" if row["p_fc"] < 0.05 else ""
            print(f"  {row['fc_var']:45s} -> {row['affect_var']:15s}: "
                  f"B = {row['beta_fc']:7.3f}, p = {row['p_fc']:.4f}{sig_marker:3s}, "
                  f"n = {int(row['n'])}")
    else:
        print("\n  No regressions computed (insufficient data)")

    reg_file = RESULTS_DIR / f"05_fc_affect_regressions_{sample_name}.csv"
    reg_results.to_csv(reg_file, index=False)
    print(f"\n  Regression results saved to {reg_file}")

    return corr_results, reg_results


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Analysis 05: Task-Based FC (Amygdala-vmPFC) x Daily Affect")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df)} participants")

    # ========================================================================
    # Compute Derived FC Variables (safety - threat)
    # ========================================================================
    print("\nComputing derived FC variables (safety - threat)...")

    df["conn_l_amyg_safety_vs_threat"] = (
        df["conn_l_amyg_ant_vmPFC_neg_vs_neu"]
        - df["conn_l_amyg_post_vmPFC_neg_vs_neu"]
    )
    df["conn_r_amyg_safety_vs_threat"] = (
        df["conn_r_amyg_ant_vmPFC_neg_vs_neu"]
        - df["conn_r_amyg_post_vmPFC_neg_vs_neu"]
    )

    print("  conn_l_amyg_safety_vs_threat = ant_vmPFC - post_vmPFC (left amygdala)")
    print("  conn_r_amyg_safety_vs_threat = ant_vmPFC - post_vmPFC (right amygdala)")

    # ========================================================================
    # Define Variables
    # ========================================================================
    print("\n" + "=" * 80)
    print("Analysis Variables")
    print("=" * 80)

    # FC variables (already Fisher z-transformed from runBStaskFC.sh)
    fc_vars = [
        # Left amygdala seeds
        "conn_l_amyg_ant_vmPFC_neg_vs_neu",     # L amyg -> anterior vmPFC (safety)
        "conn_l_amyg_post_vmPFC_neg_vs_neu",    # L amyg -> posterior vmPFC (threat)
        "conn_l_amyg_safety_vs_threat",          # L amyg: safety - threat
        # Right amygdala seeds
        "conn_r_amyg_ant_vmPFC_neg_vs_neu",     # R amyg -> anterior vmPFC (safety)
        "conn_r_amyg_post_vmPFC_neg_vs_neu",    # R amyg -> posterior vmPFC (threat)
        "conn_r_amyg_safety_vs_threat",          # R amyg: safety - threat
    ]

    print(f"\nFC measures (already Fisher z-transformed, neg > neu):")
    print(f"  Left amygdala:")
    print(f"  - conn_l_amyg_ant_vmPFC_neg_vs_neu  (safety)")
    print(f"  - conn_l_amyg_post_vmPFC_neg_vs_neu (threat)")
    print(f"  - conn_l_amyg_safety_vs_threat       (relative)")
    print(f"  Right amygdala:")
    print(f"  - conn_r_amyg_ant_vmPFC_neg_vs_neu  (safety)")
    print(f"  - conn_r_amyg_post_vmPFC_neg_vs_neu (threat)")
    print(f"  - conn_r_amyg_safety_vs_threat       (relative)")

    # Affect measures
    affect_vars = [
        # Daily diary (primary)
        "PA_score",
        "NA_score",
        "NA_score_log",
        # PANAS (secondary)
        "C5SPGP",
        "C5SPGN",
        "C5SPGN_log",
    ]

    print(f"\nAffect measures:")
    print(f"  Primary (Daily Diary): PA_score, NA_score, NA_score_log")
    print(f"  Secondary (PANAS): C5SPGP, C5SPGN, C5SPGN_log")

    # Covariates — diary-specific covariates only for daily diary outcomes
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

    diary_affect_vars = {"PA_score", "NA_score", "NA_score_log"}

    print(f"\nCovariates:")
    print(f"  Base (all models): C5PAGE, sex, {len(race_dummies)} race, {len(twin_dummies)} twin dummies")
    print(f"  Diary-only (PA/NA outcomes): time_P2_P5, n_days_complete")

    # ========================================================================
    # Define Samples
    # ========================================================================
    full_sample = get_full_sample(df)
    conservative_sample = get_conservative_sample(df)

    # ========================================================================
    # Run Analyses
    # ========================================================================
    full_corr, full_reg = run_sample_analysis(
        full_sample, "full", fc_vars, affect_vars,
        base_covariates, diary_covariates, diary_affect_vars
    )

    cons_corr, cons_reg = run_sample_analysis(
        conservative_sample, "conservative", fc_vars, affect_vars,
        base_covariates, diary_covariates, diary_affect_vars
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)

    print(f"\nFull sample:")
    print(f"  N = {len(full_sample)}")
    print(f"  Correlations: {len(full_corr)}")
    print(f"  Regressions: {len(full_reg)}")
    if len(full_corr) > 0:
        print(f"  Significant correlations (p < 0.05): {(full_corr['p'] < 0.05).sum()}")
    if len(full_reg) > 0:
        print(f"  Significant regressions (p < 0.05): {(full_reg['p_fc'] < 0.05).sum()}")

    print(f"\nConservative sample:")
    print(f"  N = {len(conservative_sample)}")
    print(f"  Correlations: {len(cons_corr)}")
    print(f"  Regressions: {len(cons_reg)}")
    if len(cons_corr) > 0:
        print(f"  Significant correlations (p < 0.05): {(cons_corr['p'] < 0.05).sum()}")
    if len(cons_reg) > 0:
        print(f"  Significant regressions (p < 0.05): {(cons_reg['p_fc'] < 0.05).sum()}")


if __name__ == "__main__":
    main()
