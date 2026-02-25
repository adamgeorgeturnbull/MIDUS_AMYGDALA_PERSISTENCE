#!/usr/bin/env python3
"""
06_fc_persistence.py

Test associations between amygdala-vmPFC task-based functional connectivity
(beta-series, neg > neu) and amygdala persistence.

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

Note: FC values are already Fisher z-transformed from runBStaskFC.sh.
Persistence values are Fisher z-transformed at script start (r -> z).

Analysis plan:
1. Zero-order correlations between FC and persistence
2. Multiple linear regressions controlling for:
   - Age (C5PAGE)
   - Gender (sex)
   - Race (dummy-coded)
   - Twin status (dummy-coded)
   - Time between visits (time_P2_P5)
   - Number of diary interviews completed (n_days_complete)

Two versions:
- Full sample: All participants with FC + persistence data
- Conservative sample: Full + mean FD < 0.5 AND all 3 runs pass QC

Inputs:
- data/processed/midus_with_fmri.csv

Outputs:
- results/tables/06_fc_persistence_correlations_full.csv
- results/tables/06_fc_persistence_regressions_full.csv
- results/tables/06_fc_persistence_correlations_conservative.csv
- results/tables/06_fc_persistence_regressions_conservative.csv

Run from project root directory.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ============================================================================
# Helper Functions for Fisher z-transform
# ============================================================================
def fisher_z(r):
    """Apply Fisher z-transformation to correlation coefficient."""
    return 0.5 * np.log((1 + r) / (1 - r))


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
    Define full analysis sample: participants with both FC and persistence data.
    """
    has_fc = df["has_beta_series"] == 1
    has_persistence = df["has_neg_persistence"] == 1

    sample = df[has_fc & has_persistence].copy()

    print(f"\nFull sample: {len(sample)} participants")

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

    if "all_runs_pass" in sample.columns and "fd_pass" in sample.columns:
        excluded = sample[~qc_pass]
        runs_fail = (excluded["all_runs_pass"] == 0).sum()
        fd_fail = (excluded["fd_pass"] == 0).sum()
        both_fail = ((excluded["all_runs_pass"] == 0) & (excluded["fd_pass"] == 0)).sum()
        print(f"  - Failed run QC only: {runs_fail - both_fail}")
        print(f"  - Failed FD criterion only: {fd_fail - both_fail}")
        print(f"  - Failed both: {both_fail}")

    return conservative


def compute_correlations(df, fc_vars, persistence_vars):
    """
    Compute zero-order correlations between FC and persistence measures.
    """
    results = []

    for fc_var in fc_vars:
        for persist_var in persistence_vars:
            data = df[[fc_var, persist_var]].dropna()
            n = len(data)

            if n < MIN_N_CORR:
                continue

            r, p = stats.pearsonr(data[fc_var], data[persist_var])

            results.append({
                "fc_var": fc_var,
                "persistence_var": persist_var,
                "n": n,
                "r": r,
                "p": p,
            })

    return pd.DataFrame(results)


def run_regression(df, fc_var, persistence_var, covariates):
    """
    Run OLS regression: FC ~ persistence + covariates.
    """
    vars_needed = [fc_var, persistence_var] + covariates
    data = df[vars_needed].dropna()

    if len(data) < MIN_N_REG:
        return None

    y = data[fc_var]
    X = data[[persistence_var] + covariates]
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"    Error fitting model for {fc_var} ~ {persistence_var}: {e}")
        return None

    persist_idx = 1  # First column after intercept

    result = {
        "fc_var": fc_var,
        "persistence_var": persistence_var,
        "n": int(model.nobs),
        "beta_persistence": model.params.iloc[persist_idx],
        "se_persistence": model.bse.iloc[persist_idx],
        "t_persistence": model.tvalues.iloc[persist_idx],
        "p_persistence": model.pvalues.iloc[persist_idx],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue,
    }

    return result


def run_all_regressions(df, fc_vars, persistence_vars, covariates):
    """
    Run all FC x persistence regressions.
    """
    results = []

    for fc_var in fc_vars:
        for persist_var in persistence_vars:
            result = run_regression(df, fc_var, persist_var, covariates)
            if result is not None:
                results.append(result)

    return pd.DataFrame(results)


def run_sample_analysis(sample, sample_name, fc_vars, persistence_vars, covariates):
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

    corr_results = compute_correlations(sample, fc_vars, persistence_vars)

    if len(corr_results) > 0:
        print(f"\n  Computed {len(corr_results)} correlations")
        print("\nCorrelation results:")
        for _, row in corr_results.iterrows():
            sig_marker = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            print(f"  {row['fc_var']:45s} x {row['persistence_var']:40s}: "
                  f"r = {row['r']:6.3f}, p = {row['p']:.4f}{sig_marker:3s}, n = {int(row['n'])}")
    else:
        print("\n  No correlations computed (insufficient data)")

    corr_file = RESULTS_DIR / f"06_fc_persistence_correlations_{sample_name}.csv"
    corr_results.to_csv(corr_file, index=False)
    print(f"\n  Correlation results saved to {corr_file}")

    # ========================================================================
    # OLS Regressions with Covariates
    # ========================================================================
    print("\nRunning OLS regressions (controlling for covariates)...")

    reg_results = run_all_regressions(sample, fc_vars, persistence_vars, covariates)

    if len(reg_results) > 0:
        print(f"\n  Computed {len(reg_results)} regressions")
        print("\nRegression results (persistence effect):")
        for _, row in reg_results.iterrows():
            sig_marker = "***" if row["p_persistence"] < 0.001 else "**" if row["p_persistence"] < 0.01 else "*" if row["p_persistence"] < 0.05 else ""
            print(f"  {row['fc_var']:45s} ~ {row['persistence_var']:40s}: "
                  f"B = {row['beta_persistence']:7.3f}, p = {row['p_persistence']:.4f}{sig_marker:3s}, "
                  f"n = {int(row['n'])}")
    else:
        print("\n  No regressions computed (insufficient data)")

    reg_file = RESULTS_DIR / f"06_fc_persistence_regressions_{sample_name}.csv"
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
    print("Analysis 06: Task-Based FC (Amygdala-vmPFC) x Amygdala Persistence")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df)} participants")

    # ========================================================================
    # Create Fisher z-transformed persistence measures
    # ========================================================================
    print("\nCreating Fisher z-transformed persistence measures...")

    persistence_r_vars = [
        # Cross-run negative persistence (primary)
        "neg_persist_crossrun_mean_r_L",
        "neg_persist_crossrun_mean_r_R",
        "neg_persist_crossrun_mean_r_bilateral",
        # Cross-run positive persistence (sensitivity)
        "pos_persist_crossrun_mean_r_L",
        "pos_persist_crossrun_mean_r_R",
        "pos_persist_crossrun_mean_r_bilateral",
        # Concatenated negative persistence (sensitivity)
        "neg_persist_concat_r_L",
        "neg_persist_concat_r_R",
        "neg_persist_concat_r_bilateral",
    ]

    for var in persistence_r_vars:
        z_var = var.replace("_r_", "_z_")
        df[z_var] = fisher_z(df[var])
        print(f"    {z_var}")

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

    # FC variables
    fc_vars = [
        # Left amygdala seeds
        "conn_l_amyg_ant_vmPFC_neg_vs_neu",
        "conn_l_amyg_post_vmPFC_neg_vs_neu",
        "conn_l_amyg_safety_vs_threat",
        # Right amygdala seeds
        "conn_r_amyg_ant_vmPFC_neg_vs_neu",
        "conn_r_amyg_post_vmPFC_neg_vs_neu",
        "conn_r_amyg_safety_vs_threat",
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

    # Persistence variables (Fisher z-transformed)
    persistence_vars = [
        # Cross-run negative persistence (PRIMARY)
        "neg_persist_crossrun_mean_z_L",
        "neg_persist_crossrun_mean_z_R",
        "neg_persist_crossrun_mean_z_bilateral",
        # Cross-run positive persistence (SENSITIVITY)
        "pos_persist_crossrun_mean_z_L",
        "pos_persist_crossrun_mean_z_R",
        "pos_persist_crossrun_mean_z_bilateral",
        # Concatenated negative persistence (SENSITIVITY)
        "neg_persist_concat_z_L",
        "neg_persist_concat_z_R",
        "neg_persist_concat_z_bilateral",
    ]

    print(f"\nPersistence measures (Fisher z-transformed):")
    print(f"  Primary (Cross-run Negative):")
    print(f"  - neg_persist_crossrun_mean_z_L")
    print(f"  - neg_persist_crossrun_mean_z_R")
    print(f"  - neg_persist_crossrun_mean_z_bilateral")
    print(f"  Sensitivity (Cross-run Positive):")
    print(f"  - pos_persist_crossrun_mean_z_L")
    print(f"  - pos_persist_crossrun_mean_z_R")
    print(f"  - pos_persist_crossrun_mean_z_bilateral")
    print(f"  Sensitivity (Concatenated Negative):")
    print(f"  - neg_persist_concat_z_L")
    print(f"  - neg_persist_concat_z_R")
    print(f"  - neg_persist_concat_z_bilateral")

    # Covariates — no diary-specific covariates (time_P2_P5, n_days_complete)
    # since this analysis is entirely neuroscience-based (FC x persistence).
    race_dummies = [col for col in df.columns if col.startswith("race_")]
    twin_dummies = [col for col in df.columns if col.startswith("twin_pair_")]

    covariates = [
        "C5PAGE",
        "sex",
    ] + race_dummies + twin_dummies

    print(f"\nCovariates:")
    print(f"  - Age: C5PAGE")
    print(f"  - Gender: sex")
    print(f"  - Race dummies: {len(race_dummies)}")
    print(f"  - Twin pair dummies: {len(twin_dummies)}")
    print(f"  - Total covariates: {len(covariates)}")

    # ========================================================================
    # Define Samples
    # ========================================================================
    full_sample = get_full_sample(df)
    conservative_sample = get_conservative_sample(df)

    # ========================================================================
    # Run Analyses
    # ========================================================================
    full_corr, full_reg = run_sample_analysis(
        full_sample, "full", fc_vars, persistence_vars, covariates
    )

    cons_corr, cons_reg = run_sample_analysis(
        conservative_sample, "conservative", fc_vars, persistence_vars, covariates
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
        print(f"  Significant regressions (p < 0.05): {(full_reg['p_persistence'] < 0.05).sum()}")

    print(f"\nConservative sample:")
    print(f"  N = {len(conservative_sample)}")
    print(f"  Correlations: {len(cons_corr)}")
    print(f"  Regressions: {len(cons_reg)}")
    if len(cons_corr) > 0:
        print(f"  Significant correlations (p < 0.05): {(cons_corr['p'] < 0.05).sum()}")
    if len(cons_reg) > 0:
        print(f"  Significant regressions (p < 0.05): {(cons_reg['p_persistence'] < 0.05).sum()}")


if __name__ == "__main__":
    main()
