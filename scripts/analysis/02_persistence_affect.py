#!/usr/bin/env python3
"""
02_persistence_affect.py

Test associations between amygdala persistence to negative images and daily
life affect (replication of Puccetti et al., 2021).

Confirmatory analysis testing whether amygdala persistence to negative images
is associated with daily negative and positive affect.

Analysis plan (following Puccetti et al., 2021):
1. Zero-order correlations between persistence and affect
2. Multiple linear regressions controlling for:
   - Age (C5PAGE)
   - Gender (sex)
   - Race (dummy-coded)
   - Twin status (dummy-coded for each twin pair)
   - Time between visits (time_P2_P5)
   - Number of diary interviews completed (n_days_complete)

Key analysis decisions:
- Persistence measures are Fisher z-transformed before analysis
- Primary persistence: Cross-run negative persistence (replication)
- Sensitivity persistence: Cross-run positive, concatenated negative
- Primary affect outcomes: Daily diary PA, NA, and NA_log
- Secondary affect outcomes: PANAS (C5SPGP, C5SPGN, C5SPGN_log)

Two versions:
- Full sample: All participants with imaging + affect data
- Conservative sample: Mean FD < 0.5 AND all 3 runs available

Inputs:
- data/processed/midus_with_fmri.csv

Outputs:
- results/tables/02_persistence_affect_correlations_full.csv
- results/tables/02_persistence_affect_regressions_full.csv
- results/tables/02_persistence_affect_correlations_conservative.csv
- results/tables/02_persistence_affect_regressions_conservative.csv

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
    """
    Apply Fisher z-transformation to correlation coefficient.

    Args:
        r: Correlation coefficient (scalar or array)

    Returns:
        Fisher z-transformed value
    """
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
    Define full analysis sample: participants with both imaging and affect data.

    Args:
        df: Full dataset

    Returns:
        DataFrame filtered to analysis sample
    """
    # Must have negative persistence data and at least one affect measure
    has_persistence = df["has_neg_persistence"] == 1
    has_affect = df[["PA_score", "NA_score"]].notna().any(axis=1)

    sample = df[has_persistence & has_affect].copy()

    print(f"\nFull sample: {len(sample)} participants")
    print(f"  - With daily diary affect: {sample['PA_score'].notna().sum()}")

    return sample


def get_conservative_sample(df):
    """
    Define conservative analysis sample with strict quality criteria.

    Uses QC flags from manual inspection (script 08_process_fmri_qc.py):
    1. All 3 runs pass visual QC
    2. Mean FD < 0.5 across all runs

    Args:
        df: Full dataset

    Returns:
        DataFrame filtered to conservative sample
    """
    # Start with full sample
    sample = get_full_sample(df)

    # Apply QC criteria using qc_conservative flag
    qc_pass = sample["qc_conservative"] == 1

    conservative = sample[qc_pass].copy()

    print(f"\nConservative sample: {len(conservative)} participants")
    print(f"  - Excluded for failed QC: {(~qc_pass).sum()}")
    print(f"  - With daily diary affect: {conservative['PA_score'].notna().sum()}")

    # Additional breakdown if available
    if "all_runs_pass" in sample.columns and "fd_pass" in sample.columns:
        excluded = sample[~qc_pass]
        runs_fail = (excluded["all_runs_pass"] == 0).sum()
        fd_fail = (excluded["fd_pass"] == 0).sum()
        both_fail = ((excluded["all_runs_pass"] == 0) & (excluded["fd_pass"] == 0)).sum()
        print(f"  - Failed run QC only: {runs_fail - both_fail}")
        print(f"  - Failed FD criterion only: {fd_fail - both_fail}")
        print(f"  - Failed both: {both_fail}")

    return conservative


def compute_correlations(df, persistence_vars, affect_vars):
    """
    Compute zero-order correlations between persistence and affect measures.

    Args:
        df: Dataset
        persistence_vars: List of persistence variable names
        affect_vars: List of affect variable names

    Returns:
        DataFrame with correlation results
    """
    results = []

    for persist_var in persistence_vars:
        for affect_var in affect_vars:
            # Get complete cases
            data = df[[persist_var, affect_var]].dropna()
            n = len(data)

            if n < MIN_N_CORR:
                continue

            # Compute correlation
            r, p = stats.pearsonr(data[persist_var], data[affect_var])

            results.append({
                "persistence_var": persist_var,
                "affect_var": affect_var,
                "n": n,
                "r": r,
                "p": p,
            })

    return pd.DataFrame(results)


def run_regression(df, persistence_var, affect_var, covariates):
    """
    Run OLS regression: affect ~ persistence + covariates.

    Following Puccetti et al. (2021), controls for:
    - Age, gender, race, twin status, time between visits, number of diary days

    Args:
        df: Dataset
        persistence_var: Name of persistence predictor
        affect_var: Name of affect outcome
        covariates: List of covariate names

    Returns:
        Dictionary with regression results or None if insufficient data
    """
    # Prepare data
    vars_needed = [affect_var, persistence_var] + covariates
    data = df[vars_needed].dropna()

    if len(data) < MIN_N_REG:
        return None

    # Prepare design matrix
    y = data[affect_var]
    X = data[[persistence_var] + covariates]
    X = sm.add_constant(X)

    # Fit model
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"    Error fitting model for {affect_var} ~ {persistence_var}: {e}")
        return None

    # Extract results for persistence variable
    persist_idx = 1  # First column after intercept

    result = {
        "persistence_var": persistence_var,
        "affect_var": affect_var,
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


def run_all_regressions(df, persistence_vars, affect_vars, covariates):
    """
    Run all persistence × affect regressions.

    Args:
        df: Dataset
        persistence_vars: List of persistence variable names
        affect_vars: List of affect variable names
        covariates: List of covariate names

    Returns:
        DataFrame with regression results
    """
    results = []

    for persist_var in persistence_vars:
        for affect_var in affect_vars:
            result = run_regression(df, persist_var, affect_var, covariates)
            if result is not None:
                results.append(result)

    return pd.DataFrame(results)


def run_sample_analysis(sample, sample_name, persistence_vars, affect_vars, covariates):
    """
    Run complete analysis (correlations + regressions) for a given sample.

    Args:
        sample: DataFrame with sample data
        sample_name: Name of sample (for output files)
        persistence_vars: List of persistence variables
        affect_vars: List of affect variables
        covariates: List of covariates

    Returns:
        Tuple of (corr_results, reg_results)
    """
    print("\n" + "=" * 80)
    print(f"Analysis: {sample_name}")
    print("=" * 80)

    # ========================================================================
    # Zero-Order Correlations
    # ========================================================================
    print("\nComputing zero-order correlations...")

    corr_results = compute_correlations(sample, persistence_vars, affect_vars)

    if len(corr_results) > 0:
        print(f"\n✓ Computed {len(corr_results)} correlations")
        print("\nCorrelation results:")
        for _, row in corr_results.iterrows():
            sig_marker = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            print(f"  {row['persistence_var']:40s} × {row['affect_var']:15s}: "
                  f"r = {row['r']:6.3f}, p = {row['p']:.4f}{sig_marker:3s}, n = {int(row['n'])}")
    else:
        print("\n✗ No correlations computed (insufficient data)")

    # Save correlations
    corr_file = RESULTS_DIR / f"02_persistence_affect_correlations_{sample_name}.csv"
    corr_results.to_csv(corr_file, index=False)
    print(f"\n✓ Correlation results saved to {corr_file}")

    # ========================================================================
    # OLS Regressions with Covariates
    # ========================================================================
    print("\nRunning OLS regressions (controlling for covariates)...")

    reg_results = run_all_regressions(sample, persistence_vars, affect_vars, covariates)

    if len(reg_results) > 0:
        print(f"\n✓ Computed {len(reg_results)} regressions")
        print("\nRegression results (persistence effect):")
        for _, row in reg_results.iterrows():
            sig_marker = "***" if row["p_persistence"] < 0.001 else "**" if row["p_persistence"] < 0.01 else "*" if row["p_persistence"] < 0.05 else ""
            print(f"  {row['persistence_var']:40s} → {row['affect_var']:15s}: "
                  f"β = {row['beta_persistence']:6.3f}, p = {row['p_persistence']:.4f}{sig_marker:3s}, "
                  f"n = {int(row['n'])}")
    else:
        print("\n✗ No regressions computed (insufficient data)")

    # Save regressions
    reg_file = RESULTS_DIR / f"02_persistence_affect_regressions_{sample_name}.csv"
    reg_results.to_csv(reg_file, index=False)
    print(f"\n✓ Regression results saved to {reg_file}")

    return corr_results, reg_results


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Analysis 02: Amygdala Persistence × Daily Affect Associations")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"✓ Loaded {len(df)} participants")

    # ========================================================================
    # Create Fisher z-transformed persistence measures
    # ========================================================================
    print("\nCreating Fisher z-transformed persistence measures...")

    # Cross-run negative persistence (primary measure)
    persistence_r_vars = [
        "neg_persist_crossrun_mean_r_L",
        "neg_persist_crossrun_mean_r_R",
        "neg_persist_crossrun_mean_r_bilateral",
    ]

    # Cross-run positive persistence (sensitivity analysis)
    persistence_r_vars += [
        "pos_persist_crossrun_mean_r_L",
        "pos_persist_crossrun_mean_r_R",
        "pos_persist_crossrun_mean_r_bilateral",
    ]

    # Concatenated negative persistence (sensitivity analysis)
    # Note: concat data already has z values from original computation
    persistence_r_vars += [
        "neg_persist_concat_r_L",
        "neg_persist_concat_r_R",
        "neg_persist_concat_r_bilateral",
    ]

    for var in persistence_r_vars:
        z_var = var.replace("_r_", "_z_")
        df[z_var] = fisher_z(df[var])
        print(f"  ✓ Created {z_var}")

    # ========================================================================
    # Define Variables
    # ========================================================================
    print("\n" + "=" * 80)
    print("Analysis Variables")
    print("=" * 80)

    # Persistence measures (Fisher z-transformed)
    # Primary: Cross-run negative persistence
    # Sensitivity: Cross-run positive persistence, concatenated negative persistence
    persistence_vars = [
        # Cross-run negative persistence (PRIMARY - replication of Puccetti et al., 2021)
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

    # Affect measures
    # Primary: Daily diary (following Puccetti et al., 2021)
    # Secondary: PANAS (at neuroscience visit)
    affect_vars = [
        # Daily diary
        "PA_score",          # Daily diary positive affect
        "NA_score",          # Daily diary negative affect
        "NA_score_log",      # Daily diary negative affect (log-transformed)
        # PANAS (secondary)
        "C5SPGP",            # PANAS positive affect
        "C5SPGN",            # PANAS negative affect
        "C5SPGN_log",        # PANAS negative affect (log-transformed)
    ]

    print(f"\nAffect measures:")
    print(f"  Primary (Daily Diary):")
    print(f"  - PA_score")
    print(f"  - NA_score")
    print(f"  - NA_score_log")
    print(f"  Secondary (PANAS):")
    print(f"  - C5SPGP")
    print(f"  - C5SPGN")
    print(f"  - C5SPGN_log")

    # Covariates (following Puccetti et al., 2021)
    race_dummies = [col for col in df.columns if col.startswith("race_")]
    twin_dummies = [col for col in df.columns if col.startswith("twin_pair_")]

    covariates = [
        "C5PAGE",           # Age at neuroscience visit
        "sex",              # Gender
        "time_P2_P5",       # Time between P2 and P5 visits (months)
        "n_days_complete",  # Number of diary interviews completed
    ] + race_dummies + twin_dummies

    print(f"\nCovariates (following Puccetti et al., 2021):")
    print(f"  - Age: C5PAGE")
    print(f"  - Gender: sex")
    print(f"  - Time between visits: time_P2_P5")
    print(f"  - Number of diary days: n_days_complete")
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
    # Full sample
    full_corr, full_reg = run_sample_analysis(
        full_sample, "full", persistence_vars, affect_vars, covariates
    )

    # Conservative sample
    cons_corr, cons_reg = run_sample_analysis(
        conservative_sample, "conservative", persistence_vars, affect_vars, covariates
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
