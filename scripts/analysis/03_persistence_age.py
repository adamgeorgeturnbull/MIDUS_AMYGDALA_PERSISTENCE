#!/usr/bin/env python3
"""
03_persistence_age.py

Test whether amygdala persistence to negative images decreases with age
(Extension #1, Confirmatory).

Hypothesis: Amygdala persistence to negative images is negatively associated
with age at the neuroscience visit (C5PAGE).

Analysis plan:
1. Zero-order correlations between persistence and age
2. OLS regressions: persistence ~ C5PAGE + covariates

Covariates:
- Gender (sex)
- Race (dummy-coded)
- Twin pairs (dummy-coded)
- Time between visits (time_P2_P5)
- Number of diary days completed (n_days_complete)

Note: Age (C5PAGE) is the predictor, NOT a covariate.

Key analysis decisions:
- Persistence measures are Fisher z-transformed before analysis
- Primary: Cross-run negative persistence (L, R, bilateral)
- Sensitivity: Cross-run positive, concatenated negative
- Two-tailed tests reported; hypothesis is directional (negative association)

Inputs:
- data/processed/midus_with_fmri.csv

Outputs:
- results/tables/03_persistence_age_correlations_full.csv
- results/tables/03_persistence_age_regressions_full.csv
- results/tables/03_persistence_age_correlations_conservative.csv
- results/tables/03_persistence_age_regressions_conservative.csv

Run from project root directory.
"""

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

MIN_N_CORR = 10
MIN_N_REG = 20


# ============================================================================
# Helper Functions
# ============================================================================
def fisher_z(r):
    """Apply Fisher z-transformation to correlation coefficient."""
    return 0.5 * np.log((1 + r) / (1 - r))


def get_full_sample(df):
    """Define full analysis sample: participants with imaging data and valid age."""
    has_persistence = df["has_neg_persistence"] == 1
    has_age = df["C5PAGE"].notna()
    sample = df[has_persistence & has_age].copy()

    print(f"\nFull sample: {len(sample)} participants")
    print(f"  Age range: {sample['C5PAGE'].min():.0f} - {sample['C5PAGE'].max():.0f}")
    print(f"  Age mean (SD): {sample['C5PAGE'].mean():.1f} ({sample['C5PAGE'].std():.1f})")

    return sample


def get_conservative_sample(df):
    """Define conservative sample with strict QC criteria."""
    sample = get_full_sample(df)
    qc_pass = sample["qc_conservative"] == 1
    conservative = sample[qc_pass].copy()

    print(f"Conservative sample: {len(conservative)} participants")
    if len(conservative) > 0:
        print(f"  Age range: {conservative['C5PAGE'].min():.0f} - {conservative['C5PAGE'].max():.0f}")
        print(f"  Age mean (SD): {conservative['C5PAGE'].mean():.1f} ({conservative['C5PAGE'].std():.1f})")

    return conservative


def compute_correlations(df, persistence_vars, age_var="C5PAGE"):
    """Compute zero-order Pearson correlations between persistence and age."""
    results = []

    for persist_var in persistence_vars:
        data = df[[persist_var, age_var]].dropna()
        n = len(data)

        if n < MIN_N_CORR:
            continue

        r, p = stats.pearsonr(data[persist_var], data[age_var])
        results.append({
            "persistence_var": persist_var,
            "n": n,
            "r": r,
            "p": p,
        })

    return pd.DataFrame(results)


def run_regression(df, persistence_var, age_var, covariates):
    """
    Run OLS regression: persistence ~ age + covariates.

    Returns dictionary with results, or None if insufficient data.
    """
    vars_needed = [persistence_var, age_var] + covariates
    data = df[vars_needed].dropna()

    if len(data) < MIN_N_REG:
        return None

    y = data[persistence_var]
    X = data[[age_var] + covariates]
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"    Error fitting model for {persistence_var} ~ {age_var}: {e}")
        return None

    return {
        "persistence_var": persistence_var,
        "n": int(model.nobs),
        "beta_age": model.params[age_var],
        "se_age": model.bse[age_var],
        "t_age": model.tvalues[age_var],
        "p_age": model.pvalues[age_var],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }


def run_all_regressions(df, persistence_vars, age_var, covariates):
    """Run all persistence ~ age regressions."""
    results = []
    for persist_var in persistence_vars:
        result = run_regression(df, persist_var, age_var, covariates)
        if result is not None:
            results.append(result)
    return pd.DataFrame(results)


def run_sample_analysis(sample, sample_name, persistence_vars, age_var, covariates):
    """Run complete analysis (correlations + regressions) for a given sample."""
    print("\n" + "=" * 80)
    print(f"Analysis: {sample_name}")
    print("=" * 80)

    # Zero-order correlations
    print("\nComputing zero-order correlations (persistence ~ age)...")
    corr_results = compute_correlations(sample, persistence_vars, age_var)

    if len(corr_results) > 0:
        print(f"  Computed {len(corr_results)} correlations\n")
        for _, row in corr_results.iterrows():
            sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            print(f"  {row['persistence_var']:45s}: "
                  f"r = {row['r']:7.4f}, p = {row['p']:.4f}{sig:3s}, n = {int(row['n'])}")
    else:
        print("  No correlations computed (insufficient data)")

    corr_file = RESULTS_DIR / f"03_persistence_age_correlations_{sample_name}.csv"
    corr_results.to_csv(corr_file, index=False)
    print(f"\n  Saved to {corr_file}")

    # OLS regressions
    print("\nRunning OLS regressions (persistence ~ age + covariates)...")
    reg_results = run_all_regressions(sample, persistence_vars, age_var, covariates)

    if len(reg_results) > 0:
        print(f"  Computed {len(reg_results)} regressions\n")
        for _, row in reg_results.iterrows():
            sig = "***" if row["p_age"] < 0.001 else "**" if row["p_age"] < 0.01 else "*" if row["p_age"] < 0.05 else ""
            print(f"  {row['persistence_var']:45s}: "
                  f"b = {row['beta_age']:8.5f}, p = {row['p_age']:.4f}{sig:3s}, "
                  f"n = {int(row['n'])}")
    else:
        print("  No regressions computed (insufficient data)")

    reg_file = RESULTS_DIR / f"03_persistence_age_regressions_{sample_name}.csv"
    reg_results.to_csv(reg_file, index=False)
    print(f"\n  Saved to {reg_file}")

    return corr_results, reg_results


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Analysis 03: Amygdala Persistence ~ Age (Extension #1, Confirmatory)")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} participants")

    # ========================================================================
    # Fisher z-transform persistence measures
    # ========================================================================
    persistence_r_vars = [
        "neg_persist_crossrun_mean_r_L",
        "neg_persist_crossrun_mean_r_R",
        "neg_persist_crossrun_mean_r_bilateral",
        "pos_persist_crossrun_mean_r_L",
        "pos_persist_crossrun_mean_r_R",
        "pos_persist_crossrun_mean_r_bilateral",
        "neg_persist_concat_r_L",
        "neg_persist_concat_r_R",
        "neg_persist_concat_r_bilateral",
    ]

    for var in persistence_r_vars:
        z_var = var.replace("_r_", "_z_")
        df[z_var] = fisher_z(df[var])

    # ========================================================================
    # Define Variables
    # ========================================================================
    age_var = "C5PAGE"

    persistence_vars = [
        "neg_persist_crossrun_mean_z_L",
        "neg_persist_crossrun_mean_z_R",
        "neg_persist_crossrun_mean_z_bilateral",
        "pos_persist_crossrun_mean_z_L",
        "pos_persist_crossrun_mean_z_R",
        "pos_persist_crossrun_mean_z_bilateral",
        "neg_persist_concat_z_L",
        "neg_persist_concat_z_R",
        "neg_persist_concat_z_bilateral",
    ]

    race_dummies = [col for col in df.columns if col.startswith("race_")]
    twin_dummies = [col for col in df.columns if col.startswith("twin_pair_")]

    covariates = [
        "sex",
        "time_P2_P5",
        "n_days_complete",
    ] + race_dummies + twin_dummies

    print(f"\nPredictor: {age_var} (age at neuroscience visit)")
    print(f"Outcomes: {len(persistence_vars)} persistence measures (Fisher z)")
    print(f"  Primary: 3 cross-run negative (L, R, bilateral)")
    print(f"  Sensitivity: 3 cross-run positive + 3 concatenated negative")
    print(f"Covariates: sex, time_P2_P5, n_days_complete, "
          f"{len(race_dummies)} race, {len(twin_dummies)} twin dummies")
    print(f"  NOTE: C5PAGE is the predictor, not a covariate")

    # ========================================================================
    # Define Samples and Run Analyses
    # ========================================================================
    full_sample = get_full_sample(df)
    conservative_sample = get_conservative_sample(df)

    full_corr, full_reg = run_sample_analysis(
        full_sample, "full", persistence_vars, age_var, covariates
    )

    cons_corr, cons_reg = run_sample_analysis(
        conservative_sample, "conservative", persistence_vars, age_var, covariates
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nHypothesis: Negative persistence decreases with age (negative r expected)")

    for label, corr_df, reg_df, n in [
        ("Full sample", full_corr, full_reg, len(full_sample)),
        ("Conservative sample", cons_corr, cons_reg, len(conservative_sample)),
    ]:
        print(f"\n{label} (N={n}):")
        if len(corr_df) > 0:
            n_sig_corr = (corr_df["p"] < 0.05).sum()
            print(f"  Significant correlations: {n_sig_corr}/{len(corr_df)}")
            primary = corr_df[corr_df["persistence_var"].str.contains("neg_persist_crossrun")]
            for _, row in primary.iterrows():
                direction = "neg (as hypothesized)" if row["r"] < 0 else "pos (opposite)"
                print(f"    {row['persistence_var']}: r = {row['r']:.4f}, p = {row['p']:.4f} [{direction}]")
        else:
            print("  No correlations (insufficient data)")

        if len(reg_df) > 0:
            n_sig_reg = (reg_df["p_age"] < 0.05).sum()
            print(f"  Significant regressions: {n_sig_reg}/{len(reg_df)}")
        else:
            print("  No regressions (insufficient data)")


if __name__ == "__main__":
    main()
