#!/usr/bin/env python3
"""
01_affect_age_replication.py

Replication of age-related differences in affect in MIDUS 3.

Tests associations between age and affect in two samples:
1) Full daily diary sample: Daily affect × age at P2 (C2PAGE)
2) Neuroscience sample: Daily affect × age at P2/P5, PANAS × age at P5

Inputs:
- data/processed/midus_merged_clean.csv (cleaned master dataset)

Outputs:
- results/tables/01_affect_age_correlations.csv (zero-order correlations)
- results/tables/01_affect_age_regressions.csv (regression results with covariates)

Run from project root directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

# ============================================================================
# Paths and Constants
# ============================================================================
DATA_FILE = Path("data/processed/midus_merged_clean.csv")
OUT_DIR = Path("results/tables")

# Affect outcome variables
DAILY_AFFECT = ["PA_score", "NA_score", "NA_score_log"]
PANAS_AFFECT = ["C5SPGP", "C5SPGN"]

# Minimum sample sizes for analyses
MIN_N_CORRELATION = 10
MIN_N_REGRESSION = 20

# ============================================================================
# Helper Functions
# ============================================================================
def zero_order_corr(data, x, y, min_n=MIN_N_CORRELATION):
    """
    Compute zero-order Pearson correlation between two variables.

    Args:
        data: DataFrame containing the variables
        x: Name of predictor variable
        y: Name of outcome variable
        min_n: Minimum sample size required (default: 10)

    Returns:
        Tuple of (correlation, p-value, n)
    """
    tmp = data[[x, y]].dropna()
    if len(tmp) < min_n:
        return np.nan, np.nan, len(tmp)
    r, p = pearsonr(tmp[x], tmp[y])
    return r, p, len(tmp)


def run_regression(data, outcome, age_var, covars, min_n=MIN_N_REGRESSION):
    """
    Run OLS regression: outcome ~ age + covariates.

    Args:
        data: DataFrame containing all variables
        outcome: Name of outcome variable
        age_var: Name of age predictor
        covars: List of covariate names
        min_n: Minimum sample size required (default: 20)

    Returns:
        statsmodels RegressionResults object, or None if insufficient N
    """
    cols = [outcome, age_var] + covars
    tmp = data[cols].dropna()
    if len(tmp) < min_n:
        return None

    formula = f"{outcome} ~ {age_var} + " + " + ".join(covars)
    return smf.ols(formula, data=tmp).fit()

# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"✓ Loaded {len(df)} participants")

    # ========================================================================
    # Define Samples
    # ========================================================================
    samples = {
        "daily_diary_full": {
            "filter": df["PA_score"].notna(),
            "outcomes": DAILY_AFFECT,
            "age_vars": ["C2PAGE"]
        },
        "neuro_sample": {
            "filter": df["C5PAGE"].notna(),
            "outcomes": DAILY_AFFECT + PANAS_AFFECT,
            "age_vars": {  # Dict to handle daily vs PANAS
                "daily": ["C2PAGE", "C5PAGE"],
                "panas": ["C5PAGE"]
            }
        }
    }

    # ========================================================================
    # Prepare Covariates
    # ========================================================================
    race_covars = [c for c in df.columns if c.startswith("race_")]
    twin_covars = [c for c in df.columns if c.startswith("twin_pair_")]
    base_covars = ["sex", "educ"] + race_covars + twin_covars

    # Ensure covariates are numeric
    df[base_covars] = df[base_covars].apply(pd.to_numeric, errors='coerce')

    print(f"Covariates: sex, educ, {len(race_covars)} race dummies, {len(twin_covars)} twin pair dummies")

    # ========================================================================
    # Run Analyses
    # ========================================================================
    print("\nRunning analyses...")
    corr_rows = []
    reg_rows = []

    # Daily diary full sample
    print("\n  Sample: daily_diary_full")
    sample_df = df.loc[samples["daily_diary_full"]["filter"]].copy()
    print(f"    N = {len(sample_df)}")

    for outcome in samples["daily_diary_full"]["outcomes"]:
        for age_var in samples["daily_diary_full"]["age_vars"]:
            # Zero-order correlation
            r, p, n = zero_order_corr(sample_df, age_var, outcome)
            corr_rows.append({
                "sample": "daily_diary_full",
                "outcome": outcome,
                "age_var": age_var,
                "r": r,
                "p": p,
                "n": n
            })

            # Regression with covariates
            model = run_regression(sample_df, outcome, age_var, base_covars)
            if model is not None:
                reg_rows.append({
                    "sample": "daily_diary_full",
                    "outcome": outcome,
                    "age_var": age_var,
                    "beta_age": model.params.get(age_var, np.nan),
                    "se_age": model.bse.get(age_var, np.nan),
                    "p_age": model.pvalues.get(age_var, np.nan),
                    "n": int(model.nobs),
                    "r2": model.rsquared
                })

    # Neuroscience sample
    print("\n  Sample: neuro_sample")
    sample_df = df.loc[samples["neuro_sample"]["filter"]].copy()
    print(f"    N = {len(sample_df)}")

    for outcome in samples["neuro_sample"]["outcomes"]:
        # Determine appropriate age variable(s) based on outcome type
        if outcome in PANAS_AFFECT:
            age_vars = samples["neuro_sample"]["age_vars"]["panas"]
        else:
            age_vars = samples["neuro_sample"]["age_vars"]["daily"]

        for age_var in age_vars:
            # Zero-order correlation
            r, p, n = zero_order_corr(sample_df, age_var, outcome)
            corr_rows.append({
                "sample": "neuro_sample",
                "outcome": outcome,
                "age_var": age_var,
                "r": r,
                "p": p,
                "n": n
            })

            # Regression with covariates
            model = run_regression(sample_df, outcome, age_var, base_covars)
            if model is not None:
                reg_rows.append({
                    "sample": "neuro_sample",
                    "outcome": outcome,
                    "age_var": age_var,
                    "beta_age": model.params.get(age_var, np.nan),
                    "se_age": model.bse.get(age_var, np.nan),
                    "p_age": model.pvalues.get(age_var, np.nan),
                    "n": int(model.nobs),
                    "r2": model.rsquared
                })

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\nSaving results...")

    corr_file = OUT_DIR / "01_affect_age_correlations.csv"
    pd.DataFrame(corr_rows).to_csv(corr_file, index=False)
    print(f"✓ Correlations saved to {corr_file}")
    print(f"  {len(corr_rows)} correlation tests")

    reg_file = OUT_DIR / "01_affect_age_regressions.csv"
    pd.DataFrame(reg_rows).to_csv(reg_file, index=False)
    print(f"✓ Regressions saved to {reg_file}")
    print(f"  {len(reg_rows)} regression models")

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
