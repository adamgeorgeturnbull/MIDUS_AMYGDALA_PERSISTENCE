#!/usr/bin/env python3
"""
01b_affect_age_replication_mlm.py

Mixed-effects (multilevel) reanalysis of age-related differences in affect.

Replaces the OLS + twin-pair-dummy approach from 01_affect_age_replication.py
with linear mixed-effects models (random intercept for family). This avoids
the degrees-of-freedom cost of one dummy per twin pair while properly
accounting for non-independence within twin families.

Tests associations between age and affect in two samples:
1) Full daily diary sample: Daily affect x age at P2 (C2PAGE)
2) Neuroscience sample: Daily affect x age at P2/P5, PANAS x age at P5

Grouping variable (family_id):
  - Twins (SAMPLMAJ == 3 AND 2+ members share M2FAMNUM): M2FAMNUM
  - Everyone else: M2ID (cluster of size 1)

Fixed effects: age + sex + educ + race dummies
Random effects: random intercept for family_id

Correlations are identical to 01_affect_age_replication.py and are NOT re-run
here. Only the covariate-adjusted models differ (MLM vs OLS).

Inputs:
- data/processed/midus_merged_clean.csv (cleaned master dataset)

Outputs:
- results/tables/01b_affect_age_mlm.csv (MLM regression results)

Run from project root directory.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ============================================================================
# Paths and Constants
# ============================================================================
DATA_FILE = Path("data/processed/midus_merged_clean.csv")
OUT_DIR = Path("results/tables")

# Affect outcome variables
DAILY_AFFECT = ["PA_score", "NA_score", "NA_score_log"]
PANAS_AFFECT = ["C5SPGP", "C5SPGN"]

MIN_N_REGRESSION = 20


# ============================================================================
# Helper Functions
# ============================================================================
def create_family_id(df):
    """
    Create a family_id column for mixed-effects grouping.

    Twins (SAMPLMAJ == 3) who share a M2FAMNUM with at least one other
    participant in the dataset are grouped by that family number.
    All other participants get their own unique group (M2ID).
    """
    df = df.copy()

    is_twin_sample = df["SAMPLMAJ"] == 3
    twin_fam_counts = df.loc[is_twin_sample, "M2FAMNUM"].value_counts()
    paired_families = twin_fam_counts[twin_fam_counts > 1].index

    df["family_id"] = df["M2ID"].astype(str)
    paired_mask = is_twin_sample & df["M2FAMNUM"].isin(paired_families)
    df.loc[paired_mask, "family_id"] = "fam_" + df.loc[paired_mask, "M2FAMNUM"].astype(int).astype(str)

    n_paired = paired_mask.sum()
    n_families = df.loc[paired_mask, "family_id"].nunique()
    print(f"  Family grouping: {n_paired} participants in {n_families} twin families, "
          f"{(~paired_mask).sum()} singletons")

    return df


def run_mlm(data, outcome, age_var, covars, min_n=MIN_N_REGRESSION):
    """
    Run mixed-effects model: outcome ~ age + covariates, (1 | family_id).

    Args:
        data: DataFrame with family_id column
        outcome: Name of outcome variable
        age_var: Name of age predictor
        covars: List of covariate names
        min_n: Minimum sample size required

    Returns:
        Dictionary with results, or None if insufficient data
    """
    cols = [outcome, age_var, "family_id"] + covars
    tmp = data[cols].dropna()
    if len(tmp) < min_n:
        return None

    # Drop zero-variance covariates
    active_covars = [c for c in covars if tmp[c].std() > 0]

    formula = f"{outcome} ~ {age_var} + " + " + ".join(active_covars)

    # Try multiple optimizers
    methods = ["lbfgs", "powell"]
    result = None
    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula, data=tmp, groups=tmp["family_id"])
                result = model.fit(reml=True, method=method)
                break
        except Exception:
            continue

    if result is None:
        print(f"    Error fitting MLM for {outcome} ~ {age_var}: all optimizers failed")
        return None

    return {
        "outcome": outcome,
        "age_var": age_var,
        "beta_age": result.fe_params[age_var],
        "se_age": result.bse_fe[age_var],
        "z_age": result.tvalues[age_var],
        "p_age": result.pvalues[age_var],
        "n": int(result.nobs),
        "n_groups": int(result.nobs - result.df_resid),
        "group_var": result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else float(result.cov_re),
        "log_likelihood": result.llf,
        "converged": result.converged,
        "optimizer": method,
    }


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Analysis 01b (MLM): Affect x Age -- Mixed-Effects Models")
    print("=" * 70)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} participants")

    # Create family grouping for the full dataset
    df = create_family_id(df)

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
            "age_vars": {
                "daily": ["C2PAGE", "C5PAGE"],
                "panas": ["C5PAGE"]
            }
        }
    }

    # ========================================================================
    # Prepare Covariates
    # ========================================================================
    race_covars = [c for c in df.columns if c.startswith("race_")]
    # No twin dummies -- handled by random effect
    base_covars = ["sex", "educ"] + race_covars

    df[base_covars] = df[base_covars].apply(pd.to_numeric, errors='coerce')

    print(f"\nCovariates (no twin dummies -- handled by random effect):")
    print(f"  sex, educ, {len(race_covars)} race dummies")
    print(f"  Random: intercept | family_id")

    # ========================================================================
    # Run Analyses
    # ========================================================================
    print("\nRunning mixed-effects models...")
    mlm_rows = []

    # Daily diary full sample
    print("\n  Sample: daily_diary_full")
    sample_df = df.loc[samples["daily_diary_full"]["filter"]].copy()
    print(f"    N = {len(sample_df)}")

    for outcome in samples["daily_diary_full"]["outcomes"]:
        for age_var in samples["daily_diary_full"]["age_vars"]:
            result = run_mlm(sample_df, outcome, age_var, base_covars)
            if result is not None:
                result["sample"] = "daily_diary_full"
                mlm_rows.append(result)
                sig = "*" if result["p_age"] < 0.05 else ""
                print(f"    {outcome} ~ {age_var}: b = {result['beta_age']:.4f}, "
                      f"p = {result['p_age']:.4f}{sig}, n = {result['n']}")

    # Neuroscience sample
    print("\n  Sample: neuro_sample")
    sample_df = df.loc[samples["neuro_sample"]["filter"]].copy()
    print(f"    N = {len(sample_df)}")

    for outcome in samples["neuro_sample"]["outcomes"]:
        if outcome in PANAS_AFFECT:
            age_vars = samples["neuro_sample"]["age_vars"]["panas"]
        else:
            age_vars = samples["neuro_sample"]["age_vars"]["daily"]

        for age_var in age_vars:
            result = run_mlm(sample_df, outcome, age_var, base_covars)
            if result is not None:
                result["sample"] = "neuro_sample"
                mlm_rows.append(result)
                sig = "*" if result["p_age"] < 0.05 else ""
                print(f"    {outcome} ~ {age_var}: b = {result['beta_age']:.4f}, "
                      f"p = {result['p_age']:.4f}{sig}, n = {result['n']}")

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\nSaving results...")

    mlm_file = OUT_DIR / "01b_affect_age_mlm.csv"
    pd.DataFrame(mlm_rows).to_csv(mlm_file, index=False)
    print(f"  MLM results saved to {mlm_file}")
    print(f"  {len(mlm_rows)} models")

    print("\nMLM analysis complete!")


if __name__ == "__main__":
    main()
