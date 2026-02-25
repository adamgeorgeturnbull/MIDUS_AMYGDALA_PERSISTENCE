#!/usr/bin/env python3
"""
02_persistence_affect_mlm.py

Mixed-effects (multilevel) reanalysis of persistence x affect associations.

Replaces the OLS + twin-pair-dummy approach from 02_persistence_affect.py with
linear mixed-effects models (random intercept for family). This avoids the
degrees-of-freedom cost of one dummy per twin pair while properly accounting
for non-independence within twin families.

Grouping variable (family_id):
  - Twins (SAMPLMAJ == 3 AND 2+ members share M2FAMNUM): M2FAMNUM
  - Everyone else: M2ID (cluster of size 1)

Fixed effects: persistence + age + sex + race dummies + time_P2_P5 + n_days_complete
Random effects: random intercept for family_id

Correlations are identical to 02_persistence_affect.py and are NOT re-run here.
Only the covariate-adjusted models differ (MLM vs OLS).

Inputs:
- data/processed/midus_with_fmri.csv

Outputs:
- results/tables/02_persistence_affect_mlm_full.csv
- results/tables/02_persistence_affect_mlm_conservative.csv

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
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")

DATA_FILE = PROCESSED_DIR / "midus_with_fmri.csv"

MIN_N_REG = 20


# ============================================================================
# Helper Functions
# ============================================================================
def fisher_z(r):
    """Apply Fisher z-transformation to correlation coefficient."""
    return 0.5 * np.log((1 + r) / (1 - r))


def create_family_id(df):
    """
    Create a family_id column for mixed-effects grouping.

    Twins (SAMPLMAJ == 3) who share a M2FAMNUM with at least one other
    participant in the dataset are grouped by that family number.
    All other participants get their own unique group (M2ID).

    Args:
        df: DataFrame with SAMPLMAJ, M2FAMNUM, and M2ID columns

    Returns:
        DataFrame with added family_id column
    """
    df = df.copy()

    # Identify twin-sample participants
    is_twin_sample = df["SAMPLMAJ"] == 3

    # Among twin-sample participants, find families with 2+ members in data
    twin_fam_counts = df.loc[is_twin_sample, "M2FAMNUM"].value_counts()
    paired_families = twin_fam_counts[twin_fam_counts > 1].index

    # Assign family_id
    # Twins in paired families: use M2FAMNUM (as string to avoid collision with M2ID)
    # Everyone else: use M2ID (each person is their own cluster)
    df["family_id"] = df["M2ID"].astype(str)
    paired_mask = is_twin_sample & df["M2FAMNUM"].isin(paired_families)
    df.loc[paired_mask, "family_id"] = "fam_" + df.loc[paired_mask, "M2FAMNUM"].astype(int).astype(str)

    n_paired = paired_mask.sum()
    n_families = df.loc[paired_mask, "family_id"].nunique()
    print(f"  Family grouping: {n_paired} participants in {n_families} twin families, "
          f"{(~paired_mask).sum()} singletons")

    return df


def get_full_sample(df):
    """Define full analysis sample: participants with imaging data.

    Per-model dropna() handles missing affect outcomes, so we do NOT
    pre-filter on affect availability. This lets PANAS analyses include
    participants who have MRI data but no daily diary data.
    """
    has_persistence = df["has_neg_persistence"] == 1
    sample = df[has_persistence].copy()
    print(f"\nFull sample: {len(sample)} participants")
    return sample


def get_conservative_sample(df):
    """Define conservative sample with strict QC criteria."""
    sample = get_full_sample(df)
    qc_pass = sample["qc_conservative"] == 1
    conservative = sample[qc_pass].copy()
    print(f"Conservative sample: {len(conservative)} participants")
    return conservative


def run_mlm(df, persistence_var, affect_var, covariates):
    """
    Run linear mixed-effects model: affect ~ persistence + covariates, (1 | family_id).

    Args:
        df: Dataset with family_id column
        persistence_var: Name of persistence predictor
        affect_var: Name of affect outcome
        covariates: List of covariate column names

    Returns:
        Dictionary with results, or None if insufficient data
    """
    vars_needed = [affect_var, persistence_var, "family_id"] + covariates
    data = df[vars_needed].dropna()

    if len(data) < MIN_N_REG:
        return None

    # Drop zero-variance covariates (e.g., race dummies with no cases in sample)
    active_covariates = [c for c in covariates if data[c].std() > 0]

    # Build formula (no twin dummies — handled by random effect)
    fixed = f"{affect_var} ~ {persistence_var} + " + " + ".join(active_covariates)

    # Try multiple optimizers: LBFGS needs the Hessian which can be singular
    # when the random effect variance is near zero; Powell is derivative-free.
    methods = ["lbfgs", "powell"]
    result = None
    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(fixed, data=data, groups=data["family_id"])
                result = model.fit(reml=True, method=method)
                break
        except Exception:
            continue

    if result is None:
        print(f"    Error fitting MLM for {affect_var} ~ {persistence_var}: "
              f"all optimizers failed ({', '.join(methods)})")
        return None

    # Extract persistence effect
    return {
        "persistence_var": persistence_var,
        "affect_var": affect_var,
        "n": int(result.nobs),
        "n_groups": int(result.nobs - result.df_resid),  # approximate
        "beta_persistence": result.fe_params[persistence_var],
        "se_persistence": result.bse_fe[persistence_var],
        "z_persistence": result.tvalues[persistence_var],
        "p_persistence": result.pvalues[persistence_var],
        "group_var": result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else float(result.cov_re),
        "log_likelihood": result.llf,
        "converged": result.converged,
        "optimizer": method,
    }


def run_all_mlm(df, persistence_vars, affect_vars, base_covariates,
                diary_covariates, diary_affect_vars):
    """Run all persistence x affect mixed-effects models.

    Diary-specific covariates only included for daily diary outcomes.
    """
    results = []
    for persist_var in persistence_vars:
        for affect_var in affect_vars:
            if affect_var in diary_affect_vars:
                covariates = base_covariates + diary_covariates
            else:
                covariates = base_covariates
            result = run_mlm(df, persist_var, affect_var, covariates)
            if result is not None:
                results.append(result)
    return pd.DataFrame(results)


def run_sample_analysis(sample, sample_name, persistence_vars, affect_vars,
                        base_covariates, diary_covariates, diary_affect_vars):
    """Run MLM analysis for a given sample and save results."""
    print("\n" + "=" * 80)
    print(f"Mixed-Effects Analysis: {sample_name}")
    print("=" * 80)

    # Create family grouping
    sample = create_family_id(sample)

    print(f"\nRunning mixed-effects models (random intercept for family)...")

    results = run_all_mlm(sample, persistence_vars, affect_vars,
                          base_covariates, diary_covariates, diary_affect_vars)

    if len(results) > 0:
        print(f"\n  Computed {len(results)} models")
        n_sig = (results["p_persistence"] < 0.05).sum()
        print(f"  Significant (p < 0.05): {n_sig}")

        # Print results
        for _, row in results.iterrows():
            sig = "***" if row["p_persistence"] < 0.001 else "**" if row["p_persistence"] < 0.01 else "*" if row["p_persistence"] < 0.05 else ""
            print(f"  {row['persistence_var']:40s} -> {row['affect_var']:15s}: "
                  f"b = {row['beta_persistence']:6.3f}, p = {row['p_persistence']:.4f}{sig:3s}, "
                  f"n = {int(row['n'])}")
    else:
        print("\n  No models computed (insufficient data)")

    # Save
    out_file = RESULTS_DIR / f"02_persistence_affect_mlm_{sample_name}.csv"
    results.to_csv(out_file, index=False)
    print(f"\n  Saved to {out_file}")

    return results


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Analysis 02 (MLM): Persistence x Affect — Mixed-Effects Models")
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

    affect_vars = [
        "PA_score",
        "NA_score",
        "NA_score_log",
        "C5SPGP",
        "C5SPGN",
        "C5SPGN_log",
    ]

    # Covariates: same as OLS but WITHOUT twin dummies (handled by random effect).
    # Diary-specific covariates only for daily diary outcomes.
    race_dummies = [col for col in df.columns if col.startswith("race_")]

    base_covariates = [
        "C5PAGE",
        "sex",
    ] + race_dummies

    diary_covariates = [
        "time_P2_P5",
        "n_days_complete",
    ]

    diary_affect_vars = {"PA_score", "NA_score", "NA_score_log"}

    print(f"\nCovariates (no twin dummies — handled by random effect):")
    print(f"  Base (all models): C5PAGE, sex, {len(race_dummies)} race dummies")
    print(f"  Diary-only (PA/NA outcomes): time_P2_P5, n_days_complete")
    print(f"  Random: intercept | family_id")

    # ========================================================================
    # Run Analyses
    # ========================================================================
    full_sample = get_full_sample(df)
    conservative_sample = get_conservative_sample(df)

    full_results = run_sample_analysis(
        full_sample, "full", persistence_vars, affect_vars,
        base_covariates, diary_covariates, diary_affect_vars
    )

    cons_results = run_sample_analysis(
        conservative_sample, "conservative", persistence_vars, affect_vars,
        base_covariates, diary_covariates, diary_affect_vars
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("MLM Analysis Complete")
    print("=" * 80)

    print(f"\nFull sample: {len(full_results)} models")
    if len(full_results) > 0:
        print(f"  Significant (p < 0.05): {(full_results['p_persistence'] < 0.05).sum()}")

    print(f"\nConservative sample: {len(cons_results)} models")
    if len(cons_results) > 0:
        print(f"  Significant (p < 0.05): {(cons_results['p_persistence'] < 0.05).sum()}")


if __name__ == "__main__":
    main()
