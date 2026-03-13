#!/usr/bin/env python3
"""
03b_persistence_age_mlm.py

Mixed-effects (multilevel) reanalysis of persistence ~ age associations.

Replaces the OLS + twin-pair-dummy approach from 03_persistence_age.py with
linear mixed-effects models (random intercept for family). This avoids the
degrees-of-freedom cost of one dummy per twin pair while properly accounting
for non-independence within twin families.

Grouping variable (family_id):
  - Twins (SAMPLMAJ == 3 AND 2+ members share M2FAMNUM): M2FAMNUM
  - Everyone else: M2ID (cluster of size 1)

Fixed effects: C5PAGE + sex + race dummies
Random effects: random intercept for family_id

Correlations are identical to 03_persistence_age.py and are NOT re-run here.
Only the covariate-adjusted models differ (MLM vs OLS).

Key analysis decisions:
- One-tailed p-values for age effect (directional hypothesis: persistence
  decreases with age)
- Results tiered: primary (L amyg), secondary (R amyg), sensitivity (rest)

Inputs:
- data/processed/midus_with_fmri.csv

Outputs:
- results/tables/03b_persistence_age_mlm_full.csv
- results/tables/03b_persistence_age_mlm_conservative.csv

Run from project root directory.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier_utils import get_persistence_tier, print_by_tier

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

def run_mlm(df, persistence_var, age_var, covariates):
    """
    Run linear mixed-effects model: persistence ~ age + covariates, (1 | family_id).

    Returns dictionary with results, or None if insufficient data.
    """
    vars_needed = [persistence_var, age_var, "family_id"] + covariates
    data = df[vars_needed].dropna()

    if len(data) < MIN_N_REG:
        return None

    # Drop zero-variance covariates (e.g., race dummies with no cases in sample)
    active_covariates = [c for c in covariates if data[c].std() > 0]

    # Build formula (no twin dummies -- handled by random effect)
    fixed = f"{persistence_var} ~ {age_var} + " + " + ".join(active_covariates)

    # Try multiple optimizers
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
        print(f"    Error fitting MLM for {persistence_var} ~ {age_var}: "
              f"all optimizers failed ({', '.join(methods)})")
        return None

    beta_age = result.fe_params[age_var]

    # Extract SE; if NaN (degenerate random effect with near-zero variance),
    # recover from absolute value of covariance diagonal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        se_age = result.bse_fe[age_var]
    degenerate_re = False
    if np.isnan(se_age):
        degenerate_re = True
        idx = list(result.fe_params.index).index(age_var)
        var = result.cov_params().iloc[idx, idx]
        se_age = np.sqrt(abs(var)) if abs(var) > 0 else np.nan
    z_age = beta_age / se_age if not np.isnan(se_age) else np.nan
    if not np.isnan(z_age):
        from scipy import stats as _stats
        p_age = 2 * (1 - _stats.norm.cdf(abs(z_age)))
    else:
        p_age = np.nan

    # Directional one-tailed: hypothesis is beta < 0
    if not np.isnan(p_age):
        p_age_one_tailed = p_age / 2 if beta_age < 0 else 1 - p_age / 2
    else:
        p_age_one_tailed = np.nan

    return {
        "persistence_var": persistence_var,
        "n": int(result.nobs),
        "n_groups": int(result.nobs - result.df_resid),
        "beta_age": beta_age,
        "se_age": se_age,
        "z_age": z_age,
        "p_age": p_age,
        "p_age_one_tailed": p_age_one_tailed,
        "group_var": result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else float(result.cov_re),
        "log_likelihood": result.llf,
        "converged": result.converged,
        "degenerate_re": degenerate_re,
        "optimizer": method,
        "tier": get_persistence_tier(persistence_var),
    }

def run_all_mlm(df, persistence_vars, age_var, covariates):
    """Run all persistence ~ age mixed-effects models."""
    results = []
    for persist_var in persistence_vars:
        result = run_mlm(df, persist_var, age_var, covariates)
        if result is not None:
            results.append(result)
    return pd.DataFrame(results)

def run_sample_analysis(sample, sample_name, persistence_vars, age_var, covariates):
    """Run MLM analysis for a given sample and save results."""
    print("\n" + "=" * 80)
    print(f"Mixed-Effects Analysis: {sample_name}")
    print("=" * 80)

    # Create family grouping
    sample = create_family_id(sample)

    print(f"\nRunning mixed-effects models (random intercept for family)...")

    results = run_all_mlm(sample, persistence_vars, age_var, covariates)

    if len(results) > 0:
        print(f"\n  Computed {len(results)} models")

        def _fmt_mlm(row):
            sig = ("***" if row["p_age"] < 0.001
                   else "**" if row["p_age"] < 0.01
                   else "*" if row["p_age"] < 0.05 else "")
            direction = "neg" if row["beta_age"] < 0 else "pos"
            return (f"{row['persistence_var']:45s}: "
                    f"b = {row['beta_age']:8.5f}, p = {row['p_age']:.4f}{sig:3s}, "
                    f"p(1t) = {row['p_age_one_tailed']:.4f} [{direction}], n = {int(row['n'])}")

        print_by_tier(results, _fmt_mlm, p_col="p_age")
    else:
        print("\n  No models computed (insufficient data)")

    # Save
    out_file = RESULTS_DIR / f"03b_persistence_age_mlm_{sample_name}.csv"
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
    print("Analysis 03b (MLM): Persistence ~ Age -- Mixed-Effects Models")
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
        "pos_persist_crossrun_mean_r_L",
        "pos_persist_crossrun_mean_r_R",
        "neg_persist_concat_r_L",
        "neg_persist_concat_r_R",
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
        "pos_persist_crossrun_mean_z_L",
        "pos_persist_crossrun_mean_z_R",
        "neg_persist_concat_z_L",
        "neg_persist_concat_z_R",
    ]

    # vmPFC persistence (secondary comparison ROI — available after Sherlock jobs complete)
    vmPFC_r_cols = sorted([c for c in df.columns
                           if "vmPFC" in c and "neg_image" in c and c.endswith("_mean_r")])
    if vmPFC_r_cols:
        for var in vmPFC_r_cols:
            z_var = var.replace("_mean_r", "_mean_z")
            df[z_var] = fisher_z(df[var])
            persistence_vars.append(z_var)
        print(f"  Added {len(vmPFC_r_cols)} vmPFC persistence variables (secondary)")

    # Covariates: same as OLS but WITHOUT twin dummies (handled by random effect).
    # No diary-specific covariates (this analysis is entirely neuroscience-based).
    race_dummies = [col for col in df.columns if col.startswith("race_")]

    covariates = [
        "sex",
    ] + race_dummies

    print(f"\nPredictor: {age_var} (age at neuroscience visit)")
    print(f"Outcomes: {len(persistence_vars)} persistence measures (Fisher z)")
    print(f"Covariates (no twin dummies -- handled by random effect):")
    print(f"  sex, {len(race_dummies)} race dummies")
    print(f"  Random: intercept | family_id")
    print(f"  NOTE: C5PAGE is the predictor, not a covariate")

    # ========================================================================
    # Run Analyses
    # ========================================================================
    full_sample = get_full_sample(df)
    conservative_sample = get_conservative_sample(df)

    full_results = run_sample_analysis(
        full_sample, "full", persistence_vars, age_var, covariates
    )

    cons_results = run_sample_analysis(
        conservative_sample, "conservative", persistence_vars, age_var, covariates
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nHypothesis: Negative persistence decreases with age (negative beta expected)")
    print("One-tailed p-values: p/2 when beta < 0 (as hypothesized), 1 - p/2 otherwise")

    for label, results, n_sample in [
        ("Full sample", full_results, len(full_sample)),
        ("Conservative sample", cons_results, len(conservative_sample)),
    ]:
        print(f"\n{label} (N={n_sample}):")
        if len(results) == 0:
            print("  No models computed (insufficient data)")
            continue
        for tier in ["primary", "secondary", "sensitivity"]:
            tier_res = results[results["tier"] == tier]
            if len(tier_res) == 0:
                continue
            n_sig = (tier_res["p_age"] < 0.05).sum()
            print(f"  {tier.upper()}: {n_sig}/{len(tier_res)} significant (two-tailed)")
            if tier != "sensitivity":
                for _, row in tier_res.iterrows():
                    direction = "neg (as hypothesized)" if row["beta_age"] < 0 else "pos (opposite)"
                    print(f"    {row['persistence_var']}: b = {row['beta_age']:.5f}, "
                          f"p(1t) = {row['p_age_one_tailed']:.4f} [{direction}]")

if __name__ == "__main__":
    main()
