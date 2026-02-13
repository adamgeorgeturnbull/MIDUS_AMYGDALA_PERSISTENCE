#!/usr/bin/env python3
"""
04_persistence_affect_moderation.py

Test whether emotion regulation strategy moderates the persistence–affect
association (Exploratory Analysis #3).

Moderators:
- C5SER: ERQ Reappraisal (1–7 scale)
- C5SES: ERQ Suppression (1–7 scale)

Model:
  affect ~ persistence + moderator + persistence×moderator + covariates

Both persistence and moderator are mean-centered before creating the
interaction term to reduce multicollinearity and aid interpretation.

Covariates (following Puccetti et al., 2021):
- Age (C5PAGE)
- Gender (sex)
- Race (dummy-coded)
- Twin pairs (dummy-coded)
- Time between visits (time_P2_P5)
- Number of diary days completed (n_days_complete)

Key analysis decisions:
- Persistence measures are Fisher z-transformed before analysis
- Primary: Cross-run negative persistence (L, R, bilateral)
- Sensitivity: Cross-run positive, concatenated negative
- Mean-centering done within each complete-case subset
- Two-tailed tests for all interaction effects (exploratory)

Inputs:
- data/processed/midus_with_fmri.csv

Outputs:
- results/tables/04_persistence_affect_moderation_full.csv
- results/tables/04_persistence_affect_moderation_conservative.csv

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

MIN_N_REG = 20


# ============================================================================
# Helper Functions
# ============================================================================
def fisher_z(r):
    """Apply Fisher z-transformation to correlation coefficient."""
    return 0.5 * np.log((1 + r) / (1 - r))


def get_full_sample(df):
    """Define full analysis sample: participants with imaging data.

    Per-model dropna() handles missing affect outcomes, so we do NOT
    pre-filter on affect availability. This lets PANAS analyses include
    participants who have MRI data but no daily diary data.
    """
    has_persistence = df["has_neg_persistence"] == 1
    sample = df[has_persistence].copy()

    print(f"\nFull sample: {len(sample)} participants")
    print(f"  With daily diary affect: {sample['PA_score'].notna().sum()}")

    return sample


def get_conservative_sample(df):
    """Define conservative sample with strict QC criteria."""
    sample = get_full_sample(df)
    qc_pass = sample["qc_conservative"] == 1
    conservative = sample[qc_pass].copy()

    print(f"Conservative sample: {len(conservative)} participants")

    return conservative


def run_moderation(df, persistence_var, affect_var, moderator, covariates):
    """
    Run OLS moderation: affect ~ persistence + moderator + persistence×moderator + covariates.

    Persistence and moderator are mean-centered within the complete-case subset.

    Returns dictionary with results, or None if insufficient data.
    """
    vars_needed = [affect_var, persistence_var, moderator] + covariates
    data = df[vars_needed].dropna()

    if len(data) < MIN_N_REG:
        return None

    # Mean-center persistence and moderator
    persist_centered = data[persistence_var] - data[persistence_var].mean()
    mod_centered = data[moderator] - data[moderator].mean()
    interaction = persist_centered * mod_centered

    # Build design matrix: [persist_c, mod_c, interaction, covariates]
    X = pd.concat([
        pd.DataFrame({
            persistence_var: persist_centered,
            moderator: mod_centered,
            f"{persistence_var}_x_{moderator}": interaction,
        }, index=data.index),
        data[covariates],
    ], axis=1)
    X = sm.add_constant(X)
    y = data[affect_var]

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"    Error fitting model for {affect_var} ~ "
              f"{persistence_var} * {moderator}: {e}")
        return None

    interaction_col = f"{persistence_var}_x_{moderator}"

    return {
        "persistence_var": persistence_var,
        "affect_var": affect_var,
        "moderator": moderator,
        "n": int(model.nobs),
        "beta_interaction": model.params[interaction_col],
        "se_interaction": model.bse[interaction_col],
        "t_interaction": model.tvalues[interaction_col],
        "p_interaction": model.pvalues[interaction_col],
        "beta_persistence": model.params[persistence_var],
        "p_persistence": model.pvalues[persistence_var],
        "beta_moderator": model.params[moderator],
        "p_moderator": model.pvalues[moderator],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }


def run_all_moderations(df, persistence_vars, affect_vars, moderators,
                        base_covariates, diary_covariates, diary_affect_vars):
    """Run all persistence × affect × moderator interaction models.

    Diary-specific covariates only included for daily diary outcomes.
    """
    results = []
    for moderator in moderators:
        for persist_var in persistence_vars:
            for affect_var in affect_vars:
                if affect_var in diary_affect_vars:
                    covariates = base_covariates + diary_covariates
                else:
                    covariates = base_covariates
                result = run_moderation(df, persist_var, affect_var, moderator, covariates)
                if result is not None:
                    results.append(result)
    return pd.DataFrame(results)


def run_sample_analysis(sample, sample_name, persistence_vars, affect_vars,
                        moderators, moderator_labels,
                        base_covariates, diary_covariates, diary_affect_vars):
    """Run moderation analysis for a given sample and save results."""
    print("\n" + "=" * 80)
    print(f"Moderation Analysis: {sample_name}")
    print("=" * 80)

    # Moderator descriptives
    for mod, label in zip(moderators, moderator_labels):
        valid = sample[mod].dropna()
        print(f"\n  {label} ({mod}): N={len(valid)}, "
              f"M={valid.mean():.2f}, SD={valid.std():.2f}, "
              f"range={valid.min():.1f}–{valid.max():.1f}")

    print(f"\nRunning moderation models (interaction term is key test)...")

    results = run_all_moderations(
        sample, persistence_vars, affect_vars, moderators,
        base_covariates, diary_covariates, diary_affect_vars
    )

    if len(results) > 0:
        print(f"\n  Computed {len(results)} models")

        for mod, label in zip(moderators, moderator_labels):
            mod_results = results[results["moderator"] == mod]
            n_sig = (mod_results["p_interaction"] < 0.05).sum()
            print(f"\n  {label} ({mod}): {n_sig}/{len(mod_results)} significant interactions")

            for _, row in mod_results.iterrows():
                sig = ("***" if row["p_interaction"] < 0.001
                       else "**" if row["p_interaction"] < 0.01
                       else "*" if row["p_interaction"] < 0.05
                       else "")
                print(f"    {row['persistence_var']:40s} × {row['affect_var']:15s}: "
                      f"b = {row['beta_interaction']:8.5f}, "
                      f"p = {row['p_interaction']:.4f}{sig:3s}, "
                      f"n = {int(row['n'])}")
    else:
        print("\n  No models computed (insufficient data)")

    out_file = RESULTS_DIR / f"04_persistence_affect_moderation_{sample_name}.csv"
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
    print("Analysis 04: Persistence × Affect — Moderation by Emotion Regulation")
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

    moderators = ["C5SER", "C5SES"]
    moderator_labels = ["ERQ Reappraisal", "ERQ Suppression"]

    race_dummies = [col for col in df.columns if col.startswith("race_")]
    twin_dummies = [col for col in df.columns if col.startswith("twin_pair_")]

    # Diary-specific covariates only for daily diary outcomes
    base_covariates = [
        "C5PAGE",
        "sex",
    ] + race_dummies + twin_dummies

    diary_covariates = [
        "time_P2_P5",
        "n_days_complete",
    ]

    diary_affect_vars = {"PA_score", "NA_score", "NA_score_log"}

    print(f"\nPredictors: {len(persistence_vars)} persistence measures (Fisher z)")
    print(f"Outcomes: {len(affect_vars)} affect measures")
    print(f"Moderators: {', '.join(f'{l} ({m})' for m, l in zip(moderators, moderator_labels))}")
    print(f"Covariates:")
    print(f"  Base (all models): C5PAGE, sex, {len(race_dummies)} race, {len(twin_dummies)} twin dummies")
    print(f"  Diary-only (PA/NA outcomes): time_P2_P5, n_days_complete")
    print(f"Total models per sample: {len(persistence_vars)} × {len(affect_vars)} "
          f"× {len(moderators)} = {len(persistence_vars) * len(affect_vars) * len(moderators)}")

    # ========================================================================
    # Run Analyses
    # ========================================================================
    full_sample = get_full_sample(df)
    conservative_sample = get_conservative_sample(df)

    full_results = run_sample_analysis(
        full_sample, "full", persistence_vars, affect_vars,
        moderators, moderator_labels,
        base_covariates, diary_covariates, diary_affect_vars
    )

    cons_results = run_sample_analysis(
        conservative_sample, "conservative", persistence_vars, affect_vars,
        moderators, moderator_labels,
        base_covariates, diary_covariates, diary_affect_vars
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for label, results, n_sample in [
        ("Full sample", full_results, len(full_sample)),
        ("Conservative sample", cons_results, len(conservative_sample)),
    ]:
        print(f"\n{label} (N={n_sample}):")
        if len(results) == 0:
            print("  No models computed (insufficient data)")
            continue

        for mod, mod_label in zip(moderators, moderator_labels):
            mod_results = results[results["moderator"] == mod]
            if len(mod_results) == 0:
                print(f"  {mod_label}: no models")
                continue

            n_sig = (mod_results["p_interaction"] < 0.05).sum()
            print(f"  {mod_label} ({mod}): {n_sig}/{len(mod_results)} significant interactions")

            # Highlight any significant primary results (cross-run negative)
            primary = mod_results[
                mod_results["persistence_var"].str.contains("neg_persist_crossrun")
            ]
            sig_primary = primary[primary["p_interaction"] < 0.05]
            if len(sig_primary) > 0:
                for _, row in sig_primary.iterrows():
                    print(f"    * {row['persistence_var']} × {row['affect_var']}: "
                          f"b = {row['beta_interaction']:.5f}, p = {row['p_interaction']:.4f}")


if __name__ == "__main__":
    main()
