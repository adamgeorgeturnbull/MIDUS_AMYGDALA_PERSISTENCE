#!/usr/bin/env python3
"""
07_fc_affect_moderation.py

Test whether emotion regulation strategy moderates the FC–affect association
(pre-registered exploratory analysis).

Model:
  affect ~ FC + moderator + FC×moderator + covariates

Moderators:
- C5SER: ERQ Reappraisal (1–7 scale)
- C5SES: ERQ Suppression (1–7 scale)

FC variables (6, already Fisher z-transformed neg > neu from runBStaskFC.sh):
  - L amygdala -> anterior vmPFC (safety)
  - L amygdala -> posterior vmPFC (threat)
  - R amygdala -> anterior vmPFC (safety)
  - R amygdala -> posterior vmPFC (threat)
  - L amygdala: safety - threat (relative connectivity)
  - R amygdala: safety - threat (relative connectivity)

Affect outcomes (6):
  - Primary (daily diary): PA_score, NA_score, NA_score_log
  - Secondary (PANAS): C5SPGP, C5SPGN, C5SPGN_log

Both FC and moderator are mean-centered before creating the interaction term
to reduce multicollinearity and aid interpretation.

Covariates (following Puccetti et al., 2021):
- Age (C5PAGE)
- Gender (sex)
- Race (dummy-coded)
- Twin pairs (dummy-coded)
- Time between visits (time_P2_P5) — diary outcomes only
- Number of diary days completed (n_days_complete) — diary outcomes only

Two versions:
- Full sample: All participants with FC data
- Conservative sample: Full + mean FD < 0.5 AND all 3 runs pass QC

Inputs:
- data/processed/midus_with_fmri.csv

Outputs:
- results/tables/07_fc_affect_moderation_full.csv
- results/tables/07_fc_affect_moderation_conservative.csv

Run from project root directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
def get_full_sample(df):
    """Define full analysis sample: participants with FC data.

    Per-model dropna() handles missing affect/moderator outcomes, so we do
    NOT pre-filter on affect availability.
    """
    has_fc = df["has_beta_series"] == 1
    sample = df[has_fc].copy()

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


def run_moderation(df, fc_var, affect_var, moderator, covariates):
    """
    Run OLS moderation: affect ~ FC + moderator + FC×moderator + covariates.

    FC and moderator are mean-centered within the complete-case subset.

    Returns dictionary with results, or None if insufficient data.
    """
    vars_needed = [affect_var, fc_var, moderator] + covariates
    data = df[vars_needed].dropna()

    if len(data) < MIN_N_REG:
        return None

    # Mean-center FC and moderator
    fc_centered = data[fc_var] - data[fc_var].mean()
    mod_centered = data[moderator] - data[moderator].mean()
    interaction = fc_centered * mod_centered

    interaction_col = f"{fc_var}_x_{moderator}"

    # Build design matrix: [fc_c, mod_c, interaction, covariates]
    X = pd.concat([
        pd.DataFrame({
            fc_var: fc_centered,
            moderator: mod_centered,
            interaction_col: interaction,
        }, index=data.index),
        data[covariates],
    ], axis=1)
    X = sm.add_constant(X)
    y = data[affect_var]

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"    Error fitting model for {affect_var} ~ "
              f"{fc_var} * {moderator}: {e}")
        return None

    return {
        "fc_var": fc_var,
        "affect_var": affect_var,
        "moderator": moderator,
        "n": int(model.nobs),
        "beta_interaction": model.params[interaction_col],
        "se_interaction": model.bse[interaction_col],
        "t_interaction": model.tvalues[interaction_col],
        "p_interaction": model.pvalues[interaction_col],
        "beta_fc": model.params[fc_var],
        "p_fc": model.pvalues[fc_var],
        "beta_moderator": model.params[moderator],
        "p_moderator": model.pvalues[moderator],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }


def run_all_moderations(df, fc_vars, affect_vars, moderators,
                        base_covariates, diary_covariates, diary_affect_vars):
    """Run all FC × affect × moderator interaction models.

    Diary-specific covariates only included for daily diary outcomes.
    """
    results = []
    for moderator in moderators:
        for fc_var in fc_vars:
            for affect_var in affect_vars:
                if affect_var in diary_affect_vars:
                    covariates = base_covariates + diary_covariates
                else:
                    covariates = base_covariates
                result = run_moderation(df, fc_var, affect_var, moderator, covariates)
                if result is not None:
                    results.append(result)
    return pd.DataFrame(results)


def run_sample_analysis(sample, sample_name, fc_vars, affect_vars,
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
              f"range={valid.min():.1f}-{valid.max():.1f}")

    print(f"\nRunning moderation models (interaction term is key test)...")

    results = run_all_moderations(
        sample, fc_vars, affect_vars, moderators,
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
                print(f"    {row['fc_var']:45s} x {row['affect_var']:15s}: "
                      f"b = {row['beta_interaction']:8.5f}, "
                      f"p = {row['p_interaction']:.4f}{sig:3s}, "
                      f"n = {int(row['n'])}")
    else:
        print("\n  No models computed (insufficient data)")

    out_file = RESULTS_DIR / f"07_fc_affect_moderation_{sample_name}.csv"
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
    print("Analysis 07: FC x Affect — Moderation by Emotion Regulation")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} participants")

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
    fc_vars = [
        "conn_l_amyg_ant_vmPFC_neg_vs_neu",
        "conn_l_amyg_post_vmPFC_neg_vs_neu",
        "conn_l_amyg_safety_vs_threat",
        "conn_r_amyg_ant_vmPFC_neg_vs_neu",
        "conn_r_amyg_post_vmPFC_neg_vs_neu",
        "conn_r_amyg_safety_vs_threat",
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

    base_covariates = [
        "C5PAGE",
        "sex",
    ] + race_dummies + twin_dummies

    diary_covariates = [
        "time_P2_P5",
        "n_days_complete",
    ]

    diary_affect_vars = {"PA_score", "NA_score", "NA_score_log"}

    print(f"\nFC predictors: {len(fc_vars)} (already Fisher z, neg > neu)")
    print(f"Affect outcomes: {len(affect_vars)}")
    print(f"Moderators: {', '.join(f'{l} ({m})' for m, l in zip(moderators, moderator_labels))}")
    print(f"Covariates:")
    print(f"  Base (all models): C5PAGE, sex, {len(race_dummies)} race, {len(twin_dummies)} twin dummies")
    print(f"  Diary-only (PA/NA outcomes): time_P2_P5, n_days_complete")
    print(f"Total models per sample: {len(fc_vars)} x {len(affect_vars)} "
          f"x {len(moderators)} = {len(fc_vars) * len(affect_vars) * len(moderators)}")

    # ========================================================================
    # Run Analyses
    # ========================================================================
    full_sample = get_full_sample(df)
    conservative_sample = get_conservative_sample(df)

    full_results = run_sample_analysis(
        full_sample, "full", fc_vars, affect_vars,
        moderators, moderator_labels,
        base_covariates, diary_covariates, diary_affect_vars
    )

    cons_results = run_sample_analysis(
        conservative_sample, "conservative", fc_vars, affect_vars,
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

            sig_results = mod_results[mod_results["p_interaction"] < 0.05]
            if len(sig_results) > 0:
                for _, row in sig_results.iterrows():
                    print(f"    * {row['fc_var']} x {row['affect_var']}: "
                          f"b = {row['beta_interaction']:.5f}, p = {row['p_interaction']:.4f}")


if __name__ == "__main__":
    main()
