#!/usr/bin/env python3
"""
07_fc_affect_moderation.py

Test whether emotion regulation strategy moderates the FC-affect association
(pre-registered exploratory analysis).

Model:
  affect ~ FC + moderator + FC*moderator + covariates

Moderators:
- C5SER: ERQ Reappraisal (1-7 scale)
- C5SES: ERQ Suppression (1-7 scale)

FC measures are per-condition ROI-level beta-series correlations (Fisher
z-transformed) between amygdala seeds (L, R) and vmPFC targets (anterior =
safety signaling, posterior = threat signaling), based on Tashjian et al.
(2021, TICS). Conditions: neg, neu, pos, neg_vs_neu, neg_vs_pos.

Both FC and moderator are mean-centered before creating the interaction term
to reduce multicollinearity and aid interpretation.

Covariates (following Puccetti et al., 2021):
- Age (C5PAGE)
- Gender (sex)
- Race (dummy-coded)
- Twin pairs (dummy-coded)
- Time between visits (time_P2_P5) -- diary outcomes only
- Number of diary days completed (n_days_complete) -- diary outcomes only

Key analysis decisions:
- Per-condition FC from betaSeries_all_conditions.csv (not neg>neu contrast)
- Two-tailed tests for all interaction and FC effects (exploratory)
- Results tiered: primary (L amyg neg, PA/NA), secondary (R amyg neg),
  sensitivity (bilateral, other conditions, contrasts, PANAS, log-transforms)

Inputs:
- data/processed/midus_with_fmri.csv
- data/fMRI/betaSeries_all_conditions.csv

Outputs:
- results/tables/07_fc_affect_moderation_full.csv
- results/tables/07_fc_affect_moderation_conservative.csv

Run from project root directory.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier_utils import get_fc_tier, get_affect_tier, combine_tiers, print_by_tier

# ============================================================================
# Paths and Constants
# ============================================================================
PROCESSED_DIR = Path("data/processed")
FMRI_DIR = Path("data/fMRI")
RESULTS_DIR = Path("results/tables")

MASTER_FILE = PROCESSED_DIR / "midus_with_fmri.csv"
# Prefer LSS (primary method) over LSA if available
_LSS_FILE = FMRI_DIR / "all_subjects_betaSeries_LSS_all_conditions_M2ID.csv"
_LSA_FILE = FMRI_DIR / "all_subjects_betaSeries_all_conditions_M2ID.csv"
FC_FILE = _LSS_FILE if _LSS_FILE.exists() else _LSA_FILE

MIN_N_REG = 20

ROI_PAIRS = [
    ("l_amyg", "ant_vmPFC"),
    ("l_amyg", "post_vmPFC"),
    ("r_amyg", "ant_vmPFC"),
    ("r_amyg", "post_vmPFC"),
]

CONDITIONS = ["neg", "neu", "pos"]  # contrasts removed: GLM already encodes condition differences


# ============================================================================
# Helper Functions
# ============================================================================
def parse_fc_var(fc_var):
    """Parse condition and seed_target from FC variable name."""
    if "safety_vs_threat" in fc_var:
        condition = fc_var.rsplit("_", 1)[-1]
        seed_target = fc_var.rsplit("_", 1)[0]
    elif "_vs_" in fc_var:
        condition = "_".join(fc_var.rsplit("_", 3)[-3:])
        seed_target = fc_var.rsplit("_", 3)[0]
    else:
        condition = fc_var.rsplit("_", 1)[-1]
        seed_target = fc_var.rsplit("_", 1)[0]
    return condition, seed_target


def run_moderation(df, fc_var, affect_var, moderator, covariates):
    """
    Run OLS moderation: affect ~ FC + moderator + FC*moderator + covariates.

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

    # Build design matrix
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

    condition, seed_target = parse_fc_var(fc_var)

    return {
        "fc_var": fc_var,
        "condition": condition,
        "seed_target": seed_target,
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
        "tier": combine_tiers(
            get_fc_tier(fc_var, condition=condition, seed_target=seed_target),
            get_affect_tier(affect_var),
        ),
    }


def run_all_moderations(df, fc_vars, affect_vars, moderators,
                        base_covariates, diary_covariates, diary_affect_vars):
    """Run all FC x affect x moderator interaction models.

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
            print(f"\n  --- {label} ({mod}): {n_sig}/{len(mod_results)} significant interactions ---")

            def _fmt_mod(row):
                sig = ("***" if row["p_interaction"] < 0.001
                       else "**" if row["p_interaction"] < 0.01
                       else "*" if row["p_interaction"] < 0.05
                       else "")
                return (f"{row['fc_var']:45s} x {row['affect_var']:15s}: "
                        f"b_int = {row['beta_interaction']:8.5f}, "
                        f"p_int = {row['p_interaction']:.4f}{sig:3s}, "
                        f"n = {int(row['n'])}")

            print_by_tier(mod_results, _fmt_mod, p_col="p_interaction")
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
    print("Analysis 07: FC x Affect -- Moderation by Emotion Regulation")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading behavioral data from {MASTER_FILE}...")
    master = pd.read_csv(MASTER_FILE)
    print(f"  Loaded {len(master)} participants")

    print(f"Loading condition-level FC from {FC_FILE}...")
    fc = pd.read_csv(FC_FILE)
    print(f"  Loaded {len(fc)} participants, {len(fc.columns) - 1} FC columns")

    # Ensure M2ID types match for merge
    master["M2ID"] = master["M2ID"].astype(str)
    fc["M2ID"] = fc["M2ID"].astype(str)

    # Merge
    df = master.merge(fc, on="M2ID", how="inner")
    print(f"  Merged: {len(df)} participants with both FC and behavioral data")

    # ========================================================================
    # Build FC Variable List
    # ========================================================================
    fc_vars = []
    for seed, target in ROI_PAIRS:
        pair = f"{seed}-{target}"
        for cond in CONDITIONS:
            col = f"{pair}_{cond}"
            if col in df.columns:
                fc_vars.append(col)

    # Add safety-vs-threat derived variables per condition
    for cond in CONDITIONS:
        l_ant = f"l_amyg-ant_vmPFC_{cond}"
        l_post = f"l_amyg-post_vmPFC_{cond}"
        r_ant = f"r_amyg-ant_vmPFC_{cond}"
        r_post = f"r_amyg-post_vmPFC_{cond}"
        if l_ant in df.columns and l_post in df.columns:
            derived = f"l_amyg_safety_vs_threat_{cond}"
            df[derived] = df[l_ant] - df[l_post]
            fc_vars.append(derived)
        if r_ant in df.columns and r_post in df.columns:
            derived = f"r_amyg_safety_vs_threat_{cond}"
            df[derived] = df[r_ant] - df[r_post]
            fc_vars.append(derived)

    # ========================================================================
    # Define Variables
    # ========================================================================
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

    print(f"\nFC predictors: {len(fc_vars)} (per-condition, Fisher z)")
    print(f"Affect outcomes: {len(affect_vars)}")
    print(f"Moderators: {', '.join(f'{l} ({m})' for m, l in zip(moderators, moderator_labels))}")
    print(f"Covariates:")
    print(f"  Base (all models): C5PAGE, sex, {len(race_dummies)} race, {len(twin_dummies)} twin dummies")
    print(f"  Diary-only (PA/NA outcomes): time_P2_P5, n_days_complete")
    print(f"Total models per sample: {len(fc_vars)} x {len(affect_vars)} "
          f"x {len(moderators)} = {len(fc_vars) * len(affect_vars) * len(moderators)}")

    # ========================================================================
    # Define Samples
    # ========================================================================
    has_fc = df[fc_vars[0]].notna() if fc_vars else pd.Series(False, index=df.index)
    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1

    full_sample = df[has_fc & has_persist].copy()
    conservative_sample = full_sample[full_sample.get("qc_conservative", 0) == 1].copy()

    print(f"\nFull sample: {len(full_sample)} participants")
    print(f"  With daily diary affect: {full_sample['PA_score'].notna().sum()}")
    print(f"Conservative sample: {len(conservative_sample)} participants")

    # ========================================================================
    # Run Analyses
    # ========================================================================
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

            print(f"\n  {mod_label} ({mod}):")
            for tier in ["primary", "secondary", "sensitivity"]:
                tier_res = mod_results[mod_results["tier"] == tier]
                if len(tier_res) == 0:
                    continue
                n_sig = (tier_res["p_interaction"] < 0.05).sum()
                print(f"    {tier.upper()}: {n_sig}/{len(tier_res)} significant interactions")
                if tier != "sensitivity":
                    sig = tier_res[tier_res["p_interaction"] < 0.05]
                    for _, row in sig.iterrows():
                        print(f"      {row['fc_var']} x {row['affect_var']}: "
                              f"b = {row['beta_interaction']:.5f}, p = {row['p_interaction']:.4f}")


if __name__ == "__main__":
    main()
