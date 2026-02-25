#!/usr/bin/env python3
"""
04b_fc_affect_mlm.py

Mixed-effects (multilevel) reanalysis of FC x affect associations.

Replaces the OLS + twin-pair-dummy approach from 04_fc_affect.py with
linear mixed-effects models (random intercept for family). This avoids the
degrees-of-freedom cost of one dummy per twin pair while properly accounting
for non-independence within twin families.

Model:
  affect ~ FC + covariates, (1 | family_id)

Grouping variable (family_id):
  - Twins (SAMPLMAJ == 3 AND 2+ members share M2FAMNUM): M2FAMNUM
  - Everyone else: M2ID (cluster of size 1)

Fixed effects: FC + C5PAGE + sex + race dummies [+ time_P2_P5 + n_days_complete]
Random effects: random intercept for family_id

Correlations are identical to 04_fc_affect.py and are NOT re-run here.
Only the covariate-adjusted models differ (MLM vs OLS).

Key analysis decisions:
- Two-tailed tests (FC hypotheses are exploratory)
- Results tiered: primary (L amyg neg, PA/NA), secondary (R amyg neg),
  sensitivity (bilateral, other conditions, contrasts, PANAS, log-transforms)

Inputs:
- data/processed/midus_with_fmri.csv
- data/fMRI/betaSeries_all_conditions.csv

Outputs:
- results/tables/04b_fc_affect_mlm_full.csv
- results/tables/04b_fc_affect_mlm_conservative.csv

Run from project root directory.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier_utils import get_fc_tier, get_affect_tier, combine_tiers, print_by_tier

# ============================================================================
# Paths and Constants
# ============================================================================
PROCESSED_DIR = Path("data/processed")
FMRI_DIR = Path("data/fMRI")
RESULTS_DIR = Path("results/tables")

MASTER_FILE = PROCESSED_DIR / "midus_with_fmri.csv"
FC_FILE = FMRI_DIR / "betaSeries_all_conditions.csv"

MIN_N_REG = 20

AFFECT_VARS = {
    "PA_score": "Positive Affect (Diary)",
    "NA_score": "Negative Affect (Diary)",
    "NA_score_log": "Negative Affect [log] (Diary)",
    "C5SPGP": "Positive Affect (PANAS)",
    "C5SPGN": "Negative Affect (PANAS)",
    "C5SPGN_log": "Negative Affect [log] (PANAS)",
}

ROI_PAIRS = [
    ("l_amyg", "ant_vmPFC"),
    ("l_amyg", "post_vmPFC"),
    ("r_amyg", "ant_vmPFC"),
    ("r_amyg", "post_vmPFC"),
]

CONDITIONS = ["neg", "neu", "pos", "neg_vs_neu", "neg_vs_pos"]

DIARY_AFFECT_VARS = {"PA_score", "NA_score", "NA_score_log"}


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


def run_mlm(df, fc_var, affect_var, covariates):
    """
    Run linear mixed-effects model: affect ~ FC + covariates, (1 | family_id).

    Returns dictionary with results, or None if insufficient data.
    """
    vars_needed = [affect_var, fc_var, "family_id"] + covariates
    data = df[vars_needed].dropna()

    if len(data) < MIN_N_REG:
        return None

    # Drop zero-variance covariates
    active_covariates = [c for c in covariates if data[c].std() > 0]

    # Build formula (no twin dummies -- handled by random effect)
    fixed = f"{affect_var} ~ {fc_var} + " + " + ".join(active_covariates)

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
        print(f"    Error fitting MLM for {affect_var} ~ {fc_var}: "
              f"all optimizers failed ({', '.join(methods)})")
        return None

    condition, seed_target = parse_fc_var(fc_var)

    return {
        "fc_var": fc_var,
        "condition": condition,
        "seed_target": seed_target,
        "affect_var": affect_var,
        "n": int(result.nobs),
        "n_groups": int(result.nobs - result.df_resid),
        "beta_fc": result.fe_params[fc_var],
        "se_fc": result.bse_fe[fc_var],
        "z_fc": result.tvalues[fc_var],
        "p_fc": result.pvalues[fc_var],
        "group_var": result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else float(result.cov_re),
        "log_likelihood": result.llf,
        "converged": result.converged,
        "optimizer": method,
        "tier": combine_tiers(
            get_fc_tier(fc_var, condition=condition, seed_target=seed_target),
            get_affect_tier(affect_var),
        ),
    }


def run_all_mlm(df, fc_vars, affect_vars, base_covariates, diary_covariates):
    """Run all FC x affect mixed-effects models.

    Diary-specific covariates only included for daily diary outcomes.
    """
    results = []
    for fc_var in fc_vars:
        for affect_var in affect_vars:
            if affect_var in DIARY_AFFECT_VARS:
                covariates = base_covariates + diary_covariates
            else:
                covariates = base_covariates
            result = run_mlm(df, fc_var, affect_var, covariates)
            if result is not None:
                results.append(result)
    return pd.DataFrame(results)


def run_sample_analysis(sample, sample_name, fc_vars, affect_vars,
                        base_covariates, diary_covariates):
    """Run MLM analysis for a given sample and save results."""
    print("\n" + "=" * 70)
    print(f"Mixed-Effects Analysis: {sample_name}")
    print("=" * 70)

    # Create family grouping
    sample = create_family_id(sample)

    print(f"\nRunning mixed-effects models (random intercept for family)...")

    results = run_all_mlm(sample, fc_vars, list(affect_vars.keys()),
                          base_covariates, diary_covariates)

    if len(results) > 0:
        print(f"\n  Computed {len(results)} models")

        def _fmt_mlm(row):
            sig = ("***" if row["p_fc"] < 0.001
                   else "**" if row["p_fc"] < 0.01
                   else "*" if row["p_fc"] < 0.05 else "")
            return (f"{row['fc_var']:45s} -> {row['affect_var']:15s}: "
                    f"b = {row['beta_fc']:7.3f}, p = {row['p_fc']:.4f}{sig:3s}, "
                    f"n = {int(row['n'])}")

        print_by_tier(results, _fmt_mlm, p_col="p_fc")
    else:
        print("\n  No models computed (insufficient data)")

    # Save
    out_file = RESULTS_DIR / f"04b_fc_affect_mlm_{sample_name}.csv"
    results.to_csv(out_file, index=False)
    print(f"\n  Saved to {out_file}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Analysis 04b (MLM): Condition-Level FC and Affect")
    print("=" * 70)

    # Load master behavioral data
    print(f"\nLoading behavioral data from {MASTER_FILE}...")
    master = pd.read_csv(MASTER_FILE)
    print(f"  Loaded {len(master)} participants")

    # Load condition-level FC data
    print(f"Loading condition-level FC from {FC_FILE}...")
    fc = pd.read_csv(FC_FILE)
    print(f"  Loaded {len(fc)} participants, {len(fc.columns) - 1} FC columns")

    # Ensure M2ID types match for merge
    master["M2ID"] = master["M2ID"].astype(str)
    fc["M2ID"] = fc["M2ID"].astype(str)

    # Merge
    df = master.merge(fc, on="M2ID", how="inner")
    print(f"  Merged: {len(df)} participants with both FC and behavioral data")

    # Build FC variable list from available columns
    fc_vars = []
    for seed, target in ROI_PAIRS:
        pair = f"{seed}-{target}"
        for cond in CONDITIONS:
            col = f"{pair}_{cond}"
            if col in df.columns:
                fc_vars.append(col)

    # Also add safety-vs-threat derived variables per condition
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

    print(f"\n  FC variables: {len(fc_vars)}")
    print(f"  Affect variables: {len(AFFECT_VARS)}")

    # Covariates: same as OLS but WITHOUT twin dummies
    race_dummies = [col for col in df.columns if col.startswith("race_")]

    base_covariates = [
        "C5PAGE",
        "sex",
    ] + race_dummies

    diary_covariates = [
        "time_P2_P5",
        "n_days_complete",
    ]

    print(f"\nCovariates (no twin dummies -- handled by random effect):")
    print(f"  Base (all models): C5PAGE, sex, {len(race_dummies)} race dummies")
    print(f"  Diary-only (PA/NA outcomes): time_P2_P5, n_days_complete")
    print(f"  Random: intercept | family_id")

    # Define samples
    has_fc = df[fc_vars[0]].notna() if fc_vars else pd.Series(False, index=df.index)
    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1

    full_sample = df[has_fc & has_persist].copy()
    conservative_sample = full_sample[full_sample.get("qc_conservative", 0) == 1].copy()

    # Run analyses
    full_results = run_sample_analysis(
        full_sample, "full", fc_vars, AFFECT_VARS,
        base_covariates, diary_covariates
    )
    cons_results = run_sample_analysis(
        conservative_sample, "conservative", fc_vars, AFFECT_VARS,
        base_covariates, diary_covariates
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    for label, results in [("Full", full_results), ("Conservative", cons_results)]:
        if len(results) == 0:
            print(f"\n{label}: no results")
            continue
        print(f"\n{label}:")
        for tier in ["primary", "secondary", "sensitivity"]:
            tier_res = results[results["tier"] == tier]
            if len(tier_res) == 0:
                continue
            n_sig = (tier_res["p_fc"] < 0.05).sum()
            print(f"  {tier.upper()}: {n_sig}/{len(tier_res)} significant")


if __name__ == "__main__":
    main()
