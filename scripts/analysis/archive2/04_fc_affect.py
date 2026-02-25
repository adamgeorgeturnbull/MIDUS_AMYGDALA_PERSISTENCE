#!/usr/bin/env python3
"""
04_fc_affect.py

Condition-level analysis of amygdala-vmPFC functional connectivity and affect.

Tests whether amygdala-vmPFC connectivity relates to daily-life affect when
examining each condition (negative, neutral, positive) separately, as well as
the neg > neu and neg > pos contrasts. This approach uses a 3-condition GLM
(neg, neu, pos) and tests condition-specific associations rather than relying
solely on the neg > neu contrast.

FC measures are ROI-level beta-series correlations (Fisher z-transformed)
between amygdala seeds (L, R) and vmPFC targets (anterior = safety signaling,
posterior = threat signaling), based on Tashjian et al. (2021, TICS).

Input:
  - data/fMRI/betaSeries_all_conditions.csv
    (per-condition Fisher z correlations from runBStaskFC.sh v2)
  - data/processed/midus_with_fmri.csv

Output:
  - results/tables/04_fc_affect_correlations_conservative.csv
  - results/tables/04_fc_affect_correlations_full.csv

Run from project root directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# Paths and Constants
# ============================================================================
PROCESSED_DIR = Path("data/processed")
FMRI_DIR = Path("data/fMRI")
RESULTS_DIR = Path("results/tables")

MASTER_FILE = PROCESSED_DIR / "midus_with_fmri.csv"
FC_FILE = FMRI_DIR / "betaSeries_all_conditions.csv"

MIN_N_CORR = 10

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


# ============================================================================
# Helper Functions
# ============================================================================
def compute_correlations(df, fc_vars, affect_vars, min_n=MIN_N_CORR):
    """Compute bivariate Pearson correlations between FC and affect vars."""
    results = []
    for fc_var in fc_vars:
        for affect_var in affect_vars:
            data = df[[fc_var, affect_var]].dropna()
            n = len(data)
            if n < min_n:
                continue
            r, p = stats.pearsonr(data[fc_var], data[affect_var])
            # Parse condition and seed_target from variable name
            if "safety_vs_threat" in fc_var:
                # e.g., l_amyg_safety_vs_threat_neg → cond=neg
                condition = fc_var.rsplit("_", 1)[-1]
                seed_target = fc_var.rsplit("_", 1)[0]
            elif "_vs_" in fc_var:
                # e.g., l_amyg-ant_vmPFC_neg_vs_neu → cond=neg_vs_neu
                condition = "_".join(fc_var.rsplit("_", 3)[-3:])
                seed_target = fc_var.rsplit("_", 3)[0]
            else:
                # e.g., l_amyg-ant_vmPFC_neg → cond=neg
                condition = fc_var.rsplit("_", 1)[-1]
                seed_target = fc_var.rsplit("_", 1)[0]

            results.append({
                "fc_var": fc_var,
                "condition": condition,
                "seed_target": seed_target,
                "affect_var": affect_var,
                "n": n,
                "r": r,
                "p": p,
            })
    return pd.DataFrame(results)


def run_sample_analysis(df, sample_name, fc_vars, affect_vars):
    """Run correlation analysis for a given sample."""
    print(f"\n{'=' * 70}")
    print(f"Condition-Level FC Analysis: {sample_name}")
    print(f"{'=' * 70}")
    print(f"  N = {len(df)}")

    results = compute_correlations(df, fc_vars, list(affect_vars.keys()))

    if len(results) == 0:
        print("  No results (insufficient data)")
        return pd.DataFrame()

    # Print summary per condition
    for cond in CONDITIONS:
        cond_results = results[results["condition"] == cond]
        if len(cond_results) == 0:
            continue
        n_sig = (cond_results["p"] < 0.05).sum()
        n_total = len(cond_results)
        print(f"\n  {cond}: {n_sig}/{n_total} significant (p < .05)")

        for _, row in cond_results.iterrows():
            sig = ("***" if row["p"] < 0.001
                   else "**" if row["p"] < 0.01
                   else "*" if row["p"] < 0.05
                   else "")
            print(f"    {row['fc_var']:45s} x {row['affect_var']:15s}: "
                  f"r = {row['r']:7.3f}, p = {row['p']:.4f}{sig:3s}, "
                  f"n = {int(row['n'])}")

    # Compare neg vs neu vs pos for any affect var with a significant hit
    print(f"\n  --- Condition comparison for significant associations ---")
    sig_affects = results.loc[results["p"] < 0.05, "affect_var"].unique()
    sig_seeds = results.loc[results["p"] < 0.05, "seed_target"].unique()
    for seed_target in sig_seeds:
        for affect_var in sig_affects:
            subset = results[
                (results["seed_target"] == seed_target) &
                (results["affect_var"] == affect_var) &
                (results["condition"].isin(["neg", "neu", "pos"]))
            ]
            if len(subset) > 0:
                print(f"\n    {seed_target} -> {affect_var}:")
                for _, row in subset.iterrows():
                    sig = "*" if row["p"] < 0.05 else ""
                    print(f"      {row['condition']:6s}: r = {row['r']:7.3f}, "
                          f"p = {row['p']:.4f}{sig}")

    # Save
    out_file = RESULTS_DIR / f"04_fc_affect_correlations_{sample_name}.csv"
    results.to_csv(out_file, index=False)
    print(f"\n  Saved to {out_file}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Analysis 04: Condition-Level FC and Affect")
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

    # Define samples
    has_fc = df[fc_vars[0]].notna() if fc_vars else pd.Series(False, index=df.index)
    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1

    full_sample = df[has_fc & has_persist].copy()
    conservative_sample = full_sample[full_sample.get("qc_conservative", 0) == 1].copy()

    # Run analyses
    full_results = run_sample_analysis(full_sample, "full", fc_vars, AFFECT_VARS)
    cons_results = run_sample_analysis(conservative_sample, "conservative", fc_vars, AFFECT_VARS)

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    for label, results in [("Full", full_results), ("Conservative", cons_results)]:
        if len(results) == 0:
            print(f"\n{label}: no results")
            continue
        for cond in ["neg", "neu", "pos"]:
            cond_res = results[results["condition"] == cond]
            n_sig = (cond_res["p"] < 0.05).sum() if len(cond_res) > 0 else 0
            n_total = len(cond_res)
            print(f"  {label} {cond:6s}: {n_sig}/{n_total} significant")
        contrast_res = results[results["condition"] == "neg_vs_neu"]
        n_sig = (contrast_res["p"] < 0.05).sum() if len(contrast_res) > 0 else 0
        print(f"  {label} neg>neu: {n_sig}/{len(contrast_res)} significant")


if __name__ == "__main__":
    main()
