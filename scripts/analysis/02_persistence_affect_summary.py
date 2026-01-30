#!/usr/bin/env python3
"""
02_persistence_affect_summary.py

Summarize results from Analysis 02 (persistence × affect associations).

Focuses on consistency across analytical methods:
- Identifies findings significant in BOTH correlations AND regressions
- Notes that conservative sample (N=3) is primary but unavailable
- Full sample results reported as exploratory only

Inputs:
- results/tables/02_persistence_affect_correlations_full.csv
- results/tables/02_persistence_affect_regressions_full.csv
- results/tables/02_persistence_affect_correlations_conservative.csv
- results/tables/02_persistence_affect_regressions_conservative.csv

Outputs:
- results/tables/02_persistence_affect_summary.txt

Run from project root directory.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Paths and Constants
# ============================================================================
RESULTS_DIR = Path("results/tables")

# Input files
CORR_FILE_FULL = RESULTS_DIR / "02_persistence_affect_correlations_full.csv"
REG_FILE_FULL = RESULTS_DIR / "02_persistence_affect_regressions_full.csv"
CORR_FILE_CONS = RESULTS_DIR / "02_persistence_affect_correlations_conservative.csv"
REG_FILE_CONS = RESULTS_DIR / "02_persistence_affect_regressions_conservative.csv"

# Output files
SUMMARY_TEXT = RESULTS_DIR / "02_persistence_affect_summary.txt"

# Analysis categorization
PRIMARY_PERSIST = ["neg_persist_crossrun_mean_z_L", "neg_persist_crossrun_mean_z_R",
                   "neg_persist_crossrun_mean_z_bilateral"]
PRIMARY_AFFECT = ["PA_score", "NA_score", "NA_score_log"]

SENSITIVITY_PERSIST_POS = ["pos_persist_crossrun_mean_z_L", "pos_persist_crossrun_mean_z_R",
                           "pos_persist_crossrun_mean_z_bilateral"]
SENSITIVITY_PERSIST_CONCAT = ["neg_persist_concat_z_L", "neg_persist_concat_z_R",
                              "neg_persist_concat_z_bilateral"]
SECONDARY_AFFECT = ["C5SPGP", "C5SPGN", "C5SPGN_log"]


# ============================================================================
# Helper Functions
# ============================================================================
def categorize_analysis(row):
    """
    Categorize an analysis as primary or sensitivity based on variables used.

    Args:
        row: DataFrame row with 'persistence_var' and 'affect_var'

    Returns:
        String category
    """
    persist_var = row["persistence_var"]
    affect_var = row["affect_var"]

    # Primary: Cross-run negative × daily diary affect
    if persist_var in PRIMARY_PERSIST and affect_var in PRIMARY_AFFECT:
        return "primary"

    # Sensitivity: Positive persistence
    elif persist_var in SENSITIVITY_PERSIST_POS:
        return "sensitivity_positive"

    # Sensitivity: Concatenated negative
    elif persist_var in SENSITIVITY_PERSIST_CONCAT:
        return "sensitivity_concat"

    # Sensitivity: PANAS outcomes
    elif affect_var in SECONDARY_AFFECT:
        return "sensitivity_panas"

    else:
        return "other"


def find_consistent_results(corr_df, reg_df, p_threshold=0.05):
    """
    Find analyses significant in BOTH correlations AND regressions.

    Args:
        corr_df: DataFrame with correlation results
        reg_df: DataFrame with regression results
        p_threshold: P-value threshold for significance

    Returns:
        DataFrame with consistently significant results
    """
    # Get significant correlations
    sig_corr = corr_df[corr_df["p"] < p_threshold].copy()

    # Get significant regressions
    sig_reg = reg_df[reg_df["p_persistence"] < p_threshold].copy()

    # Find matches on persistence_var and affect_var
    consistent = []

    for idx, corr_row in sig_corr.iterrows():
        # Look for matching regression
        match = sig_reg[
            (sig_reg["persistence_var"] == corr_row["persistence_var"]) &
            (sig_reg["affect_var"] == corr_row["affect_var"])
        ]

        if len(match) > 0:
            reg_row = match.iloc[0]
            consistent.append({
                "persistence_var": corr_row["persistence_var"],
                "affect_var": corr_row["affect_var"],
                "category": corr_row["category"],
                "r": corr_row["r"],
                "p_corr": corr_row["p"],
                "n_corr": corr_row["n"],
                "beta": reg_row["beta_persistence"],
                "p_reg": reg_row["p_persistence"],
                "n_reg": reg_row["n"],
            })

    return pd.DataFrame(consistent)


def summarize_by_category(df, category_col="category"):
    """
    Summarize number of tests and significant findings by category.

    Args:
        df: DataFrame with results
        category_col: Column name for category

    Returns:
        DataFrame with summary by category
    """
    if len(df) == 0:
        return pd.DataFrame()

    summaries = []
    for category in sorted(df[category_col].unique()):
        cat_data = df[df[category_col] == category]
        summaries.append({
            "category": category,
            "n_consistent": len(cat_data),
        })

    return pd.DataFrame(summaries)


def find_single_method_results(corr_df, reg_df, p_threshold=0.05):
    """
    Find analyses significant in ONLY one method (correlation OR regression).

    Args:
        corr_df: DataFrame with correlation results
        reg_df: DataFrame with regression results
        p_threshold: P-value threshold for significance

    Returns:
        Two DataFrames: (sig_in_corr_only, sig_in_reg_only)
    """
    # Significant in correlation only
    sig_corr = corr_df[corr_df["p"] < p_threshold].copy()
    corr_only = []

    for idx, corr_row in sig_corr.iterrows():
        # Find matching regression
        match = reg_df[
            (reg_df["persistence_var"] == corr_row["persistence_var"]) &
            (reg_df["affect_var"] == corr_row["affect_var"])
        ]

        if len(match) > 0:
            reg_row = match.iloc[0]
            # Check if NOT significant in regression
            if reg_row["p_persistence"] >= p_threshold:
                corr_only.append({
                    "persistence_var": corr_row["persistence_var"],
                    "affect_var": corr_row["affect_var"],
                    "category": corr_row["category"],
                    "r": corr_row["r"],
                    "p_corr": corr_row["p"],
                    "beta": reg_row["beta_persistence"],
                    "p_reg": reg_row["p_persistence"],
                })

    # Significant in regression only
    sig_reg = reg_df[reg_df["p_persistence"] < p_threshold].copy()
    reg_only = []

    for idx, reg_row in sig_reg.iterrows():
        # Find matching correlation
        match = corr_df[
            (corr_df["persistence_var"] == reg_row["persistence_var"]) &
            (corr_df["affect_var"] == reg_row["affect_var"])
        ]

        if len(match) > 0:
            corr_row = match.iloc[0]
            # Check if NOT significant in correlation
            if corr_row["p"] >= p_threshold:
                reg_only.append({
                    "persistence_var": reg_row["persistence_var"],
                    "affect_var": reg_row["affect_var"],
                    "category": corr_row["category"],
                    "r": corr_row["r"],
                    "p_corr": corr_row["p"],
                    "beta": reg_row["beta_persistence"],
                    "p_reg": reg_row["p_persistence"],
                })

    return pd.DataFrame(corr_only), pd.DataFrame(reg_only)


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    print("=" * 80)
    print("Analysis 02: Persistence × Affect Summary")
    print("=" * 80)

    # ========================================================================
    # Load Results
    # ========================================================================
    print("\nLoading results...")

    # Conservative sample (primary) - likely unavailable
    try:
        corr_cons = pd.read_csv(CORR_FILE_CONS)
        reg_cons = pd.read_csv(REG_FILE_CONS)
        cons_available = len(corr_cons) > 0 and len(reg_cons) > 0
        print(f"✓ Conservative sample: {len(corr_cons)} correlations, {len(reg_cons)} regressions")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        corr_cons = pd.DataFrame()
        reg_cons = pd.DataFrame()
        cons_available = False
        print("⚠ Conservative sample: No results (N=3, insufficient data)")

    # Full sample (exploratory)
    corr_full = pd.read_csv(CORR_FILE_FULL)
    reg_full = pd.read_csv(REG_FILE_FULL)
    print(f"✓ Full sample: {len(corr_full)} correlations, {len(reg_full)} regressions")

    # ========================================================================
    # Categorize Analyses
    # ========================================================================
    print("\nCategorizing analyses...")

    corr_full["category"] = corr_full.apply(categorize_analysis, axis=1)
    reg_full["category"] = reg_full.apply(categorize_analysis, axis=1)

    if cons_available:
        corr_cons["category"] = corr_cons.apply(categorize_analysis, axis=1)
        reg_cons["category"] = reg_cons.apply(categorize_analysis, axis=1)

    # ========================================================================
    # Find Consistent Results (sig in BOTH methods)
    # ========================================================================
    print("\nIdentifying consistent findings...")

    # Conservative sample
    if cons_available:
        consistent_cons = find_consistent_results(corr_cons, reg_cons)
        print(f"  Conservative: {len(consistent_cons)} findings significant in BOTH methods")
    else:
        consistent_cons = pd.DataFrame()
        print(f"  Conservative: Not available")

    # Full sample
    consistent_full = find_consistent_results(corr_full, reg_full)
    print(f"  Full sample: {len(consistent_full)} findings significant in BOTH methods")

    # ========================================================================
    # Find Single-Method Results (sig in ONE method only)
    # ========================================================================
    print("\nIdentifying single-method findings...")

    # Conservative sample
    if cons_available:
        corr_only_cons, reg_only_cons = find_single_method_results(corr_cons, reg_cons)
        print(f"  Conservative: {len(corr_only_cons)} sig in corr only, {len(reg_only_cons)} sig in reg only")
    else:
        corr_only_cons = pd.DataFrame()
        reg_only_cons = pd.DataFrame()

    # Full sample
    corr_only_full, reg_only_full = find_single_method_results(corr_full, reg_full)
    print(f"  Full sample: {len(corr_only_full)} sig in corr only, {len(reg_only_full)} sig in reg only")

    # ========================================================================
    # Generate Text Summary
    # ========================================================================
    print("\nGenerating text summary...")

    with open(SUMMARY_TEXT, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Analysis 02: Persistence × Affect Summary\n")
        f.write("=" * 80 + "\n\n")

        # Note about sample status
        f.write("SAMPLE STATUS\n")
        f.write("-" * 80 + "\n")
        if cons_available:
            f.write("Conservative sample (PRIMARY): Available\n")
            f.write(f"  N = {reg_cons.iloc[0]['n'] if len(reg_cons) > 0 else 'N/A'}\n")
        else:
            f.write("Conservative sample (PRIMARY): Unavailable\n")
            f.write("  N = 3 (insufficient for analysis)\n")
            f.write("  Requires: All 3 runs pass QC AND mean FD < 0.5 mm\n")

        f.write(f"\nFull sample (EXPLORATORY): Available\n")
        f.write(f"  N = {reg_full.iloc[0]['n'] if len(reg_full) > 0 else 'N/A'}\n")
        f.write("  Results below are exploratory only\n")
        f.write("\n")

        # Analysis approach
        f.write("ANALYSIS APPROACH\n")
        f.write("-" * 80 + "\n")
        f.write("This summary focuses on CONSISTENT findings:\n")
        f.write("  - Significant in BOTH zero-order correlations AND covariate-adjusted\n")
        f.write("    regressions (p < 0.05)\n")
        f.write("  - Provides stronger evidence than either method alone\n\n")

        # Category breakdown
        f.write("ANALYSIS CATEGORIES\n")
        f.write("-" * 80 + "\n")
        f.write("Primary (Confirmatory):\n")
        f.write("  - Cross-run negative persistence (L, R, bilateral)\n")
        f.write("  - Daily diary affect (PA, NA, NA_log)\n")
        f.write("  - Total: 9 tests\n\n")

        f.write("Sensitivity:\n")
        f.write("  - Positive persistence: 18 tests\n")
        f.write("  - Concatenated negative: 18 tests\n")
        f.write("  - PANAS outcomes: 9 tests\n")
        f.write("  - Total: 45 tests\n\n")

        # Conservative sample results (if available)
        if cons_available:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("PRIMARY ANALYSES (Conservative Sample)\n")
            f.write("=" * 80 + "\n\n")

            # Overall counts (handle empty case for dummy data)
            if len(consistent_cons) > 0:
                primary_cons = consistent_cons[consistent_cons["category"] == "primary"]
                sens_cons = consistent_cons[consistent_cons["category"] != "primary"]
            else:
                primary_cons = pd.DataFrame()
                sens_cons = pd.DataFrame()

            f.write(f"Consistent findings (significant in BOTH methods):\n")
            f.write(f"  Primary: {len(primary_cons)}/9 tests\n")
            f.write(f"  Sensitivity: {len(sens_cons)}/45 tests\n\n")

            # List primary findings
            if len(primary_cons) > 0:
                f.write("Primary findings (sig in BOTH correlation and regression):\n")
                for idx, row in primary_cons.iterrows():
                    f.write(f"\n  {row['persistence_var']} × {row['affect_var']}:\n")
                    f.write(f"    Correlation: r = {row['r']:.3f}, p = {row['p_corr']:.4f}, n = {row['n_corr']}\n")
                    f.write(f"    Regression:  β = {row['beta']:.3f}, p = {row['p_reg']:.4f}, n = {row['n_reg']}\n")
            else:
                f.write("No primary findings consistent across both methods\n")

            f.write("\n")

            # List sensitivity findings
            if len(sens_cons) > 0:
                f.write(f"Sensitivity findings ({len(sens_cons)} total):\n")
                for category in sens_cons["category"].unique():
                    cat_findings = sens_cons[sens_cons["category"] == category]
                    f.write(f"\n  {category.replace('sensitivity_', '').title()} ({len(cat_findings)} findings):\n")
                    for idx, row in cat_findings.iterrows():
                        f.write(f"    {row['persistence_var']} × {row['affect_var']}\n")
                        f.write(f"      Corr: r={row['r']:.3f}, p={row['p_corr']:.4f} | ")
                        f.write(f"Reg: β={row['beta']:.3f}, p={row['p_reg']:.4f}\n")
            else:
                f.write("No sensitivity findings consistent across both methods\n")

        else:
            # Conservative not available - report full sample as exploratory
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("EXPLORATORY FINDINGS (Full Sample)\n")
            f.write("=" * 80 + "\n")
            f.write("NOTE: Conservative sample (primary) unavailable (N=3)\n")
            f.write("Results below are from full sample and are EXPLORATORY ONLY\n")
            f.write("Final conclusions require conservative sample analysis\n\n")

            # Overall counts (handle empty case for dummy data)
            if len(consistent_full) > 0:
                primary_full = consistent_full[consistent_full["category"] == "primary"]
                sens_full = consistent_full[consistent_full["category"] != "primary"]
            else:
                primary_full = pd.DataFrame()
                sens_full = pd.DataFrame()

            f.write(f"Consistent findings (significant in BOTH methods):\n")
            f.write(f"  Primary: {len(primary_full)}/9 tests\n")
            f.write(f"  Sensitivity: {len(sens_full)}/45 tests\n\n")

            # List primary findings
            if len(primary_full) > 0:
                f.write("Primary analyses (sig in BOTH correlation and regression):\n")
                for idx, row in primary_full.iterrows():
                    f.write(f"\n  {row['persistence_var']} × {row['affect_var']}:\n")
                    f.write(f"    Correlation: r = {row['r']:.3f}, p = {row['p_corr']:.4f}, n = {row['n_corr']}\n")
                    f.write(f"    Regression:  β = {row['beta']:.3f}, p = {row['p_reg']:.4f}, n = {row['n_reg']}\n")
            else:
                f.write("No primary findings consistent across both methods\n")

            f.write("\n")

            # List sensitivity findings by category
            if len(sens_full) > 0:
                f.write(f"Sensitivity analyses ({len(sens_full)} total consistent findings):\n")
                for category in sorted(sens_full["category"].unique()):
                    cat_findings = sens_full[sens_full["category"] == category]
                    f.write(f"\n  {category.replace('sensitivity_', '').title()} ({len(cat_findings)} findings):\n")
                    for idx, row in cat_findings.iterrows():
                        f.write(f"    {row['persistence_var']} × {row['affect_var']}\n")
                        f.write(f"      Corr: r={row['r']:.3f}, p={row['p_corr']:.4f} | ")
                        f.write(f"Reg: β={row['beta']:.3f}, p={row['p_reg']:.4f}\n")
            else:
                f.write("No sensitivity findings consistent across both methods\n")

        # ====================================================================
        # Single-Method Findings
        # ====================================================================
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("SIGNIFICANT IN ONLY ONE METHOD\n")
        f.write("=" * 80 + "\n")
        f.write("Findings significant in ONE method but not the other (for comparison):\n\n")

        # Use appropriate sample based on availability
        if cons_available:
            corr_only = corr_only_cons
            reg_only = reg_only_cons
            sample_label = "Conservative Sample"
        else:
            corr_only = corr_only_full
            reg_only = reg_only_full
            sample_label = "Full Sample (Exploratory)"

        # Significant in correlation only
        if len(corr_only) > 0:
            f.write("SIGNIFICANT IN ZERO-ORDER CORRELATIONS ONLY:\n")
            f.write("-" * 80 + "\n")

            # Separate by category
            primary_corr_only = corr_only[corr_only["category"] == "primary"]
            sens_corr_only = corr_only[corr_only["category"] != "primary"]

            if len(primary_corr_only) > 0:
                f.write("Primary analyses:\n\n")
                for idx, row in primary_corr_only.iterrows():
                    # Create readable labels
                    persist_label = row["persistence_var"].replace("_mean_z_", " ").replace("neg_persist_crossrun", "Negative persistence").replace("_L", "(L)").replace("_R", "(R)").replace("_bilateral", "(Bilateral)")
                    affect_label = row["affect_var"].replace("PA_score", "Daily Positive Affect").replace("NA_score_log", "Daily Negative Affect (log)").replace("NA_score", "Daily Negative Affect")

                    f.write(f"{persist_label} × {affect_label}:\n")
                    f.write(f"  - Correlation: r = {row['r']:.3f}, p = {row['p_corr']:.3f}*\n")
                    f.write(f"  - Regression:  β = {row['beta']:.3f}, p = {row['p_reg']:.3f}\n\n")

            if len(sens_corr_only) > 0:
                f.write("Sensitivity analyses:\n\n")
                for idx, row in sens_corr_only.iterrows():
                    # Create readable labels
                    persist_label = row["persistence_var"].replace("_mean_z_", " ").replace("pos_persist_crossrun", "Positive persistence").replace("neg_persist_concat_z", "Concatenated Negative persistence").replace("_L", "(L)").replace("_R", "(R)").replace("_bilateral", "(Bilateral)")
                    affect_label = row["affect_var"].replace("PA_score", "Daily Positive Affect").replace("NA_score_log", "Daily Negative Affect (log)").replace("NA_score", "Daily Negative Affect").replace("C5SPGP", "PANAS Positive Affect").replace("C5SPGN_log", "PANAS Negative Affect (log)").replace("C5SPGN", "PANAS Negative Affect")

                    f.write(f"{persist_label} × {affect_label}:\n")
                    f.write(f"  - Correlation: r = {row['r']:.3f}, p = {row['p_corr']:.3f}*\n")
                    f.write(f"  - Regression:  β = {row['beta']:.4f}, p = {row['p_reg']:.3f}\n\n")
        else:
            f.write("SIGNIFICANT IN ZERO-ORDER CORRELATIONS ONLY:\n")
            f.write("-" * 80 + "\n")
            f.write("None\n\n")

        f.write("\n")

        # Significant in regression only
        if len(reg_only) > 0:
            f.write("SIGNIFICANT IN COVARIATE-ADJUSTED REGRESSIONS ONLY:\n")
            f.write("-" * 80 + "\n")

            # Separate by category
            primary_reg_only = reg_only[reg_only["category"] == "primary"]
            sens_reg_only = reg_only[reg_only["category"] != "primary"]

            if len(primary_reg_only) > 0:
                f.write("Primary analyses:\n\n")
                for idx, row in primary_reg_only.iterrows():
                    # Create readable labels
                    persist_label = row["persistence_var"].replace("_mean_z_", " ").replace("neg_persist_crossrun", "Negative persistence").replace("_L", "(L)").replace("_R", "(R)").replace("_bilateral", "(Bilateral)")
                    affect_label = row["affect_var"].replace("PA_score", "Daily Positive Affect").replace("NA_score_log", "Daily Negative Affect (log)").replace("NA_score", "Daily Negative Affect")

                    f.write(f"{persist_label} × {affect_label}:\n")
                    f.write(f"  - Correlation: r = {row['r']:.3f}, p = {row['p_corr']:.3f}\n")
                    f.write(f"  - Regression:  β = {row['beta']:.3f}, p = {row['p_reg']:.3f}*\n\n")

            if len(sens_reg_only) > 0:
                f.write("Sensitivity analyses:\n\n")
                for idx, row in sens_reg_only.iterrows():
                    # Create readable labels
                    persist_label = row["persistence_var"].replace("_mean_z_", " ").replace("pos_persist_crossrun", "Positive persistence").replace("neg_persist_concat_z", "Concatenated Negative persistence").replace("_L", "(L)").replace("_R", "(R)").replace("_bilateral", "(Bilateral)")
                    affect_label = row["affect_var"].replace("PA_score", "Daily Positive Affect").replace("NA_score_log", "Daily Negative Affect (log)").replace("NA_score", "Daily Negative Affect").replace("C5SPGP", "PANAS Positive Affect").replace("C5SPGN_log", "PANAS Negative Affect (log)").replace("C5SPGN", "PANAS Negative Affect")

                    f.write(f"{persist_label} × {affect_label}:\n")
                    f.write(f"  - Correlation: r = {row['r']:.3f}, p = {row['p_corr']:.3f}\n")
                    f.write(f"  - Regression:  β = {row['beta']:.3f}, p = {row['p_reg']:.3f}*\n\n")
        else:
            f.write("SIGNIFICANT IN COVARIATE-ADJUSTED REGRESSIONS ONLY:\n")
            f.write("-" * 80 + "\n")
            f.write("None\n\n")

        f.write("Note: * indicates p < 0.05. These findings are not consistent across methods and\n")
        f.write("should be interpreted with caution.\n")

    print(f"✓ Text summary saved to {SUMMARY_TEXT}")

    # ========================================================================
    # Print Summary to Console
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if cons_available:
        print(f"\nConservative sample (PRIMARY): {len(consistent_cons)} consistent findings")
        if len(consistent_cons) > 0:
            primary_cons = consistent_cons[consistent_cons["category"] == "primary"]
            sens_cons = consistent_cons[consistent_cons["category"] != "primary"]
        else:
            primary_cons = pd.DataFrame()
            sens_cons = pd.DataFrame()
        print(f"  Primary: {len(primary_cons)}/9")
        print(f"  Sensitivity: {len(sens_cons)}/45")
    else:
        print("\nConservative sample (PRIMARY): Unavailable (N=3)")
        print(f"\nFull sample (EXPLORATORY): {len(consistent_full)} consistent findings")
        if len(consistent_full) > 0:
            primary_full = consistent_full[consistent_full["category"] == "primary"]
            sens_full = consistent_full[consistent_full["category"] != "primary"]
        else:
            primary_full = pd.DataFrame()
            sens_full = pd.DataFrame()
        print(f"  Primary: {len(primary_full)}/9")
        print(f"  Sensitivity: {len(sens_full)}/45")

    print("\n" + "=" * 80)
    print("✓ Summary complete!")
    print(f"\nOutput: {SUMMARY_TEXT}")


if __name__ == "__main__":
    main()
