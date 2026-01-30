#!/usr/bin/env python3
"""
02_persistence_affect_figures.py

Create publication-quality figures for Analysis 02: persistence × affect associations.

Generates:
1. Individual scatterplots for each persistence × affect relationship
2. Forest plots showing regression coefficients (primary vs sensitivity analyses)

Run from project root directory.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ============================================================================
# Paths and Constants
# ============================================================================
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
FIG_DIR = Path("results/figures/analysis_02")
FIG_DIR_PRIMARY = FIG_DIR / "primary"
FIG_DIR_SENS_FULL = FIG_DIR / "sensitivity/full_sample"
FIG_DIR_SENS_POS = FIG_DIR / "sensitivity/positive_persistence"
FIG_DIR_SENS_CONCAT = FIG_DIR / "sensitivity/concatenated_negative"

DATA_FILE = DATA_DIR / "midus_with_fmri.csv"
REGRESSION_FILE_FULL = RESULTS_DIR / "02_persistence_affect_regressions_full.csv"
REGRESSION_FILE_CONS = RESULTS_DIR / "02_persistence_affect_regressions_conservative.csv"

# Figure settings
FIGURE_FORMAT = "png"
FIGURE_DPI = 300

# Color palette
COLORS = {
    "full": "#2E86AB",           # Blue for full sample
    "conservative": "#A23B72",   # Purple for conservative sample
    "neg_crossrun": "#E63946",   # Red for negative cross-run (primary)
    "pos_crossrun": "#06A77D",   # Green for positive cross-run
    "neg_concat": "#F77F00",     # Orange for negative concat
}

SAMPLE_LABELS = {
    "full": "Full Sample",
    "conservative": "Conservative Sample",
}

# Variable labels for plotting
PERSISTENCE_LABELS = {
    # Primary (cross-run negative)
    "neg_persist_crossrun_mean_z_L": "Negative Persistence (L)",
    "neg_persist_crossrun_mean_z_R": "Negative Persistence (R)",
    "neg_persist_crossrun_mean_z_bilateral": "Negative Persistence (Bilateral)",
    # Sensitivity (cross-run positive)
    "pos_persist_crossrun_mean_z_L": "Positive Persistence (L)",
    "pos_persist_crossrun_mean_z_R": "Positive Persistence (R)",
    "pos_persist_crossrun_mean_z_bilateral": "Positive Persistence (Bilateral)",
    # Sensitivity (concatenated negative)
    "neg_persist_concat_z_L": "Negative Concat (L)",
    "neg_persist_concat_z_R": "Negative Concat (R)",
    "neg_persist_concat_z_bilateral": "Negative Concat (Bilateral)",
}

AFFECT_LABELS = {
    "PA_score": "Positive Affect (Daily Diary)",
    "NA_score": "Negative Affect (Daily Diary)",
    "NA_score_log": "Negative Affect [log] (Daily Diary)",
    "C5SPGP": "Positive Affect (PANAS)",
    "C5SPGN": "Negative Affect (PANAS)",
    "C5SPGN_log": "Negative Affect [log] (PANAS)",
}


# ============================================================================
# Helper Functions
# ============================================================================
def fisher_z(r):
    """
    Apply Fisher z-transformation to correlation coefficient.

    Args:
        r: Correlation coefficient (scalar or array)

    Returns:
        Fisher z-transformed value
    """
    return 0.5 * np.log((1 + r) / (1 - r))


def create_scatterplot_panel(ax, data, persist_var, affect_var, sample_name, show_ylabel=True):
    """
    Create a single scatterplot panel showing persistence × affect relationship.

    Args:
        ax: Matplotlib axis
        data: DataFrame with data
        persist_var: Name of persistence variable (x-axis)
        affect_var: Name of affect variable (y-axis)
        sample_name: Name of sample (for color)
        show_ylabel: Whether to show y-axis label

    Returns:
        None (modifies ax in place)
    """
    # Get complete cases
    plot_data = data[[persist_var, affect_var]].dropna()

    if len(plot_data) < 10:
        ax.text(0.5, 0.5, "Insufficient data",
                ha="center", va="center", transform=ax.transAxes)
        return

    # Extract arrays
    x = plot_data[persist_var].values
    y = plot_data[affect_var].values

    # Compute regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Create predictions for line
    x_pred = np.linspace(x.min(), x.max(), 100)
    y_pred = slope * x_pred + intercept

    # Compute confidence interval
    # Standard error of prediction
    n = len(x)
    x_mean = np.mean(x)
    se = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n - 2))
    margin = 1.96 * se * np.sqrt(1/n + (x_pred - x_mean)**2 / np.sum((x - x_mean)**2))

    # Plot
    color = COLORS.get(sample_name, "#333333")

    # Scatter points with transparency
    ax.scatter(x, y, alpha=0.4, s=20, color=color, edgecolors="none")

    # Regression line
    ax.plot(x_pred, y_pred, color=color, linewidth=2, label=f"r = {r_value:.2f}")

    # Confidence interval
    ax.fill_between(x_pred, y_pred - margin, y_pred + margin,
                     alpha=0.2, color=color)

    # Labels and formatting
    ax.set_xlabel(PERSISTENCE_LABELS.get(persist_var, persist_var), fontsize=10)
    if show_ylabel:
        ax.set_ylabel(AFFECT_LABELS.get(affect_var, affect_var), fontsize=10)

    # Add significance stars if p < 0.05
    if p_value < 0.001:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    elif p_value < 0.05:
        sig_text = "*"
    else:
        sig_text = ""

    # Legend with r and p-value
    if sig_text:
        ax.legend([f"r = {r_value:.2f}{sig_text}\np < {p_value:.3f}"],
                  loc="best", frameon=False, fontsize=9)
    else:
        ax.legend([f"r = {r_value:.2f}\np = {p_value:.3f}"],
                  loc="best", frameon=False, fontsize=9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Spines
    sns.despine(ax=ax)


def create_scatterplots_by_type(data):
    """
    Create scatterplots organized by analysis type.

    Organization:
    - Primary: conservative sample × cross-run negative persistence
    - Sensitivity full sample: full sample × all persistence types
    - Sensitivity positive: conservative sample × positive persistence
    - Sensitivity concat: conservative sample × concatenated negative

    Returns:
        Dictionary with counts by folder
    """
    # Define persistence variables by type
    persist_neg_cross = [
        "neg_persist_crossrun_mean_z_L",
        "neg_persist_crossrun_mean_z_R",
        "neg_persist_crossrun_mean_z_bilateral",
    ]

    persist_pos_cross = [
        "pos_persist_crossrun_mean_z_L",
        "pos_persist_crossrun_mean_z_R",
        "pos_persist_crossrun_mean_z_bilateral",
    ]

    persist_neg_concat = [
        "neg_persist_concat_z_L",
        "neg_persist_concat_z_R",
        "neg_persist_concat_z_bilateral",
    ]

    # All affect variables
    affect_vars = [
        "PA_score",
        "NA_score",
        "NA_score_log",
        "C5SPGP",
        "C5SPGN",
        "C5SPGN_log",
    ]

    # Filter data
    data_full = data[data["has_neg_persistence"] == 1].copy()
    data_cons = data[data["qc_conservative"] == 1].copy()

    counts = {}

    # PRIMARY: Conservative × cross-run negative
    print("  Creating primary scatterplots (conservative × cross-run negative)...")
    FIG_DIR_PRIMARY.mkdir(parents=True, exist_ok=True)
    count = 0
    for persist_var in persist_neg_cross:
        for affect_var in affect_vars:
            fig, ax = plt.subplots(figsize=(5, 4))
            create_scatterplot_panel(ax, data_cons, persist_var, affect_var, "conservative", show_ylabel=True)
            plt.tight_layout()
            filename = f"scatter_conservative_{persist_var}_{affect_var}.{FIGURE_FORMAT}"
            fig.savefig(FIG_DIR_PRIMARY / filename, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            count += 1
    counts["primary"] = count

    # SENSITIVITY: Full sample × all persistence
    print("  Creating sensitivity scatterplots (full sample × all persistence)...")
    FIG_DIR_SENS_FULL.mkdir(parents=True, exist_ok=True)
    count = 0
    all_persist = persist_neg_cross + persist_pos_cross + persist_neg_concat
    for persist_var in all_persist:
        for affect_var in affect_vars:
            fig, ax = plt.subplots(figsize=(5, 4))
            create_scatterplot_panel(ax, data_full, persist_var, affect_var, "full", show_ylabel=True)
            plt.tight_layout()
            filename = f"scatter_full_{persist_var}_{affect_var}.{FIGURE_FORMAT}"
            fig.savefig(FIG_DIR_SENS_FULL / filename, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            count += 1
    counts["sensitivity_full"] = count

    # SENSITIVITY: Conservative × positive persistence
    print("  Creating sensitivity scatterplots (conservative × positive persistence)...")
    FIG_DIR_SENS_POS.mkdir(parents=True, exist_ok=True)
    count = 0
    for persist_var in persist_pos_cross:
        for affect_var in affect_vars:
            fig, ax = plt.subplots(figsize=(5, 4))
            create_scatterplot_panel(ax, data_cons, persist_var, affect_var, "conservative", show_ylabel=True)
            plt.tight_layout()
            filename = f"scatter_conservative_{persist_var}_{affect_var}.{FIGURE_FORMAT}"
            fig.savefig(FIG_DIR_SENS_POS / filename, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            count += 1
    counts["sensitivity_positive"] = count

    # SENSITIVITY: Conservative × concatenated negative
    print("  Creating sensitivity scatterplots (conservative × concatenated negative)...")
    FIG_DIR_SENS_CONCAT.mkdir(parents=True, exist_ok=True)
    count = 0
    for persist_var in persist_neg_concat:
        for affect_var in affect_vars:
            fig, ax = plt.subplots(figsize=(5, 4))
            create_scatterplot_panel(ax, data_cons, persist_var, affect_var, "conservative", show_ylabel=True)
            plt.tight_layout()
            filename = f"scatter_conservative_{persist_var}_{affect_var}.{FIGURE_FORMAT}"
            fig.savefig(FIG_DIR_SENS_CONCAT / filename, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            count += 1
    counts["sensitivity_concat"] = count

    return counts


def create_forest_plot(regression_results, analysis_type, sample_name):
    """
    Create forest plot showing regression coefficients for persistence effects.

    Args:
        regression_results: DataFrame with regression results
        analysis_type: "primary", "all", "positive", or "concat"
        sample_name: "full" or "conservative"

    Returns:
        Matplotlib figure
    """
    # Filter to analysis type
    if analysis_type == "primary":
        # Only cross-run negative persistence
        reg_subset = regression_results[
            regression_results["persistence_var"].str.contains("neg_persist_crossrun")
        ].copy()
    elif analysis_type == "all":
        # All persistence types
        reg_subset = regression_results.copy()
    elif analysis_type == "positive":
        # Only positive cross-run persistence
        reg_subset = regression_results[
            regression_results["persistence_var"].str.contains("pos_persist_crossrun")
        ].copy()
    elif analysis_type == "concat":
        # Only concatenated negative persistence
        reg_subset = regression_results[
            regression_results["persistence_var"].str.contains("neg_persist_concat")
        ].copy()
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    if len(reg_subset) == 0:
        return None

    # Prepare plot data
    plot_data = reg_subset.copy()

    # Compute confidence intervals
    plot_data["ci_lower"] = (
        plot_data["beta_persistence"] - 1.96 * plot_data["se_persistence"]
    )
    plot_data["ci_upper"] = (
        plot_data["beta_persistence"] + 1.96 * plot_data["se_persistence"]
    )

    # Create labels
    plot_data["persist_label"] = plot_data["persistence_var"].map(PERSISTENCE_LABELS)
    plot_data["affect_label"] = plot_data["affect_var"].map(AFFECT_LABELS)
    plot_data["label"] = plot_data["persist_label"] + " → " + plot_data["affect_label"]

    # Sort by persistence type then affect
    plot_data = plot_data.sort_values(["persistence_var", "affect_var"])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, len(plot_data) * 0.3 + 2))

    y_positions = range(len(plot_data))

    # Plot each result
    for idx, row in plot_data.iterrows():
        y_pos = list(plot_data.index).index(idx)

        # Determine color based on persistence type
        if "neg_persist_crossrun" in row["persistence_var"]:
            color = COLORS["neg_crossrun"]
        elif "pos_persist_crossrun" in row["persistence_var"]:
            color = COLORS["pos_crossrun"]
        else:  # neg_concat
            color = COLORS["neg_concat"]

        # Confidence interval line
        ax.plot(
            [row["ci_lower"], row["ci_upper"]],
            [y_pos, y_pos],
            color=color,
            linewidth=2,
            alpha=0.7,
        )

        # Point estimate - filled for p<0.05, hollow for p≥0.05
        is_significant = row["p_persistence"] < 0.05
        markersize = 80

        if is_significant:
            # Filled marker
            ax.scatter(
                row["beta_persistence"],
                y_pos,
                s=markersize,
                color=color,
                marker="o",
                zorder=3,
                edgecolors="white",
                linewidths=1.5,
            )
        else:
            # Hollow marker
            ax.scatter(
                row["beta_persistence"],
                y_pos,
                s=markersize,
                facecolors="none",
                marker="o",
                zorder=3,
                edgecolors=color,
                linewidths=2,
            )

    # Vertical line at zero
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["label"], fontsize=9)
    ax.set_xlabel("Standardized Beta Coefficient (Persistence Effect)", fontsize=11)

    # Title
    title = f"{analysis_type.title()} Analyses - {SAMPLE_LABELS[sample_name]}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    # Legend
    if analysis_type == "primary":
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS["neg_crossrun"], label="Cross-run Negative"),
        ]
    else:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS["pos_crossrun"], label="Cross-run Positive"),
            Patch(facecolor=COLORS["neg_concat"], label="Concatenated Negative"),
        ]

    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=10)

    # Add note about significance
    ax.text(
        0.02,
        0.98,
        "Filled circles: p < 0.05 | Hollow circles: p ≥ 0.05",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        style="italic",
    )

    sns.despine(ax=ax, left=True)
    plt.tight_layout()

    return fig


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    # Ensure output directory exists
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Figures for Analysis 02: Persistence × Affect Associations")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading data from {DATA_FILE}...")
    data = pd.read_csv(DATA_FILE)
    print(f"✓ Loaded {len(data)} participants")

    # ========================================================================
    # Create Fisher z-transformed persistence measures
    # ========================================================================
    print("\nCreating Fisher z-transformed persistence measures...")

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
        data[z_var] = fisher_z(data[var])

    print(f"✓ Created {len(persistence_r_vars)} z-transformed variables")

    print(f"\nLoading regression results...")
    reg_full = pd.read_csv(REGRESSION_FILE_FULL)
    print(f"✓ Loaded {len(reg_full)} full sample regressions")

    # Try to load conservative sample results (may be empty if N too small)
    try:
        reg_cons = pd.read_csv(REGRESSION_FILE_CONS)
        print(f"✓ Loaded {len(reg_cons)} conservative sample regressions")
    except pd.errors.EmptyDataError:
        reg_cons = pd.DataFrame()
        print(f"⚠ Conservative sample file is empty (likely insufficient data)")

    # ========================================================================
    # Create Individual Scatterplots (Organized by Analysis Type)
    # ========================================================================
    print("\nCreating individual scatterplots...")
    counts = create_scatterplots_by_type(data)
    print(f"✓ Scatterplots saved by analysis type:")
    print(f"  - Primary (conservative × neg cross-run): {counts['primary']}")
    print(f"  - Sensitivity full sample: {counts['sensitivity_full']}")
    print(f"  - Sensitivity positive: {counts['sensitivity_positive']}")
    print(f"  - Sensitivity concat: {counts['sensitivity_concat']}")

    # ========================================================================
    # Create Forest Plots (Organized by Analysis Type)
    # ========================================================================
    print("\nCreating forest plots...")

    # PRIMARY: Conservative sample × cross-run negative
    if len(reg_cons) > 0:
        fig_primary = create_forest_plot(reg_cons, "primary", "conservative")
        if fig_primary:
            forest_file = FIG_DIR_PRIMARY / f"forest_primary_conservative.{FIGURE_FORMAT}"
            fig_primary.savefig(forest_file, dpi=FIGURE_DPI, bbox_inches="tight")
            print(f"✓ Saved primary forest plot: {forest_file.name}")
            plt.close(fig_primary)
    else:
        print(f"⚠ Skipped primary forest plot (no conservative sample data)")

    # SENSITIVITY: Full sample (all persistence types)
    fig_full = create_forest_plot(reg_full, "all", "full")
    if fig_full:
        forest_file = FIG_DIR_SENS_FULL / f"forest_full_all.{FIGURE_FORMAT}"
        fig_full.savefig(forest_file, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"✓ Saved full sample forest plot: {forest_file.name}")
        plt.close(fig_full)

    # SENSITIVITY: Conservative × positive persistence
    if len(reg_cons) > 0:
        fig_pos = create_forest_plot(reg_cons, "positive", "conservative")
        if fig_pos:
            forest_file = FIG_DIR_SENS_POS / f"forest_positive_conservative.{FIGURE_FORMAT}"
            fig_pos.savefig(forest_file, dpi=FIGURE_DPI, bbox_inches="tight")
            print(f"✓ Saved positive persistence forest plot: {forest_file.name}")
            plt.close(fig_pos)
    else:
        print(f"⚠ Skipped positive persistence forest plot (no conservative sample data)")

    # SENSITIVITY: Conservative × concatenated negative
    if len(reg_cons) > 0:
        fig_concat = create_forest_plot(reg_cons, "concat", "conservative")
        if fig_concat:
            forest_file = FIG_DIR_SENS_CONCAT / f"forest_concat_conservative.{FIGURE_FORMAT}"
            fig_concat.savefig(forest_file, dpi=FIGURE_DPI, bbox_inches="tight")
            print(f"✓ Saved concatenated negative forest plot: {forest_file.name}")
            plt.close(fig_concat)
    else:
        print(f"⚠ Skipped concatenated negative forest plot (no conservative sample data)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Figure Generation Complete")
    print("=" * 80)
    print(f"\nAll figures saved to: {FIG_DIR}")
    print(f"\nOrganization:")
    print(f"  - Primary: {FIG_DIR_PRIMARY}")
    print(f"    (Conservative sample × cross-run negative)")
    print(f"  - Sensitivity full sample: {FIG_DIR_SENS_FULL}")
    print(f"    (Full sample × all persistence types)")
    print(f"  - Sensitivity positive: {FIG_DIR_SENS_POS}")
    print(f"    (Conservative sample × positive persistence)")
    print(f"  - Sensitivity concat: {FIG_DIR_SENS_CONCAT}")
    print(f"    (Conservative sample × concatenated negative)")
    total_plots = sum(counts.values()) + 4  # scatterplots + 4 forest plots
    print(f"\nTotal figures: {total_plots}")


if __name__ == "__main__":
    main()
