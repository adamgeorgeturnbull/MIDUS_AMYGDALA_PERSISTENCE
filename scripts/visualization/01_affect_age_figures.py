#!/usr/bin/env python3
"""
01_affect_age_figures.py

Generate publication-ready figures for the affect-age replication analysis.

Creates:
1. Scatterplots with regression lines showing age × affect relationships
2. Coefficient plot (forest plot) summarizing all regression results

Inputs:
- data/processed/midus_merged_clean.csv (for scatterplots)
- results/tables/01_affect_age_regressions.csv (for coefficient plot)

Outputs:
- results/figures/01_affect_age_scatterplots.pdf
- results/figures/01_affect_age_coefficients.pdf

Run from project root directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ============================================================================
# Paths and Constants
# ============================================================================
DATA_FILE = Path("data/processed/midus_merged_clean.csv")
REGRESSION_FILE = Path("results/tables/01_affect_age_regressions.csv")
FIG_DIR = Path("results/figures/analysis_01")

# Figure parameters
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"

# Color palette
COLORS = {
    "daily_diary_full": "#2E86AB",  # Blue
    "neuro_sample": "#A23B72",      # Purple
}

# Outcome labels for plotting
OUTCOME_LABELS = {
    "PA_score": "Positive Affect (Daily Diary)",
    "NA_score": "Negative Affect (Daily Diary)",
    "NA_score_log": "Negative Affect [log-transformed] (Daily Diary)",
    "C5SPGP": "Positive Affect (PANAS)",
    "C5SPGN": "Negative Affect (PANAS)",
}

AGE_LABELS = {
    "C2PAGE": "Age at Daily Diary (years)",
    "C5PAGE": "Age at Neuroscience Visit (years)",
}

SAMPLE_LABELS = {
    "daily_diary_full": "Daily Diary Sample",
    "neuro_sample": "Neuroscience Sample",
}

# ============================================================================
# Helper Functions
# ============================================================================
def create_scatterplot_panel(ax, data, x_var, y_var, sample_name, show_ylabel=True):
    """
    Create a single scatterplot panel with regression line and confidence interval.

    Args:
        ax: Matplotlib axis object
        data: DataFrame with data
        x_var: Name of x variable (age)
        y_var: Name of y variable (affect)
        sample_name: Name of sample for coloring
        show_ylabel: Whether to show y-axis label

    Returns:
        None (modifies ax in place)
    """
    # Get complete cases
    plot_data = data[[x_var, y_var]].dropna()

    if len(plot_data) < 10:
        ax.text(0.5, 0.5, "Insufficient data",
                ha="center", va="center", transform=ax.transAxes)
        return

    # Extract arrays
    x = plot_data[x_var].values
    y = plot_data[y_var].values

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
    ax.set_xlabel(AGE_LABELS.get(x_var, x_var), fontsize=10)
    if show_ylabel:
        ax.set_ylabel(OUTCOME_LABELS.get(y_var, y_var), fontsize=10)

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


def create_individual_scatterplots(data, output_dir):
    """
    Create individual scatterplot files for each age × affect relationship.

    Args:
        data: DataFrame with merged data
        output_dir: Directory to save individual plot files

    Returns:
        List of saved file paths
    """
    # Define what to plot
    # For daily diary sample: PA, NA, NA_log × C2PAGE
    # For neuro sample: PA, NA, NA_log × C2PAGE (NOT C5PAGE), PANAS × C5PAGE

    plots_config = [
        # Daily diary sample - daily diary outcomes × diary age
        ("daily_diary_full", "C2PAGE", "PA_score"),
        ("daily_diary_full", "C2PAGE", "NA_score"),
        ("daily_diary_full", "C2PAGE", "NA_score_log"),

        # Neuro sample - daily diary outcomes × diary age (NOT neuroscience age)
        ("neuro_sample", "C2PAGE", "PA_score"),
        ("neuro_sample", "C2PAGE", "NA_score"),
        ("neuro_sample", "C2PAGE", "NA_score_log"),

        # Neuro sample - PANAS × neuroscience age
        ("neuro_sample", "C5PAGE", "C5SPGP"),
        ("neuro_sample", "C5PAGE", "C5SPGN"),
    ]

    # Define sample filters
    sample_filters = {
        "daily_diary_full": data["PA_score"].notna(),
        "neuro_sample": data["C5PAGE"].notna(),
    }

    saved_files = []

    # Create each scatterplot as individual file
    for idx, (sample, age_var, outcome) in enumerate(plots_config):
        # Create single panel figure
        fig, ax = plt.subplots(figsize=(5, 4))

        sample_data = data.loc[sample_filters[sample]]
        create_scatterplot_panel(ax, sample_data, age_var, outcome, sample, show_ylabel=True)

        plt.tight_layout()

        # Create descriptive filename
        filename = f"scatter_{sample}_{age_var}_{outcome}.{FIGURE_FORMAT}"
        filepath = output_dir / filename

        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

        saved_files.append(filepath)

    return saved_files


def create_coefficient_plot(regression_results):
    """
    Create forest plot showing regression coefficients for age effects.

    Only includes primary analyses:
    - Daily diary outcomes × C2PAGE
    - PANAS outcomes × C5PAGE

    Args:
        regression_results: DataFrame with regression results

    Returns:
        matplotlib Figure object
    """
    # Filter to primary analyses only
    # Daily diary outcomes should use C2PAGE, PANAS should use C5PAGE
    primary_analyses = regression_results[
        ((regression_results["outcome"].isin(["PA_score", "NA_score", "NA_score_log"])) &
         (regression_results["age_var"] == "C2PAGE")) |
        ((regression_results["outcome"].isin(["C5SPGP", "C5SPGN"])) &
         (regression_results["age_var"] == "C5PAGE"))
    ].copy()

    # Prepare data for plotting
    plot_data = primary_analyses.copy()

    # Calculate confidence intervals
    plot_data["ci_lower"] = plot_data["beta_age"] - 1.96 * plot_data["se_age"]
    plot_data["ci_upper"] = plot_data["beta_age"] + 1.96 * plot_data["se_age"]

    # Create labels combining outcome and age variable
    plot_data["label"] = (
        plot_data["outcome"].map(OUTCOME_LABELS) +
        "\n(" + plot_data["age_var"].map(AGE_LABELS) + ")"
    )

    # Add sample info
    plot_data["sample_label"] = plot_data["sample"].map(SAMPLE_LABELS)

    # Sort by sample and outcome
    plot_data = plot_data.sort_values(["sample", "outcome", "age_var"])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Y positions for each coefficient
    y_positions = np.arange(len(plot_data))

    # Plot coefficients and CIs
    for idx, row in plot_data.iterrows():
        y_pos = list(plot_data.index).index(idx)
        color = COLORS.get(row["sample"], "#333333")

        # Confidence interval line
        ax.plot([row["ci_lower"], row["ci_upper"]], [y_pos, y_pos],
                color=color, linewidth=2, alpha=0.7)

        # Point estimate - filled for p<0.05, hollow for p≥0.05
        is_significant = row["p_age"] < 0.05
        markersize = 80  # s parameter for scatter

        if is_significant:
            # Filled marker
            ax.scatter(row["beta_age"], y_pos, s=markersize,
                       color=color, marker="o", zorder=3,
                       edgecolors="white", linewidths=1.5)
        else:
            # Hollow marker
            ax.scatter(row["beta_age"], y_pos, s=markersize,
                       facecolors="none", marker="o", zorder=3,
                       edgecolors=color, linewidths=2)

    # Vertical line at zero
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["label"], fontsize=9)
    ax.set_xlabel("Standardized Beta Coefficient (Age Effect)", fontsize=11)
    ax.set_ylabel("Outcome Measure", fontsize=11)
    ax.set_title("Age Effects on Affect Measures\n(Controlling for sex, education, race, twin pairs)",
                 fontsize=12, fontweight="bold", pad=20)

    # Grid
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend for samples
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["daily_diary_full"], label=SAMPLE_LABELS["daily_diary_full"]),
        Patch(facecolor=COLORS["neuro_sample"], label=SAMPLE_LABELS["neuro_sample"]),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=10)

    # Add note about significance
    ax.text(0.02, 0.98, "Filled circles: p < 0.05 | Hollow circles: p ≥ 0.05",
            transform=ax.transAxes, fontsize=9, va="top", style="italic")

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

    print("Generating affect-age replication figures...")

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading data from {DATA_FILE}...")
    data = pd.read_csv(DATA_FILE)
    print(f"✓ Loaded {len(data)} participants")

    print(f"\nLoading regression results from {REGRESSION_FILE}...")
    regression_results = pd.read_csv(REGRESSION_FILE)
    print(f"✓ Loaded {len(regression_results)} regression results")

    # ========================================================================
    # Create Individual Scatterplots
    # ========================================================================
    print("\nCreating individual scatterplots...")
    saved_files = create_individual_scatterplots(data, FIG_DIR)
    print(f"✓ {len(saved_files)} individual scatterplots saved to {FIG_DIR}")
    for f in saved_files:
        print(f"  - {f.name}")

    # ========================================================================
    # Create Coefficient Plot
    # ========================================================================
    print("\nCreating coefficient plot...")
    fig_coef = create_coefficient_plot(regression_results)

    coef_file = FIG_DIR / f"01_affect_age_coefficients.{FIGURE_FORMAT}"
    fig_coef.savefig(coef_file, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"✓ Coefficient plot saved to {coef_file}")
    plt.close(fig_coef)

    print("\n✓ All figures generated successfully!")


if __name__ == "__main__":
    main()
