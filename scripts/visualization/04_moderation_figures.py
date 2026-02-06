#!/usr/bin/env python3
"""
04_moderation_figures.py

Create figures for Analysis 04: persistence x affect moderation by emotion
regulation strategy (reappraisal, suppression).

Organization mirrors 02_persistence_affect_figures.py:
  primary/              — conservative sample × cross-run negative × daily diary
  sensitivity/full_sample/         — full sample × all persistence × all affect
  sensitivity/positive_persistence/  — conservative × positive cross-run
  sensitivity/concatenated_negative/ — conservative × concatenated negative

Generates per folder:
1. Forest plots showing interaction coefficient magnitude and significance
   (one per moderator)
2. Median-split scatterplots for significant interactions, showing the
   persistence-affect relationship separately for high vs low ER groups

Run from project root directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# ============================================================================
# Paths and Constants
# ============================================================================
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
FIG_DIR = Path("results/figures/analysis_04")
FIG_DIR_PRIMARY = FIG_DIR / "primary"
FIG_DIR_SENS_FULL = FIG_DIR / "sensitivity/full_sample"
FIG_DIR_SENS_POS = FIG_DIR / "sensitivity/positive_persistence"
FIG_DIR_SENS_CONCAT = FIG_DIR / "sensitivity/concatenated_negative"

DATA_FILE = DATA_DIR / "midus_with_fmri.csv"
MOD_FILE_FULL = RESULTS_DIR / "04_persistence_affect_moderation_full.csv"
MOD_FILE_CONS = RESULTS_DIR / "04_persistence_affect_moderation_conservative.csv"

FIGURE_FORMAT = "png"
FIGURE_DPI = 300

# Colors
COLORS = {
    "neg_crossrun": "#E63946",
    "pos_crossrun": "#06A77D",
    "neg_concat": "#F77F00",
    "high_er": "#2E86AB",
    "low_er": "#E63946",
}

MODERATOR_LABELS = {
    "C5SER": "ERQ Reappraisal",
    "C5SES": "ERQ Suppression",
}

PERSISTENCE_LABELS = {
    "neg_persist_crossrun_mean_z_L": "Neg Persist Cross-Run (L)",
    "neg_persist_crossrun_mean_z_R": "Neg Persist Cross-Run (R)",
    "neg_persist_crossrun_mean_z_bilateral": "Neg Persist Cross-Run (Bilat)",
    "pos_persist_crossrun_mean_z_L": "Pos Persist Cross-Run (L)",
    "pos_persist_crossrun_mean_z_R": "Pos Persist Cross-Run (R)",
    "pos_persist_crossrun_mean_z_bilateral": "Pos Persist Cross-Run (Bilat)",
    "neg_persist_concat_z_L": "Neg Persist Concat (L)",
    "neg_persist_concat_z_R": "Neg Persist Concat (R)",
    "neg_persist_concat_z_bilateral": "Neg Persist Concat (Bilat)",
}

AFFECT_LABELS = {
    "PA_score": "Positive Affect (Diary)",
    "NA_score": "Negative Affect (Diary)",
    "NA_score_log": "Negative Affect [log] (Diary)",
    "C5SPGP": "Positive Affect (PANAS)",
    "C5SPGN": "Negative Affect (PANAS)",
    "C5SPGN_log": "Negative Affect [log] (PANAS)",
}

# Variable groupings
PERSIST_NEG_CROSS = [
    "neg_persist_crossrun_mean_z_L",
    "neg_persist_crossrun_mean_z_R",
    "neg_persist_crossrun_mean_z_bilateral",
]
PERSIST_POS_CROSS = [
    "pos_persist_crossrun_mean_z_L",
    "pos_persist_crossrun_mean_z_R",
    "pos_persist_crossrun_mean_z_bilateral",
]
PERSIST_NEG_CONCAT = [
    "neg_persist_concat_z_L",
    "neg_persist_concat_z_R",
    "neg_persist_concat_z_bilateral",
]

AFFECT_DIARY = ["PA_score", "NA_score", "NA_score_log"]
AFFECT_ALL = ["PA_score", "NA_score", "NA_score_log", "C5SPGP", "C5SPGN", "C5SPGN_log"]

MODERATORS = ["C5SER", "C5SES"]


# ============================================================================
# Helper Functions
# ============================================================================
def fisher_z(r):
    """Apply Fisher z-transformation to correlation coefficient."""
    return 0.5 * np.log((1 + r) / (1 - r))


def get_persist_color(persist_var):
    """Return color based on persistence variable type."""
    if "neg_persist_crossrun" in persist_var:
        return COLORS["neg_crossrun"]
    elif "pos_persist_crossrun" in persist_var:
        return COLORS["pos_crossrun"]
    return COLORS["neg_concat"]


def filter_results(mod_results, persist_vars, affect_vars, moderators):
    """Filter moderation results to a subset of variables."""
    mask = (
        mod_results["persistence_var"].isin(persist_vars) &
        mod_results["affect_var"].isin(affect_vars) &
        mod_results["moderator"].isin(moderators)
    )
    return mod_results[mask].copy()


def create_forest_plot(mod_results, moderator, title_suffix=""):
    """
    Create forest plot of interaction coefficients for one moderator.

    Rows: each persistence x affect combination.
    X-axis: interaction beta (with 95% CI).
    Filled = significant (p < .05), hollow = non-significant.
    """
    subset = mod_results[mod_results["moderator"] == moderator].copy()
    if len(subset) == 0:
        return None

    # Compute CIs
    subset["ci_lower"] = subset["beta_interaction"] - 1.96 * subset["se_interaction"]
    subset["ci_upper"] = subset["beta_interaction"] + 1.96 * subset["se_interaction"]

    # Create row labels
    subset["persist_label"] = subset["persistence_var"].map(PERSISTENCE_LABELS)
    subset["affect_label"] = subset["affect_var"].map(AFFECT_LABELS)
    subset["label"] = subset["persist_label"] + " \u2192 " + subset["affect_label"]

    # Sort by persistence type then affect
    subset = subset.sort_values(["persistence_var", "affect_var"]).reset_index(drop=True)

    fig_height = max(len(subset) * 0.35 + 2, 4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    for i, (_, row) in enumerate(subset.iterrows()):
        color = get_persist_color(row["persistence_var"])
        is_sig = row["p_interaction"] < 0.05

        # CI line
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i],
                color=color, linewidth=2, alpha=0.7)

        # Point estimate
        if is_sig:
            ax.scatter(row["beta_interaction"], i, s=80, color=color,
                       marker="o", zorder=3, edgecolors="white", linewidths=1.5)
        else:
            ax.scatter(row["beta_interaction"], i, s=80, facecolors="none",
                       marker="o", zorder=3, edgecolors=color, linewidths=2)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(subset["label"], fontsize=9)
    ax.set_xlabel("Interaction Coefficient (Persistence \u00d7 Moderator)", fontsize=11)

    mod_label = MODERATOR_LABELS.get(moderator, moderator)
    ax.set_title(f"Moderation by {mod_label}{title_suffix}",
                 fontsize=12, fontweight="bold", pad=15)

    # Build legend based on what persistence types are present
    legend_elements = []
    persist_types = subset["persistence_var"].unique()
    if any("neg_persist_crossrun" in p for p in persist_types):
        legend_elements.append(Patch(facecolor=COLORS["neg_crossrun"], label="Cross-Run Negative"))
    if any("pos_persist_crossrun" in p for p in persist_types):
        legend_elements.append(Patch(facecolor=COLORS["pos_crossrun"], label="Cross-Run Positive"))
    if any("neg_persist_concat" in p for p in persist_types):
        legend_elements.append(Patch(facecolor=COLORS["neg_concat"], label="Concatenated Negative"))
    legend_elements += [
        Line2D([0], [0], marker="o", color="grey", markerfacecolor="grey",
               markersize=8, linestyle="None", label="p < .05"),
        Line2D([0], [0], marker="o", color="grey", markerfacecolor="none",
               markersize=8, linestyle="None", markeredgewidth=2, label="p \u2265 .05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

    sns.despine(ax=ax, left=True)
    plt.tight_layout()

    return fig


def create_median_split_scatter(data, persist_var, affect_var, moderator):
    """
    Create a scatterplot showing persistence-affect relationship separately
    for participants above vs below the median on the moderator.

    Two regression lines + scatter: high ER (blue) vs low ER (red).
    """
    plot_data = data[[persist_var, affect_var, moderator]].dropna()
    if len(plot_data) < 20:
        return None

    median_val = plot_data[moderator].median()
    high = plot_data[plot_data[moderator] >= median_val]
    low = plot_data[plot_data[moderator] < median_val]

    mod_label = MODERATOR_LABELS.get(moderator, moderator)
    persist_label = PERSISTENCE_LABELS.get(persist_var, persist_var)
    affect_label = AFFECT_LABELS.get(affect_var, affect_var)

    fig, ax = plt.subplots(figsize=(6, 5))

    for group_data, color, label_prefix in [
        (high, COLORS["high_er"], f"High {mod_label}"),
        (low, COLORS["low_er"], f"Low {mod_label}"),
    ]:
        x = group_data[persist_var].values
        y = group_data[affect_var].values

        if len(x) < 5:
            continue

        r, p = stats.pearsonr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        x_pred = np.linspace(x.min(), x.max(), 100)
        y_pred = slope * x_pred + intercept

        sig = "*" if p < 0.05 else ""
        label = f"{label_prefix} (n={len(x)}, r={r:.2f}{sig})"

        ax.scatter(x, y, alpha=0.4, s=25, color=color, edgecolors="none")
        ax.plot(x_pred, y_pred, color=color, linewidth=2, label=label)

        # CI band
        n = len(x)
        x_mean = np.mean(x)
        se = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n - 2))
        margin = 1.96 * se * np.sqrt(1/n + (x_pred - x_mean)**2 / np.sum((x - x_mean)**2))
        ax.fill_between(x_pred, y_pred - margin, y_pred + margin, alpha=0.12, color=color)

    ax.set_xlabel(persist_label, fontsize=10)
    ax.set_ylabel(affect_label, fontsize=10)
    ax.set_title(f"Median Split on {mod_label}\n({persist_label} \u2192 {affect_label})",
                 fontsize=11, fontweight="bold")

    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()

    return fig


def save_forest_and_scatters(mod_results, sample_data, out_dir, persist_vars,
                              affect_vars, title_suffix, label):
    """
    Save forest plots + median-split scatterplots for a set of results.

    Returns dict with counts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = filter_results(mod_results, persist_vars, affect_vars, MODERATORS)

    if len(subset) == 0:
        print(f"  {label}: no models (insufficient data)")
        return {"forest": 0, "scatter": 0}

    forest_count = 0
    scatter_count = 0

    # Forest plots (one per moderator)
    for moderator in MODERATORS:
        fig = create_forest_plot(subset, moderator, title_suffix=title_suffix)
        if fig:
            fname = out_dir / f"forest_{moderator}.{FIGURE_FORMAT}"
            fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            forest_count += 1

    # Median-split scatterplots for significant interactions
    sig = subset[subset["p_interaction"] < 0.05]
    for _, row in sig.iterrows():
        fig = create_median_split_scatter(
            sample_data, row["persistence_var"], row["affect_var"], row["moderator"]
        )
        if fig:
            fname = (out_dir /
                     f"mediansplit_{row['moderator']}_{row['persistence_var']}"
                     f"_{row['affect_var']}.{FIGURE_FORMAT}")
            fig.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            scatter_count += 1

    n_sig = len(sig)
    print(f"  {label}: {forest_count} forest plots, "
          f"{scatter_count} median-split scatterplots ({n_sig} sig interactions)")

    return {"forest": forest_count, "scatter": scatter_count}


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Figures for Analysis 04: Moderation by Emotion Regulation")
    print("=" * 80)

    # ========================================================================
    # Load Data
    # ========================================================================
    print(f"\nLoading data from {DATA_FILE}...")
    data = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(data)} participants")

    # Fisher z-transform persistence
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

    # Define samples
    has_persist = data["has_neg_persistence"] == 1
    has_affect = data[["PA_score", "NA_score"]].notna().any(axis=1)

    data_full = data[has_persist & has_affect].copy()
    data_cons = data[has_persist & has_affect & (data["qc_conservative"] == 1)].copy()
    print(f"Full sample: {len(data_full)} | Conservative sample: {len(data_cons)}")

    # Load moderation results
    print(f"\nLoading moderation results...")
    try:
        mod_full = pd.read_csv(MOD_FILE_FULL)
        print(f"Full sample: {len(mod_full)} models")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("No full sample results found.")
        return

    try:
        mod_cons = pd.read_csv(MOD_FILE_CONS)
        if len(mod_cons) == 0:
            mod_cons = pd.DataFrame()
        print(f"Conservative sample: {len(mod_cons)} models")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        mod_cons = pd.DataFrame()
        print("Conservative sample: not available")

    # ========================================================================
    # PRIMARY: Conservative × cross-run negative × daily diary
    # ========================================================================
    print("\n--- Primary (Conservative x Cross-Run Negative x Daily Diary) ---")
    if len(mod_cons) > 0:
        save_forest_and_scatters(
            mod_cons, data_cons, FIG_DIR_PRIMARY,
            persist_vars=PERSIST_NEG_CROSS,
            affect_vars=AFFECT_DIARY,
            title_suffix=" \u2014 Primary (Conservative)",
            label="Primary",
        )
    else:
        FIG_DIR_PRIMARY.mkdir(parents=True, exist_ok=True)
        print("  Skipped (no conservative sample models)")

    # ========================================================================
    # SENSITIVITY: Full sample × all persistence × all affect
    # ========================================================================
    print("\n--- Sensitivity: Full Sample (all persistence x all affect) ---")
    all_persist = PERSIST_NEG_CROSS + PERSIST_POS_CROSS + PERSIST_NEG_CONCAT
    save_forest_and_scatters(
        mod_full, data_full, FIG_DIR_SENS_FULL,
        persist_vars=all_persist,
        affect_vars=AFFECT_ALL,
        title_suffix=" \u2014 Full Sample",
        label="Full sample",
    )

    # ========================================================================
    # SENSITIVITY: Conservative × positive cross-run
    # ========================================================================
    print("\n--- Sensitivity: Conservative x Positive Persistence ---")
    if len(mod_cons) > 0:
        save_forest_and_scatters(
            mod_cons, data_cons, FIG_DIR_SENS_POS,
            persist_vars=PERSIST_POS_CROSS,
            affect_vars=AFFECT_ALL,
            title_suffix=" \u2014 Positive Persistence (Conservative)",
            label="Positive persistence",
        )
    else:
        FIG_DIR_SENS_POS.mkdir(parents=True, exist_ok=True)
        print("  Skipped (no conservative sample models)")

    # ========================================================================
    # SENSITIVITY: Conservative × concatenated negative
    # ========================================================================
    print("\n--- Sensitivity: Conservative x Concatenated Negative ---")
    if len(mod_cons) > 0:
        save_forest_and_scatters(
            mod_cons, data_cons, FIG_DIR_SENS_CONCAT,
            persist_vars=PERSIST_NEG_CONCAT,
            affect_vars=AFFECT_ALL,
            title_suffix=" \u2014 Concatenated Negative (Conservative)",
            label="Concatenated negative",
        )
    else:
        FIG_DIR_SENS_CONCAT.mkdir(parents=True, exist_ok=True)
        print("  Skipped (no conservative sample models)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Figure Generation Complete")
    print("=" * 80)
    print(f"\nAll figures saved to: {FIG_DIR}")
    print(f"  Primary: {FIG_DIR_PRIMARY}")
    print(f"    (Conservative x cross-run negative x daily diary)")
    print(f"  Sensitivity full sample: {FIG_DIR_SENS_FULL}")
    print(f"    (Full sample x all persistence x all affect)")
    print(f"  Sensitivity positive: {FIG_DIR_SENS_POS}")
    print(f"    (Conservative x positive cross-run)")
    print(f"  Sensitivity concat: {FIG_DIR_SENS_CONCAT}")
    print(f"    (Conservative x concatenated negative)")


if __name__ == "__main__":
    main()
