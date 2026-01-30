#!/usr/bin/env python3
"""
07_sample_descriptives.py

Compute sample descriptives for 5 MIDUS analytic samples and generate
publication-ready summary table.

Samples:
1) Full daily diary sample
2) Full neuroscience sample
3) Daily diary + neuroscience overlap
4) Neuroimaging completers (C5IC == 1)
5) Imaging + diary overlap

Inputs:
- data/processed/midus_merged_clean.csv (cleaned master dataset)

Outputs:
- results/tables/sample_descriptives.csv (N, age, sex, education, ethnicity, race)

Run from project root directory.
"""

import os

import pandas as pd

# ============================================================================
# Paths and Constants
# ============================================================================
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results/tables"

CLEAN_FILE = os.path.join(PROCESSED_DIR, "midus_merged_clean.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "sample_descriptives.csv")

# ============================================================================
# Helper Functions
# ============================================================================
def summarize_sample(df_sample, use_age_col):
    """
    Compute descriptive statistics for a sample.

    Args:
        df_sample: DataFrame subset representing the sample
        use_age_col: Column name to use for age (C2PAGE or C5PAGE)

    Returns:
        Series with N, age stats, sex %, education, ethnicity %, and race %s
    """
    summary = {}

    # Age statistics (N based on non-missing age)
    age_series = df_sample[use_age_col].dropna()
    summary["N"] = len(age_series)
    summary["Age_mean"] = f"{age_series.mean():.1f}"
    summary["Age_SD"] = f"{age_series.std():.1f}"
    summary["Age_range"] = f"{age_series.min():.0f}-{age_series.max():.0f}"

    # Sex (1 = Male, 2 = Female)
    sex_counts = df_sample["sex"].value_counts(dropna=False)
    n_total = len(df_sample)
    summary["%Female"] = f"{(sex_counts.get(2, 0) / n_total * 100):.1f}"

    # Education (1-12 scale)
    educ_series = df_sample["educ"].dropna()
    summary["Educ_mean"] = f"{educ_series.mean():.1f}"
    summary["Educ_SD"] = f"{educ_series.std():.1f}"

    # Ethnicity (Hispanic/Latino)
    eth_counts = df_sample["ethnicity"].value_counts(dropna=False)
    summary["%Hispanic"] = f"{(eth_counts.get(1, 0) / len(df_sample) * 100):.1f}"

    # Race categories
    race_counts = df_sample["race"].value_counts(dropna=False)
    summary["%White"] = f"{(race_counts.get(1, 0) / len(df_sample) * 100):.1f}"
    summary["%Black"] = f"{(race_counts.get(2, 0) / len(df_sample) * 100):.1f}"
    summary["%NativeAmerican"] = f"{(race_counts.get(3, 0) / len(df_sample) * 100):.1f}"
    summary["%Asian"] = f"{(race_counts.get(4, 0) / len(df_sample) * 100):.1f}"
    summary["%PacificIslander"] = f"{(race_counts.get(5, 0) / len(df_sample) * 100):.1f}"
    summary["%Other"] = f"{(race_counts.get(6, 0) / len(df_sample) * 100):.1f}"

    return pd.Series(summary)


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    # Ensure output directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load cleaned merged dataset
    df = pd.read_csv(CLEAN_FILE)
    print(f"Loaded cleaned merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Define analytic samples
    samples = {
        "daily_diary_full": df[df["StartYear"].notna()],
        "neuro_full": df[df["C5PDATE_YR"].notna()],
        "daily_neuro_overlap": df[df["StartYear"].notna() & df["C5PDATE_YR"].notna()],
        "neuro_imaging": df[df["C5IC"] == 1],
        "imaging_daily_overlap": df[(df["C5IC"] == 1) & df["StartYear"].notna()]
    }

    # Compute descriptives for each sample
    rows = []
    for name, sample_df in samples.items():
        # Use C2PAGE for diary-only sample, C5PAGE for neuroscience samples
        age_col = "C2PAGE" if name == "daily_diary_full" else "C5PAGE"
        rows.append(summarize_sample(sample_df, age_col).rename(name))

    # Create and save table
    descriptives_table = pd.DataFrame(rows).T
    descriptives_table.to_csv(OUTPUT_FILE)
    print(f"✓ Sample descriptives saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
