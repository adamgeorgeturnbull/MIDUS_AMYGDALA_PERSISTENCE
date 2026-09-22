#!/usr/bin/env python3
"""
00b_task_condition_differences.py

Task validity check: are ROI activations and task-based FC (LSS)
condition-dependent?

Sections:
  1. ROI activations (one-sample t vs 0; paired t for neg vs neu, neg vs pos)
     ROIs:        l_amyg, r_amyg, ant_vmPFC, post_vmPFC
     Trial types: image only (face trials are neutral regardless of condition)
  2. Beta-series FC / LSS (one-sample t vs 0; paired t for neg vs neu, neg vs pos)
     Connections: l_amyg-ant_vmPFC, l_amyg-post_vmPFC,
                  r_amyg-ant_vmPFC, r_amyg-post_vmPFC

Sample   : conservative fMRI sample (qc_conservative == 1)
Tests    : two-tailed

Outputs:
  results/tables/00b_task_conditions/roi_one_sample.csv
  results/tables/00b_task_conditions/roi_paired.csv
  results/tables/00b_task_conditions/fc_one_sample.csv
  results/tables/00b_task_conditions/fc_paired.csv

Run from project root directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from confidence_intervals import mean_ci

# ============================================================================
# Paths / constants
# ============================================================================
MASTER_FILE  = Path("data/processed/midus_with_fmri.csv")
ROI_FILE     = Path("data/fMRI/all_subjects_roi_activations.csv")
_LSS_FILE    = Path("data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv")
_LSA_FILE    = Path("data/fMRI/all_subjects_betaSeries_all_conditions_M2ID.csv")
FC_FILE      = _LSS_FILE if _LSS_FILE.exists() else _LSA_FILE
RESULTS_DIR  = Path("results/tables/00b_task_conditions")
MIN_N        = 20

ROIS       = ["l_amyg", "r_amyg", "ant_vmPFC", "post_vmPFC"]
CONDITIONS = ["neg", "neu", "pos"]
TRIAL_TYPES = ["image"]  # face trials are neutral regardless of condition

FC_CONNECTIONS = [
    "l_amyg-ant_vmPFC",
    "l_amyg-post_vmPFC",
    "r_amyg-ant_vmPFC",
    "r_amyg-post_vmPFC",
]


# ============================================================================
# Statistical helpers
# ============================================================================
def one_sample_t(data, col):
    """One-sample t-test against zero for a single column."""
    x = data[col].dropna()
    if len(x) < MIN_N:
        return None
    t, p = stats.ttest_1samp(x, 0)
    return {
        **mean_ci(x),
        "variable": col,
        "n":        len(x),
        "mean":     float(x.mean()),
        "sd":       float(x.std(ddof=1)),
        "t":        float(t),
        "df":       len(x) - 1,
        "p":        float(p),
    }


def paired_t(data, col1, col2, contrast_label):
    """Paired t-test for col1 − col2."""
    d = data[[col1, col2]].dropna()
    if len(d) < MIN_N:
        return None
    diff = d[col1] - d[col2]
    t, p = stats.ttest_rel(d[col1], d[col2])
    return {
        **mean_ci(diff),
        "contrast":  contrast_label,
        "col1":      col1,
        "col2":      col2,
        "n":         len(d),
        "mean_diff": float(diff.mean()),
        "sd_diff":   float(diff.std(ddof=1)),
        "t":         float(t),
        "df":        len(d) - 1,
        "p":         float(p),
    }


def _save(rows, out_path, label):
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    sig = df[df["p"] < 0.05]
    print(f"\n  {label}  ({len(sig)}/{len(df)} p < .05)")
    for _, row in sig.iterrows():
        var = row.get("variable") or row.get("contrast")
        t, df_val, p = row["t"], row["df"], row["p"]
        print(f"    {var:40s}  t({int(df_val)})={t:+.3f}  p={p:.4f}")
    print(f"  Saved → {out_path}")
    return df


# ============================================================================
# Load data
# ============================================================================
def load_data():
    """
    Returns (roi_cons, fc_cons): conservative-sample DataFrames
    for ROI activations and LSS FC respectively.
    """
    # Master for QC flags
    master = pd.read_csv(MASTER_FILE)
    master["M2ID"] = master["M2ID"].astype(str)
    qc = master[["M2ID", "qc_conservative"]].copy()

    # ---- ROI activations ----
    if not ROI_FILE.exists():
        print(f"  WARNING: {ROI_FILE} not found — skipping ROI section.")
        roi_cons = None
    else:
        roi_raw = pd.read_csv(ROI_FILE)
        roi_raw["M2ID"] = roi_raw["M2ID"].astype(str)
        roi_df   = qc.merge(roi_raw, on="M2ID", how="inner")
        roi_cons = roi_df[roi_df["qc_conservative"] == 1].copy()
        print(f"  ROI activations — conservative N = {len(roi_cons)}")

    # ---- LSS FC ----
    if not FC_FILE.exists():
        print(f"  WARNING: {FC_FILE} not found — skipping FC section.")
        fc_cons = None
    else:
        fc_raw = pd.read_csv(FC_FILE)
        fc_raw["M2ID"] = fc_raw["M2ID"].astype(str)
        fc_df   = qc.merge(fc_raw, on="M2ID", how="inner")
        fc_cons = fc_df[fc_df["qc_conservative"] == 1].copy()
        fc_label = "LSS" if "LSS" in FC_FILE.name else "LSA"
        print(f"  FC ({fc_label}) — conservative N = {len(fc_cons)}")

    return roi_cons, fc_cons


# ============================================================================
# Section 1: ROI Activations
# ============================================================================
def run_roi_activations(df):
    print("\n" + "=" * 70)
    print("  Section 1: ROI Activations")
    print("=" * 70)

    one_sample_rows, paired_rows = [], []

    for trial_type in TRIAL_TYPES:
        for roi in ROIS:
            # One-sample t-tests for each condition
            for cond in CONDITIONS:
                col = f"{roi}_{cond}_{trial_type}"
                if col not in df.columns:
                    continue
                row = one_sample_t(df, col)
                if row:
                    row["roi"] = roi
                    row["condition"] = cond
                    row["trial_type"] = trial_type
                    one_sample_rows.append(row)

            # Paired contrasts: neg vs neu, neg vs pos
            for ref_cond in ["neu", "pos"]:
                col_neg = f"{roi}_neg_{trial_type}"
                col_ref = f"{roi}_{ref_cond}_{trial_type}"
                if col_neg not in df.columns or col_ref not in df.columns:
                    continue
                label = f"{roi}_neg_vs_{ref_cond}_{trial_type}"
                row = paired_t(df, col_neg, col_ref, label)
                if row:
                    row["roi"] = roi
                    row["trial_type"] = trial_type
                    paired_rows.append(row)

    _save(one_sample_rows, RESULTS_DIR / "roi_one_sample.csv",
          "ROI activations — one-sample t vs 0")
    _save(paired_rows, RESULTS_DIR / "roi_paired.csv",
          "ROI activations — paired t (neg vs neu / pos)")


# ============================================================================
# Section 2: Task-based FC (LSS)
# ============================================================================
def run_fc_conditions(df):
    print("\n" + "=" * 70)
    print("  Section 2: Task-based FC (LSS)")
    print("=" * 70)

    one_sample_rows, paired_rows = [], []

    for conn in FC_CONNECTIONS:
        # One-sample t-tests for each condition
        for cond in CONDITIONS:
            col = f"{conn}_{cond}"
            if col not in df.columns:
                continue
            row = one_sample_t(df, col)
            if row:
                row["connection"] = conn
                row["condition"] = cond
                one_sample_rows.append(row)

        # Paired contrasts: neg vs neu, neg vs pos
        for ref_cond in ["neu", "pos"]:
            col_neg = f"{conn}_neg"
            col_ref = f"{conn}_{ref_cond}"
            if col_neg not in df.columns or col_ref not in df.columns:
                continue
            label = f"{conn}_neg_vs_{ref_cond}"
            row = paired_t(df, col_neg, col_ref, label)
            if row:
                row["connection"] = conn
                paired_rows.append(row)

    _save(one_sample_rows, RESULTS_DIR / "fc_one_sample.csv",
          "FC — one-sample t vs 0")
    _save(paired_rows, RESULTS_DIR / "fc_paired.csv",
          "FC — paired t (neg vs neu / pos)")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 00b: Task Condition Differences")
    print("  (ROI activations + LSS FC, conservative fMRI sample)")
    print("=" * 70)

    roi_cons, fc_cons = load_data()

    if roi_cons is not None and len(roi_cons) >= MIN_N:
        run_roi_activations(roi_cons)
    else:
        print("\n  Skipping ROI activations (no data or N < 20).")
        print("  Run extract_roi_activations.sh → combineROIActivations.py →")
        print("  09_merge_fmri_data.py first.")

    if fc_cons is not None and len(fc_cons) >= MIN_N:
        run_fc_conditions(fc_cons)
    else:
        print("\n  Skipping LSS FC (no data or N < 20).")
        print("  Run runBStaskFC_LSS.sh → combineBTS_LSS.py first.")


if __name__ == "__main__":
    main()
