#!/usr/bin/env python3
"""
00c_vmPFC_convergence.py

Are anterior and posterior vmPFC distinct during this task?

For each measurement method and condition, correlates the anterior and
posterior vmPFC measures within participants. A high r means the two ROIs
are tracking the same signal; a low r means they carry unique variance.

Methods × conditions tested:
  1. ROI activation (neg, neu, pos image trials)
  2. Spatial persistence (neg, neu, pos)
  3. Task-based FC / LSS (neg, neu, pos)

Sample  : conservative fMRI sample (qc_conservative == 1, no diary required)
Tests   : two-tailed Pearson r

Output:
  results/tables/00c_vmPFC_convergence/ant_post_correlations.csv

Run from project root directory.
"""

from pathlib import Path

import pandas as pd
from scipy import stats

# ============================================================================
# Paths
# ============================================================================
MASTER_FILE  = Path("data/processed/midus_with_fmri.csv")
ROI_FILE     = Path("data/fMRI/all_subjects_roi_activations.csv")
_LSS_FILE    = Path("data/fMRI/all_subjects_betaSeries_LSS_all_conditions_M2ID.csv")
_LSA_FILE    = Path("data/fMRI/all_subjects_betaSeries_all_conditions_M2ID.csv")
FC_FILE      = _LSS_FILE if _LSS_FILE.exists() else _LSA_FILE
PERSIST_FILE = Path("data/fMRI/vmPFC_persistence_wide.csv")
RESULTS_DIR  = Path("results/tables/00c_vmPFC_convergence")
MIN_N        = 20

# ant–post pairs per method and condition
# Format: (method_label, condition, ant_col, post_col)
PAIRS = [
    # ROI activations
    ("activation", "neg", "ant_vmPFC_neg_image",        "post_vmPFC_neg_image"),
    ("activation", "neu", "ant_vmPFC_neu_image",        "post_vmPFC_neu_image"),
    ("activation", "pos", "ant_vmPFC_pos_image",        "post_vmPFC_pos_image"),
    # Spatial persistence
    ("persistence", "neg", "ant_vmPFC_neg_image_mean_r", "post_vmPFC_neg_image_mean_r"),
    ("persistence", "neu", "ant_vmPFC_neu_image_mean_r", "post_vmPFC_neu_image_mean_r"),
    ("persistence", "pos", "ant_vmPFC_pos_image_mean_r", "post_vmPFC_pos_image_mean_r"),
    # Task-based FC (LSS)
    ("fc_lss", "neg", "l_amyg-ant_vmPFC_neg", "l_amyg-post_vmPFC_neg"),
    ("fc_lss", "neu", "l_amyg-ant_vmPFC_neu", "l_amyg-post_vmPFC_neu"),
    ("fc_lss", "pos", "l_amyg-ant_vmPFC_pos", "l_amyg-post_vmPFC_pos"),
]


# ============================================================================
# Load / merge data
# ============================================================================
def load_data():
    master = pd.read_csv(MASTER_FILE)
    master["M2ID"] = master["M2ID"].astype(str)
    frames = [master[["M2ID", "qc_conservative"]].copy()]

    if ROI_FILE.exists():
        roi = pd.read_csv(ROI_FILE)
        roi["M2ID"] = roi["M2ID"].astype(str)
        frames.append(roi)
    else:
        print(f"  WARNING: {ROI_FILE} not found — activation pairs will be skipped.")

    if PERSIST_FILE.exists():
        pers = pd.read_csv(PERSIST_FILE)
        pers = pers.rename(columns={"subject": "M2ID"})
        pers["M2ID"] = pers["M2ID"].astype(str).str.replace("sub-", "", regex=False)
        frames.append(pers)
    else:
        print(f"  WARNING: {PERSIST_FILE} not found — persistence pairs will be skipped.")

    if FC_FILE.exists():
        fc = pd.read_csv(FC_FILE)
        fc["M2ID"] = fc["M2ID"].astype(str)
        frames.append(fc)
    else:
        print(f"  WARNING: {FC_FILE} not found — FC pairs will be skipped.")

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="M2ID", how="left")

    df = df[df["qc_conservative"] == 1].copy()
    print(f"  Conservative fMRI N = {len(df)}")
    return df


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 00c: Anterior vs Posterior vmPFC — Within-Method Correlations")
    print("=" * 70)

    df = load_data()

    rows = []
    for method, condition, ant_col, post_col in PAIRS:
        if ant_col not in df.columns or post_col not in df.columns:
            continue
        d = df[[ant_col, post_col]].dropna()
        if len(d) < MIN_N:
            continue
        r, p = stats.pearsonr(d[ant_col], d[post_col])
        rows.append({
            "method":    method,
            "condition": condition,
            "ant_col":   ant_col,
            "post_col":  post_col,
            "n":         len(d),
            "r":         round(float(r), 4),
            "p":         round(float(p), 4),
        })

    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "ant_post_correlations.csv", index=False)

    print(f"\n  {'Method':<12} {'Cond':<6} {'n':>4}  {'r':>7}  {'p':>7}")
    print("  " + "-" * 42)
    for _, row in out.iterrows():
        print(f"  {row['method']:<12} {row['condition']:<6} {row['n']:>4}  "
              f"{row['r']:>7.3f}  {row['p']:>7.4f}")
    print(f"\n  Saved → {RESULTS_DIR / 'ant_post_correlations.csv'}")


if __name__ == "__main__":
    main()
