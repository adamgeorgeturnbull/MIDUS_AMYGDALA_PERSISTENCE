#!/usr/bin/env python3
"""
simple_slopes_07_reappraisal_antFC.py

Quick simple slopes check for the reappraisal x anterior vmPFC FC -> PA
interaction (Analysis 07). Uses OLS (not MLM) since we only need the direction.

Run from project root directory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from analysis_utils import load_master, get_samples, get_covariates

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

FC_VAR   = "l_amyg-ant_vmPFC_neg_vs_neu"
MOD_VAR  = "C5SER"
OUTCOME  = "PA_score"
FONT     = "Arial"
COL_LOW  = "#2E86AB"
COL_MID  = "#888888"
COL_HIGH = "#A23B72"


def main():
    df = load_master(fc=True)
    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1
    has_qc      = df.get("qc_conservative",    pd.Series(0, index=df.index)) == 1
    has_diary   = df[["PA_score", "NA_score"]].notna().any(axis=1)
    cons = df[has_persist & has_qc & has_diary & df[FC_VAR].notna()].copy()

    covs = [c for c in get_covariates(cons) if not c.startswith("twin_pair_")]
    cols = [OUTCOME, FC_VAR, MOD_VAR] + covs
    data = cons[cols].dropna().copy()
    print(f"N = {len(data)}")

    fc_c   = data[FC_VAR] - data[FC_VAR].mean()
    reap_c = data[MOD_VAR] - data[MOD_VAR].mean()
    mod_sd = reap_c.std()

    cov_cols = [c for c in covs if data[c].std() > 0]
    X = np.column_stack([
        np.ones(len(data)),
        fc_c,
        reap_c,
        fc_c * reap_c,
        data[cov_cols].values,
    ])
    y = data[OUTCOME].values
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    b_int, b_fc, b_mod, b_inter = b[0], b[1], b[2], b[3]

    print(f"\nOLS coefficients:")
    print(f"  FC main effect:  {b_fc:.3f}")
    print(f"  Reappraisal:     {b_mod:.3f}")
    print(f"  Interaction:     {b_inter:.3f}")
    print(f"\nSimple slopes (FC -> PA):")
    mod_vals = {"−1 SD (low reappraisal)": -mod_sd,
                "Mean reappraisal":          0.0,
                "+1 SD (high reappraisal)": +mod_sd}
    for label, m in mod_vals.items():
        slope = b_fc + b_inter * m
        print(f"  {label}: slope = {slope:.3f}")

    # Plot
    fc_range = np.linspace(fc_c.min(), fc_c.max(), 200)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for (label, m), color in zip(mod_vals.items(), [COL_LOW, COL_MID, COL_HIGH]):
        slope     = b_fc + b_inter * m
        intercept = b_int + b_mod * m
        ax.plot(fc_range, intercept + slope * fc_range,
                color=color, linewidth=2.2, label=label)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Anterior vmPFC–amygdala FC (mean-centered, neg−neu)",
                  fontfamily=FONT, fontsize=10)
    ax.set_ylabel("Positive Affect (Daily Diary)", fontfamily=FONT, fontsize=10)
    ax.set_title("Reappraisal × Anterior vmPFC FC → PA",
                 fontfamily=FONT, fontsize=11)
    ax.legend(title="Reappraisal (C5SER)", fontsize=9,
              title_fontsize=9, frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "simple_slopes_07_reappraisal_antFC.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
