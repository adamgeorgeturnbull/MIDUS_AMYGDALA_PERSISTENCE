#!/usr/bin/env python3
"""
suppression_vmPFC_moderation_residualized.py

Residualized median-split scatterplot for the suppression × posterior
vmPFC-amygdala FC interaction predicting PANAS negative affect (N = 127).

Residualizes FC, suppression, and PANAS NA on covariates (age, sex, race)
before plotting, so the visualization matches the regression estimate.

Run from project root directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# ============================================================================
# Paths and settings
# ============================================================================
DATA_FILE = Path("data/processed/midus_with_fmri.csv")
FIG_DIR   = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

FC_VAR  = "conn_l_amyg_post_vmPFC_neg_vs_neu"
MOD_VAR = "C5SES"
OUTCOMES = [("C5SPGN",     "PANAS Negative Affect (raw)"),
            ("C5SPGN_log", "PANAS Negative Affect (log)")]

COL_LOW  = "#2E86AB"
COL_HIGH = "#A23B72"

# ============================================================================
# Load and prepare data
# ============================================================================
df = pd.read_csv(DATA_FILE)

cov_cols = (["C5PAGE", "sex"] +
            [c for c in df.columns if c.startswith("race_")])
cov_cols = [c for c in cov_cols if c in df.columns]

vars_needed = [FC_VAR, MOD_VAR] + [o for o, _ in OUTCOMES] + cov_cols
df_clean = df[vars_needed].dropna()
print(f"N = {len(df_clean)}")

# ============================================================================
# Residualize on covariates
# ============================================================================
def residualize(y, X):
    reg = LinearRegression().fit(X, y)
    return y - reg.predict(X)

X = df_clean[cov_cols].values

df_clean["fc_resid"]  = residualize(df_clean[FC_VAR].values, X)
df_clean["sup_resid"] = residualize(df_clean[MOD_VAR].values, X)
for out_var, _ in OUTCOMES:
    df_clean[f"{out_var}_resid"] = residualize(df_clean[out_var].values, X)

# Median split on residualized suppression
median_sup = df_clean["sup_resid"].median()
df_clean["suppression_group"] = np.where(
    df_clean["sup_resid"] <= median_sup, "Low suppression", "High suppression"
)
print(f"Groups: {df_clean['suppression_group'].value_counts().to_dict()}")

# ============================================================================
# Within-group correlations
# ============================================================================
for out_var, out_label in OUTCOMES:
    print(f"\n{out_label} (residualized):")
    for grp in ["Low suppression", "High suppression"]:
        sub = df_clean[df_clean["suppression_group"] == grp]
        r, p = stats.pearsonr(sub["fc_resid"], sub[f"{out_var}_resid"])
        print(f"  {grp}: r = {r:.3f}, p = {p:.3f}, n = {len(sub)}")

# ============================================================================
# Plot
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, (out_var, out_label) in zip(axes, OUTCOMES):
    resid_out = f"{out_var}_resid"
    for grp, color in [("Low suppression", COL_LOW), ("High suppression", COL_HIGH)]:
        sub = df_clean[df_clean["suppression_group"] == grp]
        ax.scatter(sub["fc_resid"], sub[resid_out], color=color,
                   alpha=0.5, s=30, label=grp, zorder=2)
        m, b = np.polyfit(sub["fc_resid"], sub[resid_out], 1)
        x_range = np.linspace(df_clean["fc_resid"].min(),
                              df_clean["fc_resid"].max(), 100)
        ax.plot(x_range, m * x_range + b, color=color, linewidth=2, zorder=3)

    ax.set_xlabel("Posterior vmPFC–amygdala FC\n(residualized)", fontsize=11)
    ax.set_ylabel(f"{out_label}\n(residualized)", fontsize=11)
    ax.set_title(f"Suppression × Post vmPFC FC\n→ {out_label}", fontsize=11)
    ax.legend(title="Suppression use", fontsize=9, title_fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    sns.despine(ax=ax)

plt.tight_layout()
out_path = FIG_DIR / "suppression_vmPFC_moderation_residualized.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
plt.show()
