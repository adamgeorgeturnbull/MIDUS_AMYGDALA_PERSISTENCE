#!/usr/bin/env python3
"""
simple_slopes_07_suppression_vmPFC.py

Simple slopes plot for the significant MLM interaction:
  suppression (C5SES) × posterior vmPFC–amygdala FC (neg−neu)
  → PANAS negative affect (C5SPGN)

Predicted values are computed from fixed effects only (random intercept = 0).
Three lines: suppression at −1 SD, mean, +1 SD.
Confidence bands via parametric SE of the linear combination of coefficients.

Run from project root directory.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from analysis_utils import load_master, get_covariates

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PREDICTOR = "l_amyg-post_vmPFC_neg_vs_neu"
MODERATOR = "C5SES"
OUTCOME   = "C5SPGN"

COL_LOW  = "#2E86AB"   # low suppression
COL_MID  = "#888888"   # mean suppression
COL_HIGH = "#A23B72"   # high suppression
FONT     = "Arial"


def main():
    df = load_master(fc=True)

    # Same sample as 07_sensitivity PANAS section
    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1
    has_qc      = df.get("qc_conservative",    pd.Series(0, index=df.index)) == 1
    anchor_fc   = "l_amyg-ant_vmPFC_neg_vs_neu"
    sample = df[has_persist & has_qc & df[anchor_fc].notna()].copy()

    covs = get_covariates(sample)
    cov_list = [c for c in covs if not c.startswith("twin_pair_")]

    # Family grouping (mirrors run_moderation_mlm)
    if "SAMPLMAJ" in sample.columns and "M2FAMNUM" in sample.columns:
        is_twin = sample["SAMPLMAJ"] == 3
        paired  = sample.loc[is_twin, "M2FAMNUM"].value_counts()
        paired  = paired[paired > 1].index
        sample["_family_id"] = sample["M2ID"].astype(str)
        mask = is_twin & sample["M2FAMNUM"].isin(paired)
        sample.loc[mask, "_family_id"] = "fam_" + sample.loc[mask, "M2FAMNUM"].astype(int).astype(str)
    else:
        sample["_family_id"] = sample["M2ID"].astype(str)

    cols = [OUTCOME, PREDICTOR, MODERATOR, "_family_id"] + cov_list
    data = sample[cols].dropna().copy()
    print(f"N = {len(data)}")

    # Mean-center predictor and moderator
    pred_c = PREDICTOR.replace("-", "_").replace(".", "_") + "_c"
    mod_c  = MODERATOR + "_c"
    int_c  = pred_c + "_x_" + mod_c

    data[pred_c] = data[PREDICTOR] - data[PREDICTOR].mean()
    data[mod_c]  = data[MODERATOR]  - data[MODERATOR].mean()
    data[int_c]  = data[pred_c] * data[mod_c]

    cov_list = [c for c in cov_list if data[c].std() > 0]
    cov_terms = (" + " + " + ".join(cov_list)) if cov_list else ""
    formula = f"{OUTCOME} ~ {pred_c} + {mod_c} + {int_c}{cov_terms}"

    result = None
    for method in ["lbfgs", "powell", "nm", "bfgs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = smf.mixedlm(formula, data=data, groups=data["_family_id"])
                result = mdl.fit(reml=True, method=method)
                print(f"  Converged with {method}")
                break
        except Exception:
            continue

    if result is None:
        print("MLM failed — exiting.")
        return

    b_pred = result.fe_params[pred_c]
    b_mod  = result.fe_params[mod_c]
    b_int  = result.fe_params[int_c]
    b_con  = result.fe_params["Intercept"]

    mod_sd   = data[mod_c].std()
    mod_vals = {"−1 SD": -mod_sd, "Mean": 0.0, "+1 SD": +mod_sd}

    # Simple slope at a given moderator value = b_pred + b_int * mod_val
    # Predicted outcome = b_con + (b_pred + b_int*mod_val)*x + b_mod*mod_val

    fc_range = np.linspace(data[pred_c].min(), data[pred_c].max(), 200)

    # ── Confidence bands via delta method ────────────────────────────────────
    # Var(b_pred + b_int*m) = Var(b_pred) + m²·Var(b_int) + 2m·Cov(b_pred,b_int)
    cov_mat  = result.cov_params()
    var_pred = cov_mat.loc[pred_c, pred_c]
    var_int  = cov_mat.loc[int_c,  int_c]
    cov_pi   = cov_mat.loc[pred_c, int_c]

    fig, ax = plt.subplots(figsize=(6.5, 5))

    colors = [COL_LOW, COL_MID, COL_HIGH]
    for (label, m_val), color in zip(mod_vals.items(), colors):
        slope    = b_pred + b_int * m_val
        intercept = b_con + b_mod * m_val
        y_hat    = intercept + slope * fc_range

        # SE of predicted values
        var_slope = var_pred + m_val**2 * var_int + 2 * m_val * cov_pi
        se_slope  = np.sqrt(var_slope)
        # simple SE of line: se * |x|  (intercept SE ignored for band shape)
        se_y = se_slope * np.abs(fc_range)
        ci   = 1.96 * se_y

        ax.plot(fc_range, y_hat, color=color, linewidth=2.2, label=label, zorder=3)
        ax.fill_between(fc_range, y_hat - ci, y_hat + ci,
                        color=color, alpha=0.12, zorder=2)

        # Simple slope test
        z_slope = slope / se_slope
        p_slope = float(2 * (1 - stats.norm.cdf(abs(z_slope))))
        print(f"  Suppression {label:>5}: slope = {slope:.3f}, SE = {se_slope:.3f}, "
              f"z = {z_slope:.2f}, p = {p_slope:.3f}")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    ax.set_xlabel("Posterior vmPFC–amygdala FC\n(mean-centered, neg−neu)", fontfamily=FONT, fontsize=11)
    ax.set_ylabel("PANAS Negative Affect", fontfamily=FONT, fontsize=11)
    ax.set_title("Suppression × Posterior vmPFC FC → PANAS Negative Affect",
                 fontfamily=FONT, fontsize=11, pad=10)

    leg = ax.legend(title="Suppression (C5SES)", fontsize=9,
                    title_fontsize=9, frameon=False)
    for text in leg.get_texts():
        text.set_fontfamily(FONT)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "simple_slopes_07_suppression_vmPFC.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
