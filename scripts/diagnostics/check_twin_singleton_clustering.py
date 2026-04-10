#!/usr/bin/env python3
"""
check_twin_singleton_clustering.py

Checks whether assigning twin family IDs before vs. after dropna makes any
meaningful difference to MLM estimates for the primary analyses.

Current approach: family_id assigned on the full sample before dropna.
  → If twin B is dropped due to missing data, twin A retains "fam_XXX"
    but is now a singleton cluster in the analytic sample.

Alternative:      family_id reassigned after dropna.
  → Post-dropna singletons (formerly paired twins) get their own M2ID cluster.

Runs the comparison for Analysis 02 (persistence → affect) and
Analysis 04 (FC → affect) primary outcomes.

Run from project root directory.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from analysis_utils import (
    load_master, get_samples, get_covariates,
    prepare_persistence_vars, fisher_z,
)

# ============================================================================
# Helpers
# ============================================================================
def assign_family_id(df):
    """Assign family_id on df as currently done in run_mlm (before dropna)."""
    df = df.copy()
    if "SAMPLMAJ" in df.columns and "M2FAMNUM" in df.columns:
        is_twin = df["SAMPLMAJ"] == 3
        paired = df.loc[is_twin, "M2FAMNUM"].value_counts()
        paired = paired[paired > 1].index
        df["_family_id"] = df["M2ID"].astype(str)
        mask = is_twin & df["M2FAMNUM"].isin(paired)
        df.loc[mask, "_family_id"] = "fam_" + df.loc[mask, "M2FAMNUM"].astype(int).astype(str)
    else:
        df["_family_id"] = df["M2ID"].astype(str)
    return df


def reassign_family_id_post_dropna(data):
    """Reassign family_id after dropna so post-dropna singletons get own cluster."""
    data = data.copy()
    fam_counts = data["_family_id"].value_counts()
    singletons = fam_counts[fam_counts == 1].index
    # Only fix those that were originally paired (fam_ prefix)
    originally_paired_singletons = [f for f in singletons if f.startswith("fam_")]
    if originally_paired_singletons:
        for fam in originally_paired_singletons:
            idx = data[data["_family_id"] == fam].index
            # Re-assign to their M2ID (stored in M2ID column if present)
            # Use the row index as fallback
            data.loc[idx, "_family_id"] = data.loc[idx, "M2ID"].astype(str) if "M2ID" in data.columns else str(idx[0])
    return data, len(originally_paired_singletons)


def run_mlm(data, outcome, predictor, cov_list):
    """Run MLM and return (beta, se, z, p) for predictor."""
    pred_safe = predictor.replace("-", "_").replace(".", "_")
    if pred_safe != predictor:
        data = data.copy()
        data[pred_safe] = data[predictor]

    cov_list = [c for c in cov_list if c in data.columns and data[c].std() > 0
                and not (c.startswith("twin_pair_") and data[c].sum() < 2)]
    cov_terms = " + ".join(cov_list) if cov_list else ""
    fixed = f"{outcome} ~ {pred_safe}" + (f" + {cov_terms}" if cov_terms else "")

    for method in ["lbfgs", "powell", "nm", "bfgs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = smf.mixedlm(fixed, data=data, groups=data["_family_id"])
                result = mdl.fit(reml=True, method=method)
                beta = result.fe_params[pred_safe]
                se = result.bse_fe.get(pred_safe, np.nan)
                z = beta / se if se > 0 else np.nan
                p = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan
                return beta, se, z, p, method
        except Exception:
            continue
    return np.nan, np.nan, np.nan, np.nan, "failed"


def compare(df_with_fam, outcome, predictor, cov_list, label):
    diary_covs = ["time_P2_P5", "n_days_complete"]
    covs = cov_list + [c for c in diary_covs
                       if c in df_with_fam.columns and outcome in {"PA_score","NA_score","NA_score_log"}]
    # Drop twin_pair_ from cov_list for MLM (handled by random effect)
    mlm_covs = [c for c in covs if not c.startswith("twin_pair_")]

    cols = [outcome, predictor, "_family_id"] + mlm_covs
    if "M2ID" in df_with_fam.columns:
        cols += ["M2ID"]
    data = df_with_fam[cols].dropna()

    # Current approach
    b1, se1, z1, p1, m1 = run_mlm(data, outcome, predictor, mlm_covs)

    # Alternative: reassign post-dropna singletons
    data_fixed, n_fixed = reassign_family_id_post_dropna(data)
    b2, se2, z2, p2, m2 = run_mlm(data_fixed, outcome, predictor, mlm_covs)

    print(f"\n  {label}  (n={len(data)})")
    print(f"    Post-dropna singleton pairs reassigned: {n_fixed}")
    print(f"    Current approach : β={b1:+.4f}  SE={se1:.4f}  z={z1:+.3f}  p={p1:.4f}  [{m1}]")
    print(f"    Post-dropna fix  : β={b2:+.4f}  SE={se2:.4f}  z={z2:+.3f}  p={p2:.4f}  [{m2}]")
    delta = abs(b1 - b2)
    print(f"    Δβ = {delta:.6f}  {'<< negligible' if delta < 0.001 else '*** CHECK THIS'}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Diagnostic: Twin singleton clustering before vs. after dropna")
    print("=" * 70)

    # --- Analysis 02: Persistence → Affect ---
    df = load_master()
    prepare_persistence_vars(df)
    _, cons = get_samples(df)
    df_fam = assign_family_id(cons)
    base_covs = get_covariates(cons)

    print("\n--- Analysis 02: Persistence → Affect ---")
    for outcome in ["PA_score", "NA_score", "NA_score_log"]:
        compare(df_fam, outcome, "neg_persist_crossrun_mean_z_L", base_covs,
                f"neg_persist_crossrun_mean_z_L → {outcome}")

    # --- Analysis 04: FC → Affect ---
    df_fc = load_master(fc=True)
    prepare_persistence_vars(df_fc)
    _, cons_fc = get_samples(df_fc, check_fc_col="l_amyg-ant_vmPFC_neg_vs_neu")
    df_fc_fam = assign_family_id(cons_fc)
    base_covs_fc = get_covariates(cons_fc)

    print("\n--- Analysis 04: FC → Affect ---")
    for pred in ["l_amyg-ant_vmPFC_neg_vs_neu", "l_amyg-post_vmPFC_neg_vs_neu"]:
        for outcome in ["PA_score", "NA_score", "NA_score_log"]:
            compare(df_fc_fam, outcome, pred, base_covs_fc,
                    f"{pred} → {outcome}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
