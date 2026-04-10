#!/usr/bin/env python3
"""
check_mlm_pvalues.py

Diagnostic to:
1. Check how often the degenerate RE fallback triggers across all MLM results
2. Re-run primary MLM models and compare our manually computed p-values
   against statsmodels' result.pvalues to confirm they match.

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
    prepare_persistence_vars, RESULTS_DIR,
)

# ============================================================================
# Part 1: Check degenerate_re flag across all saved MLM results
# ============================================================================
def check_degenerate_flag():
    print("=" * 70)
    print("Part 1: Degenerate RE fallback frequency in saved MLM results")
    print("=" * 70)

    mlm_files = list(RESULTS_DIR.rglob("mlm.csv"))
    mlm_files += list(RESULTS_DIR.rglob("moderation_mlm.csv"))

    total_rows = 0
    degenerate_rows = 0
    degenerate_cases = []

    for f in sorted(mlm_files):
        df = pd.read_csv(f)
        if "degenerate_re" not in df.columns:
            continue
        n = len(df)
        n_deg = int(df["degenerate_re"].fillna(False).astype(bool).sum())
        total_rows += n
        degenerate_rows += n_deg
        if n_deg > 0:
            for _, row in df[df["degenerate_re"].fillna(False).astype(bool)].iterrows():
                degenerate_cases.append({
                    "file": str(f.relative_to(RESULTS_DIR)),
                    "predictor": row.get("predictor", ""),
                    "outcome": row.get("outcome", ""),
                    "p": row.get("p", np.nan),
                })

    print(f"\n  Total MLM results rows: {total_rows}")
    print(f"  Degenerate RE (fallback SE used): {degenerate_rows}")
    if degenerate_rows == 0:
        print("  >> Fallback never triggered. All SEs from bse_fe directly.")
    else:
        print(f"  >> Fallback triggered {degenerate_rows} times:")
        for c in degenerate_cases:
            print(f"     {c['file']}")
            print(f"       {c['predictor']} -> {c['outcome']}  p={c['p']:.4f}")


# ============================================================================
# Part 2: Compare our p-values to statsmodels result.pvalues
# ============================================================================
def assign_family_id(df):
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


def run_and_compare(data, outcome, predictor, cov_list, label):
    pred_safe = predictor.replace("-", "_").replace(".", "_")
    if pred_safe != predictor:
        data = data.copy()
        data[pred_safe] = data[predictor]

    cov_list = [c for c in cov_list
                if c in data.columns and data[c].std() > 0
                and not c.startswith("twin_pair_")]
    cov_terms = " + ".join(cov_list) if cov_list else ""
    fixed = f"{outcome} ~ {pred_safe}" + (f" + {cov_terms}" if cov_terms else "")

    result = None
    for method in ["lbfgs", "powell", "nm", "bfgs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = smf.mixedlm(fixed, data=data, groups=data["_family_id"])
                result = mdl.fit(reml=True, method=method)
                break
        except Exception:
            continue

    if result is None:
        print(f"  {label}: all optimizers failed")
        return

    # Our computed p
    beta = result.fe_params[pred_safe]
    se = result.bse_fe.get(pred_safe, np.nan)
    degenerate = np.isnan(se)
    if degenerate:
        idx = list(result.fe_params.index).index(pred_safe)
        var = result.cov_params().iloc[idx, idx]
        se = np.sqrt(abs(var)) if abs(var) > 0 else np.nan
    z_ours = beta / se if not np.isnan(se) and se > 0 else np.nan
    p_ours = float(2 * (1 - stats.norm.cdf(abs(z_ours)))) if not np.isnan(z_ours) else np.nan

    # statsmodels p
    p_sm = result.pvalues.get(pred_safe, np.nan)

    match = abs(p_ours - p_sm) < 1e-6 if not (np.isnan(p_ours) or np.isnan(p_sm)) else False
    flag = "OK" if match else "*** MISMATCH ***"
    deg_flag = " [degenerate RE]" if degenerate else ""
    print(f"  {label}{deg_flag}")
    print(f"    β={beta:+.4f}  SE={se:.4f}  our p={p_ours:.6f}  sm p={p_sm:.6f}  {flag}")


def check_pvalue_consistency():
    print("\n" + "=" * 70)
    print("Part 2: Our p-values vs statsmodels result.pvalues")
    print("=" * 70)

    diary_covs = ["time_P2_P5", "n_days_complete"]

    # Analysis 02: persistence → affect
    df = load_master()
    prepare_persistence_vars(df)
    _, cons = get_samples(df)
    cons = assign_family_id(cons)
    base_covs = get_covariates(cons)
    mlm_covs = [c for c in base_covs if not c.startswith("twin_pair_")]

    print("\n--- Analysis 02: Persistence → Affect ---")
    for outcome in ["PA_score", "NA_score", "NA_score_log"]:
        covs = mlm_covs + [c for c in diary_covs if c in cons.columns]
        cols = [outcome, "neg_persist_crossrun_mean_z_L", "_family_id"] + covs
        data = cons[cols].dropna()
        run_and_compare(data, outcome, "neg_persist_crossrun_mean_z_L", covs,
                        f"persistence → {outcome} (n={len(data)})")

    # Analysis 04: FC → affect
    df_fc = load_master(fc=True)
    prepare_persistence_vars(df_fc)
    _, cons_fc = get_samples(df_fc, check_fc_col="l_amyg-ant_vmPFC_neg_vs_neu")
    cons_fc = assign_family_id(cons_fc)
    base_covs_fc = get_covariates(cons_fc)
    mlm_covs_fc = [c for c in base_covs_fc if not c.startswith("twin_pair_")]

    print("\n--- Analysis 04: FC → Affect ---")
    for pred in ["l_amyg-ant_vmPFC_neg_vs_neu", "l_amyg-post_vmPFC_neg_vs_neu"]:
        for outcome in ["PA_score", "NA_score", "NA_score_log"]:
            covs = mlm_covs_fc + [c for c in diary_covs if c in cons_fc.columns]
            cols = [outcome, pred, "_family_id"] + covs
            data = cons_fc[cols].dropna()
            run_and_compare(data, outcome, pred, covs,
                            f"{pred} → {outcome} (n={len(data)})")

    # Analysis 03: age → persistence
    df3 = load_master()
    prepare_persistence_vars(df3)
    _, cons3 = get_samples(df3, require_diary=False)
    cons3 = assign_family_id(cons3)
    base_covs3 = get_covariates(cons3)
    mlm_covs3 = [c for c in base_covs3 if not c.startswith("twin_pair_")]

    print("\n--- Analysis 03: Age → Persistence ---")
    mlm_covs3_no_age = [c for c in mlm_covs3 if c != "C5PAGE"]
    cols = ["neg_persist_crossrun_mean_z_L", "C5PAGE", "_family_id"] + mlm_covs3_no_age
    data = cons3[cols].dropna()
    run_and_compare(data, "neg_persist_crossrun_mean_z_L", "C5PAGE", mlm_covs3_no_age,
                    f"age → persistence (n={len(data)})")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    check_degenerate_flag()
    check_pvalue_consistency()
    print("\nDone.")
