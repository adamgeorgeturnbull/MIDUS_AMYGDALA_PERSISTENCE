#!/usr/bin/env python3
"""
01_sensitivity.py

Sensitivity analyses for Analysis 01: Age → Affect.

Run ONLY if primary findings in 01_affect_age.py show meaningful signal.

Sections:
  1. PANAS convergent validity (C5PAGE, neuro subsample)
       → sensitivity_panas/

Note: log-transformed NA is now a primary outcome in 01_affect_age.py,
so a separate log-transform sensitivity section is not needed here.

Run from project root directory.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import write_methods_note, one_tailed_p

# ============================================================================
# Paths
# ============================================================================
DATA_FILE = Path("data/processed/midus_merged_clean.csv")
BASE_DIR  = Path("results/tables/01_affect_age")
MIN_N     = 20


# ============================================================================
# Helpers (mirror 01_affect_age.py)
# ============================================================================
def _get_covariates(df, include_diary_covs=True, include_time_p2_p5=False):
    """
    include_diary_covs: include n_days_complete (only for analyses with diary outcomes)
    include_time_p2_p5: include time between P2 and P5 visits (neuro + diary only)
    """
    race_dummies = [c for c in df.columns if c.startswith("race_")]
    twin_dummies = [c for c in df.columns if c.startswith("twin_pair_")]
    covs = ["sex"] + race_dummies + twin_dummies
    if include_diary_covs:
        covs += ["n_days_complete"]
    if include_time_p2_p5:
        covs += ["time_P2_P5"]
    return [c for c in covs if c in df.columns]


def _create_family_id(df):
    df = df.copy()
    if "SAMPLMAJ" in df.columns and "M2FAMNUM" in df.columns:
        is_twin = df["SAMPLMAJ"] == 3
        paired  = df.loc[is_twin, "M2FAMNUM"].value_counts()
        paired  = paired[paired > 1].index
        df["_family_id"] = df["M2ID"].astype(str)
        mask = is_twin & df["M2FAMNUM"].isin(paired)
        df.loc[mask, "_family_id"] = "fam_" + df.loc[mask, "M2FAMNUM"].astype(int).astype(str)
    elif "M2FAMNUM" in df.columns:
        df["_family_id"] = df["M2FAMNUM"].astype(str)
    else:
        df["_family_id"] = df["M2ID"].astype(str)
    return df


def run_correlation(df, predictor, outcome, one_tailed=False, expected_positive=True):
    data = df[[predictor, outcome]].dropna()
    if len(data) < MIN_N:
        return None
    r, p_two = stats.pearsonr(data[predictor], data[outcome])
    p = one_tailed_p(r, p_two, expected_positive) if one_tailed else p_two
    row = {"predictor": predictor, "outcome": outcome, "n": len(data), "r": r, "p": p}
    if one_tailed:
        row["p_two_tailed"] = p_two
    return row


def run_ols(df, predictor, outcome, covariates, one_tailed=False, expected_positive=True):
    cov_list = [c for c in covariates if c in df.columns and c != predictor]
    data = df[[outcome, predictor] + cov_list].dropna()
    if len(data) < MIN_N:
        return None
    cov_list = [c for c in cov_list
                if data[c].std() > 0
                and not (c.startswith("twin_pair_") and data[c].sum() < 2)]
    X = sm.add_constant(data[[predictor] + cov_list])
    try:
        model = sm.OLS(data[outcome], X).fit()
    except Exception as e:
        print(f"    OLS error ({outcome} ~ {predictor}): {e}")
        return None
    beta  = model.params[predictor]
    p_two = model.pvalues[predictor]
    p     = one_tailed_p(beta, p_two, expected_positive) if one_tailed else p_two
    row = {
        "predictor": predictor, "outcome": outcome,
        "n": int(model.nobs), "df_resid": int(model.df_resid), "beta": beta,
        "se": model.bse[predictor], "t": model.tvalues[predictor],
        "p": p,
        "r_squared": model.rsquared, "adj_r_squared": model.rsquared_adj,
    }
    if one_tailed:
        row["p_two_tailed"] = p_two
    return row


def run_mlm(df, predictor, outcome, covariates, one_tailed=False, expected_positive=True):
    df = _create_family_id(df)
    cov_list = [c for c in covariates
                if c in df.columns and c != predictor and not c.startswith("twin_pair_")]
    data = df[[outcome, predictor, "_family_id"] + cov_list].dropna()
    if len(data) < MIN_N:
        return None
    cov_list = [c for c in cov_list if data[c].std() > 0]
    cov_terms = " + ".join(cov_list) if cov_list else ""
    fixed = f"{outcome} ~ {predictor}" + (f" + {cov_terms}" if cov_terms else "")

    result = None
    for method in ["lbfgs", "powell", "nm", "bfgs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = smf.mixedlm(fixed, data=data, groups=data["_family_id"]).fit(
                    reml=True, method=method)
                break
        except Exception:
            continue

    if result is None:
        print(f"    MLM error ({outcome} ~ {predictor}): all optimizers failed")
        return None

    beta = result.fe_params[predictor]
    se   = result.bse_fe.get(predictor, np.nan)
    if np.isnan(se):
        try:
            idx = list(result.fe_params.index).index(predictor)
            var = result.cov_params().iloc[idx, idx]
            se  = np.sqrt(abs(var)) if abs(var) > 0 else np.nan
        except Exception:
            se = np.nan
    z     = beta / se if not np.isnan(se) and se > 0 else np.nan
    p_two = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else np.nan
    p     = one_tailed_p(beta, p_two, expected_positive) if one_tailed else p_two
    row = {
        "predictor": predictor, "outcome": outcome,
        "n": int(result.nobs),
        "n_groups": result.ngroups if hasattr(result, "ngroups") else np.nan,
        "beta": beta, "se": se, "z": z, "p": p,
    }
    if one_tailed:
        row["p_two_tailed"] = p_two
    return row


def run_all(df, predictor, outcomes, covariates, one_tailed=False, expected_directions=None):
    if expected_directions is None:
        expected_directions = {}
    corr_rows, ols_rows, mlm_rows = [], [], []
    for out in outcomes:
        exp_pos = expected_directions.get(out, 1) > 0
        c = run_correlation(df, predictor, out, one_tailed=one_tailed, expected_positive=exp_pos)
        if c: corr_rows.append(c)
        o = run_ols(df, predictor, out, covariates, one_tailed=one_tailed, expected_positive=exp_pos)
        if o: ols_rows.append(o)
        m = run_mlm(df, predictor, out, covariates, one_tailed=one_tailed, expected_positive=exp_pos)
        if m: mlm_rows.append(m)
    return pd.DataFrame(corr_rows), pd.DataFrame(ols_rows), pd.DataFrame(mlm_rows)


def save_results(corr_df, ols_df, mlm_df, out_dir, label="",
                 predictors=None, outcomes=None, covariates=None, n=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_dir / "correlations.csv", index=False)
    ols_df.to_csv(out_dir  / "regressions.csv",  index=False)
    mlm_df.to_csv(out_dir  / "mlm.csv",          index=False)
    if predictors and outcomes and covariates:
        _n = n if n is not None else (int(corr_df["n"].max()) if len(corr_df) else 0)
        write_methods_note(out_dir, label, predictors, outcomes, covariates, _n)
    if label:
        print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
    for name, df in [("Correlations", corr_df), ("Regressions", ols_df), ("MLM", mlm_df)]:
        if df is None or len(df) == 0:
            continue
        sig = df[df["p"].apply(lambda x: isinstance(x, float) and x < 0.05)]
        print(f"\n  {name}  ({len(sig)}/{len(df)} p < .05)")
        for _, row in sig.iterrows():
            stat_key = "r" if "r" in row.index else "beta"
            print(f"    {row['predictor']:20s} -> {row['outcome']:20s} "
                  f"stat={row[stat_key]:+.3f}  p={row['p']:.4f}  n={int(row['n'])}")
    print(f"  Saved to: {out_dir}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 01 — Sensitivity Analyses")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    df["M2ID"] = df["M2ID"].astype(str)

    # All participants with neuroscience-visit age (C5PAGE); diary affect not required
    neuro           = df[df["C5PAGE"].notna()].copy()
    # Covariates for diary-inclusive analyses (e.g., diary PA/NA + neuro predictor)
    covs_neuro_diary = _get_covariates(neuro, include_diary_covs=True, include_time_p2_p5=True)
    # Covariates for neuro-visit-only analyses (no diary metrics)
    covs_neuro_visit = _get_covariates(neuro, include_diary_covs=False, include_time_p2_p5=False)
    print(f"  Neuro subsample N = {len(neuro)}")

    # -------------------------------------------------------------------------
    # 1. PANAS (convergent validity — C5PAGE, neuro subsample)
    # -------------------------------------------------------------------------
    # Older age → higher PANAS PA, lower PANAS NA (mirrors analysis 01 directions)
    panas_expected = {"C5SPGP": +1, "C5SPGN": -1, "C5SPGN_log": -1}

    print("\n--- Sensitivity: PANAS ---")
    panas_vars = [v for v in ["C5SPGP", "C5SPGN", "C5SPGN_log"] if v in neuro.columns]
    if panas_vars:
        corr, ols, mlm = run_all(neuro, "C5PAGE", panas_vars, covs_neuro_visit,
                                 one_tailed=True, expected_directions=panas_expected)
        save_results(corr, ols, mlm, BASE_DIR / "sensitivity_panas",
                     label="01 Age → PANAS  [neuro subsample, C5PAGE]",
                     predictors=["C5PAGE"], outcomes=panas_vars,
                     covariates=covs_neuro_visit, n=len(neuro))
    else:
        print("  C5SPGP / C5SPGN not found in data — skipping.")
        print(f"  Available columns matching 'C5SP': "
              f"{[c for c in neuro.columns if 'C5SP' in c]}")


if __name__ == "__main__":
    main()
