#!/usr/bin/env python3
"""
00a_diary_panas_convergence.py

Convergent validity: daily diary affect (P2) vs PANAS affect (P5 neuroimaging visit).

Tests matched construct pairs in participants who completed both assessments:
  PA_score      → C5SPGP     (positive affect, diary vs PANAS)
  NA_score      → C5SPGN     (negative affect, diary vs PANAS)
  NA_score_log  → C5SPGN_log (log negative affect, diary vs PANAS)

Sample    : participants with diary affect AND PANAS scores (overlap ~n=137)
Covariates: sex, race dummies, twin pair dummies, n_days_complete, time_P2_P5
Methods   : Pearson correlation, OLS regression, MLM (random intercept for family)
Tests     : Two-tailed

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

from confidence_intervals import coefficient_ci, pearson_ci

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import write_methods_note

# ============================================================================
# Paths / constants
# ============================================================================
DATA_FILE   = Path("data/processed/midus_merged_clean.csv")
RESULTS_DIR = Path("results/tables/00a_diary_panas")
MIN_N       = 20

# Matched construct pairs: (diary predictor, PANAS outcome)
PAIRS = [
    ("PA_score",     "C5SPGP"),
    ("NA_score",     "C5SPGN"),
    ("NA_score_log", "C5SPGN_log"),
]


# ============================================================================
# Helpers  (standalone copies — mirrors 01_affect_age.py)
# ============================================================================
def _get_covariates(df):
    """
    Covariates for cross-wave analyses (diary predictor, PANAS outcome).
    Includes n_days_complete (diary quality) and time_P2_P5 (wave separation).
    """
    race_dummies = [c for c in df.columns if c.startswith("race_")]
    twin_dummies = [c for c in df.columns if c.startswith("twin_pair_")]
    covs = ["sex"] + race_dummies + twin_dummies + ["n_days_complete", "time_P2_P5"]
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


def run_correlation(df, predictor, outcome):
    data = df[[predictor, outcome]].dropna()
    if len(data) < MIN_N:
        return None
    corr_result = stats.pearsonr(data[predictor], data[outcome])
    r, p = corr_result
    return {
        **pearson_ci(corr_result),
        "predictor": predictor, "outcome": outcome, "n": len(data), "r": r, "p": p,
    }


def run_ols(df, predictor, outcome, covariates):
    cov_list = [c for c in covariates if c in df.columns and c != predictor]
    data = df[[outcome, predictor] + cov_list].dropna()
    if len(data) < MIN_N:
        return None
    X = sm.add_constant(data[[predictor] + cov_list])
    try:
        model = sm.OLS(data[outcome], X).fit()
    except Exception as e:
        print(f"    OLS error ({outcome} ~ {predictor}): {e}")
        return None
    return {
        **coefficient_ci(model, predictor),
        "predictor":     predictor,
        "outcome":       outcome,
        "n":             int(model.nobs),
        "beta":          model.params[predictor],
        "se":            model.bse[predictor],
        "t":             model.tvalues[predictor],
        "p":             model.pvalues[predictor],
        "r_squared":     model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }


def run_mlm(df, predictor, outcome, covariates):
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
    return {
        **coefficient_ci(result, predictor),
        "predictor": predictor,
        "outcome":   outcome,
        "n":         int(result.nobs),
        "n_groups":  result.ngroups if hasattr(result, "ngroups") else np.nan,
        "beta":      beta,
        "se":        se,
        "z":         z,
        "p":         p_two,
    }


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
    print("Analysis 00a: Diary Affect ↔ PANAS Convergent Validity")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    df["M2ID"] = df["M2ID"].astype(str)

    # Participants with both diary affect and PANAS
    diary_vars = ["PA_score", "NA_score", "NA_score_log"]
    panas_vars = ["C5SPGP", "C5SPGN", "C5SPGN_log"]
    has_diary  = df[["PA_score", "NA_score"]].notna().any(axis=1)
    has_panas  = df[["C5SPGP", "C5SPGN"]].notna().any(axis=1)
    sample     = df[has_diary & has_panas].copy()
    print(f"  N (diary + PANAS overlap) = {len(sample)}")

    covariates = _get_covariates(sample)
    print(f"  Covariates: {covariates[:8]}{'...' if len(covariates) > 8 else ''}")

    # Check which PANAS variables are available
    missing = [v for v in panas_vars if v not in sample.columns]
    if missing:
        print(f"  WARNING: missing PANAS variables: {missing}")
    pairs = [(pred, out) for pred, out in PAIRS if out in sample.columns]

    # -------------------------------------------------------------------------
    # Run matched pairs
    # -------------------------------------------------------------------------
    corr_rows, ols_rows, mlm_rows = [], [], []
    for pred, out in pairs:
        c = run_correlation(sample, pred, out)
        if c: corr_rows.append(c)
        o = run_ols(sample, pred, out, covariates)
        if o: ols_rows.append(o)
        m = run_mlm(sample, pred, out, covariates)
        if m: mlm_rows.append(m)

    corr_df = pd.DataFrame(corr_rows)
    ols_df  = pd.DataFrame(ols_rows)
    mlm_df  = pd.DataFrame(mlm_rows)

    predictors_used = [p for p, _ in pairs]
    outcomes_used   = [o for _, o in pairs]

    save_results(corr_df, ols_df, mlm_df, RESULTS_DIR,
                 label="00a Diary Affect → PANAS  [diary+neuro overlap]",
                 predictors=predictors_used, outcomes=outcomes_used,
                 covariates=covariates, n=len(sample))


if __name__ == "__main__":
    main()
