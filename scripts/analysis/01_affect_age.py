#!/usr/bin/env python3
"""
01_affect_age.py

Replication: age-related differences in daily life affect.

Two samples:
  1. Full daily diary sample (primary):
       Predictor  : C2PAGE (age at diary wave / P2)
       Outcomes   : PA_score, NA_score
       Sample     : everyone with at least one daily affect measure

  2. Neuroscience subsample (saved to neuro_sample/):
       Predictor  : C5PAGE (age at MRI visit / P5)
       Outcomes   : PA_score, NA_score
       Sample     : participants with fMRI data (has_neg_persistence == 1)

Methods    : Pearson correlation, OLS regression, MLM (random intercept for family)
Tests      : Two-tailed

Sensitivity analyses (PANAS, log-transforms) are in 01_sensitivity.py.

Run from project root directory.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import write_methods_note, one_tailed_p
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from confidence_intervals import coefficient_ci, pearson_ci

# ============================================================================
# Paths
# ============================================================================
DATA_FILE   = Path("data/processed/midus_merged_clean.csv")
RESULTS_DIR = Path("results/tables/01_affect_age")
MIN_N       = 20

OUTCOMES = ["PA_score", "NA_score", "NA_score_log"]

# Older age → higher PA, lower NA
EXPECTED_DIRECTIONS = {"PA_score": +1, "NA_score": -1, "NA_score_log": -1}


# ============================================================================
# Helpers
# ============================================================================
def _get_covariates(df, include_time_p2_p5=False):
    """
    Return covariate list for OLS (includes twin pair dummies).
    time_P2_P5 is MRI-specific and must only be included for the neuro subsample.
    """
    race_dummies = [c for c in df.columns if c.startswith("race_")]
    twin_dummies = [c for c in df.columns if c.startswith("twin_pair_")]
    covs = ["sex"] + race_dummies + twin_dummies + ["n_days_complete"]
    if include_time_p2_p5:
        covs += ["time_P2_P5"]
    return [c for c in covs if c in df.columns]


def _create_family_id(df):
    """
    Assign family_id for MLM grouping.
    Paired twins (SAMPLMAJ == 3, sharing M2FAMNUM) get a shared family ID.
    Everyone else is their own cluster.
    """
    df = df.copy()
    if "SAMPLMAJ" in df.columns and "M2FAMNUM" in df.columns:
        is_twin = df["SAMPLMAJ"] == 3
        paired = df.loc[is_twin, "M2FAMNUM"].value_counts()
        paired = paired[paired > 1].index
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
    corr_result = stats.pearsonr(data[predictor], data[outcome])
    r, p_two = corr_result
    p = one_tailed_p(r, p_two, expected_positive) if one_tailed else p_two
    row = {
        **pearson_ci(corr_result),
        "predictor": predictor, "outcome": outcome, "n": len(data), "r": r, "p": p,
    }
    if one_tailed:
        row["p_two_tailed"] = p_two
    return row


def run_ols(df, predictor, outcome, covariates, one_tailed=False, expected_positive=True):
    cov_list = [c for c in covariates if c in df.columns and c != predictor]
    cols = [outcome, predictor] + cov_list
    data = df[cols].dropna()
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
        **coefficient_ci(model, predictor),
        "predictor":     predictor,
        "outcome":       outcome,
        "n":             int(model.nobs),
        "df_resid":      int(model.df_resid),
        "beta":          beta,
        "se":            model.bse[predictor],
        "t":             model.tvalues[predictor],
        "p":             p,
        "r_squared":     model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }
    if one_tailed:
        row["p_two_tailed"] = p_two
    return row


def run_mlm(df, predictor, outcome, covariates, one_tailed=False, expected_positive=True):
    df = _create_family_id(df)

    # Twin pair dummies are replaced by the random effect — exclude from fixed effects
    cov_list = [c for c in covariates
                if c in df.columns and c != predictor and not c.startswith("twin_pair_")]
    cols = [outcome, predictor, "_family_id"] + cov_list
    data = df[cols].dropna()
    if len(data) < MIN_N:
        return None

    # Drop zero-variance covariates (can cause rank deficiency / optimizer failures)
    cov_list = [c for c in cov_list if data[c].std() > 0]
    cov_terms = " + ".join(cov_list) if cov_list else ""
    fixed = f"{outcome} ~ {predictor}" + (f" + {cov_terms}" if cov_terms else "")

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
        **coefficient_ci(result, predictor),
        "predictor": predictor,
        "outcome":   outcome,
        "n":         int(result.nobs),
        "n_groups":  result.ngroups if hasattr(result, "ngroups") else np.nan,
        "beta":      beta,
        "se":        se,
        "z":         z,
        "p":         p,
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
        if c:
            corr_rows.append(c)
        o = run_ols(df, predictor, out, covariates, one_tailed=one_tailed, expected_positive=exp_pos)
        if o:
            ols_rows.append(o)
        m = run_mlm(df, predictor, out, covariates, one_tailed=one_tailed, expected_positive=exp_pos)
        if m:
            mlm_rows.append(m)
    return pd.DataFrame(corr_rows), pd.DataFrame(ols_rows), pd.DataFrame(mlm_rows)


def save_results(corr_df, ols_df, mlm_df, out_dir, label=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_dir / "correlations.csv", index=False)
    ols_df.to_csv(out_dir / "regressions.csv", index=False)
    mlm_df.to_csv(out_dir / "mlm.csv", index=False)

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
    print("Analysis 01: Age → Affect  (Replication)")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)
    df["M2ID"] = df["M2ID"].astype(str)

    # -------------------------------------------------------------------------
    # Sample 1: Full daily diary sample — C2PAGE
    # -------------------------------------------------------------------------
    has_affect = df[["PA_score", "NA_score"]].notna().any(axis=1)
    diary_full = df[has_affect].copy()
    print(f"  Full diary N = {len(diary_full)}")

    covs_diary = _get_covariates(diary_full)  # no time_P2_P5 — MRI-specific, would drop non-MRI
    corr, ols, mlm = run_all(diary_full, "C2PAGE", OUTCOMES, covs_diary,
                             one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr, ols, mlm, RESULTS_DIR,
                 label="01 Age → Affect  [full diary sample, C2PAGE]")
    write_methods_note(RESULTS_DIR, "01 Age → Affect  [full diary sample, C2PAGE]",
                       ["C2PAGE"], OUTCOMES, covs_diary, n=len(diary_full),
                       one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # -------------------------------------------------------------------------
    # Sample 2: Neuroscience subsample — C5PAGE
    # -------------------------------------------------------------------------
    neuro = df[df["C5PAGE"].notna() & has_affect].copy()
    print(f"  Neuro subsample N = {len(neuro)}")

    covs_neuro = _get_covariates(neuro, include_time_p2_p5=True)
    corr_n, ols_n, mlm_n = run_all(neuro, "C5PAGE", OUTCOMES, covs_neuro,
                                    one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)
    save_results(corr_n, ols_n, mlm_n, RESULTS_DIR / "neuro_sample",
                 label="01 Age → Affect  [neuro subsample, C5PAGE]")
    write_methods_note(RESULTS_DIR / "neuro_sample",
                       "01 Age → Affect  [neuro subsample, C5PAGE]",
                       ["C5PAGE"], OUTCOMES, covs_neuro, n=len(neuro),
                       one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
