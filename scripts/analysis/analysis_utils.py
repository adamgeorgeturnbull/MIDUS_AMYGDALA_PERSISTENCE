#!/usr/bin/env python3
"""
analysis_utils.py

Shared utilities for MIDUS amygdala persistence analyses 01-05.
Handles data loading, sample definition, correlation, OLS regression,
MLM (random intercept for twin family), and results output.

Run from project root directory.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# ============================================================================
# Paths
# ============================================================================
PROCESSED_DIR = Path("data/processed")
FMRI_DIR = Path("data/fMRI")
RESULTS_DIR = Path("results/tables")

MASTER_FILE = PROCESSED_DIR / "midus_with_fmri.csv"
_LSS_FILE = FMRI_DIR / "all_subjects_betaSeries_LSS_all_conditions_M2ID.csv"
_LSA_FILE = FMRI_DIR / "all_subjects_betaSeries_all_conditions_M2ID.csv"
FC_FILE = _LSS_FILE if _LSS_FILE.exists() else _LSA_FILE

MIN_N = 20

DIARY_OUTCOMES = {"PA_score", "NA_score", "NA_score_log"}


# ============================================================================
# Statistical helpers
# ============================================================================
def fisher_z(r):
    """Fisher z-transform a correlation or array of correlations."""
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def one_tailed_p(stat, p_two, expected_positive):
    """
    Convert two-tailed p to one-tailed for a pre-specified directional hypothesis.

    If the observed effect is in the predicted direction: p_one = p_two / 2.
    If it is in the wrong direction:                      p_one = 1 - p_two / 2.
    """
    correct = (stat > 0) == expected_positive
    return p_two / 2 if correct else 1 - p_two / 2


# ============================================================================
# Data loading
# ============================================================================
def load_master(fc=False):
    """
    Load master behavioral dataset, optionally merging FC beta-series data.

    Parameters
    ----------
    fc : bool
        If True, merge the LSS (or LSA fallback) beta-series FC file.

    Returns
    -------
    pd.DataFrame
    """
    master = pd.read_csv(MASTER_FILE)
    master["M2ID"] = master["M2ID"].astype(str)

    if fc:
        fc_data = pd.read_csv(FC_FILE)
        fc_data["M2ID"] = fc_data["M2ID"].astype(str)
        master = master.merge(fc_data, on="M2ID", how="inner")
        print(f"  FC file: {FC_FILE.name}")

    return master


def prepare_persistence_vars(df):
    """
    Fisher z-transform all *_r_* persistence columns in-place.
    Returns list of z-transformed variable names that are present.
    """
    r_vars = [c for c in df.columns if "_persist_" in c and c.endswith(("_r_L", "_r_R",
              "_mean_r_L", "_mean_r_R", "_mean_r"))]
    z_vars = []
    for var in r_vars:
        z_var = var.replace("_mean_r", "_mean_z").replace("_r_L", "_z_L").replace("_r_R", "_z_R")
        if z_var not in df.columns:
            df[z_var] = fisher_z(df[var])
        z_vars.append(z_var)
    return z_vars


def get_samples(df, check_fc_col=None, behavioral=False, require_diary=True):
    """
    Return (full_sample, conservative_sample) DataFrames.

    When behavioral=True (purely behavioral analyses, e.g. Analysis 01):
      Full sample  = all participants with at least one affect measure.
      Conservative = same as full (no fMRI QC applies).

    When behavioral=False (fMRI analyses, default):
      Full sample  = participants with fMRI data (has_neg_persistence == 1)
                     and, if require_diary=True, at least one affect measure.
      Conservative = full sample restricted to qc_conservative == 1.

    Parameters
    ----------
    check_fc_col : str, optional
        If provided, also require this FC column to be non-null (for FC analyses).
    behavioral : bool
        If True, use the full behavioral sample rather than the fMRI subsample.
    require_diary : bool
        If False, do not require diary affect data (use for neuro-only analyses
        such as age → persistence). Default True.
    """
    has_affect = df[["PA_score", "NA_score"]].notna().any(axis=1)

    if behavioral:
        full = df[has_affect].copy()
        return full, full

    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1
    mask = has_persist & has_affect if require_diary else has_persist

    if check_fc_col and check_fc_col in df.columns:
        mask = mask & df[check_fc_col].notna()

    full = df[mask].copy()
    conservative = full[full.get("qc_conservative", pd.Series(0, index=full.index)) == 1].copy()
    return full, conservative


def get_covariates(df):
    """
    Return list of base covariate column names present in df.

    Diary-specific covariates (time_P2_P5, n_days_complete) are NOT included
    here — run_analysis_set adds them automatically for diary outcomes.
    """
    race_dummies = [c for c in df.columns if c.startswith("race_")]
    twin_dummies = [c for c in df.columns if c.startswith("twin_pair_")]
    covs = ["C5PAGE", "sex"] + race_dummies + twin_dummies
    return [c for c in covs if c in df.columns]


# ============================================================================
# Single-pair analysis functions
# ============================================================================
def run_correlation(df, predictor, outcome, one_tailed=False, expected_positive=True):
    """
    Pearson correlation between predictor and outcome.

    Returns a result dict or None if n < MIN_N.
    """
    data = df[[predictor, outcome]].dropna()
    n = len(data)
    if n < MIN_N:
        return None
    r, p_two = stats.pearsonr(data[predictor], data[outcome])
    p = one_tailed_p(r, p_two, expected_positive) if one_tailed else p_two
    return {
        "predictor": predictor,
        "outcome": outcome,
        "n": n,
        "r": r,
        "p": p,
        "p_two_tailed": p_two,
    }


def run_ols(df, predictor, outcome, covariates, one_tailed=False, expected_positive=True):
    """
    OLS regression: outcome ~ predictor + covariates.

    Returns a result dict or None if n < MIN_N.
    """
    cov_list = [c for c in covariates if c in df.columns and c != predictor]
    cols = [outcome, predictor] + cov_list
    data = df[cols].dropna()
    if len(data) < MIN_N:
        return None

    # Drop zero-variance covariates and singleton twin pair dummies (sum < 2
    # means at most one twin is in this sample — the dummy becomes a person-
    # specific indicator that absorbs that individual from the regression)
    cov_list = [c for c in cov_list
                if data[c].std() > 0
                and not (c.startswith("twin_pair_") and data[c].sum() < 2)]
    X = sm.add_constant(data[[predictor] + cov_list])
    y = data[outcome]

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"    OLS error ({outcome} ~ {predictor}): {e}")
        return None

    beta = model.params[predictor]
    p_two = model.pvalues[predictor]
    p = one_tailed_p(beta, p_two, expected_positive) if one_tailed else p_two

    return {
        "predictor": predictor,
        "outcome": outcome,
        "n": int(model.nobs),
        "df_resid": int(model.df_resid),
        "beta": beta,
        "se": model.bse[predictor],
        "t": model.tvalues[predictor],
        "p": p,
        "p_two_tailed": p_two,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }


def run_mlm(df, predictor, outcome, covariates, one_tailed=False, expected_positive=True):
    """
    Mixed-effects model: outcome ~ predictor + covariates + (1 | family).

    Family grouping uses M2FAMNUM; falls back to M2ID if unavailable.
    Tries multiple optimizers in order; returns None if all fail.
    Returns a result dict or None if n < MIN_N.
    """
    df = df.copy()

    # Family grouping: paired twins share a family ID, singletons are their own cluster
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

    # Sanitize predictor name for formula (hyphens not allowed)
    pred_safe = predictor.replace("-", "_").replace(".", "_")
    if pred_safe != predictor:
        df[pred_safe] = df[predictor]

    # Twin pair dummies are replaced by the random effect — exclude from fixed effects
    cov_list = [c for c in covariates
                if c in df.columns and c != predictor and c != pred_safe
                and not c.startswith("twin_pair_")]
    cols = [outcome, pred_safe, "_family_id"] + cov_list
    data = df[cols].dropna()
    if len(data) < MIN_N:
        return None

    # Drop zero-variance covariates to avoid rank deficiency / optimizer failures
    cov_list = [c for c in cov_list if data[c].std() > 0]
    cov_terms = " + ".join(c for c in cov_list if c in data.columns)
    fixed = f"{outcome} ~ {pred_safe}" + (f" + {cov_terms}" if cov_terms else "")

    result = None
    successful_method = None
    for method in ["lbfgs", "powell", "nm", "bfgs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = smf.mixedlm(fixed, data=data, groups=data["_family_id"])
                result = mdl.fit(reml=True, method=method)
                successful_method = method
                break
        except Exception:
            continue

    if result is None:
        print(f"    MLM error ({outcome} ~ {predictor}): all optimizers failed")
        return None

    beta = result.fe_params[pred_safe]

    # SE: handle degenerate random effect (near-zero RE variance → NaN SE)
    se = result.bse_fe.get(pred_safe, np.nan)
    degenerate = False
    if np.isnan(se):
        degenerate = True
        try:
            idx = list(result.fe_params.index).index(pred_safe)
            var = result.cov_params().iloc[idx, idx]
            se = np.sqrt(abs(var)) if abs(var) > 0 else np.nan
        except Exception:
            se = np.nan

    z = beta / se if not np.isnan(se) and se > 0 else np.nan
    if not np.isnan(z):
        p_two = float(2 * (1 - stats.norm.cdf(abs(z))))
        p = one_tailed_p(beta, p_two, expected_positive) if one_tailed else p_two
    else:
        p_two = p = np.nan

    n_groups = result.ngroups if hasattr(result, "ngroups") else np.nan

    re_var = np.nan
    try:
        re_var = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, "iloc") else float(result.cov_re)
    except Exception:
        pass

    return {
        "predictor": predictor,
        "outcome": outcome,
        "n": int(result.nobs),
        "n_groups": n_groups,
        "beta": beta,
        "se": se,
        "z": z,
        "p": p,
        "p_two_tailed": p_two,
        "re_var": re_var,
        "log_likelihood": result.llf,
        "converged": result.converged,
        "degenerate_re": degenerate,
        "optimizer": successful_method,
    }


# ============================================================================
# Multi-pair loop
# ============================================================================
def run_analysis_set(df, predictors, outcomes, base_covariates,
                     one_tailed=False, expected_directions=None):
    """
    Run correlations + OLS + MLM for all predictor × outcome combinations.

    Diary-specific covariates (time_P2_P5, n_days_complete) are automatically
    added when the outcome is in DIARY_OUTCOMES.

    Parameters
    ----------
    predictors : list of str
    outcomes : list of str
    base_covariates : list of str
    one_tailed : bool
    expected_directions : dict, optional
        Maps outcome name to +1 or -1. Defaults to +1 for all.

    Returns
    -------
    corr_df, ols_df, mlm_df : pd.DataFrame (may be empty)
    """
    if expected_directions is None:
        expected_directions = {}

    diary_covs = [c for c in ["time_P2_P5", "n_days_complete"] if c in df.columns]
    corr_rows, ols_rows, mlm_rows = [], [], []

    for pred in predictors:
        for out in outcomes:
            exp_pos = expected_directions.get(out, 1) > 0
            covs = base_covariates + (diary_covs if out in DIARY_OUTCOMES else [])

            c = run_correlation(df, pred, out, one_tailed=one_tailed, expected_positive=exp_pos)
            if c:
                corr_rows.append(c)

            o = run_ols(df, pred, out, covs, one_tailed=one_tailed, expected_positive=exp_pos)
            if o:
                ols_rows.append(o)

            m = run_mlm(df, pred, out, covs, one_tailed=one_tailed, expected_positive=exp_pos)
            if m:
                mlm_rows.append(m)

    return pd.DataFrame(corr_rows), pd.DataFrame(ols_rows), pd.DataFrame(mlm_rows)


# ============================================================================
# Variable labels (for methods notes)
# ============================================================================
VAR_LABELS = {
    # Age
    "C2PAGE":  "age at daily diary wave (P2)",
    "C5PAGE":  "age at neuroscience visit (P5)",
    # Affect — daily diary
    "PA_score":       "positive affect (daily diary composite, mean across days)",
    "NA_score":       "negative affect (daily diary composite, mean across days)",
    "NA_score_log":   "log-transformed negative affect (daily diary)",
    # Affect — PANAS (neuroscience visit)
    "C5SPGP":      "positive affect (PANAS, neuroscience visit)",
    "C5SPGN":      "negative affect (PANAS, neuroscience visit)",
    "C5SPGN_log":  "log-transformed negative affect (PANAS)",
    # Covariates
    "sex":             "biological sex (0 = male, 1 = female)",
    "n_days_complete": "number of completed daily diary days",
    "time_P2_P5":      "time between diary (P2) and neuroscience (P5) waves (years)",
    # Amygdala persistence (cross-run, Fisher z)
    "neg_persist_crossrun_mean_z_L": "left amygdala negative-affect persistence (cross-run spatial correlation, Fisher z)",
    "neg_persist_crossrun_mean_z_R": "right amygdala negative-affect persistence (cross-run spatial correlation, Fisher z)",
    "pos_persist_crossrun_mean_z_L": "left amygdala positive-affect persistence (cross-run spatial correlation, Fisher z)",
    "pos_persist_crossrun_mean_z_R": "right amygdala positive-affect persistence (cross-run spatial correlation, Fisher z)",
    "neg_persist_concat_z_L": "left amygdala negative-affect persistence (concatenated runs, Fisher z)",
    "neg_persist_concat_z_R": "right amygdala negative-affect persistence (concatenated runs, Fisher z)",
    # Functional connectivity (seed-based, LSS betas, Fisher z)
    "l_amyg-ant_vmPFC_neg":  "left amygdala – anterior vmPFC FC, negative condition (Fisher z)",
    "l_amyg-post_vmPFC_neg": "left amygdala – posterior vmPFC FC, negative condition (Fisher z)",
    "r_amyg-ant_vmPFC_neg":  "right amygdala – anterior vmPFC FC, negative condition (Fisher z)",
    "r_amyg-post_vmPFC_neg": "right amygdala – posterior vmPFC FC, negative condition (Fisher z)",
    "l_amyg-ant_vmPFC_neu":  "left amygdala – anterior vmPFC FC, neutral condition (Fisher z)",
    "l_amyg-post_vmPFC_neu": "left amygdala – posterior vmPFC FC, neutral condition (Fisher z)",
    "l_amyg-ant_vmPFC_pos":  "left amygdala – anterior vmPFC FC, positive condition (Fisher z)",
    "l_amyg-post_vmPFC_pos": "left amygdala – posterior vmPFC FC, positive condition (Fisher z)",
    "r_amyg-ant_vmPFC_neu":  "right amygdala – anterior vmPFC FC, neutral condition (Fisher z)",
    "r_amyg-post_vmPFC_neu": "right amygdala – posterior vmPFC FC, neutral condition (Fisher z)",
    "r_amyg-ant_vmPFC_pos":  "right amygdala – anterior vmPFC FC, positive condition (Fisher z)",
    "r_amyg-post_vmPFC_pos": "right amygdala – posterior vmPFC FC, positive condition (Fisher z)",
}


def _var_label(v):
    """Return human-readable label for a variable, falling back to the raw name."""
    if v in VAR_LABELS:
        return VAR_LABELS[v]
    if v.startswith("race_"):
        return f"race/ethnicity dummy: {v.replace('race_', '')}"
    if v.startswith("twin_pair_"):
        return f"twin pair dummy: {v.replace('twin_pair_', '')}"
    return v


def write_methods_note(out_dir, label, predictors, outcomes, covariates, n,
                       one_tailed=False, expected_directions=None, extra_notes=None):
    """
    Write a _methods.txt file describing the exact model specification.
    Intended to support methods-section writing.
    """
    import datetime
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    race_covs = [c for c in covariates if c.startswith("race_")]
    twin_covs = [c for c in covariates if c.startswith("twin_pair_")]
    base_covs = [c for c in covariates
                 if not c.startswith("race_") and not c.startswith("twin_pair_")]

    def _fmt_cov_list(covs, race, twin, mlm=False):
        parts = [_var_label(c) for c in covs]
        if race:
            parts.append(f"race/ethnicity ({len(race)} dummy variables, ref = White)")
        if twin and not mlm:
            parts.append(f"twin pair membership ({len(twin)} dummy variables; OLS only)")
        return parts

    L = []
    L.append("MODEL SPECIFICATION")
    L.append("=" * 60)
    L.append(f"Analysis  : {label}")
    L.append(f"N         : {n}")
    L.append(f"Generated : {datetime.date.today()}")
    L.append("")

    L.append("VARIABLES")
    L.append("-" * 60)
    L.append("Predictor(s):")
    for p in predictors:
        L.append(f"  {p:<45} {_var_label(p)}")
    L.append("Outcome(s):")
    for o in outcomes:
        L.append(f"  {o:<45} {_var_label(o)}")
    L.append("")

    L.append("STATISTICAL MODELS")
    L.append("-" * 60)

    L.append("1. Pearson correlation")
    L.append("   Zero-order association; no covariates.")
    L.append("")

    ols_covs = _fmt_cov_list(base_covs, race_covs, twin_covs, mlm=False)
    L.append("2. OLS regression")
    L.append("   Formula : outcome ~ predictor + covariates")
    L.append("   Covariates:")
    for c in ols_covs:
        L.append(f"     - {c}")
    L.append("")

    mlm_covs = _fmt_cov_list(base_covs, race_covs, [], mlm=True)
    L.append("3. Linear mixed-effects model (MLM)")
    L.append("   Fixed effects  : outcome ~ predictor + covariates")
    L.append("                    (twin pair dummies excluded; handled by random effect)")
    L.append("   Random effects : random intercept | family_id")
    L.append("   Family grouping: paired twins share family_id (SAMPLMAJ == 3 and")
    L.append("                    ≥2 members sharing M2FAMNUM); all others grouped by M2ID")
    L.append("   Estimation     : REML; optimizers tried in order: lbfgs, powell, nm, bfgs")
    L.append("   Fixed-effect covariates:")
    for c in mlm_covs:
        L.append(f"     - {c}")
    L.append("")

    L.append("INFERENCE")
    L.append("-" * 60)
    if one_tailed:
        L.append("Tests: one-tailed (pre-registered directional hypotheses)")
        L.append("  p_one = p_two / 2        if effect is in predicted direction")
        L.append("  p_one = 1 - p_two / 2    if effect is in opposite direction")
        if expected_directions:
            L.append("  Predicted directions:")
            for v, d in expected_directions.items():
                direction = "positive" if d > 0 else "negative"
                L.append(f"    {v}: {direction} ({_var_label(v)})")
    else:
        L.append("Tests: two-tailed")
    L.append("")

    if extra_notes:
        L.append("NOTES")
        L.append("-" * 60)
        L.append(extra_notes)
        L.append("")

    (out_dir / "_methods.txt").write_text("\n".join(L))


# ============================================================================
# Output
# ============================================================================
def save_results(corr_df, ols_df, mlm_df, out_dir, label="",
                 predictors=None, outcomes=None, covariates=None, n=None,
                 one_tailed=False, expected_directions=None, extra_notes=None):
    """Save correlations.csv, regressions.csv, mlm.csv, _methods.txt and print summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_df.to_csv(out_dir / "correlations.csv", index=False)
    ols_df.to_csv(out_dir / "regressions.csv", index=False)
    mlm_df.to_csv(out_dir / "mlm.csv", index=False)

    if predictors is not None and outcomes is not None and covariates is not None:
        _n = n if n is not None else (int(corr_df["n"].max()) if len(corr_df) else 0)
        write_methods_note(out_dir, label, predictors, outcomes, covariates, _n,
                           one_tailed=one_tailed, expected_directions=expected_directions,
                           extra_notes=extra_notes)

    _print_summary(corr_df, ols_df, mlm_df, label=label)
    print(f"  Saved to: {out_dir}")


def _print_summary(corr_df, ols_df, mlm_df, label=""):
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    for name, df in [("Correlations", corr_df), ("Regressions", ols_df), ("MLM", mlm_df)]:
        if df is None or len(df) == 0:
            continue
        sig = df[df["p"] < 0.05]
        print(f"\n  {name}  ({len(sig)}/{len(df)} p < .05)")
        for _, row in sig.iterrows():
            stat_key = "r" if "r" in row else "beta"
            p2 = f", p_two={row['p_two_tailed']:.3f}" if "p_two_tailed" in row and not np.isnan(row["p_two_tailed"]) else ""
            print(f"    {row['predictor']:45s} -> {row['outcome']:20s} "
                  f"stat={row[stat_key]:+.3f}  p={row['p']:.4f}{p2}  n={int(row['n'])}")
