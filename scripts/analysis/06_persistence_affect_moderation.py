#!/usr/bin/env python3
"""
06_persistence_affect_moderation.py

Does emotion regulation strategy moderate the relationship between amygdala
negative-affect persistence and daily life affect?

Model (run separately for each moderator):
  affect ~ persistence_c + moderator_c + persistence_c × moderator_c + covariates

Both persistence and moderator are mean-centered within each complete-case
subset before forming the interaction term.

Predictor  : neg_persist_crossrun_mean_z_L
Moderators : C5SER (ERQ reappraisal), C5SES (ERQ suppression)
Outcomes   : PA_score, NA_score, NA_score_log  (daily diary; N~80)
Methods    : OLS, MLM (random intercept for family)
Tests      : Two-tailed throughout

Full sample results saved to full_sample/.
Sensitivity analyses (PANAS, right hemisphere) in 06_sensitivity.py.

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
from analysis_utils import (
    RESULTS_DIR, load_master, get_samples, get_covariates,
    prepare_persistence_vars, DIARY_OUTCOMES,
)

# ============================================================================
# Configuration
# ============================================================================
OUT_DIR    = RESULTS_DIR / "06_persistence_affect_moderation"
PREDICTOR  = "neg_persist_crossrun_mean_z_L"
MODERATORS = [("C5SER", "reappraisal"), ("C5SES", "suppression")]
OUTCOMES   = ["PA_score", "NA_score", "NA_score_log"]
MIN_N      = 20
DIARY_COVS = ["time_P2_P5", "n_days_complete"]


# ============================================================================
# Moderation helpers
# ============================================================================
def _covariates_for(df, outcome, base_covs):
    covs = list(base_covs)
    if outcome in DIARY_OUTCOMES:
        covs += [c for c in DIARY_COVS if c in df.columns]
    return covs


def _mlm_term_stats(result, term):
    """Extract beta, se, two-tailed p for a fixed-effect term from MLM result."""
    beta = result.fe_params.get(term, np.nan)
    se   = result.bse_fe.get(term, np.nan)
    if np.isnan(se) or se == 0:
        return beta, np.nan, np.nan
    z = beta / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return beta, se, p


def run_moderation_ols(df, predictor, outcome, moderator, base_covs):
    covariates = _covariates_for(df, outcome, base_covs)
    cols = [outcome, predictor, moderator] + covariates
    data = df[cols].dropna()
    if len(data) < MIN_N:
        return None

    pred_c  = data[predictor] - data[predictor].mean()
    mod_c   = data[moderator] - data[moderator].mean()
    int_col = f"{predictor}_x_{moderator}"

    cov_cols = {c: data[c] for c in covariates
                if c in data.columns
                and data[c].std() > 0
                and not (c.startswith("twin_pair_") and data[c].sum() < 2)}
    X = pd.concat([
        pd.DataFrame({predictor: pred_c, moderator: mod_c, int_col: pred_c * mod_c},
                     index=data.index),
        pd.DataFrame(cov_cols, index=data.index),
    ], axis=1)
    X = sm.add_constant(X)
    y = data[outcome]

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"    OLS error ({outcome} ~ {predictor}*{moderator}): {e}")
        return None

    return {
        "predictor":        predictor,
        "moderator":        moderator,
        "outcome":          outcome,
        "n":                int(model.nobs),
        "beta_interaction": model.params[int_col],
        "se_interaction":   model.bse[int_col],
        "t_interaction":    model.tvalues[int_col],
        "p_interaction":    model.pvalues[int_col],
        "beta_predictor":   model.params[predictor],
        "p_predictor":      model.pvalues[predictor],
        "beta_moderator":   model.params[moderator],
        "p_moderator":      model.pvalues[moderator],
        "r_squared":        model.rsquared,
        "adj_r_squared":    model.rsquared_adj,
    }


def run_moderation_mlm(df, predictor, outcome, moderator, base_covs):
    df = df.copy()

    # Family grouping (mirrors analysis_utils.run_mlm)
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

    covariates = _covariates_for(df, outcome, base_covs)
    # Twin pair dummies replaced by random effect — exclude from MLM fixed effects
    cov_list = [c for c in covariates
                if c in df.columns and not c.startswith("twin_pair_")]

    cols = [outcome, predictor, moderator, "_family_id"] + cov_list
    data = df[cols].dropna()
    if len(data) < MIN_N:
        return None

    pred_safe = predictor.replace("-", "_").replace(".", "_") + "_c"
    mod_safe  = moderator + "_c"
    int_safe  = pred_safe + "_x_" + mod_safe

    data = data.copy()
    data[pred_safe] = (data[predictor] - data[predictor].mean()).values
    data[mod_safe]  = (data[moderator]  - data[moderator].mean()).values
    data[int_safe]  = (data[pred_safe]  * data[mod_safe]).values

    cov_list = [c for c in cov_list if data[c].std() > 0]
    cov_terms = (" + " + " + ".join(cov_list)) if cov_list else ""
    fixed = f"{outcome} ~ {pred_safe} + {mod_safe} + {int_safe}{cov_terms}"

    result = None
    best_method = None
    for method in ["lbfgs", "powell", "nm", "bfgs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = smf.mixedlm(fixed, data=data, groups=data["_family_id"])
                result = mdl.fit(reml=True, method=method)
                best_method = method
                break
        except Exception:
            continue

    if result is None:
        print(f"    MLM error ({outcome} ~ {predictor}*{moderator}): all optimizers failed")
        return None

    beta_int,  se_int,  p_int  = _mlm_term_stats(result, int_safe)
    beta_pred, _,       p_pred = _mlm_term_stats(result, pred_safe)
    beta_mod,  _,       p_mod  = _mlm_term_stats(result, mod_safe)

    re_var = np.nan
    try:
        re_var = float(result.cov_re.iloc[0, 0])
    except Exception:
        pass

    return {
        "predictor":        predictor,
        "moderator":        moderator,
        "outcome":          outcome,
        "n":                int(result.nobs),
        "n_groups":         result.ngroups if hasattr(result, "ngroups") else np.nan,
        "beta_interaction": beta_int,
        "se_interaction":   se_int,
        "p_interaction":    p_int,
        "beta_predictor":   beta_pred,
        "p_predictor":      p_pred,
        "beta_moderator":   beta_mod,
        "p_moderator":      p_mod,
        "re_var":           re_var,
        "log_likelihood":   result.llf,
        "converged":        result.converged,
        "optimizer":        best_method,
    }


def run_moderation_set(df, predictor, outcomes, moderator, base_covs):
    ols_rows, mlm_rows = [], []
    for outcome in outcomes:
        o = run_moderation_ols(df, predictor, outcome, moderator, base_covs)
        if o:
            ols_rows.append(o)
        m = run_moderation_mlm(df, predictor, outcome, moderator, base_covs)
        if m:
            mlm_rows.append(m)
    return pd.DataFrame(ols_rows), pd.DataFrame(mlm_rows)


def save_moderation(ols_df, mlm_df, out_dir, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ols_df.to_csv(out_dir / "moderation_ols.csv", index=False)
    mlm_df.to_csv(out_dir / "moderation_mlm.csv", index=False)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    for name, res in [("OLS", ols_df), ("MLM", mlm_df)]:
        if res is None or len(res) == 0:
            continue
        print(f"  {name}:")
        for _, row in res.iterrows():
            sig = ("***" if row["p_interaction"] < 0.001
                   else "**" if row["p_interaction"] < 0.01
                   else "*"  if row["p_interaction"] < 0.05
                   else "")
            print(f"    {row['predictor']} × {row['moderator']} → {row['outcome']:20s}: "
                  f"b={row['beta_interaction']:7.4f}, p={row['p_interaction']:.3f}{sig}"
                  f"  (n={int(row['n'])})")
    print(f"  Saved to: {out_dir}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 06: Persistence × ERQ → Daily Affect  (two-tailed)")
    print("=" * 70)

    df = load_master(fc=False)
    prepare_persistence_vars(df)
    full, cons = get_samples(df)
    print(f"  Full N = {len(full)} | Conservative N = {len(cons)}")

    for mod_var, mod_name in MODERATORS:
        print(f"\n{'─' * 70}")
        print(f"  Moderator: {mod_name} ({mod_var})")

        # -- Conservative (primary) --
        base_covs = get_covariates(cons)
        ols_df, mlm_df = run_moderation_set(cons, PREDICTOR, OUTCOMES, mod_var, base_covs)
        save_moderation(ols_df, mlm_df,
                        OUT_DIR / mod_name,
                        f"06 Persistence × {mod_name} → Daily Affect  [conservative, two-tailed]")

        # -- Full sample (archive) --
        base_covs_full = get_covariates(full)
        ols_f, mlm_f = run_moderation_set(full, PREDICTOR, OUTCOMES, mod_var, base_covs_full)
        save_moderation(ols_f, mlm_f,
                        OUT_DIR / mod_name / "full_sample",
                        f"06 Persistence × {mod_name} → Daily Affect  [full sample, two-tailed]")


if __name__ == "__main__":
    main()
