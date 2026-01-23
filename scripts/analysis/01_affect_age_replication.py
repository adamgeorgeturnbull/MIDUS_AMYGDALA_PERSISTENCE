#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_affect_age_replication.py

Replication of age-related differences in affect in MIDUS 3.

Simplified: two samples
1) Daily diary full: daily affect, age C2PAGE
2) Neuro sample: daily affect (C2PAGE/C5PAGE), PANAS affect (C5PAGE)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from pathlib import Path

# =====================================================
# Paths
# =====================================================

DATA_FILE = Path("data/processed/midus_merged_clean.csv")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load data
# =====================================================

df = pd.read_csv(DATA_FILE)

# =====================================================
# Affect variables
# =====================================================

DAILY_AFFECT = ["PA_score", "NA_score", "NA_score_log"]
PANAS_AFFECT = ["C5SPGP", "C5SPGN"]

# =====================================================
# Samples
# =====================================================

SAMPLES = {
    "daily_diary_full": {
        "filter": df["PA_score"].notna(),
        "outcomes": DAILY_AFFECT,
        "age_vars": ["C2PAGE"]
    },
    "neuro_sample": {
        "filter": df["C5PAGE"].notna(),
        "outcomes": DAILY_AFFECT + PANAS_AFFECT,
        "age_vars": {  # dict to handle daily vs PANAS
            "daily": ["C2PAGE", "C5PAGE"],
            "panas": ["C5PAGE"]
        }
    }
}

# =====================================================
# Covariates
# =====================================================
RACE_COVARS = [c for c in df.columns if c.startswith("race_")]
TWIN_COVARS = [c for c in df.columns if c.startswith("twin_pair_")]
BASE_COVARS = ["sex", "educ"] + RACE_COVARS + TWIN_COVARS

df[BASE_COVARS] = df[BASE_COVARS].apply(pd.to_numeric, errors='coerce')

# =====================================================
# Helper functions
# =====================================================

def zero_order_corr(data, x, y):
    tmp = data[[x, y]].dropna()
    if len(tmp) < 10:
        return np.nan, np.nan, len(tmp)
    r, p = pearsonr(tmp[x], tmp[y])
    return r, p, len(tmp)

def run_regression(data, outcome, age_var, covars):
    cols = [outcome, age_var] + covars
    tmp = data[cols].dropna()
    if len(tmp) < 20:
        return None

    formula = f"{outcome} ~ {age_var} + " + " + ".join(covars)
    return smf.ols(formula, data=tmp).fit()

# =====================================================
# Analyses
# =====================================================

corr_rows = []
reg_rows = []

# --- Daily diary full ---
sdf = df.loc[SAMPLES["daily_diary_full"]["filter"]].copy()
for outcome in SAMPLES["daily_diary_full"]["outcomes"]:
    for age_var in SAMPLES["daily_diary_full"]["age_vars"]:
        # Correlation
        r, p, n = zero_order_corr(sdf, age_var, outcome)
        corr_rows.append({
            "sample": "daily_diary_full",
            "outcome": outcome,
            "age_var": age_var,
            "r": r,
            "p": p,
            "n": n
        })
        # Regression
        model = run_regression(sdf, outcome, age_var, BASE_COVARS)
        if model is not None:
            reg_rows.append({
                "sample": "daily_diary_full",
                "outcome": outcome,
                "age_var": age_var,
                "beta_age": model.params.get(age_var, np.nan),
                "se_age": model.bse.get(age_var, np.nan),
                "p_age": model.pvalues.get(age_var, np.nan),
                "n": int(model.nobs),
                "r2": model.rsquared
            })

# --- Neuro sample ---
sdf = df.loc[SAMPLES["neuro_sample"]["filter"]].copy()
for outcome in SAMPLES["neuro_sample"]["outcomes"]:
    # Determine appropriate age variable(s)
    if outcome in PANAS_AFFECT:
        age_vars = SAMPLES["neuro_sample"]["age_vars"]["panas"]
    else:
        age_vars = SAMPLES["neuro_sample"]["age_vars"]["daily"]

    for age_var in age_vars:
        # Correlation
        r, p, n = zero_order_corr(sdf, age_var, outcome)
        corr_rows.append({
            "sample": "neuro_sample",
            "outcome": outcome,
            "age_var": age_var,
            "r": r,
            "p": p,
            "n": n
        })
        # Regression
        model = run_regression(sdf, outcome, age_var, BASE_COVARS)
        if model is not None:
            reg_rows.append({
                "sample": "neuro_sample",
                "outcome": outcome,
                "age_var": age_var,
                "beta_age": model.params.get(age_var, np.nan),
                "se_age": model.bse.get(age_var, np.nan),
                "p_age": model.pvalues.get(age_var, np.nan),
                "n": int(model.nobs),
                "r2": model.rsquared
            })

# =====================================================
# Save outputs
# =====================================================

pd.DataFrame(corr_rows).to_csv(
    OUT_DIR / "01_affect_age_correlations.csv",
    index=False
)

pd.DataFrame(reg_rows).to_csv(
    OUT_DIR / "01_affect_age_regressions.csv",
    index=False
)

print("01_affect_age_replication complete.")
