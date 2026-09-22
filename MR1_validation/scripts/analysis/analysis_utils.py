#!/usr/bin/env python3
"""
analysis_utils.py (MR1)

Shared utilities for MR1 primary analyses (02–05).

MR1 vs M3 differences:
- Key: MIDUSID (not M2ID)
- Age: RA5PAGE (not C5PAGE)
- Sex: RA1PRSEX → harmonized to 'sex' during preprocessing
- Race dummies: race_2 through race_6 (constructed in 04_construct_covariates.py)
- PANAS: RA5SPGP, RA5SPGN (sensitivity instrument)
- ERQ: RA5SER, RA5SES
- Primary adjusted model: OLS (the available MR1 P5 analytic inputs contain no
  usable family, household, sibling, twin-pair, or other grouping identifier,
  so participant-level OLS is used)
- Master file: mr1_with_fmri.csv

Run from MR1_validation/ directory.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Share the data-free CI helpers with M3; retain MR1 analysis_utils precedence.
sys.path.append(str(Path(__file__).resolve().parents[3] / "scripts" / "analysis"))
from confidence_intervals import coefficient_ci, pearson_ci

# ============================================================================
# Paths and constants
# ============================================================================
PROCESSED_DIR = Path("data/processed")
FMRI_DIR      = Path("data/fMRI")
RESULTS_DIR   = Path("results/tables")

MASTER_FILE = PROCESSED_DIR / "mr1_with_fmri.csv"
FC_FILE     = FMRI_DIR / "all_subjects_betaSeries_LSS_all_conditions_M2ID.csv"

MIN_N = 20
DIARY_OUTCOMES = {"PA_score", "NA_score", "NA_score_log"}

# Exact ordered base covariate set for MR1 adjusted models.
BASE_COVARIATES = ["RA5PAGE", "sex", "race_2", "race_3", "race_4", "race_5", "race_6"]


# ============================================================================
# Statistical helpers
# ============================================================================
def fisher_z(r):
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def one_tailed_p(stat, p_two, expected_positive):
    correct = (stat > 0) == expected_positive
    return p_two / 2 if correct else 1 - p_two / 2


# ============================================================================
# Validation helpers
# ============================================================================
def _require_columns(df, required, label):
    """Abort with aggregate message if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: {label} is missing required column(s): {sorted(missing)}")
        sys.exit(1)


def _require_midusid_nonmissing(df, label):
    """Abort if MIDUSID has missing values."""
    n_missing = int(df["MIDUSID"].isna().sum())
    if n_missing > 0:
        print(f"ERROR: {label}: {n_missing} missing MIDUSID value(s).")
        sys.exit(1)


def _normalize_midusid(series, label):
    """
    Normalize MIDUSID to canonical integer strings.

    Accepts numeric or numeric-string inputs.  Aborts with aggregate-only
    messages if any nonmissing value is unparseable, non-finite, or
    non-integer.  Returns a Series of dtype object (str for valid IDs, NaN
    for originally-missing inputs).  Never prints individual identifiers.
    """
    raw_notna_n = int(series.notna().sum())
    numeric     = pd.to_numeric(series, errors="coerce")

    n_unparseable = raw_notna_n - int(numeric.notna().sum())
    if n_unparseable > 0:
        print(f"ERROR: {label}: {n_unparseable} MIDUSID value(s) could not be "
              f"converted to numeric.")
        sys.exit(1)

    valid = numeric.dropna()

    n_nonfinite = int((~np.isfinite(valid)).sum())
    if n_nonfinite > 0:
        print(f"ERROR: {label}: {n_nonfinite} MIDUSID value(s) are non-finite "
              f"(inf/-inf).")
        sys.exit(1)

    n_non_int = int((valid % 1 != 0).sum())
    if n_non_int > 0:
        print(f"ERROR: {label}: {n_non_int} MIDUSID value(s) are not integers.")
        sys.exit(1)

    result = pd.Series(np.nan, index=series.index, dtype=object)
    notna_mask = numeric.notna()
    result[notna_mask] = numeric[notna_mask].astype(int).astype(str)
    return result


def _require_midusid_unique(df, label):
    """Abort if MIDUSID has duplicates."""
    n_dup = int(df["MIDUSID"].duplicated().sum())
    if n_dup > 0:
        print(f"ERROR: {label}: {n_dup} duplicate MIDUSID value(s).")
        sys.exit(1)


# ============================================================================
# Data loading
# ============================================================================
def load_master(fc=False):
    """
    Load master behavioral/fMRI dataset, optionally inner-merging FC data.

    Parameters
    ----------
    fc : bool
        If True, inner-merge the LSS beta-series FC file on MIDUSID.

    Returns
    -------
    pd.DataFrame
    """
    if not MASTER_FILE.exists():
        print(f"ERROR: master file not found: {MASTER_FILE}")
        sys.exit(1)

    master = pd.read_csv(MASTER_FILE)
    _require_columns(master, ["MIDUSID"], "master file")
    _require_midusid_nonmissing(master, "master file")
    master["MIDUSID"] = _normalize_midusid(master["MIDUSID"], "master file")
    _require_midusid_unique(master, "master file")

    if fc:
        if not FC_FILE.exists():
            print(f"ERROR: FC file not found: {FC_FILE}")
            sys.exit(1)

        fc_data = pd.read_csv(FC_FILE)
        _require_columns(fc_data, ["MIDUSID"], f"FC file ({FC_FILE.name})")
        _require_midusid_nonmissing(fc_data, f"FC file ({FC_FILE.name})")
        fc_data["MIDUSID"] = _normalize_midusid(
            fc_data["MIDUSID"], f"FC file ({FC_FILE.name})"
        )
        _require_midusid_unique(fc_data, f"FC file ({FC_FILE.name})")

        # Reject overlapping non-key column names to prevent _x/_y suffix columns.
        master_cols = set(master.columns) - {"MIDUSID"}
        fc_cols     = set(fc_data.columns) - {"MIDUSID"}
        overlap     = master_cols & fc_cols
        if overlap:
            print(
                f"ERROR: {len(overlap)} non-key column(s) appear in both master and "
                f"FC file; resolve before merging: {sorted(overlap)}"
            )
            sys.exit(1)

        expected_n = len(set(master["MIDUSID"]) & set(fc_data["MIDUSID"]))
        merged = master.merge(fc_data, on="MIDUSID", how="inner", validate="one_to_one")
        if len(merged) != expected_n:
            print(
                f"ERROR: FC merge row count ({len(merged)}) != MIDUSID intersection "
                f"size ({expected_n})."
            )
            sys.exit(1)

        print(f"  FC file: {FC_FILE.name}")
        print(f"  After FC merge: {len(merged)} participants (intersection)")
        master = merged

    return master


# ============================================================================
# Persistence variable preparation
# ============================================================================
def prepare_persistence_vars(df):
    """
    Fisher z-transform MR1 persistence correlation columns in-place.

    Recognized naming convention:
      Any column containing "_persist_" and ending in _r_L, _r_R,
      _mean_r_L, _mean_r_R, or _mean_r:
          neg_persist_crossrun_mean_r_L  →  neg_persist_crossrun_mean_z_L
          pos_persist_crossrun_mean_r_R  →  pos_persist_crossrun_mean_z_R

    Raw *_r columns are preserved.  A z column is created only when not
    already present.

    Before transforming each column, nonmissing values are validated as
    numeric, finite, and within [-1, 1].  Aborts with the column name and
    aggregate problem count; never prints individual values.

    Returns
    -------
    list of str
        Names of the z-transformed columns.
    """
    r_vars = [
        c for c in df.columns
        if "_persist_" in c and c.endswith(
            ("_r_L", "_r_R", "_mean_r_L", "_mean_r_R", "_mean_r")
        )
    ]

    z_vars = []
    for var in r_vars:
        # Full-length numeric conversion for the transform; df[var] is never modified.
        numeric_ser = pd.to_numeric(df[var], errors="coerce")
        valid = df[var].dropna()
        if len(valid) > 0:
            valid_num = pd.to_numeric(valid, errors="coerce")
            n_non_numeric = int(valid_num.isna().sum())
            if n_non_numeric > 0:
                print(f"ERROR: {var}: {n_non_numeric} nonmissing value(s) are not numeric.")
                sys.exit(1)
            n_non_finite = int((~np.isfinite(valid_num)).sum())
            if n_non_finite > 0:
                print(f"ERROR: {var}: {n_non_finite} nonmissing value(s) are non-finite.")
                sys.exit(1)
            n_out_of_range = int(((valid_num < -1) | (valid_num > 1)).sum())
            if n_out_of_range > 0:
                print(
                    f"ERROR: {var}: {n_out_of_range} nonmissing value(s) outside [-1, 1]."
                )
                sys.exit(1)

        z_var = (
            var.replace("_mean_r", "_mean_z")
               .replace("_r_L", "_z_L")
               .replace("_r_R", "_z_R")
        )
        if z_var not in df.columns:
            df[z_var] = fisher_z(numeric_ser)
        z_vars.append(z_var)
    return z_vars


# ============================================================================
# Sample construction
# ============================================================================
def get_samples(df, check_fc_col=None, behavioral=False, require_diary=True):
    """
    Return (full_sample, conservative_sample) DataFrames.

    When behavioral=True:
      Full sample  = participants with at least one of PA_score / NA_score.
      Conservative = same as full (no fMRI QC applies).

    When behavioral=False (fMRI analyses, default):
      Full sample  = participants with has_neg_persistence == 1
                     and, if require_diary=True, at least one affect measure.
      Conservative = full sample restricted to qc_conservative == 1.

    Parameters
    ----------
    check_fc_col : str, optional
        If provided, require this column to be nonmissing.  Aborts if the
        column is absent from df.
    behavioral : bool
        If True, use the behavioral sample.
    require_diary : bool
        If False, omit the diary affect requirement (for neuro-only analyses).
        Default True.
    """
    if behavioral or require_diary:
        _require_columns(df, ["PA_score", "NA_score"],
                         "get_samples (behavioral/diary requirement)")

    if behavioral:
        has_affect = df[["PA_score", "NA_score"]].notna().any(axis=1)
        full = df[has_affect].copy()
        return full, full

    # fMRI path
    _require_columns(df, ["has_neg_persistence", "qc_conservative"],
                     "get_samples (fMRI)")

    # Validate flag values: must be 0, 1, or missing
    for flag in ["has_neg_persistence", "qc_conservative"]:
        observed = set(df[flag].dropna().unique())
        bad = observed - {0, 1}
        if bad:
            print(
                f"ERROR: get_samples: {flag} contains unexpected value(s) "
                f"{sorted(bad)} (allowed: {{0, 1}})."
            )
            sys.exit(1)

    if check_fc_col is not None:
        if check_fc_col not in df.columns:
            print(
                f"ERROR: get_samples: required FC column '{check_fc_col}' "
                f"not found in DataFrame."
            )
            sys.exit(1)

    has_persist = df["has_neg_persistence"] == 1

    if require_diary:
        has_affect = df[["PA_score", "NA_score"]].notna().any(axis=1)
        mask = has_persist & has_affect
    else:
        mask = has_persist

    if check_fc_col is not None:
        mask = mask & df[check_fc_col].notna()

    full         = df[mask].copy()
    conservative = full[full["qc_conservative"] == 1].copy()
    return full, conservative


# ============================================================================
# Covariate selection
# ============================================================================
def get_covariates(df):
    """
    Return the exact ordered base covariate list for MR1 adjusted models.

    Covariates (in order): RA5PAGE, sex, race_2, race_3, race_4, race_5, race_6.
    All seven must be present in df; aborts if any are absent.

    Diary-specific covariates (time_P2_P5, n_days_complete) are added by
    run_analysis_set for diary outcomes and are not included here.

    The age analyses remove RA5PAGE from this list when it is also the
    predictor; that removal is the caller's responsibility.
    """
    _require_columns(df, BASE_COVARIATES, "get_covariates")
    return list(BASE_COVARIATES)


# ============================================================================
# Single-pair analysis functions
# ============================================================================
def run_correlation(df, predictor, outcome, one_tailed=False, expected_positive=True):
    """
    Pearson correlation between predictor and outcome.

    Returns a result dict or None if n < MIN_N.
    Aborts if predictor or outcome is absent from df.
    """
    if predictor not in df.columns:
        print(f"ERROR: run_correlation: predictor '{predictor}' not in DataFrame.")
        sys.exit(1)
    if outcome not in df.columns:
        print(f"ERROR: run_correlation: outcome '{outcome}' not in DataFrame.")
        sys.exit(1)

    data = df[[predictor, outcome]].dropna()
    n = len(data)
    if n < MIN_N:
        return None
    corr_result = stats.pearsonr(data[predictor], data[outcome])
    r, p_two = corr_result
    p = one_tailed_p(r, p_two, expected_positive) if one_tailed else p_two
    return {
        **pearson_ci(corr_result),
        "predictor": predictor, "outcome": outcome,
        "n": n, "r": r, "p": p, "p_two_tailed": p_two,
    }


def run_ols(df, predictor, outcome, covariates, one_tailed=False, expected_positive=True):
    """
    OLS regression: outcome ~ predictor + covariates.

    Returns a result dict or None if n < MIN_N.
    Aborts if predictor or outcome is absent from df.
    Zero-variance covariates are silently dropped from the model.
    """
    if predictor not in df.columns:
        print(f"ERROR: run_ols: predictor '{predictor}' not in DataFrame.")
        sys.exit(1)
    if outcome not in df.columns:
        print(f"ERROR: run_ols: outcome '{outcome}' not in DataFrame.")
        sys.exit(1)

    cov_list = [c for c in covariates if c in df.columns and c != predictor]
    cols = [outcome, predictor] + cov_list
    data = df[cols].dropna()
    if len(data) < MIN_N:
        return None

    cov_list = [c for c in cov_list if data[c].std() > 0]
    X = sm.add_constant(data[[predictor] + cov_list])
    y = data[outcome]

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"    OLS error ({outcome} ~ {predictor}): {e}")
        return None

    beta  = model.params[predictor]
    p_two = model.pvalues[predictor]
    p     = one_tailed_p(beta, p_two, expected_positive) if one_tailed else p_two

    return {
        **coefficient_ci(model, predictor),
        "predictor":     predictor,
        "outcome":       outcome,
        "n":             int(model.nobs),
        "df_resid":      int(model.df_resid),
        "beta":          beta,
        "se":            model.bse[predictor],
        "t":             model.tvalues[predictor],
        "p":             p,
        "p_two_tailed":  p_two,
        "r_squared":     model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }


# ============================================================================
# Multi-pair analysis loop
# ============================================================================
def run_analysis_set(df, predictors, outcomes, base_covariates,
                     one_tailed=False, expected_directions=None):
    """
    Run Pearson correlations and OLS for all predictor × outcome combinations.

    Diary-specific covariates (time_P2_P5, n_days_complete) are automatically
    added for outcomes in DIARY_OUTCOMES.

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
    corr_df, ols_df : pd.DataFrame (may be empty)
    """
    if expected_directions is None:
        expected_directions = {}

    _require_columns(
        df,
        list(predictors) + list(outcomes) + list(base_covariates),
        "run_analysis_set",
    )

    needs_diary = any(out in DIARY_OUTCOMES for out in outcomes)
    if needs_diary:
        _require_columns(df, ["time_P2_P5", "n_days_complete"],
                         "run_analysis_set (diary outcomes)")

    diary_covs = [c for c in ["time_P2_P5", "n_days_complete"] if c in df.columns]
    corr_rows, ols_rows = [], []

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

    return pd.DataFrame(corr_rows), pd.DataFrame(ols_rows)


# ============================================================================
# Variable labels (for methods notes)
# ============================================================================
VAR_LABELS = {
    "RA5PAGE":      "age at neuroscience visit (P5)",
    "RA2PAGE":      "age at diary wave (P2, estimated)",
    "PA_score":     "positive affect (daily diary composite, mean across days)",
    "NA_score":     "negative affect (daily diary composite, mean across days)",
    "NA_score_log": "log-transformed negative affect (daily diary)",
    "RA5SPGP":      "positive affect (PANAS, neuroscience visit)",
    "RA5SPGN":      "negative affect (PANAS, neuroscience visit)",
    "RA5SPGN_log":  "log-transformed negative affect (PANAS)",
    "RA5SER":       "emotion regulation — reappraisal (ERQ)",
    "RA5SES":       "emotion regulation — suppression (ERQ)",
    "sex":              "biological sex (1=Male, 2=Female)",
    "n_days_complete":  "number of completed daily diary days",
    "time_P2_P5":       "time between diary (P2) and neuroscience (P5) waves (months)",
    "neg_persist_crossrun_mean_z_L": (
        "left amygdala negative-affect persistence "
        "(cross-run spatial correlation, Fisher z)"
    ),
    "neg_persist_crossrun_mean_z_R": (
        "right amygdala negative-affect persistence "
        "(cross-run spatial correlation, Fisher z)"
    ),
    "pos_persist_crossrun_mean_z_L": (
        "left amygdala positive-affect persistence "
        "(cross-run spatial correlation, Fisher z)"
    ),
    "pos_persist_crossrun_mean_z_R": (
        "right amygdala positive-affect persistence "
        "(cross-run spatial correlation, Fisher z)"
    ),
    # FC — negative condition
    "l_amyg-ant_vmPFC_neg":  "left amygdala – anterior vmPFC FC, negative condition (Fisher z)",
    "l_amyg-post_vmPFC_neg": "left amygdala – posterior vmPFC FC, negative condition (Fisher z)",
    "r_amyg-ant_vmPFC_neg":  "right amygdala – anterior vmPFC FC, negative condition (Fisher z)",
    "r_amyg-post_vmPFC_neg": "right amygdala – posterior vmPFC FC, negative condition (Fisher z)",
    # FC — negative vs neutral (primary contrast)
    "l_amyg-ant_vmPFC_neg_vs_neu":  (
        "left amygdala – anterior vmPFC FC, negative > neutral contrast (Fisher z)"
    ),
    "l_amyg-post_vmPFC_neg_vs_neu": (
        "left amygdala – posterior vmPFC FC, negative > neutral contrast (Fisher z)"
    ),
    "r_amyg-ant_vmPFC_neg_vs_neu":  (
        "right amygdala – anterior vmPFC FC, negative > neutral contrast (Fisher z)"
    ),
    "r_amyg-post_vmPFC_neg_vs_neu": (
        "right amygdala – posterior vmPFC FC, negative > neutral contrast (Fisher z)"
    ),
}


def _var_label(v):
    if v in VAR_LABELS:
        return VAR_LABELS[v]
    if v.startswith("race_"):
        return f"race dummy (ref = White): category {v.replace('race_', '')}"
    return v


# ============================================================================
# Methods note
# ============================================================================
def write_methods_note(out_dir, label, predictors, outcomes, covariates, n,
                       one_tailed=False, expected_directions=None, extra_notes=None):
    """Write a _methods.txt file describing the exact MR1 model specification."""
    import datetime
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand with diary covariates when any outcome uses them, matching run_analysis_set.
    diary_extra = []
    if any(o in DIARY_OUTCOMES for o in outcomes):
        for dc in ["time_P2_P5", "n_days_complete"]:
            if dc not in covariates:
                diary_extra.append(dc)
    all_covariates = list(covariates) + diary_extra

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
    L.append("2. OLS regression (participant-level)")
    L.append("   Formula : outcome ~ predictor + covariates")
    L.append("   Note    : the available MR1 P5 analytic inputs contain no usable")
    L.append("             family, household, sibling, twin-pair, or other grouping")
    L.append("             identifier; participant-level OLS is therefore used.")
    L.append("   Covariates:")
    for c in all_covariates:
        L.append(f"     - {c}: {_var_label(c)}")
    L.append("")

    L.append("INFERENCE")
    L.append("-" * 60)
    if one_tailed:
        L.append("Tests: one-tailed (pre-registered directional hypotheses)")
        L.append("  p_one = p_two / 2        if effect is in predicted direction")
        L.append("  p_one = 1 - p_two / 2    if effect is in opposite direction")
        if expected_directions:
            L.append("  Predicted directions:")
            for v, d in (expected_directions or {}).items():
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
def save_results(corr_df, ols_df, out_dir, label="",
                 predictors=None, outcomes=None, covariates=None, n=None,
                 one_tailed=False, expected_directions=None, extra_notes=None):
    """
    Save correlations.csv, regressions.csv, and (optionally) _methods.txt.

    Privacy: prints only aggregate counts and summary statistics.
    Never prints participant identifiers or rows.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_df.to_csv(out_dir / "correlations.csv", index=False)
    ols_df.to_csv(out_dir / "regressions.csv", index=False)

    if predictors is not None and outcomes is not None and covariates is not None:
        _n = n if n is not None else (int(corr_df["n"].max()) if len(corr_df) else 0)
        write_methods_note(
            out_dir, label, predictors, outcomes, covariates, _n,
            one_tailed=one_tailed, expected_directions=expected_directions,
            extra_notes=extra_notes,
        )

    _print_summary(corr_df, ols_df, label=label)
    print(f"  Saved to: {out_dir}")


def _print_summary(corr_df, ols_df, label=""):
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    for name, df in [("Correlations", corr_df), ("Regressions", ols_df)]:
        if df is None or len(df) == 0:
            continue
        sig = df[df["p"] < 0.05]
        print(f"\n  {name}  ({len(sig)}/{len(df)} p < .05)")
        for _, row in sig.iterrows():
            stat_key = "r" if "r" in row else "beta"
            p2 = (
                f", p_two={row['p_two_tailed']:.3f}"
                if "p_two_tailed" in row and not np.isnan(row["p_two_tailed"])
                else ""
            )
            print(f"    {row['predictor']:45s} -> {row['outcome']:20s} "
                  f"stat={row[stat_key]:+.3f}  p={row['p']:.4f}{p2}  n={int(row['n'])}")
