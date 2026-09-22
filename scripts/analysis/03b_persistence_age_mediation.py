#!/usr/bin/env python3
"""
03b_persistence_age_mediation.py

Prespecified confirmatory extension of Analysis 03: statistical indirect
effects of age on daily affect through left-amygdala negative persistence.

    X (age)  --a-->  M (neg_persist_crossrun_mean_z_L)  --b-->  Y (affect)
          \\_______________________ c' ________________________/

Preregistered as a confirmatory extension: "mediation model for age-related
differences in amygdala persistence -> age-related differences in
positive/negative affect."

Predicted directions, from analyses 02 and 03: older age is associated with
LOWER persistence (a < 0); lower persistence is associated with HIGHER PA
(b < 0) and LOWER NA (b > 0). The predicted indirect effect a*b is therefore
POSITIVE for PA_score and NEGATIVE for NA_score and NA_score_log.

======================================================================
TEMPORAL ORDER AND CAUSAL STATUS - READ BEFORE INTERPRETING
======================================================================

The mediator is measured AFTER the outcome. Daily diary affect (Y) is
collected in Project 2; the neuroscience visit that yields persistence (M)
happens afterwards. Preprocessing computes
time_P2_P5 = C5PDATE - diary start date (06_clean_merged_data.py), i.e. the P5
visit post-dates the P2 diary. The absolute value is taken there, which also
conceals any participant for whom that order is reversed.

The preregistration acknowledged that diary affect preceded the neuroimaging
assessment. The measurement order is therefore X -> Y -> M, not X -> M -> Y.

Everything this script produces is a STATISTICAL DECOMPOSITION OF COVARIANCE.
These are indirect associations. They cannot establish temporal precedence or
causal mediation, and must not be described as showing that age affects affect
"through" persistence: persistence had not yet been measured when affect was
reported. This constraint is a property of the study design and is not
remedied by any modelling choice below.

======================================================================

Specifications
--------------
primary_c5page      Age = C5PAGE (age at the neuroscience visit).
sensitivity_c2page  Age = C2PAGE (age at the diary wave). An age-DEFINITION
                    sensitivity, not a separate hypothesis. C2PAGE replaces
                    C5PAGE; the two never appear in the same model.

Outcomes
--------
PA_score, NA_score      outcome_role = primary
NA_score_log            outcome_role = robustness_log_na
                        an outcome-TRANSFORMATION robustness check

Models (mixed-effects, random intercept for family, REML, optimizers tried
lbfgs -> powell -> nm -> bfgs, exactly as analysis_utils.run_mlm builds them):

  primary_c5page
      a           M ~ C5PAGE + sex + race
      b, c'       Y ~ M + C5PAGE + sex + race + time_P2_P5 + n_days_complete
      c total     Y ~ C5PAGE + sex + race + time_P2_P5 + n_days_complete

  sensitivity_c2page
      a           M ~ C2PAGE + sex + race + time_P2_P5
      b, c'       Y ~ M + C2PAGE + sex + race + time_P2_P5 + n_days_complete
      c total     Y ~ C2PAGE + sex + race + time_P2_P5 + n_days_complete

The elapsed-time covariate enters the C2PAGE a path because persistence is
measured later, at P5. It remains in the affect models under the existing
diary-analysis convention.

run_mlm's own rules apply throughout: the predictor is dropped from its own
covariate list, twin_pair_* dummies are dropped because the family random
intercept represents that structure, and zero-variance covariates are pruned.

Common complete-case sample
---------------------------
Each specification is fitted on ONE participant-level complete-case frame. The
a model, all three b/c' models, all three total-effect models and the cluster
bootstrap use that same frame, so a and b can never be estimated on different
participants and the product a*b cannot mix complete-case sets. Every observed
model is asserted to have been fitted on the specification's common N.
Zero-variance covariate pruning still happens downstream and does not change
who is included.

The family id is constructed FIRST, on the full sample, and only then is the
complete-case filter applied - to the ANALYSIS variables (the age predictor,
the mediator, the three outcomes, sex, race_2 through race_6, time_P2_P5 and
n_days_complete) plus the derived _family_id.

SAMPLMAJ and M2FAMNUM are deliberately NOT part of that filter. They are raw
inputs to the family-id construction, not fixed effects. A participant missing
them is simply not a paired twin and is validly assigned their own M2ID-keyed
singleton cluster; excluding such participants would shrink the sample without
changing any model. M2ID itself IS required: an otherwise-eligible participant
with no M2ID cannot be clustered, so the run aborts rather than dropping them
silently or coercing the identifier to the string "nan".

The frame is returned with _family_id attached. fit_mlm uses that existing
column rather than rebuilding it, and the bootstrap resamples it directly.

All required columns - age predictors, mediator, outcomes, sex, race_2 through
race_6, the diary covariates and the grouping columns - must be PRESENT at
startup. Column presence is mandatory; universal nonmissingness of SAMPLMAJ
and M2FAMNUM is not. No covariate list is assembled from whichever columns
happen to exist, and the family grouping never falls back to an alternative
definition.

Execution order
---------------
Within a specification every observed model is fitted and fully validated
(convergence, common N, age-variable exclusion, stored anchors) BEFORE the
2,000-replicate bootstrap begins, so a failure surfaces immediately instead of
after thousands of fits.

Validation anchors
------------------
The primary specification's models are the published models, so they must
reproduce them exactly or the script aborts:

  * a path refitted in the N = 127 conservative fMRI sample must reproduce
    results/tables/03_persistence_age/mlm.csv
  * the b/c' model IS analysis 02's model (02 already covaries C5PAGE), so the
    fitted b for PA_score, NA_score and NA_score_log must reproduce
    results/tables/02_persistence_affect/mlm.csv

The sensitivity specification has no published counterpart and is not
anchored; it is reported as a sensitivity only.

Inference on the indirect effect a*b
------------------------------------
PRIMARY: family-level cluster bootstrap, percentile 95% interval. Families -
not individuals - are resampled, matching the clustering the random intercept
assumes. A replicate contributes only when BOTH its a model and its b model
converge.

SECONDARY: Monte Carlo distribution-of-the-product interval. This draws a and
b from INDEPENDENT normal distributions and therefore does NOT incorporate any
covariance between them. It is an approximation reported alongside the
bootstrap, never in place of it.

All indirect-effect intervals are two-sided. The directional predictions in 02
and 03 apply to a and b individually, not to the sampling distribution of
their product, so the product tests are not converted to one-tailed.

Multiplicity and precedence
---------------------------
No multiplicity adjustment is applied across the predefined sensitivities.
Sensitivity and robustness findings do not override the primary specification.

Note when reading the tables: c, c' and a*b come from separate mixed models,
so they are not guaranteed to satisfy c = c' + a*b exactly. Do not expect the
three estimates to reconcile arithmetically.

The field indirect_sign_as_predicted records only whether the point estimate
falls in the predicted direction. It is DESCRIPTIVE and is never evidence of a
statistical effect; read the bootstrap interval for that.

Outputs
-------
    results/tables/03b_persistence_age_mediation/paths.csv
    results/tables/03b_persistence_age_mediation/mediation.csv
    results/tables/03b_persistence_age_mediation/_methods.txt

Console output is aggregate only; no participant record or identifier is
printed.

Run from project root directory.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (  # noqa: E402
    DIARY_OUTCOMES,
    MIN_N,
    RESULTS_DIR,
    get_covariates,
    get_samples,
    load_master,
    prepare_persistence_vars,
)

# ============================================================================
# Configuration
# ============================================================================
OUT_DIR = RESULTS_DIR / "03b_persistence_age_mediation"

M_VAR = "neg_persist_crossrun_mean_z_L"
PRIMARY_AGE = "C5PAGE"
SENSITIVITY_AGE = "C2PAGE"
DIARY_COVS = ["time_P2_P5", "n_days_complete"]
ELAPSED_COV = "time_P2_P5"
RACE_DUMMIES = ["race_2", "race_3", "race_4", "race_5", "race_6"]
GROUPING_COLS = ["M2ID", "SAMPLMAJ", "M2FAMNUM"]

# Every column the analysis needs. Required unconditionally at startup: no
# covariate list is assembled from whichever columns happen to be present.
REQUIRED_COLUMNS = (
    [PRIMARY_AGE, SENSITIVITY_AGE, M_VAR]
    + ["PA_score", "NA_score", "NA_score_log"]
    + ["sex"]
    + RACE_DUMMIES
    + DIARY_COVS
    + GROUPING_COLS
)

# outcome -> (role, expected sign of a*b under the preregistered directions)
OUTCOMES = {
    "PA_score": ("primary", +1),
    "NA_score": ("primary", -1),
    "NA_score_log": ("robustness_log_na", -1),
}

SPEC_PRIMARY = "primary_c5page"
SPEC_SENSITIVITY = "sensitivity_c2page"

EXPECTED_N_PRIMARY = 81     # diary + fMRI conservative
EXPECTED_N_APATH_PUB = 127  # conservative fMRI, diary not required

N_MC = 200_000
N_BOOT = 2000
MIN_BOOT_FRACTION = 0.80    # abort below this share of converged replicates
CI_LEVEL = 95.0
SEED = 42

OPTIMIZERS = ["lbfgs", "powell", "nm", "bfgs"]

PUB_APATH_CSV = RESULTS_DIR / "03_persistence_age" / "mlm.csv"
PUB_BPATH_CSV = RESULTS_DIR / "02_persistence_affect" / "mlm.csv"

PUB_APATH = {"beta": -0.002598949928202176, "se": 0.0012437344840872948, "n": 127}
PUB_BPATH = {
    "PA_score": {"beta": -0.6995706141184733, "se": 0.4760595215276992, "n": 81},
    "NA_score": {"beta": 0.27792707372427095, "se": 0.1357241284390131, "n": 81},
    "NA_score_log": {"beta": 1.9500893839275775, "se": 1.2731125524149542, "n": 81},
}

TOL_REL = 1e-5
TOL_ABS = 1e-8


def _fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def _close(got, want, rel=TOL_REL, abs_=TOL_ABS):
    if got is None or want is None:
        return False
    if not (np.isfinite(got) and np.isfinite(want)):
        return False
    return abs(got - want) <= max(abs_, rel * abs(want))


# ============================================================================
# Model fitting
# ============================================================================
def assign_family_id(df):
    """
    Add _family_id exactly as analysis_utils.run_mlm derives it: paired twins
    share a family id, everyone else is their own cluster.

    Factored out so the bootstrap resamples the same clusters the model treats
    as exchangeable. Resampling on M2FAMNUM alone would be wrong, since
    non-twin siblings can share a family number without being grouped by the
    random intercept.

    The grouping inputs are REQUIRED. run_mlm falls back to M2FAMNUM-only or
    M2ID-only grouping when they are absent; that fallback is deliberately not
    reproduced here, because a silently different clustering would change both
    the random effect and the bootstrap resampling unit.
    """
    missing = [c for c in GROUPING_COLS if c not in df.columns]
    if missing:
        _fail(
            f"family-grouping column(s) {missing} absent. The specified "
            "grouping (paired twins share a family id, everyone else is their "
            "own cluster) cannot be constructed, and no alternative grouping "
            "will be substituted."
        )
    df = df.copy()
    # A missing SAMPLMAJ or M2FAMNUM is not a problem: NaN == 3 is False and
    # NaN is never in `paired`, so such a participant simply becomes their own
    # M2ID-keyed singleton cluster, which is the correct grouping for them.
    is_twin = df["SAMPLMAJ"] == 3
    paired = df.loc[is_twin, "M2FAMNUM"].value_counts()
    paired = paired[paired > 1].index

    # Never coerce a missing M2ID to the string "nan": that would invent a
    # cluster and, worse, merge every such participant into a single one.
    # Leave _family_id missing instead and let the caller decide.
    m2id_ok = df["M2ID"].notna()
    df["_family_id"] = pd.NA
    df.loc[m2id_ok, "_family_id"] = df.loc[m2id_ok, "M2ID"].astype(str)

    mask = is_twin & df["M2FAMNUM"].isin(paired)
    df.loc[mask, "_family_id"] = (
        "fam_" + df.loc[mask, "M2FAMNUM"].astype(int).astype(str)
    )
    return df


def fit_mlm(df, predictor, outcome, covariates):
    """
    Fit outcome ~ predictor + covariates + (1 | family) as run_mlm builds it.

    A fit is accepted ONLY if the optimizer reports convergence. statsmodels
    returns a results object even when the optimizer has not converged, so an
    un-gated `break` would silently accept a non-converged model. Each
    optimizer is tried in turn and the first CONVERGED fit is returned;
    if none converge, returns None.

    Returns (result, data, pred_safe, cov_used, optimizer) or None.
    """
    if "_family_id" not in df.columns:
        df = assign_family_id(df)
    else:
        df = df.copy()

    pred_safe = predictor.replace("-", "_").replace(".", "_")
    if pred_safe != predictor:
        df[pred_safe] = df[predictor]

    cov_list = [
        c
        for c in covariates
        if c in df.columns
        and c != predictor
        and c != pred_safe
        and not c.startswith("twin_pair_")
    ]
    cols = [outcome, pred_safe, "_family_id"] + cov_list
    data = df[cols].dropna()
    if len(data) < MIN_N:
        return None

    cov_list = [c for c in cov_list if data[c].std() > 0]
    cov_terms = " + ".join(cov_list)
    fixed = f"{outcome} ~ {pred_safe}" + (f" + {cov_terms}" if cov_terms else "")

    for method in OPTIMIZERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = smf.mixedlm(fixed, data=data, groups=data["_family_id"])
                res = mdl.fit(reml=True, method=method)
        except Exception:
            continue
        if bool(getattr(res, "converged", False)) is True:
            return res, data, pred_safe, cov_list, method
        # Returned but not converged: try the next optimizer.
    return None


def coef(result, name):
    """(beta, se) for a fixed effect, with run_mlm's degenerate-SE fallback."""
    beta = float(result.fe_params[name])
    se = float(result.bse_fe.get(name, np.nan))
    if np.isnan(se):
        try:
            idx = list(result.fe_params.index).index(name)
            var = result.cov_params().iloc[idx, idx]
            se = float(np.sqrt(abs(var))) if abs(var) > 0 else np.nan
        except Exception:
            se = np.nan
    return beta, se


def wald(beta, se):
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan
    z = beta / se
    return z, float(2 * (1 - stats.norm.cdf(abs(z))))


# ============================================================================
# Published-model validation
# ============================================================================
def _read_pub(csv_path, predictor, outcome):
    if not csv_path.exists():
        _fail(
            f"required published result table {csv_path} not found; cannot "
            "validate that the primary models match the manuscript."
        )
    df = pd.read_csv(csv_path)
    sel = df[(df["predictor"] == predictor) & (df["outcome"] == outcome)]
    if len(sel) != 1:
        _fail(
            f"{csv_path} has {len(sel)} rows for {predictor} -> {outcome}; "
            "expected exactly one."
        )
    return sel.iloc[0]


def validate_apath_against_published(base_covs_127, sample_127):
    """
    Refit the published a-path model (N = 127) and check it reproduces 03.

    This fit deliberately uses analysis 03's own sample and its per-model
    listwise deletion, NOT a specification common-case frame: it exists to
    reproduce the published model exactly. The common-frame rule applies to
    the mediation specifications, where a and b must share participants.
    """
    row = _read_pub(PUB_APATH_CSV, PRIMARY_AGE, M_VAR)
    for key in ("beta", "se"):
        if not _close(float(row[key]), PUB_APATH[key], rel=0.0, abs_=1e-12):
            _fail(
                f"{PUB_APATH_CSV} {key} = {float(row[key])!r} does not match the "
                f"frozen published value {PUB_APATH[key]!r}."
            )
    if int(row["n"]) != EXPECTED_N_APATH_PUB:
        _fail(f"published a-path N = {int(row['n'])}, expected {EXPECTED_N_APATH_PUB}.")

    fit = fit_mlm(sample_127, PRIMARY_AGE, M_VAR, base_covs_127)
    if fit is None:
        _fail(
            "the published a-path model (N = 127) did not converge under any "
            "optimizer; cannot validate against analysis 03."
        )
    res, _data, pred_safe, _cov, opt = fit
    if int(res.nobs) != EXPECTED_N_APATH_PUB:
        _fail(
            f"refitted a-path model used N = {int(res.nobs)}, expected "
            f"{EXPECTED_N_APATH_PUB}."
        )
    beta, se = coef(res, pred_safe)
    if not _close(beta, float(row["beta"])) or not _close(se, float(row["se"])):
        _fail(
            "the refitted a-path model does not reproduce analysis 03 "
            f"(beta {beta!r} vs {float(row['beta'])!r}, se {se!r} vs "
            f"{float(row['se'])!r})."
        )
    return beta, se, opt


def validate_bpath_against_published(res, m_safe, outcome):
    """The primary b/c' model IS analysis 02's model; b must reproduce it."""
    row = _read_pub(PUB_BPATH_CSV, M_VAR, outcome)
    exp = PUB_BPATH[outcome]
    for key in ("beta", "se"):
        if not _close(float(row[key]), exp[key], rel=0.0, abs_=1e-12):
            _fail(
                f"{PUB_BPATH_CSV} {key} for {outcome} = {float(row[key])!r} does "
                f"not match the frozen published value {exp[key]!r}."
            )
    b, se_b = coef(res, m_safe)
    if not _close(b, float(row["beta"])) or not _close(se_b, float(row["se"])):
        _fail(
            f"the fitted b path for {outcome} ({b!r}, SE {se_b!r}) does not "
            f"reproduce analysis 02 ({float(row['beta'])!r}, SE "
            f"{float(row['se'])!r}). The primary model is not the published model."
        )
    return b, se_b


# ============================================================================
# Specification covariates
# ============================================================================
def spec_covariates(age_var, base):
    """
    Return (a_covs, b_covs, c_covs) for one age specification.

    `base` is get_covariates(sample) unchanged, so the primary specification
    reproduces the published models exactly. The diary and elapsed-time
    covariates are appended UNCONDITIONALLY - their presence is guaranteed by
    the startup column check rather than tested here.

    C5PAGE and C2PAGE must never appear in the same model, so the C2PAGE
    specification explicitly strips C5PAGE from the base list.
    """
    diary = list(DIARY_COVS)

    if age_var == PRIMARY_AGE:
        # run_mlm strips the predictor from its own covariate list, so passing
        # `base` gives  M ~ C5PAGE + sex + race  for the a path and
        # Y ~ C5PAGE + ... for the total-effect model, while the b/c' model
        # retains C5PAGE as a covariate alongside the mediator.
        return list(base), list(base) + diary, list(base) + diary

    no_c5 = [c for c in base if c != PRIMARY_AGE]
    a_covs = no_c5 + [ELAPSED_COV]    # persistence measured later, at P5
    b_covs = no_c5 + [age_var] + diary
    c_covs = no_c5 + diary            # run_mlm strips the predictor (C2PAGE)
    return a_covs, b_covs, c_covs


def common_sample_columns(age_var):
    """
    The ANALYSIS variables every model in a specification needs.

    Deliberately excludes SAMPLMAJ and M2FAMNUM. Those are raw inputs to the
    family-id construction, not fixed effects; requiring them to be nonmissing
    would exclude participants whom assign_family_id validly places in their
    own M2ID-keyed singleton cluster. The derived _family_id is what the model
    actually needs, and it is required instead.
    """
    return (
        [age_var, M_VAR, "PA_score", "NA_score", "NA_score_log", "sex"]
        + RACE_DUMMIES
        + DIARY_COVS
    )


def build_common_sample(cons, age_var):
    """
    One participant-level complete-case frame per specification.

    Every model in the specification - the a model, all three b/c' models, all
    three total-effect models, and the cluster bootstrap - is fitted on this
    exact frame. Per-model listwise deletion is therefore eliminated: without
    it, a and b could be estimated on different participants and the product
    a*b would mix complete-case sets.

    The family id is constructed BEFORE complete-case filtering, and the
    filter is applied to the analysis variables plus that derived _family_id -
    never to the raw SAMPLMAJ / M2FAMNUM inputs. A participant missing those
    raw fields is validly a singleton cluster and must not be excluded.

    Zero-variance covariates are still pruned downstream by the established
    fitting logic; that happens after this frame is fixed and does not change
    which participants are included.
    """
    cols = common_sample_columns(age_var)
    missing = [c for c in cols if c not in cons.columns]
    if missing:
        _fail(
            f"column(s) {missing} required for the common complete-case sample "
            f"are absent for age predictor '{age_var}'."
        )

    tagged = assign_family_id(cons)
    eligible = tagged[cols].notna().all(axis=1)

    # An otherwise-eligible participant with no M2ID cannot be assigned a
    # cluster. Abort rather than drop them silently or invent an id for them.
    orphan = int((eligible & tagged["M2ID"].isna()).sum())
    if orphan:
        _fail(
            f"{orphan} participant(s) complete on every analysis variable for "
            f"'{age_var}' have a missing M2ID, so no family cluster can be "
            "assigned. Resolve the identifier before running the mediation."
        )

    frame = tagged.loc[eligible, cols + ["_family_id"]].copy()
    if frame["_family_id"].isna().any():
        _fail(
            "the common complete-case sample contains rows with no assigned "
            "_family_id; the family grouping is incomplete."
        )
    if len(frame) < MIN_N:
        _fail(
            f"the common complete-case sample for '{age_var}' has only "
            f"{len(frame)} participants, below MIN_N = {MIN_N}."
        )
    return frame


def assert_no_age_collision(result, age_var, label):
    """C5PAGE and C2PAGE must never be fixed effects in the same model."""
    terms = set(result.fe_params.index)
    other = PRIMARY_AGE if age_var != PRIMARY_AGE else SENSITIVITY_AGE
    if other in terms and age_var in terms:
        _fail(
            f"{label}: both '{age_var}' and '{other}' are fixed effects in the "
            "same model. The age specifications must be mutually exclusive."
        )
    if age_var != PRIMARY_AGE and PRIMARY_AGE in terms:
        _fail(
            f"{label}: '{PRIMARY_AGE}' appears in a {SENSITIVITY_AGE} model. "
            "C2PAGE must replace C5PAGE, not supplement it."
        )


# ============================================================================
# Indirect-effect inference
# ============================================================================
def mc_interval(a, se_a, b, se_b, rng):
    """
    Secondary Monte Carlo interval for a*b.

    a and b are drawn from INDEPENDENT normals, so this ignores any covariance
    between the two estimates. Reported alongside, never instead of, the
    cluster bootstrap.
    """
    lo_q, hi_q = (100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2
    draws = rng.normal(a, se_a, N_MC) * rng.normal(b, se_b, N_MC)
    lo, hi = np.percentile(draws, [lo_q, hi_q])
    tail = min((draws <= 0).mean(), (draws >= 0).mean())
    return float(lo), float(hi), float(min(1.0, 2 * tail))


def cluster_bootstrap_spec(sample, age_var, a_covs, b_covs, outcomes, rng):
    """
    Family-level cluster bootstrap for one age specification.

    Per replicate the families are drawn ONCE, the a model is fitted ONCE, and
    one b model is fitted per outcome against those same resampled families.
    A product is retained only when both required models converged.

    Returns (products_by_outcome, n_attempted, n_a_converged).
    """
    products = {o: [] for o in outcomes}
    if N_BOOT <= 0:
        return products, 0, 0

    # Reuse the _family_id already attached to the common frame; recomputing
    # it from raw grouping columns here would be both redundant and, since
    # those columns are no longer carried, impossible.
    if "_family_id" not in sample.columns:
        _fail(
            "cluster_bootstrap_spec expects the common frame to carry "
            "_family_id; it must not be reconstructed here."
        )
    tagged = sample
    keys = pd.Index(tagged["_family_id"].unique())
    groups = {k: tagged[tagged["_family_id"] == k] for k in keys}

    n_attempted = 0
    n_a_converged = 0
    for _ in range(N_BOOT):
        n_attempted += 1
        drawn = rng.choice(keys, size=len(keys), replace=True)
        rep = pd.concat([groups[k] for k in drawn], ignore_index=True)
        # A family drawn twice must remain two distinct clusters, or the random
        # intercept would merge them and understate the variance.
        rep["_family_id"] = [
            f"{k}#{j}" for j, k in enumerate(drawn) for _ in range(len(groups[k]))
        ]

        fa = fit_mlm(rep, age_var, M_VAR, a_covs)
        if fa is None:
            continue
        n_a_converged += 1
        a_i, _ = coef(fa[0], fa[2])
        if not np.isfinite(a_i):
            continue

        for outcome in outcomes:
            fb = fit_mlm(rep, M_VAR, outcome, b_covs)
            if fb is None:
                continue
            b_i, _ = coef(fb[0], fb[2])
            if np.isfinite(b_i):
                products[outcome].append(a_i * b_i)

    return products, n_attempted, n_a_converged


# ============================================================================
# One specification
# ============================================================================
def run_specification(spec_name, age_var, cons, base_covs, rng):
    """
    Fit and summarise one age specification.

    Execution order is deliberate: EVERY observed model is fitted and fully
    validated - convergence, common N, age-variable exclusion, and the stored
    published anchors - BEFORE the 2,000-replicate bootstrap starts. An
    anchor or convergence failure therefore surfaces immediately rather than
    after thousands of wasted fits.
    """
    print(f"\n  --- {spec_name}  (age = {age_var}) ---")
    a_covs, b_covs, c_covs = spec_covariates(age_var, base_covs)

    # ---- one common complete-case frame for every model in this spec ------
    common = build_common_sample(cons, age_var)
    n_common = len(common)
    if spec_name == SPEC_PRIMARY:
        if n_common != EXPECTED_N_PRIMARY:
            _fail(
                f"{spec_name}: the common complete-case sample has N = "
                f"{n_common}, expected {EXPECTED_N_PRIMARY}."
            )
        print(f"    common complete-case N = {n_common}")
    else:
        print(f"    common complete-case N = {n_common}")
        if n_common != EXPECTED_N_PRIMARY:
            print(
                f"    [note] this sensitivity uses N = {n_common}, not "
                f"{EXPECTED_N_PRIMARY}; it is NOT the same sample as the "
                "primary specification."
            )

    def _require_common_n(res, label):
        if int(res.nobs) != n_common:
            _fail(
                f"{spec_name}: {label} was fitted on N = {int(res.nobs)} but "
                f"the specification's common sample is N = {n_common}. Every "
                "model must use the identical participant set."
            )

    # ---- (a) observed a model ---------------------------------------------
    fa = fit_mlm(common, age_var, M_VAR, a_covs)
    if fa is None:
        _fail(f"{spec_name}: the a-path model did not converge under any optimizer.")
    res_a, _da, x_safe, cov_a, opt_a = fa
    assert_no_age_collision(res_a, age_var, f"{spec_name} a path")
    _require_common_n(res_a, "the a-path model")
    a, se_a = coef(res_a, x_safe)
    z_a, p_a = wald(a, se_a)
    print(
        f"    a: beta = {a:+.6f}, SE = {se_a:.6f}, z = {z_a:+.2f}, "
        f"p(two) = {p_a:.4f}, N = {n_common}, optimizer = {opt_a}"
    )

    # ---- (b) every observed b/c' and total-effect model -------------------
    observed = {}
    for outcome, (role, expected) in OUTCOMES.items():
        fb = fit_mlm(common, M_VAR, outcome, b_covs)
        if fb is None:
            _fail(
                f"{spec_name}: the b/c' model for '{outcome}' did not converge "
                "under any optimizer."
            )
        res_b, _db, m_safe, cov_b, opt_b = fb
        assert_no_age_collision(res_b, age_var, f"{spec_name} b/c' {outcome}")
        _require_common_n(res_b, f"the b/c' model for '{outcome}'")

        # (c) stored-anchor validation for the published primary models
        if spec_name == SPEC_PRIMARY:
            b, se_b = validate_bpath_against_published(res_b, m_safe, outcome)
        else:
            b, se_b = coef(res_b, m_safe)
        z_b, p_b = wald(b, se_b)

        if age_var not in res_b.fe_params.index:
            _fail(
                f"{spec_name}: '{age_var}' is absent from the b/c' model for "
                f"'{outcome}'; the direct effect cannot be read."
            )
        cprime, se_cp = coef(res_b, age_var)
        z_cp, p_cp = wald(cprime, se_cp)

        fc = fit_mlm(common, age_var, outcome, c_covs)
        if fc is None:
            _fail(
                f"{spec_name}: the total-effect model for '{outcome}' did not "
                "converge under any optimizer."
            )
        res_c, _dc, x_safe_c, cov_c, opt_c = fc
        assert_no_age_collision(res_c, age_var, f"{spec_name} total {outcome}")
        _require_common_n(res_c, f"the total-effect model for '{outcome}'")
        c_tot, se_c = coef(res_c, x_safe_c)
        z_c, p_c = wald(c_tot, se_c)

        observed[outcome] = dict(
            role=role, expected=expected,
            b=b, se_b=se_b, z_b=z_b, p_b=p_b, opt_b=opt_b, n_cov_b=len(cov_b),
            cprime=cprime, se_cp=se_cp, z_cp=z_cp, p_cp=p_cp,
            c_tot=c_tot, se_c=se_c, z_c=z_c, p_c=p_c, opt_c=opt_c,
            n_cov_c=len(cov_c),
        )

    print("    [ok] all observed models converged, share the common N, and "
          "match their stored anchors")

    # ---- (d) observed path summaries --------------------------------------
    path_rows = [
        dict(
            specification=spec_name, age_predictor=age_var, outcome=M_VAR,
            outcome_role="mediator_model", path="a", predictor=age_var,
            term=age_var, beta=a, se=se_a, z=z_a, p_two_tailed=p_a, n=n_common,
            n_covariates=len(cov_a), optimizer=opt_a, converged=True,
        )
    ]
    for outcome, o in observed.items():
        for nm, term, bb, ss, zz, pp, ncov, opt in [
            ("b", M_VAR, o["b"], o["se_b"], o["z_b"], o["p_b"], o["n_cov_b"], o["opt_b"]),
            ("c_prime", age_var, o["cprime"], o["se_cp"], o["z_cp"], o["p_cp"],
             o["n_cov_b"], o["opt_b"]),
            ("c_total", age_var, o["c_tot"], o["se_c"], o["z_c"], o["p_c"],
             o["n_cov_c"], o["opt_c"]),
        ]:
            path_rows.append(
                dict(
                    specification=spec_name, age_predictor=age_var,
                    outcome=outcome, outcome_role=o["role"], path=nm,
                    predictor=M_VAR if nm == "b" else age_var, term=term,
                    beta=bb, se=ss, z=zz, p_two_tailed=pp, n=n_common,
                    n_covariates=ncov, optimizer=opt, converged=True,
                )
            )

    # ---- (e) bootstrap, only now that every observed model has passed -----
    print(f"    bootstrapping {N_BOOT} family resamples (a fitted once each)...")
    products, n_att, n_a_ok = cluster_bootstrap_spec(
        common, age_var, a_covs, b_covs, list(OUTCOMES), rng
    )
    print(
        f"    a model converged in {n_a_ok}/{n_att} replicates "
        f"({(n_a_ok / n_att if n_att else 0):.1%})"
    )

    # ---- (f) combine observed + bootstrap + Monte Carlo -------------------
    med_rows = []
    lo_q, hi_q = (100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2
    for outcome, o in observed.items():
        b, se_b = o["b"], o["se_b"]
        ind = a * b
        prods = products[outcome]
        n_ok = len(prods)
        frac = (n_ok / n_att) if n_att else 0.0
        if n_att and frac < MIN_BOOT_FRACTION:
            _fail(
                f"{spec_name} / {outcome}: only {n_ok}/{n_att} bootstrap "
                f"replicates ({frac:.1%}) had BOTH the a and b models converge, "
                f"below the required {MIN_BOOT_FRACTION:.0%}. The bootstrap "
                "interval would not be trustworthy."
            )
        if n_ok:
            bs_lo, bs_hi = (float(v) for v in np.percentile(prods, [lo_q, hi_q]))
        else:
            bs_lo = bs_hi = np.nan
        mc_lo, mc_hi, mc_p = mc_interval(a, se_a, b, se_b, rng)

        med_rows.append(
            dict(
                specification=spec_name, age_predictor=age_var, outcome=outcome,
                outcome_role=o["role"], n=n_common, n_a_model=n_common,
                a=a, se_a=se_a, b=b, se_b=se_b, indirect_ab=ind,
                boot_ci_low=bs_lo, boot_ci_high=bs_hi,
                boot_n_attempted=n_att, boot_n_converged=n_ok,
                boot_convergence_fraction=frac,
                mc_ci_low=mc_lo, mc_ci_high=mc_hi, mc_p_two_tailed=mc_p,
                c_prime=o["cprime"], se_c_prime=o["se_cp"], p_c_prime=o["p_cp"],
                c_total=o["c_tot"], se_c_total=o["se_c"], p_c_total=o["p_c"],
                expected_indirect_sign=o["expected"],
                indirect_sign_as_predicted=bool(np.sign(ind) == o["expected"])
                if ind != 0 else False,
                ci_level=CI_LEVEL,
            )
        )

        tag = "" if o["role"] == "primary" else f"  [{o['role']}]"
        print(f"    {outcome}{tag}")
        print(
            f"      indirect a*b = {ind:+.6f}   bootstrap {CI_LEVEL:.0f}% CI "
            f"[{bs_lo:+.6f}, {bs_hi:+.6f}]  ({n_ok}/{n_att} = {frac:.1%} "
            "replicates with both models converged)"
        )
        print(
            f"      secondary MC CI [{mc_lo:+.6f}, {mc_hi:+.6f}], p = {mc_p:.4f} "
            "(independent normal draws; ignores cov(a,b))"
        )
        print(
            f"      total c = {o['c_tot']:+.6f} (p = {o['p_c']:.4f}); "
            f"direct c' = {o['cprime']:+.6f} (p = {o['p_cp']:.4f})"
        )

    return path_rows, med_rows


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 74)
    print("Analysis 03b: Age -> Persistence -> Affect")
    print("Prespecified confirmatory extension - statistical indirect effects")
    print("=" * 74)
    print("TEMPORAL ORDER: the mediator (P5 neuroscience) is measured AFTER the")
    print("outcome (P2 daily diary), as acknowledged in the preregistration.")
    print("These are statistical decompositions of covariance; they cannot")
    print("establish temporal precedence or causal mediation.")
    print("=" * 74)

    df = load_master()
    prepare_persistence_vars(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        _fail(
            "required column(s) absent from the master table:\n"
            f"    {missing}\n"
            "  Every model definition, covariate and grouping input is fixed in "
            "advance; no specification, outcome, covariate or grouping column "
            "is skipped, substituted or inferred from what happens to be "
            "present."
        )
    for o in OUTCOMES:
        if o not in DIARY_OUTCOMES:
            _fail(f"'{o}' is not in DIARY_OUTCOMES; diary covariates would not apply.")
    if M_VAR in DIARY_OUTCOMES:
        _fail(f"'{M_VAR}' is unexpectedly in DIARY_OUTCOMES.")

    # ---- validate the published a path (N = 127) --------------------------
    _f127, cons127 = get_samples(df, require_diary=False)
    if len(cons127) != EXPECTED_N_APATH_PUB:
        _fail(
            f"the conservative fMRI sample has N = {len(cons127)}, expected "
            f"{EXPECTED_N_APATH_PUB}."
        )
    a_pub, se_a_pub, opt_pub = validate_apath_against_published(
        get_covariates(cons127), cons127
    )
    print(
        f"\n  [ok] published a path reproduced in N = {len(cons127)}: "
        f"beta = {a_pub:.6f}, SE = {se_a_pub:.6f}, optimizer = {opt_pub}"
    )

    # ---- mediation sample --------------------------------------------------
    _full, cons = get_samples(df)
    if len(cons) != EXPECTED_N_PRIMARY:
        _fail(
            f"the diary + fMRI conservative sample has N = {len(cons)}, "
            f"expected {EXPECTED_N_PRIMARY}."
        )
    # get_covariates() collects race dummies by prefix. The model definitions
    # specify race_2 through race_6 exactly, so an unexpected extra dummy would
    # silently enter every model - and would also break the published anchors.
    base_covs = get_covariates(cons)
    race_found = sorted(c for c in base_covs if c.startswith("race_"))
    if race_found != sorted(RACE_DUMMIES):
        _fail(
            f"race dummies in the sample are {race_found}, but the model "
            f"definitions specify exactly {sorted(RACE_DUMMIES)}."
        )

    print(f"  Mediation sample frame N = {len(cons)} (diary + fMRI, conservative)")
    if "time_P2_P5" in cons.columns:
        gap = cons["time_P2_P5"].dropna()
        if len(gap):
            print(
                f"  Diary-to-neuroscience interval: median {gap.median():.1f} "
                f"months (IQR {gap.quantile(.25):.1f}-{gap.quantile(.75):.1f}); "
                "the mediator post-dates the outcome."
            )

    rng = np.random.default_rng(SEED)
    all_paths, all_med = [], []
    for spec_name, age_var in [
        (SPEC_PRIMARY, PRIMARY_AGE),
        (SPEC_SENSITIVITY, SENSITIVITY_AGE),
    ]:
        p_rows, m_rows = run_specification(
            spec_name, age_var, cons, base_covs, rng
        )
        all_paths.extend(p_rows)
        all_med.extend(m_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_paths).to_csv(OUT_DIR / "paths.csv", index=False)
    pd.DataFrame(all_med).to_csv(OUT_DIR / "mediation.csv", index=False)
    _write_methods_note(len(cons))
    print(f"\n  Saved to: {OUT_DIR}")
    print(
        "  Reminder: indirect associations only; the mediator post-dates the "
        "outcome. Sensitivity results do not override the primary "
        "specification."
    )
    return 0


def _write_methods_note(n_frame):
    import datetime

    lines = [
        "Analysis 03b - Age -> Persistence -> Affect",
        "Prespecified confirmatory extension (statistical indirect effects)",
        f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Mediation sample frame: diary + fMRI conservative, N = {n_frame}",
        "",
        "COMMON COMPLETE-CASE SAMPLE",
        "  Each specification is fitted on ONE participant-level complete-case",
        "  frame. The a model, every b/c' model, every total-effect model and",
        "  the cluster bootstrap all use that identical frame, so a and b are",
        "  never estimated on different participants. Every observed model is",
        "  asserted to have been fitted on the specification's common N.",
        "  Zero-variance covariates are still pruned afterwards, which does",
        "  not change who is included.",
        "",
        "  The family id is constructed FIRST, on the full sample; the",
        "  complete-case filter is then applied to the ANALYSIS variables",
        "  (age predictor, mediator, the three outcomes, sex, race_2-race_6,",
        "  time_P2_P5, n_days_complete) plus the derived _family_id.",
        "",
        "  SAMPLMAJ and M2FAMNUM are NOT part of that filter. They are raw",
        "  inputs to the family-id construction rather than fixed effects; a",
        "  participant missing them is simply not a paired twin and is validly",
        "  assigned their own M2ID-keyed singleton cluster. Excluding them",
        "  would shrink the sample without changing any model. M2ID itself is",
        "  required: an otherwise-eligible participant with no M2ID cannot be",
        "  clustered, and the run aborts rather than dropping them silently or",
        "  coercing the identifier to the string \"nan\".",
        f"Mediator: {M_VAR}",
        "Outcomes: PA_score, NA_score (primary); NA_score_log "
        "(robustness_log_na, outcome-transformation check)",
        "",
        "SPECIFICATIONS",
        f"  {SPEC_PRIMARY} - age = {PRIMARY_AGE} (age at the neuroscience visit)",
        f"      a        {M_VAR} ~ {PRIMARY_AGE} + sex + race",
        f"      b, c'    Y ~ {M_VAR} + {PRIMARY_AGE} + sex + race + "
        "time_P2_P5 + n_days_complete",
        f"      c total  Y ~ {PRIMARY_AGE} + sex + race + time_P2_P5 + "
        "n_days_complete",
        "",
        f"  {SPEC_SENSITIVITY} - age = {SENSITIVITY_AGE} (age at the diary wave);",
        "      an age-DEFINITION sensitivity. C2PAGE replaces C5PAGE; the two",
        "      never appear in the same model.",
        f"      a        {M_VAR} ~ {SENSITIVITY_AGE} + sex + race + time_P2_P5",
        f"      b, c'    Y ~ {M_VAR} + {SENSITIVITY_AGE} + sex + race + "
        "time_P2_P5 + n_days_complete",
        f"      c total  Y ~ {SENSITIVITY_AGE} + sex + race + time_P2_P5 + "
        "n_days_complete",
        "",
        "      time_P2_P5 enters the C2PAGE a path because persistence is",
        "      measured later, at P5.",
        "",
        "Mixed-effects, random intercept for family, REML. Optimizers tried",
        "lbfgs -> powell -> nm -> bfgs; a fit is ACCEPTED ONLY IF the optimizer",
        "reports convergence. twin_pair_* dummies are excluded (family random",
        "intercept); zero-variance covariates are pruned, as in run_mlm.",
        "",
        "INDIRECT EFFECT a*b",
        f"  PRIMARY:   family-level cluster bootstrap, {N_BOOT:,} resamples,",
        f"             two-sided {CI_LEVEL:.0f}% percentile interval, seed {SEED}.",
        "             Families, not individuals, are resampled; a family drawn",
        "             twice becomes two distinct clusters. Within a replicate",
        "             the families are drawn once and the a model fitted once,",
        "             shared across all three outcomes. A replicate counts only",
        "             when BOTH its a and b models converge; at least",
        f"             {MIN_BOOT_FRACTION:.0%} of replicates must qualify or the",
        "             run aborts.",
        f"  SECONDARY: Monte Carlo distribution-of-the-product, {N_MC:,} draws.",
        "             a and b are drawn from INDEPENDENT normals, so this does",
        "             NOT incorporate cov(a, b). An approximation only.",
        "  Both intervals are two-sided; the product tests are not converted to",
        "  one-tailed inference.",
        "",
        "  c, c' and a*b come from separate mixed models and are not",
        "  guaranteed to satisfy c = c' + a*b exactly; the three estimates are",
        "  not expected to reconcile arithmetically.",
        "",
        "  indirect_sign_as_predicted records only whether the point estimate",
        "  falls in the predicted direction. It is DESCRIPTIVE and is not",
        "  evidence of a statistical effect.",
        "",
        "MULTIPLICITY AND PRECEDENCE",
        "  No multiplicity adjustment is applied across the predefined",
        "  sensitivities. Sensitivity and robustness findings do not override",
        "  the primary specification.",
        "",
        "TEMPORAL ORDER AND CAUSAL STATUS",
        "  Daily affect (P2) precedes the neuroimaging assessment (P5) that",
        "  yields the mediator, as acknowledged in the preregistration. The",
        "  measurement order is X -> Y -> M. All estimates are statistical",
        "  indirect associations and cannot establish temporal precedence or",
        "  causal mediation.",
        "",
        "OTHER CONSTRAINTS",
        "  The a path is estimated here in the diary + fMRI sample; analysis 03",
        "  reports it in the larger conservative fMRI sample (N = 127). Both",
        "  appear in paths.csv / the console log.",
        "  Complete-case N is reported per model and per specification.",
    ]
    (OUT_DIR / "_methods.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
