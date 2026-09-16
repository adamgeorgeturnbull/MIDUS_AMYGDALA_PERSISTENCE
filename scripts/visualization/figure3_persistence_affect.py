#!/usr/bin/env python3
"""
figure3_persistence_affect.py

Figure 3 for the MIDUS Amygdala Persistence manuscript: the primary M3
association between left-amygdala negative persistence and daily affect.

    Panel A  Positive affect
    Panel B  Negative affect

Both outcomes are shown for transparency even though only the negative-affect
mixed-effects result reaches the prespecified threshold. Log-transformed
negative affect is deliberately excluded; it remains a robustness result.
Neither panel is labelled "significant" or "nonsignificant" - the annotated
p-values speak for themselves.

Authoritative analysis
----------------------
This script does NOT reconstruct the analysis. It imports
scripts/analysis/analysis_utils.py and calls the same functions that
scripts/analysis/02_persistence_affect.py uses, so the sample, transformation,
covariate set, family grouping, and model are identical by construction:

    load_master()                 -> master behavioural + fMRI table
    prepare_persistence_vars(df)  -> creates neg_persist_crossrun_mean_z_L
    get_samples(df)               -> (full, conservative); the conservative
                                     diary + fMRI sample is used
    get_covariates(cons)          -> C5PAGE, sex, race_*, twin_pair_*
    run_mlm(...)                  -> the authoritative MLM recomputation

`run_analysis_set` adds the diary covariates (time_P2_P5, n_days_complete) for
outcomes in DIARY_OUTCOMES; this script adds them the same way, so the
covariate vector handed to run_mlm matches the analysis exactly.

Note two behaviours inherited from run_mlm that matter for interpretation:

  - twin_pair_* dummies are returned by get_covariates but are EXCLUDED from
    the fixed effects by run_mlm, because the family random intercept replaces
    them. They are therefore absent from the fitted model here too.
  - run_mlm drops zero-variance covariates before fitting. In the N = 81
    conservative sample some race dummies have no members and are dropped.
    That is a property of the authoritative analysis, not a deviation from it;
    the dropped columns are reported at run time.

Model: outcome ~ predictor + covariates + (1 | family), REML, optimizers tried
in the order lbfgs, powell, nm, bfgs. Family grouping: paired twins
(SAMPLMAJ == 3 with more than one participant sharing M2FAMNUM) share a family
id; everyone else is their own cluster keyed by M2ID.

Inference is one-tailed with prespecified directions: negative for PA_score,
positive for NA_score.

Validation
----------
Three independent checks must all pass before anything is drawn:

  1. results/tables/02_persistence_affect/mlm.csv is read and its PA_score and
     NA_score rows are compared against the frozen expected values below.
  2. run_mlm is re-run on the rebuilt sample and its beta/SE/z/p are compared
     against the stored CSV.
  3. The model is refitted here (to obtain the fitted object needed for partial
     residuals) using the same construction, and its coefficients are compared
     against both run_mlm and the CSV.

Any mismatch, non-convergence, degenerate random effect, or unexpected N aborts
with an informative message. The script never silently falls back to a
different sample or model.

Plotted values
--------------
The displayed relationship represents the mixed-effects model. No correlation,
OLS slope, or OLS p-value is computed or shown.

For each outcome, the plotted y value is a mixed-model partial residual:

    adjusted_i = mean(y)
               + (y_i - Xb_i - u_family(i))        <- conditional residual
               + beta_predictor * (x_i - mean(x))

Xb_i is the fixed-effect prediction and u_family(i) the fitted (BLUP) family
random intercept, so the other fixed effects and the family effect are removed
while the fitted predictor association is retained, on the original outcome
scale. The plotted line is the fitted fixed-effect line implied by the
validated beta, y = mean(y) + beta * (x - mean(x)), which is exactly the
regression line of the plotted points.

No confidence band is drawn. A band would have to be derived analytically from
the fitted mixed model; rather than risk an ambiguous or subtly wrong
interval, it is omitted.

Privacy and scope
-----------------
Participant-level data are read only when this script is executed in the
authoritative Box project. Console output is limited to aggregate N values,
validation status, and output paths. No participant record or identifier is
printed.

Outputs
-------
    results/figures/figure3_persistence_affect.svg
    results/figures/figure3_persistence_affect.tiff   (300 dpi, LZW)

Run from the project root directory.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save-only; never opens a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The analysis package is the single source of truth for sample and model logic.
_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

try:
    import statsmodels.formula.api as smf
    from analysis_utils import (
        DIARY_OUTCOMES,
        MIN_N,
        RESULTS_DIR,
        get_covariates,
        get_samples,
        load_master,
        prepare_persistence_vars,
        run_mlm,
    )
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "ERROR: could not import the analysis utilities or statsmodels.\n"
        f"  Underlying error: {type(exc).__name__}: {exc}\n"
        f"  Expected analysis_utils.py in: {_ANALYSIS_DIR}\n"
        "  Run this script from the project root directory."
    ) from exc


# ============================================================================
# Configuration
# ============================================================================
MLM_CSV = RESULTS_DIR / "02_persistence_affect" / "mlm.csv"
OUT_DIR = Path("results/figures")
OUT_STEM = "figure3_persistence_affect"
FIG_DPI = 300

PREDICTOR = "neg_persist_crossrun_mean_z_L"

# (outcome, panel letter, panel title, y label, expected direction)
PANELS = [
    ("PA_score", "A", "Positive affect", "Adjusted positive affect", -1),
    ("NA_score", "B", "Negative affect", "Adjusted negative affect", +1),
]

X_LABEL = "Left-amygdala negative persistence (Fisher z)"
FIG_NOTE = "Adjusted mixed-effects partial residuals"

EXPECTED_N = 81

# Frozen authoritative values. Used for validation only; every annotated number
# is taken from the loaded CSV, not from this table.
EXPECTED_MLM = {
    "PA_score": {
        "n": 81,
        "beta": -0.6995706141184733,
        "se": 0.4760595215276992,
        "z": -1.4695024098531118,
        "p": 0.07084828433185175,
    },
    "NA_score": {
        "n": 81,
        "beta": 0.27792707372427095,
        "se": 0.1357241284390131,
        "z": 2.047735188435238,
        "p": 0.020292976965517906,
    },
}

REQUIRED_MLM_COLS = [
    "predictor",
    "outcome",
    "n",
    "beta",
    "se",
    "z",
    "p",
    "converged",
    "degenerate_re",
]

# Covariates the fitted model is expected to be offered (before run_mlm's
# zero-variance pruning). Presence as columns is required; survival into the
# final design is not, because pruning is part of the authoritative analysis.
EXPECTED_COVARIATES = [
    "C5PAGE",
    "sex",
    "race_2",
    "race_3",
    "race_4",
    "race_5",
    "race_6",
    "time_P2_P5",
    "n_days_complete",
]

# Tolerances: the CSV should reproduce the frozen constants essentially
# exactly; a refit may differ in the last few digits because of optimizer and
# BLAS variation, so it is checked slightly more loosely.
TOL_STORED_ABS = 1e-9
TOL_REFIT_REL = 1e-5
TOL_REFIT_ABS = 1e-8

# ============================================================================
# Appearance — gold / dark gold, consistent with Figure 1's amygdala colour
# ============================================================================
GOLD_FILL = "#D9BE00"  # mid gold — point faces
GOLD_LINE = "#8A7500"  # dark gold — point edges and fitted line
INK = "#1A1A1A"
INK_SOFT = "#5A5A5A"

FS_LETTER = 10.0
FS_TITLE = 8.5
FS_LABEL = 7.5
FS_ANNOT = 7.0
FS_NOTE = 7.0

FIG_W, FIG_H = 7.25, 3.10

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FS_LABEL,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": FS_ANNOT,
        "ytick.labelsize": FS_ANNOT,
        "svg.fonttype": "none",  # editable SVG text
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


# ============================================================================
# Helpers
# ============================================================================
def _fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def _close(got, want, rel=TOL_REFIT_REL, abs_=TOL_REFIT_ABS):
    if got is None or want is None:
        return False
    if not (np.isfinite(got) and np.isfinite(want)):
        return False
    return abs(got - want) <= max(abs_, rel * abs(want))


def fmt_p(p):
    """Three decimals, no leading zero; '< .001' below the printable floor."""
    if not np.isfinite(p):
        _fail("a p-value to be annotated is not finite.")
    if p < 0.0005:
        return "< .001"
    return f"{p:.3f}".lstrip("0")


def annotation_lines(row):
    """Compact annotation block, all values taken from the loaded CSV row."""
    return "\n".join(
        [
            rf"$\beta$ = {row['beta']:.3f}",
            f"SE = {row['se']:.3f}",
            rf"$z$ = {row['z']:.2f}",
            rf"$p$(one-tailed) = {fmt_p(row['p'])}",
            f"N = {int(row['n'])}",
        ]
    )


# ============================================================================
# Stored aggregate results
# ============================================================================
def load_stored_mlm():
    """Read and validate the authoritative MLM table for the two outcomes."""
    if not MLM_CSV.exists():
        _fail(
            f"required aggregate result table {MLM_CSV} not found.\n"
            "  Run scripts/analysis/02_persistence_affect.py first, or run this\n"
            "  script from the project root of the authoritative project."
        )
    try:
        df = pd.read_csv(MLM_CSV)
    except Exception as exc:
        _fail(f"could not parse {MLM_CSV}: {type(exc).__name__}: {exc}")

    missing = [c for c in REQUIRED_MLM_COLS if c not in df.columns]
    if missing:
        _fail(
            f"{MLM_CSV} is missing required column(s): {missing}\n"
            f"  Columns present: {list(df.columns)}"
        )

    stored = {}
    for outcome, exp in EXPECTED_MLM.items():
        sel = df[(df["predictor"] == PREDICTOR) & (df["outcome"] == outcome)]
        if len(sel) == 0:
            _fail(
                f"{MLM_CSV} has no row for predictor '{PREDICTOR}' and outcome "
                f"'{outcome}'."
            )
        if len(sel) > 1:
            _fail(
                f"{MLM_CSV} has {len(sel)} rows for predictor '{PREDICTOR}' and "
                f"outcome '{outcome}'; expected exactly one."
            )
        row = sel.iloc[0]

        if int(row["n"]) != exp["n"]:
            _fail(
                f"stored N for '{outcome}' is {int(row['n'])}, expected "
                f"{exp['n']}. The analytic sample has changed."
            )
        if not bool(row["converged"]):
            _fail(f"the stored MLM for '{outcome}' is marked not converged.")
        if bool(row["degenerate_re"]):
            _fail(
                f"the stored MLM for '{outcome}' is marked as having a "
                "degenerate random effect; its SE is not trustworthy."
            )
        for key in ("beta", "se", "z", "p"):
            if not _close(float(row[key]), exp[key], rel=0.0, abs_=TOL_STORED_ABS):
                _fail(
                    f"stored {key} for '{outcome}' is {float(row[key])!r} but the "
                    f"frozen expected value is {exp[key]!r}.\n"
                    f"  {MLM_CSV} does not match the manuscript results."
                )
        stored[outcome] = row

    return stored


# ============================================================================
# Sample + model
# ============================================================================
def build_sample():
    """Rebuild the conservative diary + fMRI sample via the analysis utilities."""
    try:
        df = load_master()
    except FileNotFoundError as exc:
        _fail(
            "the master analytic table could not be found.\n"
            f"  Underlying error: {exc}\n"
            "  This script must run from the project root of the authoritative\n"
            "  project, where data/processed/midus_with_fmri.csv exists."
        )

    prepare_persistence_vars(df)
    if PREDICTOR not in df.columns:
        _fail(
            f"predictor '{PREDICTOR}' is absent after "
            "prepare_persistence_vars(); the master table lacks the "
            "cross-run persistence columns."
        )

    _full, cons = get_samples(df)
    if len(cons) != EXPECTED_N:
        _fail(
            f"the conservative diary + fMRI sample has N = {len(cons)}, "
            f"expected {EXPECTED_N}.\n"
            "  The sample definition or the master table has changed; refusing "
            "to plot a different sample."
        )

    # run_analysis_set appends the diary covariates only for outcomes in
    # DIARY_OUTCOMES, so verify that assumption still holds before mirroring it.
    non_diary = [o for o, *_ in PANELS if o not in DIARY_OUTCOMES]
    if non_diary:
        _fail(
            f"outcome(s) {non_diary} are not in analysis_utils.DIARY_OUTCOMES, "
            "so run_analysis_set would not add the diary covariates "
            "(time_P2_P5, n_days_complete) for them. Mirroring that behaviour "
            "here would no longer reproduce the authoritative model."
        )

    base_covs = get_covariates(cons)
    diary_covs = [c for c in ["time_P2_P5", "n_days_complete"] if c in cons.columns]
    covariates = base_covs + diary_covs  # matches run_analysis_set for diary outcomes

    absent = [c for c in EXPECTED_COVARIATES if c not in cons.columns]
    if absent:
        _fail(
            f"expected covariate column(s) absent from the sample: {absent}\n"
            "  Refusing to fit a model with a different covariate set."
        )
    return cons, covariates


def refit_mlm(sample, outcome, covariates):
    """
    Refit the model exactly as analysis_utils.run_mlm builds it, returning the
    fitted object plus the analysis frame, which run_mlm does not expose.

    Every construction step below mirrors run_mlm; the resulting coefficients
    are cross-checked against run_mlm's own output by the caller.
    """
    df = sample.copy()

    # --- family grouping (identical to run_mlm) ---------------------------
    if "SAMPLMAJ" in df.columns and "M2FAMNUM" in df.columns:
        is_twin = df["SAMPLMAJ"] == 3
        paired = df.loc[is_twin, "M2FAMNUM"].value_counts()
        paired = paired[paired > 1].index
        df["_family_id"] = df["M2ID"].astype(str)
        mask = is_twin & df["M2FAMNUM"].isin(paired)
        df.loc[mask, "_family_id"] = (
            "fam_" + df.loc[mask, "M2FAMNUM"].astype(int).astype(str)
        )
    elif "M2FAMNUM" in df.columns:
        df["_family_id"] = df["M2FAMNUM"].astype(str)
    else:
        df["_family_id"] = df["M2ID"].astype(str)

    # --- formula-safe predictor name (identical to run_mlm) ---------------
    pred_safe = PREDICTOR.replace("-", "_").replace(".", "_")
    if pred_safe != PREDICTOR:
        df[pred_safe] = df[PREDICTOR]

    # --- covariate pruning (identical to run_mlm) -------------------------
    cov_list = [
        c
        for c in covariates
        if c in df.columns
        and c != PREDICTOR
        and c != pred_safe
        and not c.startswith("twin_pair_")
    ]
    cols = [outcome, pred_safe, "_family_id"] + cov_list
    data = df[cols].dropna()
    if len(data) < MIN_N:
        _fail(
            f"only {len(data)} complete cases for '{outcome}' after listwise "
            f"deletion; below MIN_N = {MIN_N}."
        )

    offered = list(cov_list)
    cov_list = [c for c in cov_list if data[c].std() > 0]
    dropped = [c for c in offered if c not in cov_list]

    cov_terms = " + ".join(cov_list)
    fixed = f"{outcome} ~ {pred_safe}" + (f" + {cov_terms}" if cov_terms else "")

    # --- fit, trying the same optimizers in the same order ----------------
    result = None
    optimizer = None
    for method in ["lbfgs", "powell", "nm", "bfgs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdl = smf.mixedlm(fixed, data=data, groups=data["_family_id"])
                result = mdl.fit(reml=True, method=method)
                optimizer = method
                break
        except Exception:
            continue
    if result is None:
        _fail(f"all optimizers failed when refitting the model for '{outcome}'.")
    if not result.converged:
        _fail(f"the refitted model for '{outcome}' did not converge.")
    if int(result.nobs) != len(data):
        _fail(
            f"the refitted model for '{outcome}' used {int(result.nobs)} "
            f"observations but the analysis frame has {len(data)} rows; "
            "row alignment for partial residuals cannot be guaranteed."
        )
    if int(result.nobs) != EXPECTED_N:
        _fail(
            f"the refitted model for '{outcome}' used {int(result.nobs)} "
            f"observations, expected {EXPECTED_N}."
        )
    return result, data, pred_safe, dropped, optimizer


def partial_residuals(result, data, outcome, pred_safe, beta):
    """
    Mixed-model partial residuals on the original outcome scale:

        mean(y) + (y - Xb - u_family) + beta * (x - mean(x))

    Xb is computed explicitly as exog @ fe_params and the family BLUP is
    subtracted explicitly, rather than relying on the version-dependent
    semantics of MixedLMResults.fittedvalues / .resid (which include the
    random effects in current statsmodels but did not always).
    """
    fe_index = list(result.fe_params.index)
    exog_names = list(result.model.exog_names)
    if fe_index != exog_names:
        _fail(
            "fixed-effect parameter order does not match the design matrix "
            "column order; cannot compute Xb safely."
        )

    exog = np.asarray(result.model.exog, dtype=float)
    fe = np.asarray(result.fe_params, dtype=float)
    if exog.shape[0] != len(data):
        _fail("design matrix row count does not match the analysis frame.")
    xb = exog @ fe

    # Family random intercept (BLUP) per row.
    re_map = result.random_effects
    u = np.empty(len(data), dtype=float)
    for i, grp in enumerate(data["_family_id"].to_numpy()):
        if grp not in re_map:
            _fail("a family group is missing from the fitted random effects.")
        vals = np.asarray(re_map[grp], dtype=float).ravel()
        if vals.size != 1:
            _fail(
                f"expected a single random intercept per family but found "
                f"{vals.size} random-effect terms; the model is not the "
                "random-intercept model this figure assumes."
            )
        u[i] = vals[0]

    y = data[outcome].to_numpy(dtype=float)
    x = data[pred_safe].to_numpy(dtype=float)
    resid_cond = y - xb - u
    return y.mean() + resid_cond + beta * (x - x.mean()), x


# ============================================================================
# Drawing
# ============================================================================
def draw_panel(ax, letter, title, y_label, x, y_adj, beta, y_mean, annot):
    ax.scatter(
        x,
        y_adj,
        s=13,
        facecolor=GOLD_FILL,
        edgecolor=GOLD_LINE,
        linewidth=0.4,
        alpha=0.55,
        zorder=3,
    )
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(
        xs,
        y_mean + beta * (xs - float(np.mean(x))),
        color=GOLD_LINE,
        linewidth=1.6,
        zorder=4,
    )

    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=4)
    ax.set_xlabel(X_LABEL, fontsize=FS_LABEL, labelpad=3)
    ax.set_ylabel(y_label, fontsize=FS_LABEL, labelpad=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(width=0.7, length=2.5)

    # The negative PA slope occupies the upper-left corner; its annotation is
    # placed in the open lower-left region.  The NA panel remains upper-left.
    annot_y = 0.035 if letter == "A" else 0.975
    annot_va = "bottom" if letter == "A" else "top"
    ax.text(
        0.035,
        annot_y,
        annot,
        transform=ax.transAxes,
        fontsize=FS_ANNOT,
        va=annot_va,
        ha="left",
        color=INK,
        linespacing=1.45,
    )
    ax.text(
        -0.155,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=FS_LETTER,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )


def build_figure(panel_data):
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H), facecolor="white"
    )
    for ax, pd_ in zip(axes, panel_data):
        draw_panel(
            ax,
            pd_["letter"],
            pd_["title"],
            pd_["y_label"],
            pd_["x"],
            pd_["y_adj"],
            pd_["beta"],
            pd_["y_mean"],
            pd_["annot"],
        )
    fig.text(
        0.008,
        0.012,
        FIG_NOTE,
        fontsize=FS_NOTE,
        ha="left",
        va="bottom",
        color=INK_SOFT,
        style="italic",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.905, bottom=0.215, wspace=0.28)
    return fig


def save_figure(fig):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    svg_path = OUT_DIR / f"{OUT_STEM}.svg"
    fig.savefig(svg_path, format="svg", facecolor="white")
    print(f"  Saved: {svg_path}")

    tiff_path = OUT_DIR / f"{OUT_STEM}.tiff"
    try:
        fig.savefig(
            tiff_path,
            format="tiff",
            dpi=FIG_DPI,
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except (ImportError, ValueError) as exc:
        raise SystemExit(
            f"ERROR: could not write the 300 dpi TIFF to {tiff_path}.\n"
            f"  Underlying error: {type(exc).__name__}: {exc}\n"
            "  Matplotlib needs Pillow for TIFF output:  pip install Pillow"
        ) from exc
    print(f"  Saved: {tiff_path} ({FIG_DPI} dpi, LZW)")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 72)
    print("Figure 3 — persistence → daily affect (M3 primary)")
    print("=" * 72)

    print(f"Stored results: {MLM_CSV}")
    stored = load_stored_mlm()
    print("  [ok] stored MLM rows match the frozen expected values")

    sample, covariates = build_sample()
    n_cov = len([c for c in covariates if not c.startswith("twin_pair_")])
    print(f"Sample: conservative diary + fMRI, N = {len(sample)}")
    print(f"  fixed-effect covariates offered: {n_cov} (twin_pair_* excluded by run_mlm)")

    panel_data = []
    for outcome, letter, title, y_label, direction in PANELS:
        row = stored[outcome]

        # (2) authoritative recomputation through run_mlm itself
        recomputed = run_mlm(
            sample,
            PREDICTOR,
            outcome,
            covariates,
            one_tailed=True,
            expected_positive=direction > 0,
        )
        if recomputed is None:
            _fail(f"analysis_utils.run_mlm returned no result for '{outcome}'.")
        if int(recomputed["n"]) != EXPECTED_N:
            _fail(
                f"run_mlm used N = {int(recomputed['n'])} for '{outcome}', "
                f"expected {EXPECTED_N}."
            )
        if not recomputed["converged"]:
            _fail(f"run_mlm reports non-convergence for '{outcome}'.")
        if recomputed["degenerate_re"]:
            _fail(f"run_mlm reports a degenerate random effect for '{outcome}'.")
        for key in ("beta", "se", "z", "p"):
            if not _close(float(recomputed[key]), float(row[key])):
                _fail(
                    f"recomputed {key} for '{outcome}' "
                    f"({float(recomputed[key])!r}) does not match the stored "
                    f"value ({float(row[key])!r}) within tolerance."
                )

        # (3) local refit, to obtain the fitted object for partial residuals
        result, data, pred_safe, dropped, optimizer = refit_mlm(
            sample, outcome, covariates
        )
        beta = float(result.fe_params[pred_safe])
        se = float(result.bse_fe.get(pred_safe, np.nan))
        if not _close(beta, float(row["beta"])):
            _fail(
                f"refitted beta for '{outcome}' ({beta!r}) does not match the "
                f"stored value ({float(row['beta'])!r}) within tolerance."
            )
        if not _close(se, float(row["se"])):
            _fail(
                f"refitted SE for '{outcome}' ({se!r}) does not match the "
                f"stored value ({float(row['se'])!r}) within tolerance."
            )

        y_adj, x = partial_residuals(result, data, outcome, pred_safe, beta)
        y_mean = float(data[outcome].to_numpy(dtype=float).mean())

        print(
            f"  [ok] {outcome}: N = {int(result.nobs)}, optimizer = {optimizer}"
            + (f", zero-variance covariates dropped: {len(dropped)}" if dropped else "")
        )

        panel_data.append(
            {
                "letter": letter,
                "title": title,
                "y_label": y_label,
                "x": x,
                "y_adj": y_adj,
                "beta": float(row["beta"]),  # annotate/draw the stored value
                "y_mean": y_mean,
                "annot": annotation_lines(row),
            }
        )

    print("Validation complete: stored, recomputed, and refitted models agree.")
    print(
        f"Canvas: {FIG_W:.2f} x {FIG_H:.2f} in  "
        f"({round(FIG_W * FIG_DPI)} x {round(FIG_H * FIG_DPI)} px at {FIG_DPI} dpi)"
    )

    fig = build_figure(panel_data)
    save_figure(fig)
    plt.close(fig)
    print("\nDone. Figure 3 is fully assembled; no manual composition required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
