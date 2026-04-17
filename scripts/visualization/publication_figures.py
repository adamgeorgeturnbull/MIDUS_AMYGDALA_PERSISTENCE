#!/usr/bin/env python3
"""
publication_figures.py

Generate all three main publication figures for the MIDUS Amygdala Persistence paper.

Figure 1: ROI brain visualization (amygdala seeds + vmPFC target spheres)
Figure 2: Persistence × Affect (2-panel: PA and NA, conservative diary+fMRI sample, residualized)
Figure 3: Anterior vmPFC FC × Affect (2-panel: PA and NA, conservative diary+fMRI sample, residualized)

Run from project root directory.
"""

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import datasets, image, plotting, surface
from nilearn.image import new_img_like
from scipy import stats
from sklearn.linear_model import LinearRegression

# ============================================================================
# Paths and constants
# ============================================================================
DATA_DIR  = Path("data/processed")
FMRI_DIR  = Path("data/fMRI")
FIG_DIR   = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE = DATA_DIR / "midus_with_fmri.csv"
FC_FILE     = FMRI_DIR / "all_subjects_betaSeries_LSS_all_conditions_M2ID.csv"

FIG_DPI    = 300
FIG_FORMAT = "tiff"

# Primary variables
PERSIST_VAR = "neg_persist_crossrun_mean_z_L"
FC_VAR      = "l_amyg-ant_vmPFC_neg_vs_neu"

# ROI definitions
ANT_VMFPC_COORDS  = (-2,  46, -10)
POST_VMFPC_COORDS = ( 0,  26, -12)
SPHERE_RADIUS     = 10  # mm

# Colors
COL_L_AMYG     = "#FFE600"   # bright yellow   — left amygdala
COL_R_AMYG     = "#B8A000"   # dark gold-yellow — right amygdala
COL_ANT_VMFPC  = "#4CAF35"   # medium green    — anterior vmPFC
COL_POST_VMFPC = "#2A6B1A"   # dark forest green — posterior vmPFC

COL_PERSIST = "#A23B72"      # rose/purple for persistence scatterplots
COL_FC      = "#2E86AB"      # teal for FC scatterplots


# ============================================================================
# Data loading
# ============================================================================
def load_samples():
    """
    Load master + FC data; return (sample_persist, sample_fc).

    Both are conservative diary+fMRI samples. sample_fc additionally requires
    the primary FC variable to be non-null.
    """
    master = pd.read_csv(MASTER_FILE)
    master["M2ID"] = master["M2ID"].astype(str)

    fc = pd.read_csv(FC_FILE)
    fc["M2ID"] = fc["M2ID"].astype(str)
    df = master.merge(fc, on="M2ID", how="inner")

    # Fisher z-transform persistence if not already present
    r_var = "neg_persist_crossrun_mean_r_L"
    if r_var in df.columns and PERSIST_VAR not in df.columns:
        df[PERSIST_VAR] = np.arctanh(np.clip(df[r_var], -0.9999, 0.9999))

    has_persist = df.get("has_neg_persistence", pd.Series(0, index=df.index)) == 1
    has_qc      = df.get("qc_conservative",    pd.Series(0, index=df.index)) == 1
    has_diary   = df[["PA_score", "NA_score"]].notna().any(axis=1)
    has_fc      = df[FC_VAR].notna()

    sample_persist = df[has_persist & has_qc & has_diary].copy()
    sample_fc      = df[has_persist & has_qc & has_diary & has_fc].copy()

    print(f"  Persistence sample  N = {len(sample_persist)}")
    print(f"  FC sample           N = {len(sample_fc)}")
    return sample_persist, sample_fc


# ============================================================================
# Sphere NIfTI helper
# ============================================================================
def sphere_nifti(center_mni, radius_mm, ref_img):
    """Return a binary NIfTI image with 1s inside a sphere."""
    ref_data = ref_img.get_fdata()
    affine   = ref_img.affine

    xi, yi, zi = np.mgrid[0:ref_data.shape[0],
                           0:ref_data.shape[1],
                           0:ref_data.shape[2]]
    coords_vox = np.column_stack([xi.ravel(), yi.ravel(),
                                  zi.ravel(), np.ones(xi.size)])
    coords_mni = (affine @ coords_vox.T)[:3].T
    dist = np.sqrt(np.sum((coords_mni - np.array(center_mni)) ** 2, axis=1))
    sphere = (dist <= radius_mm).astype(np.float32).reshape(ref_data.shape[:3])
    return new_img_like(ref_img, sphere)


# ============================================================================
# Figure 1: ROI brain visualization
# ============================================================================
def make_figure1():
    """
    Single glass-brain panel showing all four ROIs as filled contours.
    display_mode='lyrz': left sagittal, axial, right sagittal, coronal.
    """
    print("\nFigure 1: ROI brain visualization...")

    atlas     = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm")
    labels    = atlas.labels
    atlas_img = atlas.maps

    l_idx = labels.index("Left Amygdala")
    r_idx = labels.index("Right Amygdala")
    l_amyg_img = image.math_img(f"img == {l_idx}", img=atlas_img)
    r_amyg_img = image.math_img(f"img == {r_idx}", img=atlas_img)

    ant_vmfpc_img  = sphere_nifti(ANT_VMFPC_COORDS,  SPHERE_RADIUS, l_amyg_img)
    post_vmfpc_img = sphere_nifti(POST_VMFPC_COORDS, SPHERE_RADIUS, l_amyg_img)

    rois = [
        (l_amyg_img,     COL_L_AMYG,     "Left Amygdala (seed)"),
        (r_amyg_img,     COL_R_AMYG,     "Right Amygdala (seed)"),
        (ant_vmfpc_img,  COL_ANT_VMFPC,  "Anterior vmPFC (target)"),
        (post_vmfpc_img, COL_POST_VMFPC, "Posterior vmPFC (target)"),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        display = plotting.plot_glass_brain(
            None,
            display_mode="lyrz",
            colorbar=False,
            plot_abs=False,
        )
        for roi_img, color, _ in rois:
            display.add_contours(roi_img, filled=True, threshold=0.5,
                                 colors=[color], alpha=0.80)

    fig = display.frame_axes.get_figure()
    fig.set_facecolor("white")
    fig.set_size_inches(12, 3.5)

    handles = [mpatches.Patch(facecolor=c, label=lbl, edgecolor="none")
               for _, c, lbl in rois]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.08))

    out = FIG_DIR / f"figure1_roi.{FIG_FORMAT}"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Residualization helper
# ============================================================================
COVARIATES = ["C5PAGE", "sex", "n_days_complete", "time_P2_P5"]

def residualize(series, cov_df):
    """Partial out covariates from series via OLS; return residuals."""
    idx   = series.dropna().index.intersection(cov_df.dropna().index)
    y     = series.loc[idx].values
    X     = cov_df.loc[idx].values
    resid = y - LinearRegression().fit(X, y).predict(X)
    out   = pd.Series(np.nan, index=series.index)
    out.loc[idx] = resid
    return out


def residualize_pair(data, x_var, y_var):
    """Return (x_resid, y_resid) arrays after partialling out covariates."""
    cov_cols = [c for c in COVARIATES if c in data.columns]
    # add race dummies
    cov_cols += [c for c in data.columns if c.startswith("race_")]
    cols_needed = [x_var, y_var] + cov_cols
    d = data[cols_needed].dropna()
    cov_df = d[cov_cols]
    x_resid = residualize(d[x_var], cov_df)
    y_resid = residualize(d[y_var], cov_df)
    return x_resid.dropna().values, y_resid.dropna().values


# ============================================================================
# Scatterplot helper
# ============================================================================
def scatter_panel(ax, data, x_var, y_var, color, x_label, y_label):
    """
    Single scatter panel with residualized data: points, regression line,
    95% CI, and partial r annotation. Covariates partialled out before plotting.
    """
    x, y = residualize_pair(data, x_var, y_var)
    n = len(x)

    slope, intercept, r, _, _ = stats.linregress(x, y)

    # Regression line + 95% CI
    x_pred  = np.linspace(x.min(), x.max(), 200)
    y_pred  = slope * x_pred + intercept
    se_res  = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
    margin  = 1.96 * se_res * np.sqrt(
        1 / n + (x_pred - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    )

    ax.scatter(x, y, alpha=0.45, s=22, color=color, edgecolors="none", zorder=2)
    ax.plot(x_pred, y_pred, color=color, linewidth=2, zorder=3)
    ax.fill_between(x_pred, y_pred - margin, y_pred + margin,
                    alpha=0.18, color=color, zorder=1)

    ax.text(0.05, 0.95,
            f"partial r = {r:.2f}\nn = {n}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75))

    ax.set_xlabel(f"{x_label}\n(residualized)", fontsize=10)
    ax.set_ylabel(f"{y_label}\n(residualized)", fontsize=10)
    sns.despine(ax=ax)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


# ============================================================================
# Figure 2: Persistence × Affect
# ============================================================================
def make_figure2(sample):
    """Two-panel: left amygdala persistence → PA and NA (residualized)."""
    print("\nFigure 2: Persistence × Affect...")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor="white")

    panels = [
        ("PA_score",     "Positive Affect (Daily Diary)"),
        ("NA_score", "Negative Affect (Daily Diary)"),
    ]

    for ax, (y_var, y_label) in zip(axes, panels):
        scatter_panel(ax, sample, PERSIST_VAR, y_var, COL_PERSIST,
                      "Left Amygdala Negative Persistence (Fisher z)", y_label)

    for ax, letter in zip(axes, "AB"):
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=12)

    fig.tight_layout()
    out = FIG_DIR / f"figure2_persistence_affect.{FIG_FORMAT}"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Figure 3: FC × Affect
# ============================================================================
def make_figure3(sample):
    """Two-panel: left anterior vmPFC FC → PA and NA (residualized)."""
    print("\nFigure 3: FC × Affect...")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor="white")

    panels = [
        ("PA_score",     "Positive Affect (Daily Diary)"),
        ("NA_score", "Negative Affect (Daily Diary)"),
    ]

    for ax, (y_var, y_label) in zip(axes, panels):
        scatter_panel(ax, sample, FC_VAR, y_var, COL_FC,
                      "L Amygdala–Anterior vmPFC FC\n(neg−neu contrast, Fisher z)",
                      y_label)

    for ax, letter in zip(axes, "AB"):
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=12)

    fig.tight_layout()
    out = FIG_DIR / f"figure3_fc_affect.{FIG_FORMAT}"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Publication Figures — MIDUS Amygdala Persistence")
    print("=" * 70)

    make_figure1()

    print("\nLoading data...")
    sample_persist, sample_fc = load_samples()

    make_figure2(sample_persist)
    make_figure3(sample_fc)

    print(f"\nDone. All figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
