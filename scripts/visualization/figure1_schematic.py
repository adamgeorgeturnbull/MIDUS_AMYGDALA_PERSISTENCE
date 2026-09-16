#!/usr/bin/env python3
"""
figure1_schematic.py

Figure 1 for the MIDUS Amygdala Persistence manuscript: a single, fully
assembled, publication-ready landscape schematic at journal column width. No
manual assembly step is required.

Layout
------
A 2x2 outer grid with unequal columns, so reading order runs A -> B -> C -> D:

    +--------------+--------------------------------------+
    | A. Task      | B. Amygdala persistence              |
    +--------------+--------------------------------------+
    | C. LSS FC    | D. Hypothesized relationships        |
    +--------------+--------------------------------------+

Left column 31% of usable width, right column 69%; top row 43% of usable
height, bottom row 57%. The figure carries no global title: the manuscript
caption supplies the figure number and title.

Panels
------
A. Task structure
     Three representative trials stacked vertically - negative, neutral, and
     positive image, each followed by a neutral face. All face stimuli were
     neutral; the figure therefore never uses "negative face" / "positive face"
     wording and instead says "neutral face following a negative image".

B. Amygdala neural persistence
     Voxelwise left-amygdala beta pattern for a negative image in one run,
     spatially correlated with the pattern for the subsequent neutral face in a
     *different* run. With three runs there are six directional cross-run pairs
     (run i != run j); the pairwise correlations are averaged in Fisher-z space.
     The primary measure is left-amygdala negative persistence.

C. LSS beta-series connectivity, as a compact vertical pipeline
     image-trial strip -> paired amygdala / vmPFC beta series -> seed-target
     correlation scatter -> Fisher-z -> anatomical ROI inset with labels.

     The trial strip is truncated for space and ends in an ellipsis; the number
     of bars drawn is a schematic, NOT the actual trial count (the task
     presented 10 images per valence per run).

     Full LSS detail, kept here rather than on the figure: for the condition of
     interest the trial of interest is modelled as its own regressor, the
     remaining trials of that same condition are collapsed into a single
     "<cond>_others" regressor, and every other condition is collapsed into one
     "<cond>_all" regressor per condition. Each image event is modelled with a
     6.0 s duration. Trial-level image betas are correlated between the
     left-amygdala seed and the vmPFC target within each run and valence,
     Fisher-z transformed, and averaged across runs. Following-face estimates
     are NOT included in the connectivity calculation. The primary contrast is
     negative minus neutral connectivity between the left amygdala and the
     anterior vmPFC. (See scripts/fMRI/analysis/seedbasedBStaskFC_LSS.sh.)

     The ROI inset is a glass-brain axial projection carrying the actual
     Harvard-Oxford amygdala masks (50% probability threshold) and the actual
     10 mm vmPFC spheres. Anterior and posterior vmPFC are co-primary targets
     and are given equal visual prominence, distinguished only by green shade.
     The right amygdala is the sensitivity region and is the one ROI drawn less
     prominently. Sphere radius and MNI coordinates are recorded below rather
     than on the figure; they belong in the manuscript caption. The atlas is
     fetched exactly once (load_atlas_bundle) and shared with panel B.

D. Hypothesized relationships, as a 2x2 grid
     Rows are the neural predictor, columns the diary outcome:
         persistence   -> PA : negative      persistence  -> NA : positive
         connectivity  -> PA : positive      connectivity -> NA : negative

Everything shown is a deterministic simulated schematic. This script reads no
participant-level file, no result CSV, and no identifier. Its only external
input is the Harvard-Oxford atlas, fetched through nilearn.

Scientific definitions and ROI definitions are reused from the established
project code:
  - Harvard-Oxford "sub-maxprob-thr50-2mm", labels "Left Amygdala" /
    "Right Amygdala"
      (scripts/visualization/amygdala_voxel_schematic.py,
       scripts/visualization/publication_figures.py)
  - Anterior vmPFC sphere  MNI (-2, 46, -10), radius 10 mm
  - Posterior vmPFC sphere MNI ( 0, 26, -12), radius 10 mm
  - sphere_nifti() and the plot_glass_brain + add_contours inset approach
      (scripts/visualization/publication_figures.py)
  - LSS event construction and the 6.0 s modeled image-event duration
      (scripts/fMRI/analysis/seedbasedBStaskFC_LSS.sh)
  - Cross-run image-to-face persistence with Fisher-z averaging
      (scripts/fMRI/analysis/run_cross_corr.py)

Stimulus presentation durations are deliberately NOT depicted. The project code
is internally inconsistent about how the 6.0 s modeled image event decomposes
into image and fixation time (runBStaskFC.sh comments say "2s image + 4s
fixation"), so no trial timings are asserted here beyond the 6 s modeled
regressor duration that the LSS code actually sets.

No visible text is smaller than 7 pt.

Outputs
-------
    results/figures/figure1_schematic.svg
    results/figures/figure1_schematic.tiff   (300 dpi)

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
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ============================================================================
# Output
# ============================================================================
OUT_DIR = Path("results/figures")
OUT_STEM = "figure1_schematic"
FIG_DPI = 300

# ============================================================================
# Deterministic simulation
# ============================================================================
SEED = 42  # matches the seed used in the existing schematic scripts

# ============================================================================
# ROI definitions (reused from publication_figures.py)
# ============================================================================
HO_ATLAS = "sub-maxprob-thr50-2mm"
L_AMYG_LABEL = "Left Amygdala"
R_AMYG_LABEL = "Right Amygdala"

ANT_VMPFC_COORDS = (-2, 46, -10)
POST_VMPFC_COORDS = (0, 26, -12)
SPHERE_RADIUS = 10  # mm

# ============================================================================
# Colors
#   Amygdala = gold/yellow, vmPFC = green, used consistently in every panel.
# ============================================================================
AMYG_FILL = "#FFE600"  # bright yellow  — left amygdala
AMYG_MID = "#D9BE00"  # mid gold        — amygdala beta series
AMYG_MUTED = "#B8A000"  # dark gold-yellow — right amygdala (publication_figures)
AMYG_LINE = "#8A7500"  # dark gold      — amygdala strokes / text

VMPFC_FILL = "#4CAF35"  # medium green    — anterior vmPFC
VMPFC_LINE = "#2A6B1A"  # dark green      — posterior vmPFC / strokes

# Valence colors (from lss_schematic.py); red / grey / blue is
# colorblind-distinguishable and is used only for image valence.
COL_NEG = "#E8735A"
COL_NEU = "#AAAAAA"
COL_POS = "#5B8DB8"
COL_VALENCE = {"neg": COL_NEG, "neu": COL_NEU, "pos": COL_POS}
VALENCE_NAME = {"neg": "Negative", "neu": "Neutral", "pos": "Positive"}
DIM = 0.30  # alpha for non-focal valences (lss_schematic.py convention)

INK = "#1A1A1A"
INK_SOFT = "#5A5A5A"
HAIRLINE = "#BBBBBB"

# Diverging colormap for simulated voxel / trial betas (colorblind-safe)
CMAP = plt.get_cmap("RdBu_r")
NORM = Normalize(vmin=-1, vmax=1)

# ============================================================================
# Typography — nothing below 7 pt
# ============================================================================
FS_LETTER = 10.0  # panel letters
FS_TITLE = 8.5  # panel titles
FS_HEAD = 7.5  # column / axis headers
FS_BODY = 7.2  # body labels
FS_SMALL = 7.0  # notes
FS_TINY = 7.0  # smallest permitted

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FS_BODY,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.7,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": FS_TINY,
        "ytick.labelsize": FS_TINY,
        "legend.frameon": False,
        "svg.fonttype": "none",  # keep SVG text editable
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

# ============================================================================
# Figure geometry
#
# Landscape, journal full-page width. Panels A, B and C are equal-aspect
# drawing canvases, so each one's data-coordinate aspect must match the
# physical aspect of its grid cell or matplotlib letterboxes it. The cell
# sizes are therefore computed from the layout fractions below and the data
# limits are derived from them, rather than hand-tuned.
# ============================================================================
FIG_W, FIG_H = 7.25, 5.80
# Bottom margin must clear the two-line x labels on panel D's lower row.
LEFT, RIGHT, TOP, BOTTOM = 0.045, 0.988, 0.975, 0.058

COL_RATIOS = [0.31, 0.69]  # left / right column
ROW_RATIOS = [0.43, 0.57]  # top / bottom row
# WSPACE 0.228 puts ~0.70 in between the A/C and B/D columns (0.50 in more
# than the original 0.06), clearing the panel A/C right edge from panel B/D.
WSPACE, HSPACE = 0.228, 0.10

USABLE_W = (RIGHT - LEFT) * FIG_W
USABLE_H = (TOP - BOTTOM) * FIG_H


def _cell_sizes():
    """Physical (inches) width of each column and height of each row."""
    avg_w = sum(COL_RATIOS) / len(COL_RATIOS)
    unit_w = USABLE_W / (sum(COL_RATIOS) + (len(COL_RATIOS) - 1) * WSPACE * avg_w)
    avg_h = sum(ROW_RATIOS) / len(ROW_RATIOS)
    unit_h = USABLE_H / (sum(ROW_RATIOS) + (len(ROW_RATIOS) - 1) * HSPACE * avg_h)
    return [r * unit_w for r in COL_RATIOS], [r * unit_h for r in ROW_RATIOS]


CELL_W, CELL_H = _cell_sizes()

ASPECT_A = CELL_W[0] / CELL_H[0]
ASPECT_B = CELL_W[1] / CELL_H[0]
ASPECT_C = CELL_W[0] / CELL_H[1]

# Data limits: x span chosen for convenience, y span derived from the aspect.
A_XMAX = 10.0
A_YMAX = A_XMAX / ASPECT_A
B_XMAX = 12.0
B_YMAX = B_XMAX / ASPECT_B


# ============================================================================
# Small helpers
# ============================================================================
def panel_letter(ax, letter, x=-0.02, y=0.99):
    """Place a bold panel letter in axes coordinates."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=FS_LETTER,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=INK,
    )


def blank_axis(ax, xlim, ylim):
    """Turn an axis into a bare equal-aspect drawing canvas."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_aspect("equal")


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def neutral_face_glyph(ax, cx, cy, r, color=INK, lw=0.7):
    """
    Draw a minimal *neutral* face: circle, two eyes, straight (flat) mouth.
    The flat mouth is deliberate — every face stimulus was neutral.
    """
    ax.add_patch(
        Circle((cx, cy), r, facecolor="white", edgecolor=color, linewidth=lw, zorder=3)
    )
    eye_dx, eye_dy, eye_r = 0.38 * r, 0.28 * r, 0.11 * r
    for sx in (-1, 1):
        ax.add_patch(
            Circle(
                (cx + sx * eye_dx, cy + eye_dy),
                eye_r,
                facecolor=color,
                edgecolor="none",
                zorder=4,
            )
        )
    ax.plot(
        [cx - 0.40 * r, cx + 0.40 * r],
        [cy - 0.42 * r, cy - 0.42 * r],
        color=color,
        linewidth=lw,
        solid_capstyle="round",
        zorder=4,
    )


def gradient_bar(ax, x0, y0, width, height, n=40):
    """Draw a horizontal colormap strip as vector rectangles."""
    for i, frac in enumerate(np.linspace(0.0, 1.0, n)):
        ax.add_patch(
            Rectangle(
                (x0 + i * width / n, y0),
                width / n + 1e-4,
                height,
                facecolor=CMAP(frac),
                edgecolor="none",
                zorder=2,
            )
        )
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor="none",
            edgecolor=INK_SOFT,
            linewidth=0.5,
            zorder=3,
        )
    )


# ============================================================================
# Anatomical ROIs (Harvard-Oxford + vmPFC spheres)
# ============================================================================
def sphere_nifti(center_mni, radius_mm, ref_img):
    """
    Return a binary NIfTI image with 1s inside a sphere.

    Same construction as publication_figures.sphere_nifti, so the vmPFC targets
    drawn here are the established ones rather than stylized stand-ins.
    """
    from nilearn.image import new_img_like

    ref_data = ref_img.get_fdata()
    affine = ref_img.affine

    xi, yi, zi = np.mgrid[
        0 : ref_data.shape[0], 0 : ref_data.shape[1], 0 : ref_data.shape[2]
    ]
    coords_vox = np.column_stack([xi.ravel(), yi.ravel(), zi.ravel(), np.ones(xi.size)])
    coords_mni = (affine @ coords_vox.T)[:3].T
    dist = np.sqrt(np.sum((coords_mni - np.array(center_mni)) ** 2, axis=1))
    sphere = (dist <= radius_mm).astype(np.float32).reshape(ref_data.shape[:3])
    return new_img_like(ref_img, sphere)


def load_atlas_bundle():
    """
    Fetch the Harvard-Oxford atlas ONCE and return everything the figure needs:

        xs, ys      ordered voxel coordinates of the densest left-amygdala
                    axial slice (panel B voxel patches)
        l_amyg      left-amygdala mask
        r_amyg      right-amygdala mask (sensitivity region)
        ant_vmpfc   anterior vmPFC sphere
        post_vmpfc  posterior vmPFC sphere

    Fails loudly. A missing atlas is never silently replaced by a different ROI.
    """
    try:
        from nilearn import datasets, image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "ERROR: nilearn is required to draw the anatomical left-amygdala "
            "outline (panel B) and the ROI inset (panel C).\n"
            "  Install it with:  pip install nilearn\n"
            "  No substitute ROI will be used."
        ) from exc

    try:
        atlas = datasets.fetch_atlas_harvard_oxford(HO_ATLAS)
    except Exception as exc:  # pragma: no cover - network / cache dependent
        raise SystemExit(
            f"ERROR: could not obtain the Harvard-Oxford atlas '{HO_ATLAS}'.\n"
            f"  Underlying error: {type(exc).__name__}: {exc}\n"
            "  nilearn downloads this atlas on first use and caches it in\n"
            "  ~/nilearn_data. Provide network access or a populated cache.\n"
            "  No substitute ROI will be used."
        ) from exc

    labels = [str(lbl) for lbl in atlas.labels]
    for needed in (L_AMYG_LABEL, R_AMYG_LABEL):
        if needed not in labels:
            raise SystemExit(
                f"ERROR: label '{needed}' is absent from Harvard-Oxford "
                f"'{HO_ATLAS}'.\n"
                f"  Available labels: {labels}\n"
                "  Refusing to substitute a different ROI."
            )

    l_amyg_img = image.math_img(f"img == {labels.index(L_AMYG_LABEL)}", img=atlas.maps)
    r_amyg_img = image.math_img(f"img == {labels.index(R_AMYG_LABEL)}", img=atlas.maps)

    data = np.asarray(l_amyg_img.get_fdata())
    if data.ndim != 3 or not np.any(data > 0):
        raise SystemExit(
            f"ERROR: the '{L_AMYG_LABEL}' mask from '{HO_ATLAS}' is empty or "
            "not 3-D; cannot build the panel B voxel schematic."
        )

    best_z = int(np.argmax(data.sum(axis=(0, 1))))
    xs, ys = np.where(data[:, :, best_z] > 0)
    if xs.size < 10:
        raise SystemExit(
            f"ERROR: the densest '{L_AMYG_LABEL}' axial slice has only "
            f"{xs.size} voxels; too few for the panel B schematic."
        )

    # Fixed raster ordering so every patch unrolls its voxels identically.
    order = np.lexsort((xs, ys))
    return {
        "xs": xs[order],
        "ys": ys[order],
        "l_amyg": l_amyg_img,
        "r_amyg": r_amyg_img,
        "ant_vmpfc": sphere_nifti(ANT_VMPFC_COORDS, SPHERE_RADIUS, l_amyg_img),
        "post_vmpfc": sphere_nifti(POST_VMPFC_COORDS, SPHERE_RADIUS, l_amyg_img),
    }


# (bundle key, label, color, alpha) — one source of truth so the brain
# contours and the labels beside them can never drift apart.
#
# Anterior and posterior vmPFC are co-primary targets: they are distinguished
# only by green shade, at equal opacity. The right amygdala is the sensitivity
# region and is the one ROI drawn with reduced prominence.
ROI_INSET_SPEC = [
    ("l_amyg", "Left amygdala", AMYG_FILL, 0.95),
    ("r_amyg", "Right amygdala\n(sensitivity)", AMYG_MUTED, 0.65),
    ("ant_vmpfc", "Anterior vmPFC", VMPFC_FILL, 0.92),
    ("post_vmpfc", "Posterior vmPFC", VMPFC_LINE, 0.92),
]

ROI_LABEL_Y = [0.78, 0.545, 0.285, 0.085]  # label rows, axes fraction


def draw_roi_inset(ax_brain, ax_labels, bundle):
    """
    Anatomical ROI inset: a glass-brain axial projection carrying the real
    Harvard-Oxford amygdala masks and the real vmPFC spheres, with concise
    labels in the adjacent cell.

    The brain and the labels live in separate axes because nilearn switches off
    whatever axes it is handed.
    """
    try:
        from nilearn import plotting
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "ERROR: nilearn is required to draw the panel C ROI inset.\n"
            "  Install it with:  pip install nilearn\n"
            "  No substitute ROI will be used."
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        display = plotting.plot_glass_brain(
            None,
            display_mode="z",
            axes=ax_brain,
            colorbar=False,
            plot_abs=False,
            annotate=False,
        )
        # Reversed so the amygdala masks land on top of the vmPFC spheres.
        for key, _label, color, alpha in reversed(ROI_INSET_SPEC):
            display.add_contours(
                bundle[key], filled=True, threshold=0.5, colors=[color], alpha=alpha
            )
        display.annotate(left_right=True, positions=False, size=FS_TINY)

    # ---- labels beside the brain, keyed by the same colors -----------------
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(0, 1)
    ax_labels.axis("off")
    ax_labels.text(
        0.0,
        0.975,
        "Regions of interest",
        fontsize=FS_SMALL,
        color=INK,
        ha="left",
        va="top",
    )

    sw_x, sw_w, sw_h, sw_gap = 0.0, 0.11, 0.075, 0.055
    for (_key, label, color, alpha), y in zip(ROI_INSET_SPEC, ROI_LABEL_Y):
        ax_labels.add_patch(
            Rectangle(
                (sw_x, y - sw_h / 2),
                sw_w,
                sw_h,
                facecolor=color,
                edgecolor=INK_SOFT,
                linewidth=0.5,
                alpha=alpha,
                zorder=3,
            )
        )
        ax_labels.text(
            sw_x + sw_w + sw_gap,
            y,
            label,
            fontsize=FS_TINY,
            color=INK,
            ha="left",
            va="center",
            linespacing=1.25,
        )


def draw_voxel_patch(ax, xs, ys, values, x0, y0, height, edge_style="solid"):
    """
    Draw one amygdala voxel pattern inside a box whose lower-left corner is
    (x0, y0) and whose height is `height`. Width follows the voxel aspect ratio,
    so the anatomical outline is preserved. Returns the drawn width.
    """
    xr = xs - xs.min()
    yr = ys - ys.min()
    n_cols = int(xr.max()) + 1
    n_rows = int(yr.max()) + 1

    cell = height / n_rows
    width = cell * n_cols

    colors = CMAP(NORM(values))
    for cx, cy, color in zip(xr, yr, colors):
        ax.add_patch(
            Rectangle(
                (x0 + cx * cell, y0 + cy * cell),
                cell * 0.94,
                cell * 0.94,
                facecolor=color,
                edgecolor="none",
                zorder=3,
            )
        )

    # Gold frame marks these voxels as amygdala. Solid = image pattern,
    # dashed = subsequent-neutral-face pattern.
    ax.add_patch(
        FancyBboxPatch(
            (x0 - 0.03, y0 - 0.03),
            width + 0.06,
            height + 0.06,
            boxstyle="round,pad=0.015,rounding_size=0.05",
            facecolor="none",
            edgecolor=AMYG_LINE,
            linewidth=1.1,
            linestyle=edge_style,
            zorder=4,
        )
    )
    return width


# ============================================================================
# Panel A: task structure (upper-left, narrow)
# ============================================================================
PANEL_A_TRIALS = ["neg", "neu", "pos"]  # one representative trial per valence


def draw_panel_a(ax):
    blank_axis(ax, (0, A_XMAX), (0, A_YMAX))

    # Anchored to the top of the cell so the panel keeps its proportions if the
    # outer layout fractions change.
    title_y = A_YMAX - 0.45
    header_y = A_YMAX - 2.10
    y_centers = [A_YMAX - 3.85, A_YMAX - 6.60, A_YMAX - 9.35]

    panel_letter(ax, "A", x=-0.035, y=(title_y - 0.28) / A_YMAX)
    ax.text(
        0.75,
        title_y,
        "Task structure",
        fontsize=FS_TITLE,
        fontweight="bold",
        ha="left",
        va="center",
    )

    box = 2.25
    img_x0 = 2.45
    face_x0 = 6.40

    # ---- column headers ----------------------------------------------------
    ax.text(
        img_x0 + box / 2,
        header_y,
        "Image",
        fontsize=FS_HEAD,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        face_x0 + box / 2,
        header_y,
        "Neutral face",
        fontsize=FS_HEAD,
        fontweight="bold",
        ha="center",
        va="center",
    )

    for valence, yc in zip(PANEL_A_TRIALS, y_centers):
        y0 = yc - box / 2

        # valence label (the row identity is the image valence)
        ax.text(
            0.10,
            yc,
            VALENCE_NAME[valence],
            fontsize=FS_BODY,
            ha="left",
            va="center",
            color=INK,
        )

        # image stimulus
        ax.add_patch(
            Rectangle(
                (img_x0, y0),
                box,
                box,
                facecolor=COL_VALENCE[valence],
                edgecolor=INK,
                linewidth=0.7,
                zorder=3,
            )
        )
        # the following face (always neutral)
        ax.add_patch(
            Rectangle(
                (face_x0, y0),
                box,
                box,
                facecolor="#F4F4F4",
                edgecolor=INK,
                linewidth=0.7,
                zorder=2,
            )
        )
        neutral_face_glyph(ax, face_x0 + box / 2, yc, 0.21 * box)

        # image -> face arrow
        ax.add_patch(
            FancyArrowPatch(
                (img_x0 + box + 0.12, yc),
                (face_x0 - 0.12, yc),
                arrowstyle="-|>",
                mutation_scale=6.5,
                linewidth=0.8,
                color=INK_SOFT,
                shrinkA=0,
                shrinkB=0,
                zorder=5,
            )
        )


# ============================================================================
# Panel B: amygdala persistence (upper-right, wide)
# ============================================================================
def draw_panel_b(ax, xs, ys, rng):
    blank_axis(ax, (0, B_XMAX), (0, B_YMAX))

    # Title sits just above the column headers rather than at the very top of
    # the cell, so no gap opens up if the cell gets taller.
    title_y = 5.60
    panel_letter(ax, "B", x=-0.016, y=(title_y - 0.20) / B_YMAX)
    ax.text(
        0.35,
        title_y,
        "Amygdala neural persistence",
        fontsize=FS_TITLE,
        fontweight="bold",
        ha="left",
        va="center",
    )

    n_vox = xs.size
    n_runs = 3

    # ---- deterministic simulated beta patterns ------------------------------
    # A shared latent pattern plus independent noise, so the image and face
    # patterns are visibly similar without being identical.
    latent = rng.uniform(-1.0, 1.0, size=n_vox)
    img_vals = [np.clip(latent + rng.normal(0.0, 0.34, n_vox), -1, 1) for _ in range(n_runs)]
    face_vals = [np.clip(latent + rng.normal(0.0, 0.46, n_vox), -1, 1) for _ in range(n_runs)]

    patch_h = 1.05
    img_x0 = 0.95
    face_x0 = 4.65
    y_centers = [3.95, 2.50, 1.05]
    face_dash = (0, (2.5, 1.6))

    img_w = draw_voxel_patch(
        ax, xs, ys, img_vals[0], img_x0, y_centers[0] - patch_h / 2, patch_h, "solid"
    )
    face_w = draw_voxel_patch(
        ax, xs, ys, face_vals[0], face_x0, y_centers[0] - patch_h / 2, patch_h, face_dash
    )

    # ---- column headers -----------------------------------------------------
    ax.text(
        img_x0 + img_w / 2,
        5.12,
        "Negative image",
        fontsize=FS_HEAD,
        fontweight="bold",
        ha="center",
        va="center",
        color=AMYG_LINE,
    )
    for y, line in ((5.28, "Neutral face following"), (5.05, "a negative image")):
        ax.text(
            face_x0 + face_w / 2,
            y,
            line,
            fontsize=FS_HEAD,
            fontweight="bold",
            ha="center",
            va="center",
            color=AMYG_LINE,
        )

    # ---- remaining patches + run labels -------------------------------------
    for r_i in range(n_runs):
        yc = y_centers[r_i]
        if r_i > 0:
            draw_voxel_patch(
                ax, xs, ys, img_vals[r_i], img_x0, yc - patch_h / 2, patch_h, "solid"
            )
            draw_voxel_patch(
                ax, xs, ys, face_vals[r_i], face_x0, yc - patch_h / 2, patch_h, face_dash
            )
        ax.text(
            img_x0 - 0.12, yc, f"Run {r_i + 1}", fontsize=FS_BODY, ha="right", va="center"
        )
        ax.text(
            face_x0 + face_w + 0.12,
            yc,
            f"Run {r_i + 1}",
            fontsize=FS_BODY,
            ha="left",
            va="center",
        )

    # ---- the six directional cross-run pairs (i != j) -----------------------
    rads = {
        (0, 1): -0.18,
        (0, 2): -0.34,
        (1, 0): 0.18,
        (1, 2): -0.18,
        (2, 0): 0.34,
        (2, 1): 0.18,
    }
    x_from = img_x0 + img_w + 0.10
    x_to = face_x0 - 0.10
    for (i, j), rad in rads.items():
        ax.add_patch(
            FancyArrowPatch(
                (x_from, y_centers[i]),
                (x_to, y_centers[j]),
                arrowstyle="-|>",
                mutation_scale=6.0,
                linewidth=0.75,
                color=INK_SOFT,
                alpha=0.8,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=0,
                shrinkB=0,
                zorder=5,
            )
        )

    # ---- aggregation box ----------------------------------------------------
    # Width hugs the widest line it holds ("Left-amygdala negative", ~1.10 in
    # at FS_BODY, plus padding), which also keeps its left edge clear of the
    # Run 1/2/3 labels on the right of the face column.
    bx0, bw = 8.15, 3.70
    by0, bh = 0.95, 3.25
    ax.add_patch(
        FancyBboxPatch(
            (bx0, by0),
            bw,
            bh,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#FCFAEC",
            edgecolor=AMYG_LINE,
            linewidth=0.9,
            zorder=3,
        )
    )
    for r_i in range(n_runs):
        ax.add_patch(
            FancyArrowPatch(
                (face_x0 + face_w + 1.00, y_centers[r_i]),
                (bx0 - 0.06, y_centers[1]),
                arrowstyle="-|>",
                mutation_scale=6.0,
                linewidth=0.75,
                color=INK_SOFT,
                alpha=0.8,
                shrinkA=0,
                shrinkB=0,
                zorder=5,
            )
        )

    bxc = bx0 + bw / 2
    ax.text(bxc, 3.86, "Spatial correlation of", fontsize=FS_SMALL, ha="center", va="center")
    ax.text(bxc, 3.58, "each image–face pair", fontsize=FS_SMALL, ha="center", va="center")
    ax.text(
        bxc,
        3.12,
        "Six pairs averaged",
        fontsize=FS_BODY,
        fontweight="bold",
        ha="center",
        va="center",
        color=AMYG_LINE,
    )
    ax.text(
        bxc,
        2.84,
        "in Fisher-$z$ space",
        fontsize=FS_BODY,
        fontweight="bold",
        ha="center",
        va="center",
        color=AMYG_LINE,
    )
    ax.plot([bx0 + 0.28, bx0 + bw - 0.28], [2.50, 2.50], color=AMYG_MID, linewidth=0.7, zorder=4)
    ax.text(
        bxc, 2.14, "Left-amygdala negative", fontsize=FS_BODY, fontweight="bold", ha="center", va="center"
    )
    ax.text(
        bxc, 1.86, "persistence", fontsize=FS_BODY, fontweight="bold", ha="center", va="center"
    )
    ax.text(bxc, 1.50, "(primary measure)", fontsize=FS_SMALL, ha="center", va="center", color=INK_SOFT)

    # ---- beta gradient with minus / plus endpoints --------------------------
    cb_x0, cb_y0, cb_w, cb_h = 0.30, 0.06, 1.45, 0.16
    gradient_bar(ax, cb_x0, cb_y0, cb_w, cb_h)
    ax.text(cb_x0 - 0.10, cb_y0 + cb_h / 2, "−", fontsize=FS_SMALL, ha="right", va="center")
    ax.text(cb_x0 + cb_w + 0.10, cb_y0 + cb_h / 2, "+", fontsize=FS_SMALL, ha="left", va="center")

    # ---- the i != j condition on the arrow bundle ---------------------------
    ax.text(
        (x_from + x_to) / 2,
        0.14,
        r"run $i \neq$ run $j$",
        fontsize=FS_SMALL,
        ha="center",
        va="center",
        color=INK,
    )


# ============================================================================
# Panel C: LSS connectivity pipeline (lower-left, narrow)
# ============================================================================
# Schematic trial count only. The real task presented 10 images per valence
# per run; the strip is truncated and ends in an ellipsis.
C_PER_VALENCE = 9


def draw_panel_c(ax_strip, ax_seed, ax_targ, ax_scatter, ax_brain, ax_roilab, bundle, rng):
    conditions = np.array(
        ["neg"] * C_PER_VALENCE + ["neu"] * C_PER_VALENCE + ["pos"] * C_PER_VALENCE
    )
    conditions = conditions[rng.permutation(conditions.size)]  # deterministic
    n_trials = conditions.size
    x = np.arange(n_trials)

    # Shared latent trial signal -> amygdala and vmPFC betas covary.
    latent = rng.normal(0.0, 0.62, n_trials)
    seed_betas = np.clip(latent + rng.normal(0.0, 0.30, n_trials), -1.0, 1.0)
    targ_betas = np.clip(0.85 * latent + rng.normal(0.0, 0.34, n_trials), -1.0, 1.0)

    alphas = np.where(conditions == "neg", 1.0, DIM)
    bar_w = 0.68
    xlim = (-0.9, n_trials + 0.9)

    # ---- trial strip --------------------------------------------------------
    ax_strip.set_xlim(*xlim)
    ax_strip.set_ylim(0, 1)
    ax_strip.axis("off")
    panel_letter(ax_strip, "C", x=-0.115, y=1.55)
    ax_strip.text(
        0.0,
        1.55,
        "LSS beta-series connectivity",
        transform=ax_strip.transAxes,
        fontsize=FS_TITLE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    for xi, cond, alpha in zip(x, conditions, alphas):
        ax_strip.add_patch(
            Rectangle(
                (xi - bar_w / 2, 0.20),
                bar_w,
                0.60,
                facecolor=COL_VALENCE[cond],
                edgecolor="none",
                alpha=alpha,
                zorder=3,
            )
        )
    # Ellipsis: the strip is a truncated schematic, not the trial count.
    ax_strip.text(
        n_trials + 0.15, 0.50, "…", fontsize=FS_BODY, ha="left", va="center", color=INK_SOFT
    )
    ax_strip.text(
        -0.9,
        1.28,
        "Image trials",
        fontsize=FS_SMALL,
        ha="left",
        va="center",
        color=INK_SOFT,
    )

    # ---- paired beta series -------------------------------------------------
    for ax, betas, color, label, label_color in (
        (ax_seed, seed_betas, AMYG_MID, "Amygdala", AMYG_LINE),
        (ax_targ, targ_betas, VMPFC_FILL, "vmPFC", VMPFC_LINE),
    ):
        ax.set_xlim(*xlim)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color=HAIRLINE, linewidth=0.6, zorder=1)
        for xi, b, alpha in zip(x, betas, alphas):
            ax.add_patch(
                Rectangle(
                    (xi - bar_w / 2, min(0.0, b)),
                    bar_w,
                    abs(b),
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha,
                    zorder=3,
                )
            )
        # Tint each label to its own series so the two adjacent rotated labels
        # do not read as one merged phrase.
        ax.set_ylabel(label, fontsize=FS_SMALL, labelpad=2, color=label_color)
        despine(ax, keep=())
        ax.set_xticks([])
        ax.set_yticks([])

    # ---- arrow into the scatter --------------------------------------------
    ax_scatter.annotate(
        "",
        xy=(0.5, 1.03),
        xytext=(0.5, 1.26),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, linewidth=0.8, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )

    # ---- seed-target scatter, within run and valence -----------------------
    is_neg = conditions == "neg"
    sx, sy = seed_betas[is_neg], targ_betas[is_neg]

    ax_scatter.scatter(
        sx, sy, s=16, facecolor=AMYG_FILL, edgecolor=VMPFC_LINE, linewidth=0.7, zorder=4
    )
    fit = np.polyfit(sx, sy, 1)
    xf = np.linspace(sx.min() - 0.08, sx.max() + 0.08, 100)
    ax_scatter.plot(xf, np.polyval(fit, xf), color=INK_SOFT, linewidth=1.1, zorder=3)

    ax_scatter.set_xlabel("Amygdala beta", fontsize=FS_SMALL, labelpad=1.5)
    ax_scatter.set_ylabel("vmPFC beta", fontsize=FS_SMALL, labelpad=2)
    despine(ax_scatter)
    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])
    ax_scatter.grid(True, alpha=0.20, linestyle="--", linewidth=0.4)
    ax_scatter.set_axisbelow(True)
    # Top-left: the simulated seed-target trend is positive, so that corner is
    # the one reliably free of points and of the fit line.
    ax_scatter.text(
        0.03,
        0.95,
        r"$\rightarrow$ Fisher $z$",
        transform=ax_scatter.transAxes,
        fontsize=FS_SMALL,
        va="top",
        ha="left",
        color=INK,
    )

    # ---- anatomical ROI inset ----------------------------------------------
    draw_roi_inset(ax_brain, ax_roilab, bundle)


# ============================================================================
# Panel D: hypothesized relationships (lower-right, wide, 2x2)
# ============================================================================
D_X_PERSIST = "Left-amygdala\nnegative persistence"
D_X_FC = "L amygdala–ant. vmPFC\nconnectivity (neg − neu)"
D_Y_PA = "Daily positive\naffect"
D_Y_NA = "Daily negative\naffect"

# Row-major 2x2: (x label, y label, slope sign, (fill, line), badge)
PANEL_D_SPECS = [
    (D_X_PERSIST, D_Y_PA, -1.0, (AMYG_FILL, AMYG_LINE), "−"),
    (D_X_PERSIST, D_Y_NA, +1.0, (AMYG_FILL, AMYG_LINE), "+"),
    (D_X_FC, D_Y_PA, +1.0, (VMPFC_FILL, VMPFC_LINE), "+"),
    (D_X_FC, D_Y_NA, -1.0, (VMPFC_FILL, VMPFC_LINE), "−"),
]

D_SLOPE = 0.60  # illustrative effect size (mock_scatter_schematic.py)
D_NOISE = 0.55
D_N = 45


def draw_panel_d(ax_header, axes, rng):
    ax_header.axis("off")
    panel_letter(ax_header, "D", x=-0.016, y=0.0)
    ax_header.text(
        0.030,
        0.0,
        "Hypothesized relationships",
        transform=ax_header.transAxes,
        fontsize=FS_TITLE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    for ax, (xlab, ylab, sign, (fill, line), badge) in zip(axes, PANEL_D_SPECS):
        xv = rng.normal(0.0, 1.0, D_N)
        yv = sign * D_SLOPE * xv + rng.normal(0.0, D_NOISE, D_N)

        ax.scatter(
            xv, yv, s=8, facecolor=fill, edgecolor=line, linewidth=0.4, alpha=0.85, zorder=3
        )
        fit = np.polyfit(xv, yv, 1)
        xf = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(xf, np.polyval(fit, xf), color=line, linewidth=1.4, zorder=4)

        ax.set_xlabel(xlab, fontsize=FS_SMALL, labelpad=2, linespacing=1.25)
        ax.set_ylabel(ylab, fontsize=FS_SMALL, labelpad=2)
        despine(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.20, linestyle="--", linewidth=0.4)
        ax.set_axisbelow(True)

        # Park the badge in the corner the trend line leaves empty: a negative
        # slope frees the top right, a positive slope frees the top left.
        ax.text(
            0.87 if sign < 0 else 0.13,
            0.88,
            badge,
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            zorder=6,
            bbox=dict(boxstyle="circle,pad=0.22", facecolor=line, edgecolor="none"),
        )


# ============================================================================
# Assemble
# ============================================================================
def build_figure():
    bundle = load_atlas_bundle()  # single atlas fetch, reused by panels B and C

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=COL_RATIOS,
        height_ratios=ROW_RATIOS,
        wspace=WSPACE,
        hspace=HSPACE,
        left=LEFT,
        right=RIGHT,
        top=TOP,
        bottom=BOTTOM,
    )

    # Each panel draws from its own generator so panels are independent and
    # every value is reproducible from SEED alone.
    rng_b = np.random.default_rng(SEED)
    rng_c = np.random.default_rng(SEED + 1)
    rng_d = np.random.default_rng(SEED + 2)

    # ---- A: upper-left ------------------------------------------------------
    draw_panel_a(fig.add_subplot(gs[0, 0]))

    # ---- B: upper-right -----------------------------------------------------
    draw_panel_b(fig.add_subplot(gs[0, 1]), bundle["xs"], bundle["ys"], rng_b)

    # ---- C: lower-left — vertical pipeline, brain + labels on the last row --
    gs_c = GridSpecFromSubplotSpec(
        5,
        2,
        subplot_spec=gs[1, 0],
        width_ratios=[0.95, 1.00],
        # hspace must leave room for the scatter's x label above the brain row,
        # and separate the two rotated beta-series y labels.
        height_ratios=[0.20, 0.30, 0.30, 0.62, 1.30],
        hspace=0.34,
        wspace=0.10,
    )
    draw_panel_c(
        ax_strip=fig.add_subplot(gs_c[0, :]),
        ax_seed=fig.add_subplot(gs_c[1, :]),
        ax_targ=fig.add_subplot(gs_c[2, :]),
        ax_scatter=fig.add_subplot(gs_c[3, :]),
        ax_brain=fig.add_subplot(gs_c[4, 0]),
        ax_roilab=fig.add_subplot(gs_c[4, 1]),
        bundle=bundle,
        rng=rng_c,
    )

    # ---- D: lower-right — 2x2 hypothesis grid -------------------------------
    gs_d = GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs[1, 1],
        height_ratios=[0.12, 1.0, 1.0],
        hspace=0.42,
        wspace=0.42,
    )
    draw_panel_d(
        ax_header=fig.add_subplot(gs_d[0, :]),
        axes=[
            fig.add_subplot(gs_d[1, 0]),
            fig.add_subplot(gs_d[1, 1]),
            fig.add_subplot(gs_d[2, 0]),
            fig.add_subplot(gs_d[2, 1]),
        ],
        rng=rng_d,
    )

    return fig


def save_figure(fig):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    svg_path = OUT_DIR / f"{OUT_STEM}.svg"
    fig.savefig(svg_path, format="svg", facecolor="white")
    print(f"Saved: {svg_path}")

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
    print(f"Saved: {tiff_path} ({FIG_DPI} dpi)")


def main():
    print("=" * 72)
    print("Figure 1 — simulated schematic (MIDUS Amygdala Persistence)")
    print("=" * 72)
    print(
        f"Canvas: {FIG_W:.2f} x {FIG_H:.2f} in  "
        f"({round(FIG_W * FIG_DPI)} x {round(FIG_H * FIG_DPI)} px at {FIG_DPI} dpi)"
    )
    print(
        f"Cells:  left col {CELL_W[0]:.2f} in / right col {CELL_W[1]:.2f} in; "
        f"top row {CELL_H[0]:.2f} in / bottom row {CELL_H[1]:.2f} in"
    )
    fig = build_figure()
    save_figure(fig)
    plt.close(fig)
    print("\nDone. Figure 1 is fully assembled; no manual composition required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
