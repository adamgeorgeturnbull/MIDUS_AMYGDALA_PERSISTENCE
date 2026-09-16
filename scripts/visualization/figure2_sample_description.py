#!/usr/bin/env python3
"""
figure2_sample_description.py

Figure 2 for the MIDUS Amygdala Persistence manuscript: a single, fully
assembled landscape figure describing the analytic samples. No manual assembly
step is required.

    Panel A  Analytic sample derivation (flow diagram)
    Panel B  Principal-sample demographics (compact table)

Panels are stacked (A above B) rather than placed side by side. Both panels are
intrinsically wide - panel A opens with three parallel data sources and panel B
is a five-column table - so at 7.25 in each reads materially better with the
full page width than it would in a ~3.4 in column.

Authoritative input
-------------------
    results/tables/sample_descriptives.csv

This is the aggregate-only descriptives table. Every number drawn in either
panel is read from that file; nothing is hardcoded for display. The frozen Ns
in EXPECTED_N are used for validation only, never for display. The script reads
no participant-level data and prints no participant identifiers or rows.

Derivation shown in panel A
---------------------------
Solid arrows mark genuine subset restrictions:

    neuroscience            -> final fMRI

Dashed converging lines mark intersections, and the node label names both
contributing samples:

    daily diary + neuroscience  ->  diary INTERSECT neuroscience
    daily diary + final fMRI    ->  primary diary + fMRI

Displayed pairs with identical N are collapsed into one node rather than drawn
twice; the equality is validated in EQUAL_PAIRS rather than captioned on the
figure:

    fmri_conservative        / fmri_fc_conservative
    diary_fmri_conservative  / diary_fmri_fc_conservative

Deliberately NOT drawn, to keep the diagram legible:

  - The PANAS availability source (neuroscience_panas) and its diary overlap
    (diary_panas_overlap). The neuroscience sample is labelled simply
    "Neuroscience" rather than "neuroscience age".
  - The ERQ moderation complete-case subset (reappraisal_complete_case /
    suppression_complete_case). Those two measures each lose one participant
    relative to the primary sample; that is reported in the manuscript text.

All of those rows are still read and validated as an integrity check on the
CSV (see REQUIRED_ROWS / EXPECTED_N / EQUAL_PAIRS); they are simply not
displayed. "Final" is used as the imaging-QC descriptor rather than
"conservative", matching the wording used elsewhere in the project.

Sex is reported with the source-coded descriptive label "female"; the numerator
is n_female, the denominator N_sex_nonmissing, and the percentage
pct_female_of_nonmissing_sex, all taken from the CSV. If any principal sample
has n_sex_missing > 0 the denominator is flagged and a table note is emitted;
with no missing sex, no note is drawn. Age units are given in the column
headers.
Race percentages are deliberately not shown; they belong in the manuscript text
and the full descriptive table.

No model predictors, outcomes, covariates, or estimator descriptions appear in
any node. No visible text is smaller than 7 pt. No global title or "Figure 2"
label is drawn; the manuscript caption supplies those.

Outputs
-------
    results/figures/figure2_sample_description.svg
    results/figures/figure2_sample_description.tiff   (300 dpi, LZW)

Run from the project root directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save-only; never opens a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ============================================================================
# Paths
# ============================================================================
DESC_CSV = Path("results/tables/sample_descriptives.csv")
OUT_DIR = Path("results/figures")
OUT_STEM = "figure2_sample_description"
FIG_DPI = 300

# ============================================================================
# Validation contract
# ============================================================================
REQUIRED_ROWS = [
    "daily_diary",
    "neuroscience_age",
    "diary_neuroscience_age_overlap",
    "neuroscience_panas",
    "diary_panas_overlap",
    "fmri_conservative",
    "fmri_fc_conservative",
    "diary_fmri_conservative",
    "diary_fmri_fc_conservative",
    "reappraisal_complete_case",
    "suppression_complete_case",
]

REQUIRED_COLS = [
    "sample",
    "N",
    "N_age",
    "age_mean",
    "age_SD",
    "age_min",
    "age_max",
    "N_sex_nonmissing",
    "n_female",
    "pct_female_of_nonmissing_sex",
    "n_sex_missing",
    "N_race_nonmissing",
    "n_race_missing",
]

# Frozen Ns — validation only, covering every row in the CSV including the
# rows that are validated but not displayed. Displayed values always come
# from the CSV.
EXPECTED_N = {
    "daily_diary": 1174,
    "neuroscience_age": 231,
    "diary_neuroscience_age_overlap": 137,
    "neuroscience_panas": 230,
    "diary_panas_overlap": 136,
    "fmri_conservative": 127,
    "fmri_fc_conservative": 127,
    "diary_fmri_conservative": 81,
    "diary_fmri_fc_conservative": 81,
    "reappraisal_complete_case": 80,
    "suppression_complete_case": 80,
}

# Pairs whose Ns must match. The first two are each drawn as a single
# combined node; the third is validated but not displayed.
EQUAL_PAIRS = [
    ("fmri_conservative", "fmri_fc_conservative"),
    ("diary_fmri_conservative", "diary_fmri_fc_conservative"),
    ("reappraisal_complete_case", "suppression_complete_case"),
]

# ============================================================================
# Palette — restrained, colorblind-readable. Every fill is light with dark
# text, so the categories separate by hue AND lightness.
# ============================================================================
C_SOURCE_FILL, C_SOURCE_EDGE = "#E7EEF5", "#3D617F"  # data sources
C_OVERLAP_FILL, C_OVERLAP_EDGE = "#F2EEE4", "#877553"  # behavioural overlaps
C_IMAGING_FILL, C_IMAGING_EDGE = "#E1EDEB", "#2E6D68"  # imaging samples
C_PRIMARY_FILL, C_PRIMARY_EDGE = "#C6DBEF", "#1F4E79"  # primary sample
C_MOD_FILL, C_MOD_EDGE = "#EBE6F1", "#6B5B8A"  # moderation subset

INK = "#1A1A1A"
INK_SOFT = "#5A5A5A"
RULE = "#8C8C8C"
HAIRLINE = "#C9C9C9"
LINE_SUBSET = "#4A4A4A"
LINE_INTERSECT = "#7C7C7C"
ROW_HIGHLIGHT = "#EAF2FA"

# ============================================================================
# Typography — nothing below 7 pt
# ============================================================================
FS_LETTER = 10.0
FS_TITLE = 8.5
FS_NODE = 7.5
FS_N = 8.0
FS_TABLE = 7.2
FS_TABLE_HEAD = 7.4
FS_LEGEND = 7.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FS_TABLE,
        "text.color": INK,
        "svg.fonttype": "none",  # editable SVG text
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

# ============================================================================
# Geometry. Panel A is an equal-aspect canvas, so its data limits are derived
# from the physical cell aspect to avoid letterboxing.
# ============================================================================
FIG_W = 7.25
LEFT, RIGHT, TOP, BOTTOM = 0.035, 0.985, 0.975, 0.030

USABLE_W = (RIGHT - LEFT) * FIG_W

H_PANEL_A = USABLE_W * 0.42  # flow diagram (sets A_YMAX to A_DESIGN_Y)
H_PANEL_B = 1.42  # table (header + 5 displayed rows, conditional note)
HSPACE_IN = 0.28

_PANEL_H = [H_PANEL_A, H_PANEL_B]
_AVG_H = sum(_PANEL_H) / len(_PANEL_H)
HSPACE = HSPACE_IN / _AVG_H
FIG_H = (sum(_PANEL_H) + HSPACE_IN) / (TOP - BOTTOM)

A_XMAX = 100.0
# Panel A is drawn directly on its data grid: H_PANEL_A is chosen so the cell
# aspect makes A_YMAX exactly A_DESIGN_Y, so no rescaling of y is needed.
A_DESIGN_Y = 42.0
A_YMAX = A_XMAX * (H_PANEL_A / USABLE_W)  # equal aspect by construction
if abs(A_YMAX - A_DESIGN_Y) > 1e-9:  # guards the panel A row coordinates
    raise SystemExit(
        f"ERROR: panel A grid mismatch: A_YMAX={A_YMAX!r} but the row "
        f"coordinates in draw_panel_a assume {A_DESIGN_Y!r}. Adjust "
        "H_PANEL_A (currently USABLE_W * 0.42) to restore the invariant."
    )


# ============================================================================
# Load + validate
# ============================================================================
def _fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def _num(row, col, sample):
    """Return a finite float from the CSV or abort."""
    val = row[col]
    if pd.isna(val):
        _fail(f"'{col}' is missing for sample '{sample}' in {DESC_CSV}.")
    try:
        out = float(val)
    except (TypeError, ValueError):
        _fail(f"'{col}' for sample '{sample}' is nonnumeric ({val!r}) in {DESC_CSV}.")
    if not np.isfinite(out):
        _fail(f"'{col}' for sample '{sample}' is not finite ({val!r}) in {DESC_CSV}.")
    return out


def load_descriptives():
    """
    Read and fully validate the aggregate descriptives table.

    Returns a dict: sample name -> dict of validated numeric fields.
    Aborts with an informative error on any contract violation.
    """
    if not DESC_CSV.exists():
        _fail(
            f"required input {DESC_CSV} not found.\n"
            "  Figure 2 is built only from that aggregate table; it must be\n"
            "  present and must be the authoritative copy."
        )

    try:
        df = pd.read_csv(DESC_CSV)
    except Exception as exc:
        _fail(f"could not parse {DESC_CSV}: {type(exc).__name__}: {exc}")

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        _fail(
            f"{DESC_CSV} is missing required column(s): {missing_cols}\n"
            f"  Columns present: {list(df.columns)}"
        )

    names = df["sample"].astype(str).tolist()
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        _fail(f"duplicated sample name(s) in {DESC_CSV}: {dupes}")

    missing_rows = [r for r in REQUIRED_ROWS if r not in names]
    if missing_rows:
        _fail(
            f"{DESC_CSV} is missing required row(s): {missing_rows}\n"
            f"  Samples present: {names}"
        )

    df = df.set_index("sample")
    out = {}
    for key in REQUIRED_ROWS:
        row = df.loc[key]
        rec = {c: _num(row, c, key) for c in REQUIRED_COLS if c != "sample"}

        # --- N ------------------------------------------------------------
        if rec["N"] <= 0 or rec["N"] != int(rec["N"]):
            _fail(f"N for sample '{key}' must be a positive integer, got {rec['N']!r}.")

        # --- age ----------------------------------------------------------
        if rec["age_SD"] < 0:
            _fail(f"age_SD for '{key}' is negative ({rec['age_SD']}).")
        if rec["age_min"] > rec["age_max"]:
            _fail(
                f"age_min > age_max for '{key}' "
                f"({rec['age_min']} > {rec['age_max']})."
            )
        if not (rec["age_min"] <= rec["age_mean"] <= rec["age_max"]):
            _fail(
                f"age_mean for '{key}' ({rec['age_mean']}) falls outside its "
                f"own range [{rec['age_min']}, {rec['age_max']}]."
            )
        if rec["N_age"] <= 0 or rec["N_age"] > rec["N"]:
            _fail(
                f"N_age for '{key}' ({rec['N_age']}) must be in (0, N={rec['N']}]."
            )

        # --- sex ----------------------------------------------------------
        if rec["N_sex_nonmissing"] <= 0:
            _fail(
                f"N_sex_nonmissing for '{key}' is not positive "
                f"({rec['N_sex_nonmissing']}); cannot form a female n/N."
            )
        if rec["n_female"] < 0 or rec["n_female"] > rec["N_sex_nonmissing"]:
            _fail(
                f"n_female for '{key}' ({rec['n_female']}) must be in "
                f"[0, N_sex_nonmissing={rec['N_sex_nonmissing']}]."
            )
        if rec["n_sex_missing"] < 0:
            _fail(f"n_sex_missing for '{key}' is negative ({rec['n_sex_missing']}).")
        if rec["N_sex_nonmissing"] + rec["n_sex_missing"] != rec["N"]:
            _fail(
                f"sex counts for '{key}' do not reconcile: "
                f"N_sex_nonmissing ({rec['N_sex_nonmissing']}) + n_sex_missing "
                f"({rec['n_sex_missing']}) != N ({rec['N']})."
            )
        pct = rec["pct_female_of_nonmissing_sex"]
        if not (0.0 <= pct <= 100.0):
            _fail(
                f"pct_female_of_nonmissing_sex for '{key}' is outside 0-100 ({pct})."
            )

        # --- race counts are validated for integrity but never displayed ---
        if rec["n_race_missing"] < 0:
            _fail(f"n_race_missing for '{key}' is negative ({rec['n_race_missing']}).")
        if rec["N_race_nonmissing"] + rec["n_race_missing"] != rec["N"]:
            _fail(
                f"race counts for '{key}' do not reconcile: "
                f"N_race_nonmissing ({rec['N_race_nonmissing']}) + "
                f"n_race_missing ({rec['n_race_missing']}) != N ({rec['N']})."
            )

        out[key] = rec

    # --- frozen Ns -------------------------------------------------------
    bad = {
        k: (int(out[k]["N"]), v) for k, v in EXPECTED_N.items() if int(out[k]["N"]) != v
    }
    if bad:
        lines = "\n".join(
            f"    {k}: CSV has {got}, expected {exp}" for k, (got, exp) in bad.items()
        )
        _fail(
            "sample sizes in the CSV do not match the frozen expected values.\n"
            f"{lines}\n"
            "  Either the CSV is not the authoritative copy or the analytic\n"
            "  samples changed; resolve before drawing Figure 2."
        )

    # --- combined-pair equality ------------------------------------------
    for a, b in EQUAL_PAIRS:
        if int(out[a]["N"]) != int(out[b]["N"]):
            _fail(
                f"'{a}' (N={int(out[a]['N'])}) and '{b}' "
                f"(N={int(out[b]['N'])}) are drawn as one combined node but "
                "their Ns differ; the combined label would be wrong."
            )

    return out


# ============================================================================
# Formatting helpers (every value originates in the CSV)
# ============================================================================
def fmt_n(rec):
    return f"{int(rec['N']):,}"


def fmt_age_mean_sd(rec):
    return rf"{rec['age_mean']:.1f} $\pm$ {rec['age_SD']:.1f}"


def fmt_age_range(rec):
    return f"{int(round(rec['age_min']))}–{int(round(rec['age_max']))}"


def fmt_female(rec):
    """n/N (%) using nonmissing sex as the denominator."""
    num = int(rec["n_female"])
    den = int(rec["N_sex_nonmissing"])
    pct = rec["pct_female_of_nonmissing_sex"]
    flag = "*" if rec["n_sex_missing"] > 0 else ""
    return f"{num:,}/{den:,}{flag} ({pct:.1f}%)"


# ============================================================================
# Panel A: sample derivation
# ============================================================================
def draw_node(ax, x0, x1, yc, h, lines, fill, edge, lw=1.0, bold_label=False):
    """
    Draw one rounded node spanning x0..x1, vertically centred on yc with
    height h (all in panel A design units), stacking `lines` top-down.

    `lines` is a list of (text, fontsize, weight, color).
    """
    y0 = yc - h / 2
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            h,
            boxstyle="round,pad=0,rounding_size=1.1",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            zorder=3,
        )
    )
    xc = (x0 + x1) / 2
    total = sum(1.0 for _ in lines)
    step = h / (total + 0.55)
    top = yc + h / 2 - step * 0.78
    for i, (txt, fs, weight, color) in enumerate(lines):
        ax.text(
            xc,
            top - i * step,
            txt,
            fontsize=fs,
            fontweight="bold" if (weight == "bold" or (bold_label and i == 0)) else "normal",
            ha="center",
            va="center",
            color=color,
            zorder=4,
        )


def subset_arrow(ax, x0, y0, x1, y1):
    """Solid arrow with a head: a genuine subset restriction."""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.9,
            color=LINE_SUBSET,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def intersect_line(ax, pts):
    """Dashed headless polyline: one contribution to an intersection."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(
        xs,
        ys,
        color=LINE_INTERSECT,
        linewidth=0.8,
        linestyle=(0, (2.6, 1.7)),
        solid_capstyle="butt",
        zorder=2,
    )


def draw_panel_a(ax, d):
    ax.set_xlim(0, A_XMAX)
    ax.set_ylim(0, A_YMAX)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(
        -0.018,
        1.0,
        "A",
        transform=ax.transAxes,
        fontsize=FS_LETTER,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.030,
        1.0,
        "Analytic sample derivation",
        transform=ax.transAxes,
        fontsize=FS_TITLE,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # ---- row geometry (design grid == data grid, y 0-A_DESIGN_Y) ----------
    Y_SRC, H_SRC = 32.0, 6.4
    Y_MID, H_MID = 18.5, 6.6
    Y_PRI, H_PRI = 6.0, 7.0

    src_bot = Y_SRC - H_SRC / 2  # 28.8
    mid_top = Y_MID + H_MID / 2  # 21.8
    mid_bot = Y_MID - H_MID / 2  # 15.2
    pri_top = Y_PRI + H_PRI / 2  # 9.7

    # ---- data sources ------------------------------------------------------
    draw_node(
        ax, 4, 46, Y_SRC, H_SRC,
        [
            ("M3 Project 2", FS_NODE, "bold", INK),
            ("daily diary", FS_NODE, "bold", INK),
            (f"N = {fmt_n(d['daily_diary'])}", FS_N, "bold", INK),
        ],
        C_SOURCE_FILL, C_SOURCE_EDGE,
    )
    draw_node(
        ax, 54, 96, Y_SRC, H_SRC,
        [
            ("M3 Project 5", FS_NODE, "bold", INK),
            ("neuroscience", FS_NODE, "bold", INK),
            (f"N = {fmt_n(d['neuroscience_age'])}", FS_N, "bold", INK),
        ],
        C_SOURCE_FILL, C_SOURCE_EDGE,
    )

    # ---- overlap and the imaging sample ------------------------------------
    draw_node(
        ax, 14, 50, Y_MID, H_MID,
        [
            (r"Diary $\cap$ neuroscience", FS_NODE, "bold", INK),
            (f"N = {fmt_n(d['diary_neuroscience_age_overlap'])}", FS_N, "bold", INK),
        ],
        C_OVERLAP_FILL, C_OVERLAP_EDGE,
    )
    draw_node(
        ax, 54, 96, Y_MID, H_MID,
        [
            ("Final fMRI sample", FS_NODE, "bold", INK),
            (f"N = {fmt_n(d['fmri_conservative'])}", FS_N, "bold", INK),
        ],
        C_IMAGING_FILL, C_IMAGING_EDGE,
    )

    # ---- primary sample (visually prominent) -------------------------------
    draw_node(
        ax, 26, 74, Y_PRI, H_PRI,
        [
            ("Primary diary + fMRI sample", FS_NODE + 0.6, "bold", INK),
            (f"N = {fmt_n(d['diary_fmri_conservative'])}", FS_N + 1.0, "bold", C_PRIMARY_EDGE),
        ],
        C_PRIMARY_FILL, C_PRIMARY_EDGE, lw=2.0,
    )

    # ---- subset arrow (solid) ----------------------------------------------
    subset_arrow(ax, 75, src_bot, 75, mid_top)  # neuroscience -> final fMRI

    # ---- intersection contributions (dashed, headless) ---------------------
    # Laid out so no two connectors cross.
    intersect_line(ax, [(25, src_bot), (25, mid_top)])           # diary -> overlap
    intersect_line(ax, [(58, src_bot), (45, mid_top)])           # neuro -> overlap
    intersect_line(ax, [(8, src_bot), (8, Y_PRI), (26, Y_PRI)])  # diary spine -> primary
    intersect_line(ax, [(70, mid_bot), (70, pri_top)])           # final fMRI -> primary

    # ---- line-style legend (free lower-right corner; the lower left is
    # occupied by the diary spine) -------------------------------------------
    lx, ly = 78.0, 7.0
    ax.add_patch(
        FancyArrowPatch(
            (lx, ly),
            (lx + 6.0, ly),
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.9,
            color=LINE_SUBSET,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )
    ax.text(lx + 7.2, ly, "subset", fontsize=FS_LEGEND, va="center", ha="left", color=INK)
    ax.plot(
        [lx, lx + 6.0],
        [ly - 3.4] * 2,
        color=LINE_INTERSECT,
        linewidth=0.8,
        linestyle=(0, (2.6, 1.7)),
        zorder=4,
    )
    ax.text(
        lx + 7.2,
        ly - 3.4,
        "intersection",
        fontsize=FS_LEGEND,
        va="center",
        ha="left",
        color=INK,
    )


# ============================================================================
# Panel B: principal-sample demographics
# ============================================================================
# (display label, CSV row) — the displayed principal/combined samples.
# neuroscience_panas and reappraisal_complete_case are validated but not shown.
TABLE_ROWS = [
    ("Daily diary", "daily_diary"),
    ("Neuroscience", "neuroscience_age"),
    (r"Diary $\cap$ neuroscience", "diary_neuroscience_age_overlap"),
    ("Final fMRI", "fmri_conservative"),
    ("Primary diary + fMRI", "diary_fmri_conservative"),
]

HIGHLIGHT_ROW = "diary_fmri_conservative"  # matches panel A prominence

# Column anchors on a 0-100 grid: (header, x, horizontal alignment)
COLS = [
    ("Sample", 1.0, "left"),
    ("N", 32.0, "right"),
    (r"Age (yr.), mean $\pm$ SD", 46.0, "center"),
    ("Age range (yr.)", 63.0, "center"),
    ("Female, n/N (%)", 99.0, "right"),
]


def draw_panel_b(ax, d):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        -0.018,
        1.0,
        "B",
        transform=ax.transAxes,
        fontsize=FS_LETTER,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.030,
        1.0,
        "Principal-sample demographics",
        transform=ax.transAxes,
        fontsize=FS_TITLE,
        fontweight="bold",
        ha="left",
        va="top",
    )

    any_sex_missing = any(
        d[key]["n_sex_missing"] > 0 for _label, key in TABLE_ROWS
    )

    # Explicit vertical bands (0-100 within the panel): a title band at the
    # top, then header, then body rows, then the note. Sized so the panel
    # title cannot collide with the top rule.
    n_rows = len(TABLE_ROWS)
    y_top_rule = 88.0
    y_head = 81.0
    y_head_rule = 75.0
    y_first = 68.0
    row_h = 11.5

    # ---- header ------------------------------------------------------------
    for text, x, ha in COLS:
        ax.text(
            x, y_head, text, fontsize=FS_TABLE_HEAD, fontweight="bold",
            ha=ha, va="center", color=INK,
        )
    ax.plot([0, 100], [y_head_rule] * 2, color=RULE, linewidth=0.9, zorder=2)
    ax.plot([0, 100], [y_top_rule] * 2, color=RULE, linewidth=0.9, zorder=2)

    # ---- body --------------------------------------------------------------
    for i, (label, key) in enumerate(TABLE_ROWS):
        rec = d[key]
        y = y_first - i * row_h

        if key == HIGHLIGHT_ROW:
            ax.add_patch(
                Rectangle(
                    (0, y - row_h / 2 + row_h * 0.06),
                    100,
                    row_h * 0.88,
                    facecolor=ROW_HIGHLIGHT,
                    edgecolor="none",
                    zorder=1,
                )
            )
        elif i > 0:
            ax.plot([0, 100], [y + row_h / 2] * 2, color=HAIRLINE, linewidth=0.4, zorder=1)

        bold = "bold" if key == HIGHLIGHT_ROW else "normal"
        values = [
            label,
            fmt_n(rec),
            fmt_age_mean_sd(rec),
            fmt_age_range(rec),
            fmt_female(rec),
        ]
        for (text, x, ha), val in zip(COLS, values):
            del text
            ax.text(
                x, y, val, fontsize=FS_TABLE, fontweight=bold,
                ha=ha, va="center", color=INK, zorder=3,
            )

    y_bot = y_first - (n_rows - 1) * row_h - row_h / 2
    ax.plot([0, 100], [y_bot] * 2, color=RULE, linewidth=0.9, zorder=2)

    # ---- table note --------------------------------------------------------
    # The unconditional descriptive note is intentionally not drawn; age units
    # are carried by the column headers instead. The nonmissing-sex note is
    # still emitted whenever a displayed sample actually has missing sex.
    if any_sex_missing:
        ax.text(
            0,
            y_bot - row_h * 0.62,
            "*Denominator is the number of participants with nonmissing sex; "
            "percentages use nonmissing values only.",
            fontsize=FS_LEGEND,
            ha="left",
            va="top",
            color=INK_SOFT,
        )


# ============================================================================
# Assemble
# ============================================================================
def build_figure(d):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=_PANEL_H,
        hspace=HSPACE,
        left=LEFT,
        right=RIGHT,
        top=TOP,
        bottom=BOTTOM,
    )
    draw_panel_a(fig.add_subplot(gs[0]), d)
    draw_panel_b(fig.add_subplot(gs[1]), d)
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
    print(f"Saved: {tiff_path} ({FIG_DPI} dpi, LZW)")


def main():
    print("=" * 72)
    print("Figure 2 — sample description (MIDUS Amygdala Persistence)")
    print("=" * 72)
    print(f"Input : {DESC_CSV}")

    d = load_descriptives()
    print(f"Validated {len(REQUIRED_ROWS)} required rows against frozen Ns.")
    for a, b in EQUAL_PAIRS:
        print(f"  combined: {a} == {b}  (N = {int(d[a]['N'])})")

    print(
        f"Canvas: {FIG_W:.2f} x {FIG_H:.2f} in  "
        f"({round(FIG_W * FIG_DPI)} x {round(FIG_H * FIG_DPI)} px at {FIG_DPI} dpi)"
    )
    print(f"Panels: A {USABLE_W:.2f} x {H_PANEL_A:.2f} in; B {USABLE_W:.2f} x {H_PANEL_B:.2f} in")

    fig = build_figure(d)
    save_figure(fig)
    plt.close(fig)
    print("\nDone. Figure 2 is fully assembled; no manual composition required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
