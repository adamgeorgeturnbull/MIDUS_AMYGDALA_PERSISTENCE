#!/usr/bin/env python3
"""
04_sensitivity.py (MR1)

Sensitivity analyses for Analysis 04: FC (neg − neu contrast) → Affect.

Sections:
  1. Right-amygdala neg−neu FC → diary affect  (two-tailed, hemisphere specificity)
     → sensitivity_right_amygdala/
  2. Left-amygdala pos−neu FC  → diary affect  (two-tailed, condition specificity)
     → sensitivity_pos_vs_neu/
  3. Left-amygdala neg−neu FC  → PANAS         (one-tailed, convergent validity)
     → sensitivity_panas/
  4. Left-amygdala neg-only FC → diary affect  (one-tailed, operationalisation robustness)
     → sensitivity_neg_condition/

All sections use the conservative QC criterion (qc_conservative == 1):
  - historical visual-QC decisions (all runs pass)
  - mean FD < 0.5 mm
  - task completeness (n_pairs == 6)
  - all three task runs present with exactly 231 volumes

FC availability: l_amyg-ant_vmPFC_neg_vs_neu must be nonmissing (shared
check_fc_col). Diary sections additionally require diary participation.
The PANAS section does not require diary participation.

Methods: Pearson correlation and participant-level OLS (available MR1 P5
analytic inputs do not contain a usable family or other grouping identifier).

All required predictors and outcomes are validated strictly by run_analysis_set;
the script aborts if any are absent. No section runs silently on a subset.

Do not condition execution on whether the primary analysis is significant.

Each output directory contains: correlations.csv, regressions.csv, _methods.txt

Privacy: aggregate console output only — never prints participant IDs or rows.

Run from MR1_validation/ directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_utils import (
    RESULTS_DIR, load_master, get_samples, get_covariates,
    run_analysis_set, save_results,
)

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = RESULTS_DIR / "04_fc_affect"

PRIMARY_FC = "l_amyg-ant_vmPFC_neg_vs_neu"

DIARY_OUTCOMES = ["PA_score", "NA_score"]

RIGHT_PREDICTORS = [
    "r_amyg-ant_vmPFC_neg_vs_neu",
    "r_amyg-post_vmPFC_neg_vs_neu",
]

POS_NEU_SOURCE_COLUMNS = [
    "l_amyg-ant_vmPFC_pos",
    "l_amyg-ant_vmPFC_neu",
    "l_amyg-post_vmPFC_pos",
    "l_amyg-post_vmPFC_neu",
]

POS_NEU_PREDICTORS = [
    "l_amyg-ant_vmPFC_pos_vs_neu",
    "l_amyg-post_vmPFC_pos_vs_neu",
]

PANAS_PREDICTORS = [
    "l_amyg-ant_vmPFC_neg_vs_neu",
    "l_amyg-post_vmPFC_neg_vs_neu",
]

PANAS_OUTCOMES = [
    "RA5SPGP",
    "RA5SPGN",
    "RA5SPGN_log",
]

NEG_ONLY_PREDICTORS = [
    "l_amyg-ant_vmPFC_neg",
    "l_amyg-post_vmPFC_neg",
]

EXPECTED_DIRECTIONS = {
    "PA_score":    +1,
    "NA_score":    -1,
    "RA5SPGP":     +1,
    "RA5SPGN":     -1,
    "RA5SPGN_log": -1,
}


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Analysis 04 (MR1) — Sensitivity Analyses")
    print("=" * 70)

    df = load_master(fc=True)

    # Require all four source columns before computing pos−neu contrasts.
    missing_src = [c for c in POS_NEU_SOURCE_COLUMNS if c not in df.columns]
    if missing_src:
        print(
            f"ERROR: pos−neu contrast requires source columns absent from data: "
            f"{sorted(missing_src)}"
        )
        sys.exit(1)
    df["l_amyg-ant_vmPFC_pos_vs_neu"]  = df["l_amyg-ant_vmPFC_pos"]  - df["l_amyg-ant_vmPFC_neu"]
    df["l_amyg-post_vmPFC_pos_vs_neu"] = df["l_amyg-post_vmPFC_pos"] - df["l_amyg-post_vmPFC_neu"]

    # Shared diary + fMRI + FC sample
    _, cons = get_samples(df, check_fc_col=PRIMARY_FC, require_diary=True)
    print(f"  Conservative N (diary+fMRI+FC) = {len(cons)}")
    base_covs = get_covariates(cons)

    # ── 1. Right amygdala neg−neu (two-tailed; hemisphere specificity) ─────────
    print("\n--- Sensitivity: Right Amygdala (neg−neu) ---")
    corr, ols = run_analysis_set(cons, RIGHT_PREDICTORS, DIARY_OUTCOMES, base_covs)
    save_results(corr, ols, BASE_DIR / "sensitivity_right_amygdala",
                 label="04 FC-R (neg−neu) → Affect  [conservative, two-tailed]",
                 predictors=RIGHT_PREDICTORS, outcomes=DIARY_OUTCOMES,
                 covariates=base_covs, n=len(cons))

    # ── 2. pos−neu contrast (two-tailed; condition specificity) ───────────────
    print("\n--- Sensitivity: pos−neu contrast ---")
    corr, ols = run_analysis_set(cons, POS_NEU_PREDICTORS, DIARY_OUTCOMES, base_covs)
    save_results(corr, ols, BASE_DIR / "sensitivity_pos_vs_neu",
                 label="04 FC (pos−neu) → Affect  [conservative, two-tailed]",
                 predictors=POS_NEU_PREDICTORS, outcomes=DIARY_OUTCOMES,
                 covariates=base_covs, n=len(cons))

    # ── 3. PANAS (one-tailed; convergent validity; no diary requirement) ───────
    print("\n--- Sensitivity: PANAS ---")
    _, panas_cons = get_samples(df, check_fc_col=PRIMARY_FC, require_diary=False)
    panas_covs = get_covariates(panas_cons)
    print(f"  PANAS N (conservative fMRI) = {len(panas_cons)}")
    corr, ols = run_analysis_set(
        panas_cons, PANAS_PREDICTORS, PANAS_OUTCOMES, panas_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, BASE_DIR / "sensitivity_panas",
                 label="04 FC (neg−neu) → PANAS  [conservative fMRI, one-tailed]",
                 predictors=PANAS_PREDICTORS, outcomes=PANAS_OUTCOMES,
                 covariates=panas_covs, n=len(panas_cons),
                 one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)

    # ── 4. Neg condition only (one-tailed; operationalisation robustness) ──────
    print("\n--- Sensitivity: Neg condition only ---")
    corr, ols = run_analysis_set(
        cons, NEG_ONLY_PREDICTORS, DIARY_OUTCOMES, base_covs,
        one_tailed=True, expected_directions=EXPECTED_DIRECTIONS,
    )
    save_results(corr, ols, BASE_DIR / "sensitivity_neg_condition",
                 label="04 FC (neg only) → Affect  [conservative, one-tailed]",
                 predictors=NEG_ONLY_PREDICTORS, outcomes=DIARY_OUTCOMES,
                 covariates=base_covs, n=len(cons),
                 one_tailed=True, expected_directions=EXPECTED_DIRECTIONS)


if __name__ == "__main__":
    main()
