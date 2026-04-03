#!/usr/bin/env python3
"""
sensitivity_novel_sample.py

Sensitivity analysis for Analysis 02 (persistence–affect replication) restricted
to participants NOT present in the Puccetti et al. (2021) MIDUS II sample.

Excludes the 26 participants who completed both MIDUS II and MIDUS III
neuroimaging, leaving the 54 novel participants only. Tests whether the
persistence–affect associations hold in participants for whom this is a
genuinely new finding rather than a test-retest.

Uses the same conservative sample definition, covariates, and three statistical
approaches (correlation, OLS, MLM) as script 02_persistence_affect.py.

Inputs:
  data/raw/M2P2_variables.csv          — MIDUS II diary data (to define diary sample)
  data/raw/M2P5_variables.csv          — MIDUS II neuroimaging completion (B5IC)
  data/processed/midus_with_fmri.csv   — current study master dataset

Run from project root directory.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from analysis_utils import (
    load_master, get_samples, get_covariates, prepare_persistence_vars,
    run_analysis_set, RESULTS_DIR, DIARY_OUTCOMES,
)

M2P2_FILE = Path("data/raw/M2P2_variables.csv")
M2P5_FILE = Path("data/raw/M2P5_variables.csv")
OUT_DIR   = RESULTS_DIR / "sensitivity_novel_sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERSIST_VAR = "neg_persist_crossrun_mean_z_L"
OUTCOMES    = ["PA_score", "NA_score", "NA_score_log"]
EXPECTED    = {"PA_score": -1, "NA_score": 1, "NA_score_log": 1}


def get_midus2_ids():
    """Return set of M2IDs present in MIDUS II diary + neuroimaging sample."""
    p2 = pd.read_csv(M2P2_FILE)
    p2["M2ID"] = pd.to_numeric(p2["M2ID"], errors="coerce").astype("Int64")
    diary_cols = [c for c in p2.columns if c.upper().startswith("B2DC")]
    has_diary = set(p2.loc[p2[diary_cols].notna().any(axis=1), "M2ID"].dropna().astype(int))

    p5 = pd.read_csv(M2P5_FILE)
    p5["M2ID"] = pd.to_numeric(p5["M2ID"], errors="coerce").astype("Int64")
    ic_col = next((c for c in p5.columns if "b5ic" in c.lower()), None)
    has_imaging = set(
        p5.loc[p5[ic_col] == 1, "M2ID"].dropna().astype(int)
        if ic_col else p5["M2ID"].dropna().astype(int)
    )
    return has_diary & has_imaging


def main():
    print("=" * 70)
    print("Sensitivity: Persistence–Affect in Novel (Non-Overlapping) Sample")
    print("=" * 70)

    # ── Identify MIDUS II participants ────────────────────────────────────────
    midus2_ids = get_midus2_ids()
    print(f"\nMIDUS II diary+neuroimaging sample: N = {len(midus2_ids)}")

    # ── Load and filter to conservative sample ────────────────────────────────
    df = load_master()
    prepare_persistence_vars(df)
    _, conservative = get_samples(df)

    df_m2id = pd.to_numeric(conservative["M2ID"], errors="coerce").astype("Int64")
    overlap     = conservative[df_m2id.isin(midus2_ids)]
    novel       = conservative[~df_m2id.isin(midus2_ids)]

    print(f"Conservative diary+fMRI sample:     N = {len(conservative)}")
    print(f"  Overlapping with MIDUS II:         N = {len(overlap)}")
    print(f"  Novel participants only:            N = {len(novel)}")

    base_covs = get_covariates(novel)

    # ── Run all three analyses on novel sample ────────────────────────────────
    print("\nRunning correlation + OLS + MLM on novel sample...")
    corr_df, ols_df, mlm_df = run_analysis_set(
        df=novel,
        predictors=[PERSIST_VAR],
        outcomes=OUTCOMES,
        base_covariates=base_covs,
        one_tailed=True,
        expected_directions=EXPECTED,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    corr_df.to_csv(OUT_DIR / "novel_sample_correlations.csv", index=False)
    ols_df.to_csv(OUT_DIR  / "novel_sample_regressions.csv",  index=False)
    mlm_df.to_csv(OUT_DIR  / "novel_sample_mlm.csv",          index=False)
    print(f"Saved to: {OUT_DIR}")

    # ── Print comparison ──────────────────────────────────────────────────────
    print("\n── Correlations ──")
    print(corr_df[["predictor", "outcome", "n", "r", "p"]].to_string(index=False))
    print("\n── OLS ──")
    print(ols_df[["predictor", "outcome", "n", "beta", "se", "p"]].to_string(index=False))
    print("\n── MLM ──")
    print(mlm_df[["predictor", "outcome", "n", "beta", "se", "p"]].to_string(index=False))


if __name__ == "__main__":
    main()
