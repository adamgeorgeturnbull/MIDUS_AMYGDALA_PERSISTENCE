#!/usr/bin/env python3
"""
00e_motion_check.py

Motion sensitivity analyses for the MIDUS Amygdala Persistence project.

Part 1 — Motion as predictor:
  Test whether overall mean FD and condition-window FD (negative trials) predict
  the primary fMRI outcomes (left amygdala persistence and amygdala–vmPFC FC)
  using correlation, OLS, and MLM in the conservative diary+fMRI sample.
  Two-tailed tests (no directional prior for motion–signal relationships).

Part 2 — Motion-controlled replication:
  Candidate effects are selected from the existing conservative correlation
  results for scripts 02 (persistence) and 04 (FC): any pair significant at
  p < .05 in those unadjusted correlations is carried forward. Each selected
  pair is then re-tested with OLS and MLM, with each motion measure added as an
  extra covariate, to report whether the effect survives motion control.
  Correlations are not re-run here: run_correlation() cannot adjust for
  covariates, so there is no motion-controlled correlation to report.

Motion measures:
  fd_mean_across_runs  — mean FD averaged across all three task runs
  fd_neg_mean          — mean FD during negative-image trials (persistence window)

Run from project root directory.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Allow importing from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from analysis_utils import (
    load_master, get_samples, get_covariates, prepare_persistence_vars,
    run_ols, run_mlm,
    run_analysis_set, RESULTS_DIR, FMRI_DIR, DIARY_OUTCOMES,
)

# ============================================================================
# Config
# ============================================================================
OUT_DIR = RESULTS_DIR / "00e_motion_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

P_THRESH = 0.05

PERSIST_VAR = "neg_persist_crossrun_mean_z_L"
FC_VARS     = ["l_amyg-ant_vmPFC_neg_vs_neu", "l_amyg-post_vmPFC_neg_vs_neu"]
OUTCOMES    = ["PA_score", "NA_score", "NA_score_log"]

MOTION_VARS = {
    "fd_mean_across_runs": "Mean FD — all runs",
    "fd_neg_mean":         "Mean FD — negative-condition trials",
}

# Expected directions for primary analyses (used in Part 2)
PERSIST_DIRECTIONS = {"PA_score": False, "NA_score": True, "NA_score_log": True}
FC_DIRECTIONS      = {"PA_score": True,  "NA_score": False, "NA_score_log": False}


# ============================================================================
# Data loading
# ============================================================================
def load_data():
    df = load_master(fc=True)
    prepare_persistence_vars(df)

    for mv in ["fd_neg_mean", "fd_mean_across_runs"]:
        n = df[mv].notna().sum() if mv in df.columns else 0
        print(f"  {mv}: N={n}")

    return df


# ============================================================================
# Part 1: Motion → fMRI metrics
# ============================================================================
def part1_motion_as_predictor(df):
    print("\n" + "=" * 60)
    print("Part 1: Motion as predictor of fMRI metrics")
    print("=" * 60)

    # Two samples: persistence sample (diary + fMRI QC), FC sample (also needs FC data)
    _, cons_persist = get_samples(df, require_diary=False)
    _, cons_fc      = get_samples(df, check_fc_col=FC_VARS[0], require_diary=False)

    fmri_targets = {
        PERSIST_VAR: cons_persist,
        FC_VARS[0]:  cons_fc,
        FC_VARS[1]:  cons_fc,
    }

    for motion_var, motion_label in MOTION_VARS.items():
        print(f"\n── {motion_label} ({motion_var}) ──")

        all_corr, all_ols, all_mlm = [], [], []

        for fmri_var, sample in fmri_targets.items():
            if motion_var not in sample.columns:
                print(f"  Skipping {motion_var} — not in data")
                continue
            sub = sample.dropna(subset=[motion_var])
            if fmri_var not in sub.columns:
                print(f"  Skipping {fmri_var} — not in data")
                continue

            base_covs = get_covariates(sub)

            corr_df, ols_df, mlm_df = run_analysis_set(
                df=sub,
                predictors=[motion_var],
                outcomes=[fmri_var],
                base_covariates=base_covs,
                one_tailed=False,  # two-tailed — no directional prior
            )
            all_corr.append(corr_df)
            all_ols.append(ols_df)
            all_mlm.append(mlm_df)

        corr_out = pd.concat([d for d in all_corr if not d.empty], ignore_index=True)
        ols_out  = pd.concat([d for d in all_ols  if not d.empty], ignore_index=True)
        mlm_out  = pd.concat([d for d in all_mlm  if not d.empty], ignore_index=True)

        if not corr_out.empty:
            corr_out.to_csv(OUT_DIR / f"part1_{motion_var}_correlations.csv", index=False)
            print(corr_out[["predictor", "outcome", "n", "r", "p"]].to_string(index=False))
        if not ols_out.empty:
            ols_out.to_csv(OUT_DIR / f"part1_{motion_var}_regressions.csv", index=False)
        if not mlm_out.empty:
            mlm_out.to_csv(OUT_DIR / f"part1_{motion_var}_mlm.csv", index=False)


# ============================================================================
# Part 2: Motion-controlled replication
# ============================================================================
def get_significant_pairs():
    """
    Select the candidate effects to re-test under motion control.

    Reads the existing conservative correlation results for scripts 02 and 04 and
    returns every (predictor, outcome, expected_positive) with p < P_THRESH.
    Selection is therefore based on the unadjusted primary correlations; the
    motion-controlled tests themselves are OLS and MLM (part2_motion_controlled).
    """
    result_files = [
        RESULTS_DIR / "02_persistence_affect" / "correlations.csv",
        RESULTS_DIR / "04_fc_affect"           / "correlations.csv",
    ]

    sig_pairs = []
    for fpath in result_files:
        if not fpath.exists():
            print(f"  Results file not found (skipping): {fpath.name}")
            continue
        res = pd.read_csv(fpath)
        if "p" not in res.columns:
            continue
        for _, row in res.iterrows():
            if pd.isna(row["p"]) or row["p"] >= P_THRESH:
                continue
            pred = row["predictor"]
            out  = row["outcome"]
            if pred == PERSIST_VAR:
                exp_pos = PERSIST_DIRECTIONS.get(out, True)
            else:
                exp_pos = FC_DIRECTIONS.get(out, True)
            pair = (pred, out, exp_pos)
            if pair not in sig_pairs:
                sig_pairs.append(pair)

    return sig_pairs


def part2_motion_controlled(df, sig_pairs):
    print("\n" + "=" * 60)
    print("Part 2: Motion-controlled replication (OLS + MLM only)")
    print("=" * 60)

    if not sig_pairs:
        print("  No significant primary effects found — nothing to rerun.")
        return

    print(f"\n  Pairs significant at p < {P_THRESH} in the primary (unadjusted)")
    print(f"  correlations, to be re-tested with motion as a covariate:")
    for pred, out, exp_pos in sig_pairs:
        dirn = "positive" if exp_pos else "negative"
        print(f"    {pred}  →  {out}  (expected {dirn})")

    # Separate conservative samples for persistence and FC predictors
    _, cons_persist = get_samples(df)
    _, cons_fc      = get_samples(df, check_fc_col=FC_VARS[0])

    for motion_var, motion_label in MOTION_VARS.items():
        print(f"\n── Adding covariate: {motion_label} ({motion_var}) ──")

        rows = []
        for pred, out, exp_pos in sig_pairs:
            sample = cons_fc if pred in FC_VARS else cons_persist

            if motion_var not in sample.columns:
                print(f"  Skipping {motion_var} — not in data")
                continue
            if pred not in sample.columns or out not in sample.columns:
                continue

            sub = sample.dropna(subset=[motion_var])
            base_covs = get_covariates(sub)
            diary_covs = [c for c in ["time_P2_P5", "n_days_complete"] if c in sub.columns]
            covs = base_covs + [motion_var] + (diary_covs if out in DIARY_OUTCOMES else [])

            o = run_ols(sub, pred, out, covs, one_tailed=True, expected_positive=exp_pos)
            m = run_mlm(sub, pred, out, covs, one_tailed=True, expected_positive=exp_pos)

            for res, atype in [(o, "ols"), (m, "mlm")]:
                if res:
                    res["analysis_type"] = atype
                    res["motion_covariate"] = motion_var
                    rows.append(res)

        if not rows:
            print("  No results.")
            continue

        res_df = pd.DataFrame(rows)
        out_path = OUT_DIR / f"part2_motion_controlled_{motion_var}.csv"
        res_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.name}")

        # Summary table — Part 2 contains only OLS and MLM, so beta is the statistic
        print(f"\n  {'Analysis':<14} {'Predictor':<40} {'Outcome':<16} {'beta':>7} {'p':>7} {'sig'}")
        print("  " + "-" * 90)
        for (pred, out, _) in sig_pairs:
            for atype in ["ols", "mlm"]:
                row = res_df[(res_df["predictor"] == pred) &
                             (res_df["outcome"]   == out)  &
                             (res_df["analysis_type"] == atype)]
                if row.empty:
                    continue
                r   = row.iloc[0]
                p   = r["p"]
                beta = r.get("beta", np.nan)
                sig = "✓" if p < P_THRESH else "✗"
                print(f"  {atype:<14} {pred:<40} {out:<16} {beta:>7.3f} {p:>7.3f}  {sig}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("00e — Motion Sensitivity Analyses")
    print("=" * 70)

    df = load_data()
    part1_motion_as_predictor(df)

    sig_pairs = get_significant_pairs()
    part2_motion_controlled(df, sig_pairs)

    print(f"\nDone. All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
