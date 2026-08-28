#!/usr/bin/env python3
"""
supplementary_tables.py

Collects correlation, OLS regression, and MLM results for primary analyses
(01-05) and writes a single publication-ready supplementary table with all
three methods side by side for easy comparison.

Sample sizes
------------
Correlation, OLS, and MLM have their own complete-case Ns: each method drops
different rows depending on which covariates it includes and, for MLM, on
family structure.  The table therefore reports three separate N columns
("N correlation", "N OLS", "N MLM"), each read from that method's own result
file.  A method's N is never copied from another method.  When a method has no
row for a given predictor x outcome pair, its N and its statistics are left
blank.

The predictor x outcome pairs are the union of pairs found across the
correlation, OLS, and MLM files, so a pair present in only one file is still
reported.

Output
------
results/tables/supplementary/supp_all_methods.csv

Run from project root directory.
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results/tables")
OUT_DIR     = RESULTS_DIR / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Human-readable labels ─────────────────────────────────────────────────────
PREDICTOR_LABELS = {
    "C2PAGE":                                   "Age (Project 2)",
    "C5PAGE":                                   "Age (Project 5)",
    "neg_persist_crossrun_mean_z_L":            "Neg. persistence, left amygdala",
    "neg_persist_crossrun_mean_z_R":            "Neg. persistence, right amygdala",
    "l_amyg-ant_vmPFC_neg_vs_neu":              "Left amygdala-ant. vmPFC FC",
    "l_amyg-post_vmPFC_neg_vs_neu":             "Left amygdala-post. vmPFC FC",
    "r_amyg-ant_vmPFC_neg_vs_neu":              "Right amygdala-ant. vmPFC FC",
    "r_amyg-post_vmPFC_neg_vs_neu":             "Right amygdala-post. vmPFC FC",
    "l_amyg-ant_minus_post_vmPFC_neg_vs_neu":   "Left amygdala ant.-post. vmPFC FC contrast",
    "r_amyg-ant_minus_post_vmPFC_neg_vs_neu":   "Right amygdala ant.-post. vmPFC FC contrast",
}

OUTCOME_LABELS = {
    "PA_score":                      "Positive affect (diary)",
    "NA_score":                      "Negative affect (diary)",
    "NA_score_log":                  "Negative affect, log (diary)",
    "C5SPGP":                        "Positive affect (PANAS)",
    "C5SPGN":                        "Negative affect (PANAS)",
    "C5SPGN_log":                    "Negative affect, log (PANAS)",
    "neg_persist_crossrun_mean_z_L": "Neg. persistence, left amygdala",
    "neg_persist_crossrun_mean_z_R": "Neg. persistence, right amygdala",
}

# ── Analysis sections ─────────────────────────────────────────────────────────
# Each entry: (section label, corr_csv, reg_csv, mlm_csv)
ANALYSES = [
    ("Analysis 01: Age -> Daily Affect",
     "01_affect_age/correlations.csv",
     "01_affect_age/regressions.csv",
     "01_affect_age/mlm.csv"),

    ("Analysis 02: Persistence -> Daily Affect",
     "02_persistence_affect/correlations.csv",
     "02_persistence_affect/regressions.csv",
     "02_persistence_affect/mlm.csv"),

    ("Analysis 03: Age -> Persistence",
     "03_persistence_age/correlations.csv",
     "03_persistence_age/regressions.csv",
     "03_persistence_age/mlm.csv"),

    ("Analysis 04: FC -> Daily Affect",
     "04_fc_affect/correlations.csv",
     "04_fc_affect/regressions.csv",
     "04_fc_affect/mlm.csv"),

    ("Analysis 05: FC -> Persistence",
     "05_fc_persistence/correlations.csv",
     "05_fc_persistence/regressions.csv",
     "05_fc_persistence/mlm.csv"),
]

# ── Method specifications ─────────────────────────────────────────────────────
# Each method contributes its own N column plus its statistic and p-value.
# "stat" is the statistic column expected in that method's result file.
METHODS = [
    {"key": "corr", "label": "correlation", "stat": "r", "decimals": 3,
     "n_col": "N correlation", "stat_col": "r",     "p_col": "r p"},
    {"key": "ols",  "label": "OLS",         "stat": "t", "decimals": 2,
     "n_col": "N OLS",         "stat_col": "OLS t", "p_col": "OLS p"},
    {"key": "mlm",  "label": "MLM",         "stat": "z", "decimals": 2,
     "n_col": "N MLM",         "stat_col": "MLM z", "p_col": "MLM p"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def fmt(val, decimals=3):
    if pd.isna(val):
        return ""
    return f"{val:.{decimals}f}"


def fmt_p(p):
    """Format a p-value with its significance stars appended; blank if missing."""
    if pd.isna(p):
        return ""
    return f"{p:.3f}{stars(p)}"


def fmt_n(val):
    """Format a complete-case N as an integer string; blank if missing."""
    if pd.isna(val):
        return ""
    return str(int(val))


def label_pred(p):
    return PREDICTOR_LABELS.get(p, p)


def label_out(o):
    return OUTCOME_LABELS.get(o, o)


def load(path):
    p = RESULTS_DIR / path
    if not p.exists():
        print(f"  [skip] {p}")
        return None
    return pd.read_csv(p)


def build_lookup(df, path, stat_col, label):
    """
    Validate one method's result file and return {(predictor, outcome): row}.

    Aborts if the identifier, N, statistic, or p-value column is missing, or if
    any predictor x outcome pair appears more than once.  Silently taking the
    first of several duplicate rows would misreport which model produced the
    statistic and its N, so duplicates are a hard error.
    """
    required = ["predictor", "outcome", "n", stat_col, "p"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: {path} ({label}) is missing required column(s): {missing}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    dup_mask = df.duplicated(subset=["predictor", "outcome"], keep=False)
    if dup_mask.any():
        dup_pairs = df.loc[dup_mask, ["predictor", "outcome"]].drop_duplicates()
        print(f"ERROR: {path} ({label}) has {len(dup_pairs)} duplicated "
              f"predictor x outcome pair(s):")
        for pred, out in dup_pairs.itertuples(index=False, name=None):
            print(f"    {pred} -> {out}")
        sys.exit(1)

    return {(row["predictor"], row["outcome"]): row for _, row in df.iterrows()}


# ── Build combined table ──────────────────────────────────────────────────────
def build_table():
    rows = []

    for section, corr_path, reg_path, mlm_path in ANALYSES:
        paths = {"corr": corr_path, "ols": reg_path, "mlm": mlm_path}

        # Validate each available method file and index it by (predictor, outcome)
        lookups = {}
        for spec in METHODS:
            df = load(paths[spec["key"]])
            lookups[spec["key"]] = (
                None if df is None
                else build_lookup(df, paths[spec["key"]], spec["stat"], spec["label"])
            )

        # Index of predictor × outcome pairs = union across all three methods,
        # in first-seen order (correlation, then OLS, then MLM).  A pair present
        # in only one method's file is still reported.
        pairs = []
        for spec in METHODS:
            lut = lookups[spec["key"]]
            if lut is None:
                continue
            for pair in lut:
                if pair not in pairs:
                    pairs.append(pair)

        if not pairs:
            continue

        for pred, out in pairs:
            row = {
                "Analysis":  section,
                "Predictor": label_pred(pred),
                "Outcome":   label_out(out),
            }

            for spec in METHODS:
                lut = lookups[spec["key"]]
                res = None if lut is None else lut.get((pred, out))
                if res is None:
                    # This method has no row for the pair: leave its N and its
                    # statistics blank.  Never substitute another method's N.
                    row[spec["n_col"]]    = ""
                    row[spec["stat_col"]] = ""
                    row[spec["p_col"]]    = ""
                else:
                    row[spec["n_col"]]    = fmt_n(res["n"])
                    row[spec["stat_col"]] = fmt(res[spec["stat"]], spec["decimals"])
                    row[spec["p_col"]]    = fmt_p(res["p"])

            rows.append(row)

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Building supplementary table (correlations + OLS + MLM)…")
    df = build_table()
    out = OUT_DIR / "supp_all_methods.csv"
    df.to_csv(out, index=False)
    print(f"  Saved: {out}  ({len(df)} rows, {len(df.columns)} columns)")
    print("\nColumn groups:")
    print("  Identification: Analysis, Predictor, Outcome")
    print("  Correlation:    N correlation, r, r p")
    print("  OLS:            N OLS, OLS t, OLS p")
    print("  MLM:            N MLM, MLM z, MLM p")
    print("  (each N is the complete-case N from that method's own result file;")
    print("   Ns can differ across methods and are never copied between them)")
    print("  (a blank N and blank statistics mean that method has no row for the pair)")
    print("  (p-values are one-tailed for directional analyses, two-tailed otherwise)")
    print("  (significance appended to p: * p<.05, ** p<.01, *** p<.001)")


if __name__ == "__main__":
    main()
