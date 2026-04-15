#!/usr/bin/env python3
"""
supplementary_tables.py

Collects correlation, OLS regression, and MLM results for primary analyses
(01-05) and writes a single publication-ready supplementary table with all
three methods side by side for easy comparison.

Output
------
results/tables/supplementary/supp_all_methods.csv

Run from project root directory.
"""

from pathlib import Path
import numpy as np
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


# ── Build combined table ──────────────────────────────────────────────────────
def build_table():
    rows = []

    for section, corr_path, reg_path, mlm_path in ANALYSES:
        corr_df = load(corr_path)
        reg_df  = load(reg_path)
        mlm_df  = load(mlm_path)

        # Use correlation rows as the index of predictor × outcome pairs
        if corr_df is None:
            continue

        for _, c in corr_df.iterrows():
            pred = c["predictor"]
            out  = c["outcome"]

            def fmt_p(p):
                if pd.isna(p):
                    return ""
                s = stars(p)
                return f"{p:.3f}{s}"

            row = {
                "Analysis":  section,
                "Predictor": label_pred(pred),
                "Outcome":   label_out(out),
                "N":         int(c["n"]),
                # Correlation
                "r":   fmt(c["r"], 3),
                "r p": fmt_p(c.get("p", np.nan)),
            }

            # OLS
            if reg_df is not None:
                m = reg_df[(reg_df["predictor"] == pred) & (reg_df["outcome"] == out)]
                if len(m):
                    r = m.iloc[0]
                    row.update({
                        "OLS t": fmt(r["t"], 2),
                        "OLS p": fmt_p(r.get("p", np.nan)),
                    })
                else:
                    row.update({"OLS t": "", "OLS p": ""})

            # MLM
            if mlm_df is not None:
                m = mlm_df[(mlm_df["predictor"] == pred) & (mlm_df["outcome"] == out)]
                if len(m):
                    r = m.iloc[0]
                    row.update({
                        "MLM z": fmt(r["z"], 2),
                        "MLM p": fmt_p(r.get("p", np.nan)),
                    })
                else:
                    row.update({"MLM z": "", "MLM p": ""})

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
    print("  Identification: Analysis, Predictor, Outcome, N")
    print("  Correlation:    r, r p")
    print("  OLS:            OLS t, OLS p")
    print("  MLM:            MLM z, MLM p")
    print("  (p-values are one-tailed for directional analyses, two-tailed otherwise)")
    print("  (significance appended to p: * p<.05, ** p<.01, *** p<.001)")


if __name__ == "__main__":
    main()
