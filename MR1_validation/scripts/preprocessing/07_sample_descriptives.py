#!/usr/bin/env python3
"""
07_sample_descriptives.py (MR1)

Sample descriptives (N, age, sex, race) for the MR1 analytic samples reported
in the MIDUS Amygdala Persistence replication (Analyses 02–05).

IMPORTANT: run scripts 08 and 09 before this script to ensure that
data/processed/mr1_with_fmri.csv exists and is current.

Behavioral samples — source: data/processed/mr1_merged_clean.csv
------------------------------------------------------------------
daily_diary
    PA_score or NA_score nonmissing.  Age variable: RA2PAGE.

neuroscience_age
    RA5PAGE nonmissing.  Age variable: RA5PAGE.
    Availability of neuroscience-visit age (broader than the diary overlap).

diary_neuroscience_age_overlap
    daily_diary criterion AND RA5PAGE nonmissing.  Age variable: RA5PAGE.

neuroscience_panas
    RA5SPGP or RA5SPGN nonmissing.  Age variable: RA5PAGE.

diary_panas_overlap
    daily_diary criterion AND neuroscience_panas criterion.  Age: RA5PAGE.

fMRI samples — source: data/processed/mr1_with_fmri.csv via load_master()
---------------------------------------------------------------------------
Samples are derived using the same analysis_utils functions the analysis
scripts use, so the Ns here match analysis Ns exactly.

fmri_conservative
    load_master(fc=False) → prepare_persistence_vars() →
    get_samples(require_diary=False), conservative sample.
    Describes Analysis 03 (age → persistence) and the PANAS sensitivity
    blocks of Analysis 02.

fmri_fc_conservative
    load_master(fc=True) → prepare_persistence_vars() →
    get_samples(check_fc_col=PRIMARY_FC, require_diary=False), conservative.
    Describes Analysis 05 (FC → persistence) and the PANAS sensitivity
    blocks of Analysis 04.

diary_fmri_conservative
    load_master(fc=False) → prepare_persistence_vars() → get_samples(),
    conservative sample (diary required).
    Describes Analysis 02 (persistence → affect).

diary_fmri_fc_conservative
    load_master(fc=True) → prepare_persistence_vars() →
    get_samples(check_fc_col=PRIMARY_FC), conservative (diary required).
    Describes Analysis 04 (FC → affect).

Coding conventions
------------------
sex   1 = male, 2 = female
      (harmonized from RA1PRSEX primary, RAACRSEX fallback)
race  1 = White, 2 = Black, 3 = Native American, 4 = Asian,
      5 = Pacific Islander, 6 = Other
      (harmonized from RA1PF7A primary, RAACF7A fallback)

Sex percentages use participants with nonmissing sex as the denominator; race
percentages use participants with nonmissing race.  Every percentage column
names its denominator explicitly.  Education and ethnicity are not reported.
Race is reported as race (not "race/ethnicity").

PRIMARY_FC must be present in the fMRI file before get_samples is called for
FC samples; the column check is explicit so that a missing FC column cannot
silently inflate reported FC sample sizes.

Privacy: aggregate counts only — never prints participant IDs or rows.

Outputs
-------
results/tables/sample_descriptives.csv

Run from MR1_validation/ directory.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from analysis_utils import (
    PROCESSED_DIR, MASTER_FILE,
    load_master, prepare_persistence_vars, get_samples,
)

RESULTS_DIR = Path("results/tables")
OUTPUT_FILE = RESULTS_DIR / "sample_descriptives.csv"

BEH_FILE = PROCESSED_DIR / "mr1_merged_clean.csv"

# Primary FC predictor (Analyses 04, 05).  Required explicitly before
# get_samples to prevent a missing column from silently inflating sample Ns.
PRIMARY_FC = "l_amyg-ant_vmPFC_neg_vs_neu"

KEY = "MIDUSID"

SEX_CODES = {1: "male", 2: "female"}
RACE_CODES = {
    1: "white",
    2: "black",
    3: "native_american",
    4: "asian",
    5: "pacific_islander",
    6: "other",
}

BEH_REQUIRED = [
    "MIDUSID", "RA2PAGE", "RA5PAGE", "sex", "race",
    "PA_score", "NA_score", "RA5SPGP", "RA5SPGN",
]

FMRI_REQUIRED = [
    "MIDUSID", "RA5PAGE", "sex", "race",
    "PA_score", "NA_score",
    "has_neg_persistence", "qc_conservative",
]


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _require_columns(df, required, label):
    """Exit if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: {label} is missing required column(s): {missing}")
        sys.exit(1)


def _check_unique_key(df, label):
    """Exit unless there is exactly one row per MIDUSID.  Prints counts only."""
    if KEY not in df.columns:
        print(f"ERROR: {label}: no {KEY} column")
        sys.exit(1)
    n_dup = int(df[KEY].duplicated().sum())
    if n_dup:
        print(
            f"ERROR: {label}: {n_dup} duplicate {KEY} row(s) in {len(df)} rows "
            f"(expected one row per participant)"
        )
        sys.exit(1)


def _validate_codes(df, col, allowed, label):
    """
    Coerce df[col] to numeric and confirm every nonmissing value is in `allowed`.

    Aborts if nonmissing values are lost to numeric coercion (they would
    otherwise be silently counted as missing) or if unexpected codes appear.
    Returns the validated numeric Series.
    """
    raw     = df[col]
    numeric = pd.to_numeric(raw, errors="coerce")

    lost = raw[raw.notna() & numeric.isna()]
    if len(lost) > 0:
        print(f"ERROR: {label}: {len(lost)} nonmissing '{col}' value(s) are not numeric:")
        print(lost.value_counts().sort_index().to_string())
        sys.exit(1)

    observed   = set(numeric.dropna().unique())
    unexpected = sorted(v for v in observed if v not in allowed)
    if unexpected:
        print(
            f"ERROR: {label}: unexpected '{col}' code(s) {unexpected} "
            f"(allowed: {sorted(allowed)}).  Frequencies:"
        )
        print(numeric[numeric.isin(unexpected)].value_counts().sort_index().to_string())
        sys.exit(1)

    return numeric


def _validate_demographics(df, label):
    """Validate MIDUSID uniqueness and sex/race codes; return df with numeric codes."""
    _check_unique_key(df, label)
    df["sex"]  = _validate_codes(df, "sex",  set(SEX_CODES),  label)
    df["race"] = _validate_codes(df, "race", set(RACE_CODES), label)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Descriptives
# ─────────────────────────────────────────────────────────────────────────────
def describe_sample(df, age_col, label, analyses):
    """
    Return one descriptives row for `df`.

    Sex percentages are computed among participants with nonmissing sex; race
    percentages among participants with nonmissing race.  Internal consistency
    is asserted: male + female + sex_missing == N, and the six race counts +
    race_missing == N.
    """
    _check_unique_key(df, f"sample '{label}'")
    if age_col not in df.columns:
        print(f"ERROR: sample '{label}': age column '{age_col}' not present")
        sys.exit(1)

    n   = len(df)
    age = pd.to_numeric(df[age_col], errors="coerce").dropna()
    sex = pd.to_numeric(df["sex"],  errors="coerce")
    race = pd.to_numeric(df["race"], errors="coerce")

    n_sex         = int(sex.notna().sum())
    n_sex_missing = int(sex.isna().sum())
    n_male        = int((sex == 1).sum())
    n_female      = int((sex == 2).sum())

    if n_male + n_female + n_sex_missing != n:
        print(
            f"ERROR: sample '{label}': male({n_male}) + female({n_female}) + "
            f"sex missing({n_sex_missing}) != N({n})"
        )
        sys.exit(1)

    n_race         = int(race.notna().sum())
    n_race_missing = int(race.isna().sum())
    race_counts    = {name: int((race == code).sum()) for code, name in RACE_CODES.items()}

    if sum(race_counts.values()) + n_race_missing != n:
        print(
            f"ERROR: sample '{label}': race counts "
            f"({sum(race_counts.values())}) + race missing({n_race_missing}) != N({n})"
        )
        sys.exit(1)

    def _pct(k, denom):
        return round(k / denom * 100, 1) if denom > 0 else float("nan")

    def _stat(fn, min_n=1):
        return round(fn(), 1) if len(age) >= min_n else float("nan")

    row = {
        "sample":                       label,
        "analyses":                     analyses,
        "age_variable":                 age_col,
        "N":                            n,
        "N_age":                        len(age),
        "age_mean":                     _stat(age.mean),
        "age_SD":                       _stat(age.std, min_n=2),
        "age_min":                      _stat(age.min),
        "age_max":                      _stat(age.max),
        "N_sex_nonmissing":             n_sex,
        "n_male":                       n_male,
        "pct_male_of_nonmissing_sex":   _pct(n_male, n_sex),
        "n_female":                     n_female,
        "pct_female_of_nonmissing_sex": _pct(n_female, n_sex),
        "n_sex_missing":                n_sex_missing,
        "N_race_nonmissing":            n_race,
        "n_race_missing":               n_race_missing,
    }
    for code, name in RACE_CODES.items():
        row[f"n_{name}"]                     = race_counts[name]
        row[f"pct_{name}_of_nonmissing_race"] = _pct(race_counts[name], n_race)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("07 — Sample descriptives (MR1)")
    print("=" * 78)
    print("Aggregate counts only — never prints participant IDs or rows.")
    print(f"sex codes  : {SEX_CODES}")
    print(f"race codes : {RACE_CODES}")

    rows = []

    # ========================================================================
    # Behavioral dataset
    # ========================================================================
    print(f"\nBehavioral source: {BEH_FILE}")
    if not BEH_FILE.exists():
        sys.exit(f"FATAL: {BEH_FILE} not found. Run scripts 01–06 first.")
    beh = pd.read_csv(BEH_FILE)
    print(f"  {len(beh)} rows")

    _require_columns(beh, BEH_REQUIRED, BEH_FILE.name)
    beh = _validate_demographics(beh, BEH_FILE.name)

    mask_diary     = beh[["PA_score", "NA_score"]].notna().any(axis=1)
    mask_neuro_age = beh["RA5PAGE"].notna()
    mask_panas     = beh[["RA5SPGP", "RA5SPGN"]].notna().any(axis=1)

    behavioral_samples = [
        (beh[mask_diary],                   "RA2PAGE", "daily_diary",
         "diary affect availability (PA_score / NA_score)"),
        (beh[mask_neuro_age],               "RA5PAGE", "neuroscience_age",
         "neuroscience-visit age availability (broader than diary overlap)"),
        (beh[mask_diary & mask_neuro_age],  "RA5PAGE", "diary_neuroscience_age_overlap",
         "diary affect AND neuroscience-visit age"),
        (beh[mask_panas],                   "RA5PAGE", "neuroscience_panas",
         "PANAS availability (RA5SPGP / RA5SPGN)"),
        (beh[mask_diary & mask_panas],      "RA5PAGE", "diary_panas_overlap",
         "diary affect AND PANAS availability"),
    ]
    for sub, age_col, label, analyses in behavioral_samples:
        rows.append(describe_sample(sub.copy(), age_col, label, analyses))
        print(f"  {label}: N={len(sub)}")

    # ========================================================================
    # fMRI datasets
    # ========================================================================
    print(f"\nfMRI source: {MASTER_FILE}")
    if not MASTER_FILE.exists():
        sys.exit(
            f"FATAL: {MASTER_FILE} not found. "
            "Run scripts 08 and 09 first to generate mr1_with_fmri.csv."
        )

    # -- Without FC merge (Analyses 02, 03) -----------------------------------
    df_nofc = load_master(fc=False)
    _require_columns(df_nofc, FMRI_REQUIRED, f"{MASTER_FILE.name} (fc=False)")
    df_nofc = _validate_demographics(df_nofc, f"{MASTER_FILE.name} (fc=False)")
    prepare_persistence_vars(df_nofc)

    # -- With FC merge (Analyses 04, 05) -------------------------------------
    # PRIMARY_FC is required explicitly so a missing column cannot silently
    # inflate the reported FC sample sizes.
    df_fc = load_master(fc=True)
    _require_columns(
        df_fc, FMRI_REQUIRED + [PRIMARY_FC],
        f"{MASTER_FILE.name} + FC merge",
    )
    df_fc = _validate_demographics(df_fc, f"{MASTER_FILE.name} + FC merge")
    prepare_persistence_vars(df_fc)

    _, fmri_cons          = get_samples(df_nofc, require_diary=False)
    _, fmri_fc_cons       = get_samples(df_fc, check_fc_col=PRIMARY_FC, require_diary=False)
    _, diary_fmri_cons    = get_samples(df_nofc)
    _, diary_fmri_fc_cons = get_samples(df_fc, check_fc_col=PRIMARY_FC)

    fmri_samples = [
        (fmri_cons,          "fmri_conservative",
         "03 age → persistence; 02 PANAS sensitivity samples"),
        (fmri_fc_cons,       "fmri_fc_conservative",
         "05 FC → persistence; 04 PANAS sensitivity samples"),
        (diary_fmri_cons,    "diary_fmri_conservative",
         "02 persistence → affect"),
        (diary_fmri_fc_cons, "diary_fmri_fc_conservative",
         "04 FC → affect"),
    ]
    for sub, label, analyses in fmri_samples:
        rows.append(describe_sample(sub.copy(), "RA5PAGE", label, analyses))
        print(f"  {label}: N={len(sub)}")

    # ========================================================================
    # Save and print
    # ========================================================================
    out = pd.DataFrame(rows).set_index("sample")
    out.to_csv(OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")

    core = [
        "N", "N_age", "age_mean", "age_SD", "age_min", "age_max",
        "n_male", "n_female", "n_sex_missing", "n_white", "n_race_missing",
    ]
    print("\nCore columns:")
    print(out[core].to_string())

    print("\nFull table:")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(out.to_string())


if __name__ == "__main__":
    main()
