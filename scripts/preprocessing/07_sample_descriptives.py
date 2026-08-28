#!/usr/bin/env python3
"""
07_sample_descriptives.py

Sample descriptives (N, age, sex, race) for the analytic samples reported in the
MIDUS Amygdala Persistence paper.

Behavioral samples — source: data/processed/midus_merged_clean.csv
------------------------------------------------------------------
daily_diary
    At least one of PA_score / NA_score nonmissing.  Age variable: C2PAGE.
    The primary daily-diary sample analysed in Analysis 01.

neuroscience_age
    C5PAGE nonmissing.  Age variable: C5PAGE.
    Describes availability of neuroscience-visit age.  This is broader than the
    sample Analysis 01 actually analyses — see diary_neuroscience_age_overlap.

diary_neuroscience_age_overlap
    Daily-diary criterion AND C5PAGE nonmissing.  Age variable: C5PAGE.
    The exact neuroscience subsample analysed by Analysis 01, which requires
    both diary affect and neuroscience-visit age.

neuroscience_panas
    At least one of C5SPGP / C5SPGN nonmissing.  Age variable: C5PAGE.
    PANAS availability is defined from the PANAS variables themselves, never
    from C5PAGE availability.

diary_panas_overlap
    Daily-diary criterion AND neuroscience-PANAS criterion.  Age: C5PAGE.
    Participants with both diary affect and PANAS (e.g. Analysis 00a
    diary-PANAS convergence).

fMRI samples — source: data/processed/midus_with_fmri.csv via load_master()
---------------------------------------------------------------------------
fmri_conservative
    load_master(fc=False) → prepare_persistence_vars() →
    get_samples(require_diary=False), conservative sample.
    No diary requirement.  Analysis 03 (age → persistence).  The PANAS
    sensitivity blocks of 02_sensitivity.py and 06_sensitivity.py apply the
    same criteria (has_neg_persistence & qc_conservative, fc=False, no diary),
    so this row also describes those PANAS sensitivity samples.

fmri_fc_conservative
    load_master(fc=True) → prepare_persistence_vars() →
    get_samples(check_fc_col=PRIMARY_FC, require_diary=False), conservative.
    No diary requirement.  Analysis 05 (FC → persistence).  The PANAS
    sensitivity blocks of 04_sensitivity.py and 07_sensitivity.py apply the
    same criteria (has_neg_persistence & qc_conservative & PRIMARY_FC
    nonmissing, no diary), so this row also describes those samples.

diary_fmri_conservative
    load_master(fc=False) → prepare_persistence_vars() → get_samples(),
    conservative sample (diary required).
    Analyses 02 (persistence → affect) and 06 (persistence x ERQ moderation).

diary_fmri_fc_conservative
    load_master(fc=True) → prepare_persistence_vars() →
    get_samples(check_fc_col=PRIMARY_FC), conservative (diary required).
    Analyses 04 (FC → affect) and 07 (FC x ERQ moderation).

reappraisal_complete_case / suppression_complete_case
    diary_fmri_conservative restricted to nonmissing C5SER / C5SES.
    The complete-case Ns actually analysed by Analysis 06.  When the FC
    complete-case sample contains exactly the same participants, this single
    row describes both Analysis 06 and Analysis 07 and its analyses field says
    so explicitly.

reappraisal_complete_case_fc / suppression_complete_case_fc
    diary_fmri_fc_conservative restricted to nonmissing C5SER / C5SES.
    The complete-case Ns actually analysed by Analysis 07.  Emitted only when
    M2ID membership differs from the corresponding persistence sample above.
    Membership is compared as a set of M2IDs, because equal Ns do not imply
    identical membership.

Coding conventions
------------------
sex   1 = male, 2 = female
race  1 = White, 2 = Black, 3 = Native American, 4 = Asian,
      5 = Pacific Islander, 6 = Other

Sex percentages use participants with nonmissing sex as the denominator; race
percentages use participants with nonmissing race.  Every percentage column
names its denominator explicitly.  Ethnicity is not reported, and race is
reported as race (not "race/ethnicity").

Privacy: aggregate counts only — never prints participant IDs or rows.

Outputs
-------
results/tables/sample_descriptives.csv

Run from project root directory.
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

BEH_FILE = PROCESSED_DIR / "midus_merged_clean.csv"

# Primary FC predictor (Analyses 04, 05, 07).  Required explicitly: get_samples()
# silently drops the FC requirement when this column is absent, which would
# inflate the reported FC sample sizes.
PRIMARY_FC = "l_amyg-ant_vmPFC_neg_vs_neu"

SEX_CODES = {1: "male", 2: "female"}
RACE_CODES = {
    1: "white",
    2: "black",
    3: "native_american",
    4: "asian",
    5: "pacific_islander",
    6: "other",
}

MODERATORS = [("C5SER", "reappraisal"), ("C5SES", "suppression")]

BEH_REQUIRED = [
    "M2ID", "C2PAGE", "C5PAGE", "sex", "race",
    "PA_score", "NA_score", "C5SPGP", "C5SPGN",
]

FMRI_REQUIRED = [
    "M2ID", "C5PAGE", "sex", "race",
    "PA_score", "NA_score",
    "has_neg_persistence", "qc_conservative",
    "C5SER", "C5SES",
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


def _check_unique_m2id(df, label):
    """Exit unless there is exactly one row per M2ID.  Prints counts only."""
    if "M2ID" not in df.columns:
        print(f"ERROR: {label}: no M2ID column")
        sys.exit(1)
    n_dup_ids = int(df["M2ID"].duplicated().sum())
    if n_dup_ids:
        print(f"ERROR: {label}: {n_dup_ids} duplicate M2ID row(s) in {len(df)} rows "
              f"(expected one row per participant)")
        sys.exit(1)


def _validate_codes(df, col, allowed, label):
    """
    Coerce df[col] to numeric and confirm every nonmissing value is in `allowed`.

    Aborts if nonmissing values are lost to numeric coercion (they would
    otherwise be silently counted as missing) or if unexpected codes appear.
    Returns the validated numeric Series.
    """
    raw = df[col]
    numeric = pd.to_numeric(raw, errors="coerce")

    lost = raw[raw.notna() & numeric.isna()]
    if len(lost) > 0:
        print(f"ERROR: {label}: {len(lost)} nonmissing '{col}' value(s) are not numeric:")
        print(lost.value_counts().sort_index().to_string())
        sys.exit(1)

    observed = set(numeric.dropna().unique())
    unexpected = sorted(v for v in observed if v not in allowed)
    if unexpected:
        print(f"ERROR: {label}: unexpected '{col}' code(s) {unexpected} "
              f"(allowed: {sorted(allowed)}).  Frequencies:")
        print(numeric[numeric.isin(unexpected)].value_counts().sort_index().to_string())
        sys.exit(1)

    return numeric


def _validate_demographics(df, label):
    """Validate M2ID uniqueness and sex/race codes; return df with numeric codes."""
    _check_unique_m2id(df, label)
    df["sex"] = _validate_codes(df, "sex", set(SEX_CODES), label)
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
    is asserted: male + female + sex missing == N, and the six race counts +
    race missing == N.
    """
    _check_unique_m2id(df, f"sample '{label}'")
    if age_col not in df.columns:
        print(f"ERROR: sample '{label}': age column {age_col} not present")
        sys.exit(1)

    n = len(df)
    age = pd.to_numeric(df[age_col], errors="coerce").dropna()
    sex = pd.to_numeric(df["sex"], errors="coerce")
    race = pd.to_numeric(df["race"], errors="coerce")

    n_sex = int(sex.notna().sum())
    n_sex_missing = int(sex.isna().sum())
    n_male = int((sex == 1).sum())
    n_female = int((sex == 2).sum())

    if n_male + n_female + n_sex_missing != n:
        print(f"ERROR: sample '{label}': male({n_male}) + female({n_female}) + "
              f"sex missing({n_sex_missing}) != N({n})")
        sys.exit(1)

    n_race = int(race.notna().sum())
    n_race_missing = int(race.isna().sum())
    race_counts = {name: int((race == code).sum()) for code, name in RACE_CODES.items()}

    if sum(race_counts.values()) + n_race_missing != n:
        print(f"ERROR: sample '{label}': race counts "
              f"({sum(race_counts.values())}) + race missing({n_race_missing}) != N({n})")
        sys.exit(1)

    def _pct(k, denom):
        return round(k / denom * 100, 1) if denom > 0 else float("nan")

    def _stat(fn, min_n=1):
        return round(fn(), 1) if len(age) >= min_n else float("nan")

    row = {
        "sample":        label,
        "analyses":      analyses,
        "age_variable":  age_col,
        "N":             n,
        "N_age":         len(age),
        "age_mean":      _stat(age.mean),
        "age_SD":        _stat(age.std, min_n=2),
        "age_min":       _stat(age.min),
        "age_max":       _stat(age.max),
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
        row[f"n_{name}"] = race_counts[name]
        row[f"pct_{name}_of_nonmissing_race"] = _pct(race_counts[name], n_race)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("07 — Sample descriptives")
    print("=" * 78)
    print("Aggregate counts only — never prints participant IDs or rows.")
    print(f"sex codes  : {SEX_CODES}")
    print(f"race codes : {RACE_CODES}")

    rows = []

    # ========================================================================
    # Behavioral dataset
    # ========================================================================
    print(f"\nBehavioral source: {BEH_FILE}")
    beh = pd.read_csv(BEH_FILE)
    beh["M2ID"] = beh["M2ID"].astype(str)
    print(f"  {len(beh)} rows")

    _require_columns(beh, BEH_REQUIRED, BEH_FILE.name)
    beh = _validate_demographics(beh, BEH_FILE.name)

    mask_diary = beh[["PA_score", "NA_score"]].notna().any(axis=1)
    mask_neuro_age = beh["C5PAGE"].notna()
    mask_panas = beh[["C5SPGP", "C5SPGN"]].notna().any(axis=1)

    behavioral_samples = [
        (beh[mask_diary],                "C2PAGE", "daily_diary",
         "01 primary diary sample"),
        (beh[mask_neuro_age],            "C5PAGE", "neuroscience_age",
         "neuroscience-visit age availability (broader than the 01 neuro subsample)"),
        (beh[mask_diary & mask_neuro_age], "C5PAGE", "diary_neuroscience_age_overlap",
         "01 neuroscience-age subsample"),
        (beh[mask_panas],                "C5PAGE", "neuroscience_panas",
         "PANAS availability (C5SPGP/C5SPGN)"),
        (beh[mask_diary & mask_panas],   "C5PAGE", "diary_panas_overlap",
         "diary + PANAS availability (e.g. 00a diary-PANAS convergence)"),
    ]
    for sub, age_col, label, analyses in behavioral_samples:
        rows.append(describe_sample(sub.copy(), age_col, label, analyses))
        print(f"  {label}: N={len(sub)}")

    # ========================================================================
    # fMRI datasets
    # ========================================================================
    print(f"\nfMRI source: {MASTER_FILE}")

    # -- Without FC merge (Analyses 02, 03, 06) ------------------------------
    df_nofc = load_master(fc=False)
    _require_columns(df_nofc, FMRI_REQUIRED, f"{MASTER_FILE.name} (fc=False)")
    df_nofc = _validate_demographics(df_nofc, f"{MASTER_FILE.name} (fc=False)")
    prepare_persistence_vars(df_nofc)

    # -- With FC merge (Analyses 04, 05, 07) --------------------------------
    df_fc = load_master(fc=True)
    _require_columns(df_fc, FMRI_REQUIRED + [PRIMARY_FC],
                     f"{MASTER_FILE.name} + FC merge")
    df_fc = _validate_demographics(df_fc, f"{MASTER_FILE.name} + FC merge")
    prepare_persistence_vars(df_fc)

    _, fmri_cons = get_samples(df_nofc, require_diary=False)
    _, fmri_fc_cons = get_samples(df_fc, check_fc_col=PRIMARY_FC, require_diary=False)
    _, diary_fmri_cons = get_samples(df_nofc)
    _, diary_fmri_fc_cons = get_samples(df_fc, check_fc_col=PRIMARY_FC)

    fmri_samples = [
        (fmri_cons,            "fmri_conservative",
         "03 age -> persistence; 02 and 06 PANAS sensitivity samples"),
        (fmri_fc_cons,         "fmri_fc_conservative",
         "05 FC -> persistence; 04 and 07 PANAS sensitivity samples"),
        (diary_fmri_cons,      "diary_fmri_conservative",
         "02 persistence -> affect; 06 persistence x ERQ"),
        (diary_fmri_fc_cons,   "diary_fmri_fc_conservative",
         "04 FC -> affect; 07 FC x ERQ"),
    ]
    for sub, label, analyses in fmri_samples:
        rows.append(describe_sample(sub.copy(), "C5PAGE", label, analyses))
        print(f"  {label}: N={len(sub)}")

    # ========================================================================
    # Moderation complete-case samples
    # ========================================================================
    print("\nModeration complete-case samples:")
    for mod_var, mod_name in MODERATORS:
        persist_cc = diary_fmri_cons[diary_fmri_cons[mod_var].notna()].copy()
        fc_cc = diary_fmri_fc_cons[diary_fmri_fc_cons[mod_var].notna()].copy()

        # Compare membership as a set of M2IDs before describing either sample:
        # equal Ns do not imply equal membership, so an N-only comparison would
        # be unsafe here.  The persistence row's analyses description depends on
        # the outcome, so this must be resolved first.
        identical = set(persist_cc["M2ID"]) == set(fc_cc["M2ID"])
        print(f"  {mod_name}: persistence N={len(persist_cc)}, FC N={len(fc_cc)}, "
              f"identical membership={identical}")

        if identical:
            rows.append(describe_sample(
                persist_cc, "C5PAGE", f"{mod_name}_complete_case",
                f"06 and 07 {mod_name} moderation (complete case, {mod_var}); "
                f"persistence and FC complete-case samples are the same participants",
            ))
            print(f"    single row describes both Analysis 06 and Analysis 07")
        else:
            rows.append(describe_sample(
                persist_cc, "C5PAGE", f"{mod_name}_complete_case",
                f"06 {mod_name} moderation (complete case, {mod_var})",
            ))
            rows.append(describe_sample(
                fc_cc, "C5PAGE", f"{mod_name}_complete_case_fc",
                f"07 {mod_name} moderation (complete case, {mod_var})",
            ))

    # ========================================================================
    # Save and print
    # ========================================================================
    out = pd.DataFrame(rows).set_index("sample")
    out.to_csv(OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")

    core = ["N", "N_age", "age_mean", "age_SD", "age_min", "age_max",
            "n_male", "n_female", "n_sex_missing", "n_white", "n_race_missing"]
    print("\nCore columns:")
    print(out[core].to_string())

    print("\nFull table:")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(out.to_string())


if __name__ == "__main__":
    main()
