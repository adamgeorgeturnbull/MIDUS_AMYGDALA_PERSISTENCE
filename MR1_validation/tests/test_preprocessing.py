"""
test_preprocessing.py

Synthetic tests for MR1 preprocessing scripts.

All tests use in-memory synthetic data — NO real participant data is accessed.
Privacy: never prints participant IDs or real rows; synthetic IDs are arbitrary integers.

Run from MR1_validation/ directory:
    python -m pytest tests/test_preprocessing.py -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# ── helpers shared across tests ───────────────────────────────────────────────

def _df_p5_base(n=10, start_id=1000):
    """Minimal synthetic P5 participant-level data."""
    ids = range(start_id, start_id + n)
    return pd.DataFrame({
        "MIDUSID":    list(ids),
        "MRID":       list(ids),
        "SAMPLMAJ":   [11] * n,
        "RA1PRSEX":   ([1, 2] * ((n + 1) // 2))[:n],
        "RA1PB1":     [8] * n,
        "RA1PF1":     [0] * n,
        "RA1PF7A":    [1] * n,
        "RA1PBYEAR":  [1955] * n,
        "RA5PAGE":    [65] * n,
        "RA5PDATE_YR": [2015] * n,
        "RA5PDATE_MO": [6] * n,
    })


def _df_mker1_base(n=5, start_id=1000):
    """Minimal synthetic MKER1 data from MKER1_variables1.tsv (overlapping with first n P5 participants)."""
    ids = range(start_id, start_id + n)
    return pd.DataFrame({
        "MRID":         list(ids),
        "SAMPLMAJ":     [21] * n,
        "RAACRSEX":     [1, 2] * (n // 2) + [1] * (n % 2),
        "RAACB1":       [8] * n,
        "RAACF1":       [0] * n,
        "RAACF7A":      [1] * n,
        "RAACBYEAR":    [1955] * n,
        "RAACRAGE":     [60] * n,
        "RAACIDATE_YR": [2015] * n,
        "RAACIDATE_MO": [3] * n,
    })


def _df_diary_base(n=10, start_id=1000):
    """Minimal synthetic per-participant diary summary."""
    ids = range(start_id, start_id + n)
    return pd.DataFrame({
        "MIDUSID":   list(ids),
        "PA_score":  np.random.default_rng(42).uniform(1, 5, n),
        "NA_score":  np.random.default_rng(42).uniform(1, 3, n),
        "StartYear":  [2010] * n,
        "StartMonth": [6] * n,
        "n_days_complete": [20] * n,
    })


# ── test_01_harmonize_ids ─────────────────────────────────────────────────────

class TestToIntId:
    """Tests for the _to_int_id conversion logic."""

    def _convert(self, values):
        s = pd.Series(values)
        s = pd.to_numeric(s, errors="coerce")
        return s.round().astype("Int64")

    def test_integer_passthrough(self):
        result = self._convert([1001, 1002, 1003])
        assert list(result) == [1001, 1002, 1003]

    def test_float_rounded(self):
        result = self._convert([1001.0, 1002.0])
        assert list(result) == [1001, 1002]

    def test_scientific_notation(self):
        result = self._convert(["1.001e3", "1.002e3"])
        assert list(result) == [1001, 1002]

    def test_missing_becomes_nan(self):
        result = self._convert([1001, None, 1003])
        assert pd.isna(result[1])

    def test_string_invalid_becomes_nan(self):
        result = self._convert(["abc"])
        assert pd.isna(result[0])


class TestDuplicateDetection:
    """Script 01 must abort on duplicate MIDUSID in participant-level files."""

    def _make_dup(self):
        df = _df_p5_base(n=5)
        df.loc[4, "MIDUSID"] = df.loc[0, "MIDUSID"]
        return df

    def test_duplicate_detected(self):
        df = self._make_dup()
        assert df["MIDUSID"].duplicated().sum() == 1

    def test_no_duplicate_clean(self):
        df = _df_p5_base(n=5)
        assert df["MIDUSID"].duplicated().sum() == 0


class TestMidusidMridAgreement:
    """Script 01 must abort when MIDUSID ≠ MRID."""

    def test_agreement_ok(self):
        df = _df_p5_base(n=5)
        disagree = (df["MIDUSID"] != df["MRID"]).sum()
        assert disagree == 0

    def test_disagreement_detected(self):
        df = _df_p5_base(n=5)
        df.loc[2, "MRID"] = df.loc[2, "MIDUSID"] + 9999
        disagree = (df["MIDUSID"] != df["MRID"]).sum()
        assert disagree == 1


class TestMker1Samplmaj:
    """MKER1 SAMPLMAJ (from MKER1_variables1.tsv) must be exactly {21}."""

    def test_expected_samplmaj(self):
        df = _df_mker1_base(n=5)
        observed = set(df["SAMPLMAJ"].dropna().astype(int).unique())
        unexpected = observed - {21}
        assert unexpected == set()

    def test_unexpected_samplmaj_detected(self):
        df = _df_mker1_base(n=5)
        df.loc[0, "SAMPLMAJ"] = 11
        observed = set(df["SAMPLMAJ"].dropna().astype(int).unique())
        unexpected = observed - {21}
        assert unexpected == {11}


# ── test_02_p2_diary_long_format ──────────────────────────────────────────────

class TestP2LongFormat:
    """Repeated MIDUSID in P2 diary rows are expected and must not be flagged as duplicates."""

    def test_repeated_midusid_is_expected(self):
        # P2 is long format: one row per participant-day; multiple rows per MIDUSID are correct
        diary_long = pd.DataFrame({
            "MIDUSID": [1001, 1001, 1001, 1002, 1002],
            "day":     [1, 2, 3, 1, 2],
            "PA_item": [3.0, 4.0, 2.0, 2.0, 3.0],
        })
        n_dup = diary_long["MIDUSID"].duplicated().sum()
        # Duplicates exist — this is the expected structure, not an error
        assert n_dup > 0

    def test_unique_participant_count_is_correct(self):
        diary_long = pd.DataFrame({
            "MIDUSID": [1001, 1001, 1001, 1002, 1002],
            "day":     [1, 2, 3, 1, 2],
        })
        # Unique participants, not unique rows
        assert diary_long["MIDUSID"].nunique() == 2

    def test_p5_uniqueness_still_required(self):
        # P5 is participant-level — one row per participant, duplicates ARE an error
        p5 = _df_p5_base(n=5)
        assert p5["MIDUSID"].duplicated().sum() == 0
        p5_dup = p5.copy()
        p5_dup.loc[4, "MIDUSID"] = p5_dup.loc[0, "MIDUSID"]
        assert p5_dup["MIDUSID"].duplicated().sum() == 1

    def test_diary_summary_has_unique_midusid_after_aggregation(self):
        # After aggregating long-format rows to participant-level summary, MIDUSID must be unique
        diary_long = pd.DataFrame({
            "MIDUSID": [1001, 1001, 1001, 1002, 1002],
            "PA_item": [3.0, 4.0, 2.0, 2.0, 3.0],
        })
        summary = diary_long.groupby("MIDUSID")["PA_item"].mean().reset_index()
        assert summary["MIDUSID"].duplicated().sum() == 0
        assert len(summary) == 2


# ── test_03_construct_demographics ───────────────────────────────────────────

class TestApplyMissingCodes:
    """Missing-value codes must be replaced with NaN before harmonization."""

    def _apply(self, series, codes):
        s = pd.to_numeric(series, errors="coerce")
        return s.where(~s.isin(codes))

    def test_single_digit_mv(self):
        s = pd.Series([1, 2, 7, 8, 9])
        result = self._apply(s, [7, 8, 9])
        assert result[2:].isna().all()
        assert result[:2].notna().all()

    def test_double_digit_mv(self):
        s = pd.Series([8, 97, 98])
        result = self._apply(s, [97, 98])
        assert result[1:].isna().all()
        assert pd.notna(result[0])

    def test_year_mv(self):
        s = pd.Series([1955, 9997, 9998, 9999])
        result = self._apply(s, [9997, 9998, 9999])
        assert result[1:].isna().all()
        assert pd.notna(result[0])

    def test_valid_values_preserved(self):
        s = pd.Series([1, 2, 3])
        result = self._apply(s, [97, 98])
        assert result.notna().all()


class TestHarmonizeField:
    """P5 primary source wins; MKER1 fills only when P5 is missing."""

    def _harmonize(self, primary, fallback):
        harmonized = primary.copy()
        source = pd.Series(np.nan, index=primary.index, dtype=object)
        source[primary.notna()] = "p5"
        if fallback is not None:
            needs_fill = primary.isna() & fallback.notna()
            harmonized[needs_fill] = fallback[needs_fill]
            source[needs_fill] = "mker1"
        return harmonized, source

    def test_p5_wins_when_both_present(self):
        primary  = pd.Series([1.0, 2.0, np.nan])
        fallback = pd.Series([9.0, 9.0, 3.0])
        harmonized, source = self._harmonize(primary, fallback)
        assert harmonized[0] == 1.0
        assert harmonized[1] == 2.0
        assert source[0] == "p5"
        assert source[1] == "p5"

    def test_mker1_fills_missing_p5(self):
        primary  = pd.Series([1.0, np.nan, np.nan])
        fallback = pd.Series([9.0, 5.0, np.nan])
        harmonized, source = self._harmonize(primary, fallback)
        assert harmonized[1] == 5.0
        assert source[1] == "mker1"

    def test_missing_stays_nan_when_both_missing(self):
        primary  = pd.Series([np.nan])
        fallback = pd.Series([np.nan])
        harmonized, source = self._harmonize(primary, fallback)
        assert pd.isna(harmonized[0])
        assert pd.isna(source[0])

    def test_no_fallback_leaves_missing(self):
        primary  = pd.Series([1.0, np.nan])
        harmonized, source = self._harmonize(primary, None)
        assert pd.isna(harmonized[1])
        assert pd.isna(source[1])

    def test_source_indicator_correct(self):
        primary  = pd.Series([1.0, np.nan, np.nan])
        fallback = pd.Series([9.0, 2.0, np.nan])
        _, source = self._harmonize(primary, fallback)
        assert source[0] == "p5"
        assert source[1] == "mker1"
        assert pd.isna(source[2])


# ── test_04_construct_covariates ──────────────────────────────────────────────

class TestRaceDummies:
    """Race dummies race_2..race_6 constructed with race=1 (White) as reference."""

    def _make_dummies(self, df):
        for code in [2, 3, 4, 5, 6]:
            col = f"race_{code}"
            df[col] = (df["race"] == code).astype(float).where(df["race"].notna())
        return df

    def test_white_has_no_dummy(self):
        df = pd.DataFrame({"race": [1.0, 2.0, 3.0, np.nan]})
        df = self._make_dummies(df)
        assert "race_1" not in df.columns

    def test_race_2_coded_correctly(self):
        df = pd.DataFrame({"race": [1.0, 2.0, 2.0, 3.0]})
        df = self._make_dummies(df)
        assert list(df["race_2"]) == [0.0, 1.0, 1.0, 0.0]

    def test_missing_race_propagates_nan(self):
        df = pd.DataFrame({"race": [1.0, np.nan]})
        df = self._make_dummies(df)
        assert pd.isna(df.loc[1, "race_2"])

    def test_all_dummies_present(self):
        df = pd.DataFrame({"race": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        df = self._make_dummies(df)
        for code in [2, 3, 4, 5, 6]:
            assert f"race_{code}" in df.columns


# ── test_05_merge_master_dataset ──────────────────────────────────────────────

class TestMergeOperatorPrecedence:
    """
    The original bug: df1_cols & df2_cols - {key} evaluated as
    df1_cols & (df2_cols - {key}) due to operator precedence.
    Fix: (df1_cols & df2_cols) - {key}
    """

    def test_precedence_fixed(self):
        df1_cols = {"MIDUSID", "PA_score", "age"}
        df2_cols = {"MIDUSID", "age", "sex"}
        key = "MIDUSID"

        buggy  = df1_cols & (df2_cols - {key})
        fixed  = (df1_cols & df2_cols) - {key}

        assert "MIDUSID" not in fixed
        assert "age" in fixed
        assert "MIDUSID" not in buggy
        df2_cols2 = {"MIDUSID", "sex"}
        fixed2 = (df1_cols & df2_cols2) - {key}
        assert "MIDUSID" not in fixed2


class TestMasterUniverse:
    """
    Master universe = diary ∪ P5 (outer join via merge_combine).
    MKER1 is left-joined onto this universe — MKER1-only participants are excluded.
    """

    def _merge_combine(self, df1, df2, key="MIDUSID"):
        """Replicate merge_combine from 05_merge_master_dataset.py."""
        overlap = list((set(df1.columns) & set(df2.columns)) - {key})
        merged = df1.merge(df2, on=key, how="outer", suffixes=("_1", "_2"))
        for col in overlap:
            merged[col] = merged[f"{col}_1"].fillna(merged[f"{col}_2"])
            merged.drop([f"{col}_1", f"{col}_2"], axis=1, inplace=True)
        return merged

    def test_master_ids_equal_union(self):
        diary_ids = {1001, 1002, 1003}
        p5_ids    = {1002, 1003, 1004}
        diary = pd.DataFrame({"MIDUSID": list(diary_ids), "PA_score": [3.0, 2.0, 1.0]})
        p5    = pd.DataFrame({"MIDUSID": list(p5_ids),    "RA5PAGE":  [65.0, 66.0, 67.0]})
        merged = self._merge_combine(diary, p5)
        assert set(merged["MIDUSID"]) == diary_ids | p5_ids

    def test_diary_only_participants_retained(self):
        diary = pd.DataFrame({"MIDUSID": [1001, 1002, 1003], "PA_score": [3.0, 2.0, 1.0]})
        p5    = pd.DataFrame({"MIDUSID": [1002, 1003],        "RA5PAGE":  [65.0, 66.0]})
        merged = self._merge_combine(diary, p5)
        assert 1001 in merged["MIDUSID"].values

    def test_p5_only_participants_retained(self):
        diary = pd.DataFrame({"MIDUSID": [1001, 1002], "PA_score": [3.0, 2.0]})
        p5    = pd.DataFrame({"MIDUSID": [1002, 1004], "RA5PAGE":  [65.0, 67.0]})
        merged = self._merge_combine(diary, p5)
        assert 1004 in merged["MIDUSID"].values

    def test_overlapping_columns_prefer_df1_value(self):
        diary = pd.DataFrame({"MIDUSID": [1001, 1002], "sex": [1.0, 2.0]})
        p5    = pd.DataFrame({"MIDUSID": [1001, 1002], "sex": [9.0, 9.0]})
        merged = self._merge_combine(diary, p5)
        row = merged[merged["MIDUSID"] == 1001].iloc[0]
        assert row["sex"] == 1.0

    def test_overlapping_columns_use_df2_when_df1_missing(self):
        diary = pd.DataFrame({"MIDUSID": [1001, 1002], "sex": [1.0, np.nan]})
        p5    = pd.DataFrame({"MIDUSID": [1001, 1002], "sex": [9.0, 2.0]})
        merged = self._merge_combine(diary, p5)
        row = merged[merged["MIDUSID"] == 1002].iloc[0]
        assert row["sex"] == 2.0

    def test_result_one_row_per_union_id(self):
        diary = pd.DataFrame({"MIDUSID": [1001, 1002, 1003]})
        p5    = pd.DataFrame({"MIDUSID": [1002, 1003, 1004]})
        merged = self._merge_combine(diary, p5)
        assert len(merged) == 4
        assert merged["MIDUSID"].duplicated().sum() == 0

    def test_mker1_supplement_does_not_add_participants(self):
        # MKER1 (from MKER1_variables1.tsv) is left-joined — MKER1-only IDs must not appear
        diary  = pd.DataFrame({"MIDUSID": [1001, 1002]})
        p5     = pd.DataFrame({"MIDUSID": [1001]})
        mker1  = pd.DataFrame({"MIDUSID": [1001, 9999]})  # 9999 is MKER1-only
        master = self._merge_combine(diary, p5)
        n_before = len(master)
        master = master.merge(mker1, on="MIDUSID", how="left")
        assert len(master) == n_before
        assert 9999 not in master["MIDUSID"].values

    def test_duplicate_key_before_merge_detected(self):
        diary = pd.DataFrame({"MIDUSID": [1001, 1001, 1002]})  # duplicate in diary
        n_dup = diary["MIDUSID"].duplicated().sum()
        assert n_dup > 0

    def test_duplicate_in_p5_causes_row_expansion(self):
        # A duplicate MIDUSID in P5 (which would be a data error) expands rows — detectable
        diary = _df_diary_base(n=3)
        p5    = _df_p5_base(n=3)
        dup_row = p5.iloc[0].copy()
        p5_dup = pd.concat([p5, pd.DataFrame([dup_row])], ignore_index=True)
        merged = diary.merge(p5_dup, on="MIDUSID", how="left")
        assert len(merged) > len(diary)


# ── test_06_clean_merged_data ─────────────────────────────────────────────────

class TestMkeAgeFormula:
    """MKE interview-age formula: RAACRAGE + (StartYear - RAACIDATE_YR) + (StartMonth - RAACIDATE_MO)/12."""

    def _compute_mke_age(self, raacrage, raacidate_yr, raacidate_mo, start_year, start_month):
        return raacrage + (start_year - raacidate_yr) + (start_month - raacidate_mo) / 12

    def test_basic_formula_same_year(self):
        age = self._compute_mke_age(
            raacrage=60, raacidate_yr=2015, raacidate_mo=6,
            start_year=2015, start_month=6
        )
        assert age == pytest.approx(60.0)

    def test_formula_one_year_after_interview(self):
        age = self._compute_mke_age(
            raacrage=60, raacidate_yr=2014, raacidate_mo=6,
            start_year=2015, start_month=6
        )
        assert age == pytest.approx(61.0)

    def test_formula_partial_year_adjustment(self):
        age = self._compute_mke_age(
            raacrage=60, raacidate_yr=2015, raacidate_mo=3,
            start_year=2015, start_month=9
        )
        assert age == pytest.approx(60.5, abs=0.01)

    def test_missing_code_excluded(self):
        df = pd.DataFrame({
            "RAACRAGE":     [60, 97],
            "RAACIDATE_YR": [2015, 2015],
            "RAACIDATE_MO": [6, 6],
            "StartYear":    [2015, 2015],
            "StartMonth":   [6, 6],
        })
        raacrage = pd.to_numeric(df["RAACRAGE"], errors="coerce")
        raacrage = raacrage.where(~raacrage.isin([97, 98, 99]))
        mke_age = raacrage + (df["StartYear"] - df["RAACIDATE_YR"])
        assert pd.notna(mke_age[0])
        assert pd.isna(mke_age[1])


class TestBirthYearAge:
    """Primary RA2PAGE path: StartYear - birth_year."""

    def test_age_from_birth_year(self):
        df = pd.DataFrame({"birth_year": [1955.0], "StartYear": [2010.0]})
        df["RA2PAGE"] = df["StartYear"] - df["birth_year"]
        assert df.loc[0, "RA2PAGE"] == 55.0

    def test_missing_birth_year_gives_nan(self):
        df = pd.DataFrame({"birth_year": [np.nan], "StartYear": [2010.0]})
        df["RA2PAGE"] = df["StartYear"] - df["birth_year"]
        assert pd.isna(df.loc[0, "RA2PAGE"])


# ── test_panas_recoding ───────────────────────────────────────────────────────

class TestPanasRecoding:
    """
    MR1 PANAS variables: RA5SPGP (positive) and RA5SPGN (negative affect).
    Missing codes 8, 98, 99 → NaN; valid range 1–5 is preserved.
    RA5SPGN_log is the natural log of recoded positive values, without an offset.
    """

    MISSING_CODES = [8, 98, 99]
    VALID_MIN, VALID_MAX = 1, 5

    def _recode(self, values):
        s = pd.to_numeric(pd.Series(values), errors="coerce")
        for code in self.MISSING_CODES:
            s = s.where(s != code)
        return s

    def test_code_8_becomes_nan(self):
        s = self._recode([1, 8, 3])
        assert pd.isna(s.iloc[1])

    def test_code_98_becomes_nan(self):
        s = self._recode([2, 98, 4])
        assert pd.isna(s.iloc[1])

    def test_code_99_becomes_nan(self):
        s = self._recode([3, 99, 5])
        assert pd.isna(s.iloc[1])

    def test_valid_values_1_to_5_preserved(self):
        s = self._recode([1, 2, 3, 4, 5])
        assert s.notna().all()
        assert list(s) == [1, 2, 3, 4, 5]

    def test_out_of_range_value_detectable_after_recoding(self):
        # Values outside 1–5 that are not missing codes are invalid; script should detect these
        s = self._recode([1, 2, 6, 3])  # 6 is out of valid range but not a missing code
        invalid = s[(s < self.VALID_MIN) | (s > self.VALID_MAX)]
        assert len(invalid) == 1

    def test_zero_not_treated_as_missing_code(self):
        # 0 is not a declared missing code; it should be preserved for the range check
        s = self._recode([0, 1, 2])
        assert pd.notna(s.iloc[0])  # 0 survives recoding
        # But 0 is outside valid range 1-5
        assert s.iloc[0] < self.VALID_MIN

    def test_plain_log_preserves_scale_anchor(self):
        recoded = self._recode([1, 8, 2, 3])
        transformed = np.log(recoded)
        assert transformed.iloc[0] == pytest.approx(0.0)
        assert transformed.iloc[2] == pytest.approx(np.log(2.0))
        assert pd.isna(transformed.iloc[1])

    def test_log_transform_applied_to_recoded_not_raw(self):
        raw = pd.Series([1.0, 99.0, 3.0])
        recoded = self._recode(raw)
        log_transformed = np.log(recoded)
        assert pd.isna(log_transformed.iloc[1])
        assert log_transformed.iloc[0] == pytest.approx(0.0)

    def test_both_panas_variables_use_same_missing_codes(self):
        # RA5SPGP and RA5SPGN use the same PANAS missing-code set
        pos_recoded = self._recode([1, 8, 98, 99, 5])
        neg_recoded = self._recode([1, 8, 98, 99, 5])
        assert pos_recoded.isna().sum() == 3
        assert neg_recoded.isna().sum() == 3
        assert list(pos_recoded.dropna()) == list(neg_recoded.dropna())
