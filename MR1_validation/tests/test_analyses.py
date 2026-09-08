"""
test_analyses.py

Synthetic tests for MR1 analysis utilities.

All tests use fully in-memory synthetic data — NO real participant data accessed.
Privacy: never prints participant IDs or real rows.

Run from MR1_validation/ directory:
    python -m pytest tests/test_analyses.py -v
"""

import sys
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

# Allow importing analysis_utils without a real master file present
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "analysis"))

from analysis_utils import (
    fisher_z,
    one_tailed_p,
    get_covariates,
    get_samples,
    prepare_persistence_vars,
    run_correlation,
    run_ols,
    run_analysis_set,
    save_results,
    DIARY_OUTCOMES,
    BASE_COVARIATES,
)

RNG = np.random.default_rng(seed=2024)


def _make_synthetic_df(n=60, seed=0):
    """Synthetic participant-level dataframe with known relationships."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    return pd.DataFrame({
        "MIDUSID":     range(2000, 2000 + n),
        "pred":        x,
        "outcome_pos": x + 0.5 * noise,      # r ≈ +0.9
        "outcome_neg": -x + 0.5 * noise,     # r ≈ -0.9
        "outcome_null": noise,                # r ≈ 0
        "PA_score":    x + 0.5 * noise,
        "NA_score":    -x + 0.5 * noise,
        "RA5PAGE":     rng.uniform(50, 80, n),
        "sex":         rng.choice([1, 2], n).astype(float),
        "race":        rng.choice([1, 2, 3, 4, 5, 6], n).astype(float),
        "time_P2_P5":  rng.uniform(0, 60, n),
        "n_days_complete": rng.uniform(10, 30, n),
        "qc_conservative":    [1] * n,
        "has_neg_persistence": [1] * n,
    })


def _add_race_dummies(df):
    """Construct race_2 through race_6 with race=1 (White) as reference."""
    for code in [2, 3, 4, 5, 6]:
        df[f"race_{code}"] = (df["race"] == code).astype(float).where(df["race"].notna())
    return df


# ── fisher_z ─────────────────────────────────────────────────────────────────

class TestFisherZ:
    def test_zero_maps_to_zero(self):
        assert fisher_z(0.0) == pytest.approx(0.0)

    def test_positive_r(self):
        assert fisher_z(0.5) == pytest.approx(np.arctanh(0.5))

    def test_clips_near_one(self):
        result = fisher_z(1.0)
        assert np.isfinite(result)

    def test_negative_r(self):
        assert fisher_z(-0.5) == pytest.approx(-np.arctanh(0.5))


# ── one_tailed_p ──────────────────────────────────────────────────────────────

class TestOneTailedP:
    def test_correct_positive_direction_halves_p(self):
        p_one = one_tailed_p(stat=2.0, p_two=0.04, expected_positive=True)
        assert p_one == pytest.approx(0.02)

    def test_correct_negative_direction_halves_p(self):
        p_one = one_tailed_p(stat=-2.0, p_two=0.04, expected_positive=False)
        assert p_one == pytest.approx(0.02)

    def test_wrong_direction_gives_large_p(self):
        p_one = one_tailed_p(stat=-2.0, p_two=0.04, expected_positive=True)
        assert p_one == pytest.approx(0.98)

    def test_two_tailed_at_boundary(self):
        p_one = one_tailed_p(stat=1.0, p_two=0.10, expected_positive=True)
        assert p_one == pytest.approx(0.05)


# ── get_covariates ────────────────────────────────────────────────────────────

class TestGetCovariates:
    def _full_df(self):
        """DataFrame containing all BASE_COVARIATES."""
        return _add_race_dummies(pd.DataFrame({
            "RA5PAGE": [65.0],
            "sex":     [1.0],
            "race":    [1.0],
        }))

    def test_returns_exact_ordered_list(self):
        df = self._full_df()
        covs = get_covariates(df)
        assert covs == ["RA5PAGE", "sex", "race_2", "race_3", "race_4", "race_5", "race_6"]

    def test_order_is_exact(self):
        df = self._full_df()
        covs = get_covariates(df)
        assert covs[0] == "RA5PAGE"
        assert covs[1] == "sex"
        assert covs[2] == "race_2"
        assert covs[-1] == "race_6"
        assert len(covs) == 7

    def test_missing_required_covariate_raises_systemexit(self):
        # DataFrame missing RA5PAGE and several race dummies
        df = pd.DataFrame({"sex": [1.0], "race_2": [0.0]})
        with pytest.raises(SystemExit):
            get_covariates(df)

    def test_extra_race_column_not_included(self):
        df = self._full_df()
        df["race_7"] = 0.0
        covs = get_covariates(df)
        assert "race_7" not in covs
        assert len(covs) == 7

    def test_mlm_and_family_columns_not_included(self):
        df = self._full_df()
        df["_family_id"] = "a"
        df["M2FAMNUM"] = 1
        df["twin_pair_1"] = 0
        covs = get_covariates(df)
        for col in ("_family_id", "M2FAMNUM", "twin_pair_1"):
            assert col not in covs


# ── get_samples ───────────────────────────────────────────────────────────────

class TestGetSamples:
    def _base_df(self, n=40, seed=1):
        df = _add_race_dummies(_make_synthetic_df(n=n, seed=seed))
        df["has_neg_persistence"] = 1
        df["qc_conservative"]     = 1
        return df

    def test_diary_fmri_full_requires_persistence_and_affect(self):
        df = self._base_df(n=40)
        # Rows 0-9: no affect
        df.loc[:9, "PA_score"] = np.nan
        df.loc[:9, "NA_score"] = np.nan
        # Rows 10-14: no persistence
        df.loc[10:14, "has_neg_persistence"] = 0
        full, _ = get_samples(df, require_diary=True)
        # Only rows 15-39 have persistence==1 AND at least one affect
        assert len(full) == 25

    def test_conservative_restricts_to_qc_1(self):
        df = self._base_df(n=40)
        df.loc[20:, "qc_conservative"] = 0
        full, cons = get_samples(df, require_diary=True)
        assert len(full) == 40
        assert len(cons) == 20

    def test_conservative_is_subset_of_full(self):
        df = self._base_df(n=40)
        df.loc[30:, "qc_conservative"] = 0
        full, cons = get_samples(df, require_diary=True)
        assert len(cons) <= len(full)

    def test_require_diary_false_does_not_require_affect(self):
        df = self._base_df(n=40)
        df["PA_score"] = np.nan
        df["NA_score"] = np.nan
        # Must not raise — affect not required when require_diary=False
        full, cons = get_samples(df, require_diary=False)
        assert len(full) == 40

    def test_missing_check_fc_col_raises_systemexit(self):
        df = self._base_df(n=40)
        with pytest.raises(SystemExit):
            get_samples(df, check_fc_col="nonexistent_fc_col", require_diary=False)

    def test_invalid_has_neg_persistence_value_raises_systemexit(self):
        df = self._base_df(n=40)
        df.loc[0, "has_neg_persistence"] = 99  # not in {0, 1}
        with pytest.raises(SystemExit):
            get_samples(df, require_diary=False)

    def test_invalid_qc_conservative_value_raises_systemexit(self):
        df = self._base_df(n=40)
        df.loc[0, "qc_conservative"] = 2  # not in {0, 1}
        with pytest.raises(SystemExit):
            get_samples(df, require_diary=False)

    def test_behavioral_true_returns_affect_available_as_both(self):
        df = self._base_df(n=40)
        df.loc[:9, "PA_score"] = np.nan
        df.loc[:9, "NA_score"] = np.nan
        full, cons = get_samples(df, behavioral=True)
        assert len(full) == 30
        assert len(cons) == 30  # conservative == full for behavioral path

    def test_check_fc_col_present_restricts_to_nonmissing(self):
        df = self._base_df(n=40)
        df["my_fc"] = 1.0
        df.loc[:9, "my_fc"] = np.nan   # 10 participants missing FC
        full, cons = get_samples(df, check_fc_col="my_fc", require_diary=True)
        assert len(full) == 30


# ── prepare_persistence_vars ──────────────────────────────────────────────────

class TestPreparePersistenceVars:
    def _make_persist_df(self, r_vals=None, n=40, seed=3):
        rng = np.random.default_rng(seed)
        if r_vals is None:
            r_vals = rng.uniform(-0.5, 0.5, n)
        return pd.DataFrame({
            "MIDUSID": range(3000, 3000 + len(r_vals)),
            "neg_persist_crossrun_mean_r_L": r_vals,
        })

    def test_z_column_created(self):
        df = self._make_persist_df()
        z_vars = prepare_persistence_vars(df)
        assert "neg_persist_crossrun_mean_z_L" in df.columns
        assert "neg_persist_crossrun_mean_z_L" in z_vars

    def test_raw_r_column_unchanged(self):
        df = self._make_persist_df()
        original = df["neg_persist_crossrun_mean_r_L"].copy()
        prepare_persistence_vars(df)
        pd.testing.assert_series_equal(
            df["neg_persist_crossrun_mean_r_L"], original, check_names=True
        )

    def test_numeric_string_values_transformed(self):
        df = pd.DataFrame({
            "MIDUSID": [3001, 3002],
            "neg_persist_crossrun_mean_r_L": ["0.3", "-0.4"],
        })
        prepare_persistence_vars(df)
        assert "neg_persist_crossrun_mean_z_L" in df.columns
        z = df["neg_persist_crossrun_mean_z_L"]
        assert z.notna().all()
        assert z[0] == pytest.approx(np.arctanh(0.3), rel=1e-5)

    def test_missing_values_remain_missing(self):
        df = pd.DataFrame({
            "MIDUSID": [3001, 3002, 3003],
            "neg_persist_crossrun_mean_r_L": [0.3, np.nan, -0.2],
        })
        prepare_persistence_vars(df)
        assert pd.isna(df.loc[1, "neg_persist_crossrun_mean_z_L"])
        assert pd.notna(df.loc[0, "neg_persist_crossrun_mean_z_L"])

    def test_out_of_range_raises_systemexit(self):
        df = pd.DataFrame({
            "MIDUSID": [3001],
            "neg_persist_crossrun_mean_r_L": [1.5],  # > 1
        })
        with pytest.raises(SystemExit):
            prepare_persistence_vars(df)

    def test_nonnumeric_nonmissing_raises_systemexit(self):
        df = pd.DataFrame({
            "MIDUSID": [3001],
            "neg_persist_crossrun_mean_r_L": ["not_a_number"],
        })
        with pytest.raises(SystemExit):
            prepare_persistence_vars(df)

    def test_both_hemispheres_transformed(self):
        df = pd.DataFrame({
            "MIDUSID": [3001, 3002],
            "neg_persist_crossrun_mean_r_L": [0.3, -0.2],
            "neg_persist_crossrun_mean_r_R": [0.1, -0.1],
        })
        z_vars = prepare_persistence_vars(df)
        assert "neg_persist_crossrun_mean_z_L" in z_vars
        assert "neg_persist_crossrun_mean_z_R" in z_vars

    def test_preexisting_z_column_not_overwritten(self):
        df = pd.DataFrame({
            "MIDUSID": [3001],
            "neg_persist_crossrun_mean_r_L": [0.3],
            "neg_persist_crossrun_mean_z_L": [99.0],  # sentinel
        })
        prepare_persistence_vars(df)
        assert df.loc[0, "neg_persist_crossrun_mean_z_L"] == 99.0


# ── run_correlation ───────────────────────────────────────────────────────────

class TestRunCorrelation:
    def setup_method(self):
        self.df = _add_race_dummies(_make_synthetic_df(n=60))

    def test_returns_dict_structure(self):
        result = run_correlation(self.df, "pred", "outcome_pos")
        assert isinstance(result, dict)
        for key in ("predictor", "outcome", "n", "r", "p", "p_two_tailed"):
            assert key in result

    def test_positive_correlation_detected(self):
        result = run_correlation(self.df, "pred", "outcome_pos")
        assert result["r"] > 0.5

    def test_negative_correlation_detected(self):
        result = run_correlation(self.df, "pred", "outcome_neg")
        assert result["r"] < -0.5

    def test_one_tailed_correct_direction_lowers_p(self):
        r_result = run_correlation(self.df, "pred", "outcome_pos",
                                   one_tailed=True, expected_positive=True)
        r_two = run_correlation(self.df, "pred", "outcome_pos",
                                one_tailed=False)
        assert r_result["p"] < r_two["p"]

    def test_one_tailed_wrong_direction_raises_p(self):
        r_result = run_correlation(self.df, "pred", "outcome_pos",
                                   one_tailed=True, expected_positive=False)
        assert r_result["p"] > 0.5

    def test_returns_none_below_min_n(self):
        df_small = self.df.iloc[:5].copy()
        result = run_correlation(df_small, "pred", "outcome_pos")
        assert result is None

    def test_missing_predictor_raises_systemexit(self):
        with pytest.raises(SystemExit):
            run_correlation(self.df, "nonexistent_pred", "outcome_pos")

    def test_missing_outcome_raises_systemexit(self):
        with pytest.raises(SystemExit):
            run_correlation(self.df, "pred", "nonexistent_outcome")


# ── run_ols ───────────────────────────────────────────────────────────────────

class TestRunOLS:
    def setup_method(self):
        self.df = _add_race_dummies(_make_synthetic_df(n=60))
        self.covs = get_covariates(self.df)

    def test_returns_dict_structure(self):
        result = run_ols(self.df, "pred", "outcome_pos", self.covs)
        assert isinstance(result, dict)
        for key in ("predictor", "outcome", "n", "beta", "se", "t", "p",
                    "p_two_tailed", "r_squared"):
            assert key in result

    def test_positive_beta_detected(self):
        result = run_ols(self.df, "pred", "outcome_pos", self.covs)
        assert result["beta"] > 0

    def test_zero_variance_covariate_removed(self):
        df = self.df.copy()
        df["const_var"] = 42.0
        covs = self.covs + ["const_var"]
        result = run_ols(df, "pred", "outcome_pos", covs)
        assert result is not None
        assert result["n"] > 0

    def test_returns_none_below_min_n(self):
        df_small = self.df.iloc[:5].copy()
        result = run_ols(df_small, "pred", "outcome_pos", self.covs)
        assert result is None

    def test_one_tailed_p_correct_direction(self):
        r_one = run_ols(self.df, "pred", "outcome_pos", self.covs,
                        one_tailed=True, expected_positive=True)
        r_two = run_ols(self.df, "pred", "outcome_pos", self.covs,
                        one_tailed=False)
        assert r_one["p"] < r_two["p"]

    def test_missing_predictor_raises_systemexit(self):
        with pytest.raises(SystemExit):
            run_ols(self.df, "nonexistent_pred", "outcome_pos", self.covs)

    def test_missing_outcome_raises_systemexit(self):
        with pytest.raises(SystemExit):
            run_ols(self.df, "pred", "nonexistent_outcome", self.covs)


# ── run_analysis_set ──────────────────────────────────────────────────────────

class TestRunAnalysisSet:
    def setup_method(self):
        self.df = _add_race_dummies(_make_synthetic_df(n=60))
        self.covs = get_covariates(self.df)

    def test_returns_exactly_two_values(self):
        result = run_analysis_set(self.df, ["pred"], ["PA_score"], self.covs)
        assert len(result) == 2, (
            "run_analysis_set must return (corr_df, ols_df) — NOT three values. "
            "Singleton-group MLM has been removed from the MR1 pipeline."
        )

    def test_returns_dataframes(self):
        corr, ols = run_analysis_set(self.df, ["pred"], ["PA_score"], self.covs)
        assert isinstance(corr, pd.DataFrame)
        assert isinstance(ols, pd.DataFrame)

    def test_multiple_predictors_and_outcomes(self):
        corr, ols = run_analysis_set(
            self.df,
            ["pred", "RA5PAGE"],
            ["PA_score", "NA_score"],
            self.covs,
        )
        assert len(corr) == 4
        assert len(ols) == 4

    def test_missing_predictor_raises_systemexit(self):
        with pytest.raises(SystemExit):
            run_analysis_set(self.df, ["nonexistent_pred"], ["PA_score"], self.covs)

    def test_missing_outcome_raises_systemexit(self):
        with pytest.raises(SystemExit):
            run_analysis_set(self.df, ["pred"], ["nonexistent_outcome"], self.covs)

    def test_missing_base_covariate_raises_systemexit(self):
        with pytest.raises(SystemExit):
            run_analysis_set(self.df, ["pred"], ["PA_score"],
                             self.covs + ["nonexistent_cov"])

    def test_diary_outcome_requires_diary_covariates(self):
        df_no_diary = self.df.drop(columns=["time_P2_P5", "n_days_complete"])
        with pytest.raises(SystemExit):
            run_analysis_set(df_no_diary, ["pred"], ["PA_score"], self.covs)

    def test_nondiary_outcome_does_not_require_diary_covariates(self):
        # Drop diary covariates from df; use a non-diary outcome
        df_no_diary = self.df.drop(columns=["time_P2_P5", "n_days_complete"])
        df_no_diary["RA5SPGP"] = self.df["PA_score"].values
        # RA5SPGP is not in DIARY_OUTCOMES — should succeed without diary covariates
        corr, ols = run_analysis_set(df_no_diary, ["pred"], ["RA5SPGP"], self.covs)
        assert isinstance(corr, pd.DataFrame)

    def test_expected_directions_applied(self):
        corr, ols = run_analysis_set(
            self.df, ["pred"], ["PA_score"], self.covs,
            one_tailed=True, expected_directions={"PA_score": +1}
        )
        row = corr.iloc[0]
        assert row["p"] <= row["p_two_tailed"]

    def test_diary_outcomes_in_diary_outcomes_constant(self):
        assert "PA_score" in DIARY_OUTCOMES
        assert "NA_score" in DIARY_OUTCOMES
        assert "NA_score_log" in DIARY_OUTCOMES

    def test_nondiary_outcomes_not_in_diary_outcomes_constant(self):
        assert "RA5SPGP" not in DIARY_OUTCOMES
        assert "neg_persist_crossrun_mean_z_L" not in DIARY_OUTCOMES


# ── save_results ──────────────────────────────────────────────────────────────

class TestSaveResults:
    def setup_method(self):
        self.df = _add_race_dummies(_make_synthetic_df(n=60))
        self.covs = get_covariates(self.df)

    def test_saves_correlations_and_regressions_only(self, tmp_path):
        corr, ols = run_analysis_set(self.df, ["pred"], ["PA_score"], self.covs)
        save_results(corr, ols, tmp_path, label="test run")
        assert (tmp_path / "correlations.csv").exists()
        assert (tmp_path / "regressions.csv").exists()
        assert not (tmp_path / "mlm.csv").exists(), (
            "mlm.csv must NOT be created — singleton-group MLM removed from MR1 pipeline"
        )

    def test_correlations_csv_has_correct_columns(self, tmp_path):
        corr, ols = run_analysis_set(self.df, ["pred"], ["PA_score"], self.covs)
        save_results(corr, ols, tmp_path)
        loaded = pd.read_csv(tmp_path / "correlations.csv")
        for col in ("predictor", "outcome", "n", "r", "p"):
            assert col in loaded.columns

    def test_regressions_csv_has_correct_columns(self, tmp_path):
        corr, ols = run_analysis_set(self.df, ["pred"], ["PA_score"], self.covs)
        save_results(corr, ols, tmp_path)
        loaded = pd.read_csv(tmp_path / "regressions.csv")
        for col in ("predictor", "outcome", "n", "beta", "se", "p"):
            assert col in loaded.columns

    def test_creates_output_directory(self, tmp_path):
        new_dir = tmp_path / "nested" / "output"
        corr, ols = run_analysis_set(self.df, ["pred"], ["PA_score"], self.covs)
        save_results(corr, ols, new_dir)
        assert new_dir.exists()

    def test_empty_dataframes_saved_without_error(self, tmp_path):
        save_results(pd.DataFrame(), pd.DataFrame(), tmp_path)
        assert (tmp_path / "correlations.csv").exists()

    def test_methods_note_written_when_metadata_supplied(self, tmp_path):
        corr, ols = run_analysis_set(
            self.df, ["pred"], ["PA_score"], self.covs,
            one_tailed=True, expected_directions={"PA_score": +1},
        )
        save_results(
            corr, ols, tmp_path,
            label="test methods note",
            predictors=["pred"],
            outcomes=["PA_score"],
            covariates=self.covs,
            n=len(self.df),
            one_tailed=True,
            expected_directions={"PA_score": +1},
        )
        assert (tmp_path / "_methods.txt").exists()
        content = (tmp_path / "_methods.txt").read_text()
        assert "Pearson correlation" in content
        assert "OLS" in content
        assert "one-tailed" in content.lower()
        # Diary outcome → diary covariates listed in methods note
        assert "time_P2_P5" in content
        assert "n_days_complete" in content
        # No MLM section
        assert "MLM" not in content
        assert "MixedLM" not in content
        # Raw column names appear individually (not collapsed into generic label)
        assert "RA5PAGE" in content
        assert "sex" in content
        assert "race_2" in content
        assert "race_6" in content

    def test_methods_note_not_written_without_metadata(self, tmp_path):
        corr, ols = run_analysis_set(self.df, ["pred"], ["PA_score"], self.covs)
        save_results(corr, ols, tmp_path, label="no metadata")
        assert not (tmp_path / "_methods.txt").exists()

    def test_methods_note_two_tailed_when_not_specified(self, tmp_path):
        corr, ols = run_analysis_set(self.df, ["pred"], ["PA_score"], self.covs)
        save_results(
            corr, ols, tmp_path,
            predictors=["pred"], outcomes=["PA_score"],
            covariates=self.covs, n=len(self.df),
        )
        content = (tmp_path / "_methods.txt").read_text()
        assert "two-tailed" in content.lower()


# ── full_sample pass in Analysis 02 ──────────────────────────────────────────

class TestFullSamplePass:
    """Analysis 02 must run on both conservative AND full sample."""

    def test_conservative_subset_of_full(self):
        df = _add_race_dummies(_make_synthetic_df(n=80))
        df["qc_conservative"]     = ([1] * 40) + ([0] * 40)
        df["has_neg_persistence"]  = 1

        full_mask = df["has_neg_persistence"] == 1
        cons_mask = full_mask & (df["qc_conservative"] == 1)

        full = df[full_mask]
        cons = df[cons_mask]

        assert len(cons) < len(full)
        assert len(cons) == 40
        assert len(full) == 80
