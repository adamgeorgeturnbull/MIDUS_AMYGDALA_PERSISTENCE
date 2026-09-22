"""Synthetic-only tests. Never import an analysis entry point or load MIDUS data.

AST extraction loads function definitions only, skipping module-level path probes,
directory creation and main(). Run from the project root with unittest.
"""
import ast
import contextlib
import io
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from confidence_intervals import coefficient_ci, mean_ci, pearson_ci


def functions_only(relative, baseline=False):
    source = ROOT / relative
    if baseline:
        source = ROOT / ".sandbox-python" / "ci-baseline" / relative
    tree = ast.parse(source.read_text())
    definitions = ast.Module(body=[n for n in tree.body
                                  if isinstance(n, ast.FunctionDef)], type_ignores=[])
    env = dict(np=np, pd=pd, sm=sm, smf=smf, stats=stats, sys=sys,
               warnings=warnings, MIN_N=20,
               DIARY_OUTCOMES={"PA_score", "NA_score", "NA_score_log"},
               coefficient_ci=coefficient_ci, pearson_ci=pearson_ci, mean_ci=mean_ci)
    env["one_tailed_p"] = lambda b, p, positive: p / 2 if (b > 0) == positive else 1 - p / 2
    # Literal constants only; never evaluate Path(...), exists(), mkdir(),
    # imports of analysis modules, or any other top-level executable code.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env[target.id] = value
    exec(compile(definitions, str(source), "exec"), env)
    return env


def synthetic():
    rng = np.random.default_rng(82191)
    n = 120
    x, m, c = rng.normal(size=(3, n))
    families = np.repeat(np.arange(n // 2), 2)
    y = 0.8 * x + 0.3 * m + 0.65 * x * m + 0.4 * c
    y += rng.normal(0, 2, n // 2)[families] + rng.normal(size=n)
    return pd.DataFrame(dict(x=x, m=m, c=c, y=y, M2ID=np.arange(n),
                             SAMPLMAJ=3, M2FAMNUM=families))


class ConfidenceIntervalTests(unittest.TestCase):
    def compare_baseline(self, file, function, args, after):
        # Local pre-edit copies enable regression checks here; they are not
        # required when running the independent CI checks in another checkout.
        if (ROOT / ".sandbox-python" / "ci-baseline" / file).is_file():
            before = functions_only(file, baseline=True)[function](*args)
            self.assert_legacy_equal(before, after)

    def assert_legacy_equal(self, before, after):
        self.assertIsNotNone(before)
        for key, value in before.items():
            if isinstance(value, (float, np.floating)):
                np.testing.assert_allclose(after[key], value, rtol=1e-10, atol=1e-12,
                                           equal_nan=True, err_msg=key)
            else:
                self.assertEqual(value, after[key], key)

    def test_direct_exports_and_legacy_invariance(self):
        files = ["scripts/analysis/analysis_utils.py", "scripts/analysis/01_affect_age.py",
                 "scripts/analysis/01_sensitivity.py",
                 "scripts/analysis/00a_diary_panas_convergence.py",
                 "MR1_validation/scripts/analysis/analysis_utils.py"]
        d = synthetic()
        d.loc[2, "y"] = np.nan
        d.loc[4, "c"] = np.nan
        complete = d[["x", "y", "c"]].dropna()
        ols = sm.OLS(complete.y, sm.add_constant(complete[["x", "c"]])).fit()
        for file in files:
            with self.subTest(file=file):
                new = functions_only(file)
                names = ["run_correlation", "run_ols"]
                if "run_mlm" in new:
                    names.append("run_mlm")
                for name in names:
                    args = (d.copy(), "x", "y") + (() if name == "run_correlation" else (["c"],))
                    b = new[name](*args)
                    self.compare_baseline(file, name, args, b)
                    self.assertEqual(b["ci_status"], "ok")
                    self.assertEqual(b["ci_level"], 95)
                    self.assertEqual(b["ci_sidedness"], "two-sided")
                    if name == "run_ols":
                        np.testing.assert_allclose([b["ci_low"], b["ci_high"]], ols.conf_int().loc["x"])
                    elif name == "run_mlm":
                        # Independent normal-Wald reference from the fitted estimate/SE.
                        np.testing.assert_allclose([b["ci_low"], b["ci_high"]],
                                                   b["beta"] + np.array([-1, 1]) * stats.norm.ppf(.975) * b["se"])
                    else:
                        pair = d[["x", "y"]].dropna()
                        ci = stats.pearsonr(pair.x, pair.y).confidence_interval(.95)
                        np.testing.assert_allclose([b["ci_low"], b["ci_high"]], ci)

    def test_directional_p_does_not_change_ci(self):
        env = functions_only("scripts/analysis/analysis_utils.py")
        for name in ("run_correlation", "run_ols", "run_mlm"):
            args = (synthetic(), "x", "y") + (() if name == "run_correlation" else (["c"],))
            two = env[name](*args)
            yes = env[name](*args, one_tailed=True, expected_positive=True)
            no = env[name](*args, one_tailed=True, expected_positive=False)
            for row in (yes, no):
                np.testing.assert_allclose([row["ci_low"], row["ci_high"]],
                                           [two["ci_low"], two["ci_high"]], rtol=1e-10, atol=1e-12)
            self.assertAlmostEqual(yes["p"], two["p"] / 2)
            self.assertAlmostEqual(no["p"], 1 - two["p"] / 2)

    def test_moderation_exports(self):
        for stem in ["06_persistence_affect_moderation", "07_fc_affect_moderation",
                     "06_sensitivity", "07_sensitivity"]:
            file = "scripts/analysis/" + stem + ".py"
            for name in ["run_moderation_ols", "run_moderation_mlm"]:
                with self.subTest(file=file, function=name):
                    args = (synthetic(), "x", "y", "m", ["c"])
                    after = functions_only(file)[name](*args)
                    self.compare_baseline(file, name, args, after)
                    for term in ("interaction", "predictor", "moderator"):
                        self.assertEqual(after[term + "_ci_status"], "ok")
                        self.assertLess(after[term + "_ci_low"], after["beta_" + term])
                        self.assertGreater(after[term + "_ci_high"], after["beta_" + term])
                    if name.endswith("mlm"):
                        np.testing.assert_allclose(
                            [after["interaction_ci_low"], after["interaction_ci_high"]],
                            after["beta_interaction"] + np.array([-1, 1]) * stats.norm.ppf(.975) * after["se_interaction"])

    def test_paired_ci_uses_complete_pairs(self):
        env = functions_only("scripts/analysis/00b_task_condition_differences.py")
        d = synthetic()
        d.loc[:3, "x"] = np.nan
        row = env["paired_t"](d, "x", "y", "x_minus_y")
        pair = d[["x", "y"]].dropna()
        ci = stats.ttest_rel(pair.x, pair.y).confidence_interval(.95)
        np.testing.assert_allclose([row["ci_low"], row["ci_high"]], ci)
        row = env["one_sample_t"](d, "y")
        ci = stats.ttest_1samp(d.y, 0).confidence_interval(.95)
        np.testing.assert_allclose([row["ci_low"], row["ci_high"]], ci)

    def test_invalid_results_do_not_invent_bounds(self):
        invalid = SimpleNamespace(use_t=False, converged=False)
        self.assertEqual(coefficient_ci(invalid, "x")["ci_status"], "nonconverged")
        invalid = SimpleNamespace(use_t=False, converged=True, params={"x": 1.0},
                                  bse={"x": 0.2},
                                  cov_params=lambda: pd.DataFrame([[-.04]], index=["x"], columns=["x"]))
        row = coefficient_ci(invalid, "x")
        self.assertEqual(row["ci_status"], "invalid_variance")
        self.assertTrue(np.isnan(row["ci_low"]))
        self.assertEqual(mean_ci(np.ones(20))["ci_status"], "invalid_variance")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = pearson_ci(stats.pearsonr(np.ones(20), np.arange(20)))
        self.assertEqual(row["ci_status"], "undefined_correlation")

    def test_convergence_csv_has_unrounded_values_and_ci(self):
        env = functions_only("scripts/analysis/00c_vmPFC_convergence.py")
        d = synthetic()
        env["load_data"] = lambda: d
        env["PAIRS"] = [("synthetic", "test", "x", "y")]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            env["RESULTS_DIR"] = Path(tmp)
            env["main"]()
            row = pd.read_csv(Path(tmp) / "ant_post_correlations.csv").iloc[0]
        expected = stats.pearsonr(d.x, d.y)
        self.assertAlmostEqual(row.r_unrounded, expected.statistic)
        self.assertAlmostEqual(row.p_unrounded, expected.pvalue)
        np.testing.assert_allclose([row.ci_low, row.ci_high], expected.confidence_interval(.95))

    def test_aggregate_checker(self):
        env = functions_only("scripts/diagnostics/check_ci_exports.py")
        row = {"beta": .1, "p": .04, "ci_low": -.02, "ci_high": .22,
               "ci_level": 95, "ci_sidedness": "two-sided",
               "ci_method": "wald_normal", "ci_status": "ok"}
        frame = pd.DataFrame([row])
        env["check_frame"](frame)
        env["compare_previous"](frame[["beta", "p"]], frame)
        changed = frame.copy()
        changed.loc[0, "beta"] = .2
        with self.assertRaises(AssertionError):
            env["compare_previous"](frame[["beta", "p"]], changed)
        for key, value in [("ci_low", np.nan), ("ci_low", .3),
                           ("ci_status", "nonconverged"), ("ci_level", 90)]:
            invalid = frame.copy()
            invalid.loc[0, key] = value
            with self.assertRaises(ValueError):
                env["check_frame"](invalid)

    def test_mlm_preserves_formula_safe_outcome_fix(self):
        env = functions_only("scripts/analysis/analysis_utils.py")
        d = synthetic()
        expected = env["run_mlm"](d, "x", "y", ["c"])
        d["synthetic-predictor"] = d.x
        d["synthetic-outcome"] = d.y
        actual = env["run_mlm"](d, "synthetic-predictor", "synthetic-outcome", ["c"])
        self.assertIsNotNone(actual)
        self.assertEqual(actual["outcome"], "synthetic-outcome")
        self.assertEqual(actual["ci_status"], "ok")
        for key in ["beta", "se", "z", "p", "ci_low", "ci_high"]:
            self.assertAlmostEqual(actual[key], expected[key], places=10)
        original = "source_before_ci_review/scripts/analysis/analysis_utils.py"
        if (ROOT / original).is_file():
            before = functions_only(original)["run_mlm"](
                d, "synthetic-predictor", "synthetic-outcome", ["c"])
            self.assert_legacy_equal(before, actual)

    def test_detailed_aggregate_comparison(self):
        env = functions_only("scripts/diagnostics/check_ci_exports.py")
        old = pd.DataFrame({"term": ["x", "y"], "beta": [.1, .2], "p": [.04, .8]})
        new = old.copy()
        new.loc[0, "beta"] += .0001
        new.loc[0, "p"] = .06
        details = env["describe_changes"](old, new)
        self.assertEqual(len(details), 2)
        self.assertIn("crossings of p < .05=1", details[1])
        self.assertIn("different order", env["describe_changes"](old, old.iloc[::-1])[0])
        self.assertIn("row count changed", env["describe_changes"](old, old.iloc[:1])[0])
        self.assertIn("missing legacy columns", env["describe_changes"](old, old.drop(columns="p"))[0])
        self.assertEqual(env["describe_ci_issues"](pd.DataFrame({"ci_status": ["ok", "invalid_variance"]})),
                         ["ci_status: invalid_variance in 1 rows"])

    def test_mediation_paths_and_bootstrap_unchanged(self):
        file = "scripts/analysis/03b_persistence_age_mediation.py"

        def run(baseline=False):
            env = functions_only(file, baseline)
            env.update(M_VAR="m", PRIMARY_AGE="x", SENSITIVITY_AGE="alternate_age",
                       OUTCOMES={"y": ("primary", 1)}, N_BOOT=8, N_MC=1000,
                       EXPECTED_N_PRIMARY=120)
            d = synthetic()
            d["_family_id"] = d.M2FAMNUM.astype(str)
            env["build_common_sample"] = lambda frame, age: frame.copy()
            env["spec_covariates"] = lambda age, base: (["c"], ["x", "c"], ["c"])
            with contextlib.redirect_stdout(io.StringIO()):
                return env["run_specification"]("synthetic", "x", d, ["c"], np.random.default_rng(42))

        paths, mediation = run()
        self.assertEqual({r["path"] for r in paths}, {"a", "b", "c_prime", "c_total"})
        for row in paths:
            self.assertEqual(row["ci_status"], "ok")
            np.testing.assert_allclose([row["ci_low"], row["ci_high"]],
                                       row["beta"] + np.array([-1, 1]) * stats.norm.ppf(.975) * row["se"])
        if (ROOT / ".sandbox-python" / "ci-baseline" / file).is_file():
            old_paths, old_mediation = run(baseline=True)
            for old, new in zip(old_paths, paths):
                self.assert_legacy_equal(old, new)
            for old, new in zip(old_mediation, mediation):
                self.assert_legacy_equal(old, new)


if __name__ == "__main__":
    unittest.main()
