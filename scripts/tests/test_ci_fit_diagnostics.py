"""Synthetic-only verification of the separate two-fit diagnostic."""
import runpy
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
from test_confidence_intervals import ROOT, functions_only, synthetic

D = runpy.run_path(str(ROOT/'scripts/diagnostics/diagnose_ci_fits.py'))


class FitDiagnosticTests(unittest.TestCase):
    def test_direct_model_preserves_sample_and_groups(self):
        env = functions_only('scripts/analysis/analysis_utils.py')
        data = synthetic()
        data.loc[0, 'c'] = np.nan
        model = D['capture_model'](env['run_mlm'], data, 'x', 'y', ['c'])
        complete = data.dropna(subset=['y', 'x', 'c'])
        np.testing.assert_array_equal(model.endog, complete.y)
        np.testing.assert_allclose(model.exog[:, 1:], complete[['x', 'c']])
        self.assertEqual(model.nobs, 119)
        self.assertEqual(model.n_groups, 60)
        snapshot = model.exog.copy()
        output = D['fit_diagnostics'](model, 'synthetic', ['x'], methods=('lbfgs', 'powell'))
        self.assertEqual(set(output.optimizer), {'lbfgs', 'powell'})
        self.assertTrue(output.fit_status.eq('returned').all())
        self.assertTrue(output.ci_status.eq('ok').all())
        self.assertTrue(output.n.eq(119).all())
        np.testing.assert_array_equal(snapshot, model.exog)
        self.assertNotIn('M2ID', output.columns)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'aggregate.csv'
            output.to_csv(target, index=False)
            self.assertEqual(len(pd.read_csv(target)), 2)

    def test_moderation_uses_original_centering(self):
        env = functions_only('scripts/analysis/analysis_utils.py')
        mod = D['load_moderation_functions'](ROOT/'scripts/analysis/06_sensitivity.py', env)
        data = synthetic()
        data.loc[1, 'm'] = np.nan
        model = D['capture_model'](mod['run_moderation_mlm'], data, 'x', 'y', 'm', ['c'])
        complete = data.dropna(subset=['x', 'y', 'm', 'c'])
        x = complete.x - complete.x.mean()
        m = complete.m - complete.m.mean()
        expected = np.column_stack([np.ones(len(complete)), x, m, x*m, complete.c])
        np.testing.assert_allclose(model.exog, expected)
        np.testing.assert_array_equal(model.endog, complete.y)
        self.assertEqual(model.exog_names, ['Intercept', 'x_c', 'm_c', 'x_c_x_m_c', 'c'])
        output = D['fit_diagnostics'](model, 'synthetic_mod', ['x_c', 'm_c', 'x_c_x_m_c'], methods=('powell',))
        self.assertEqual(len(output), 3)
        self.assertTrue(output.ci_status.eq('ok').all())

    def test_iteration_override_is_scoped_to_named_predictor(self):
        env = functions_only('scripts/analysis/analysis_utils.py')
        calls = []
        env['run_correlation'] = lambda *a, **kw: {'test': 1}
        env['run_ols'] = lambda *a, **kw: {'test': 1}
        def record(*args, **kwargs):
            calls.append((args[1], kwargs))
            return {'test': 1}
        env['run_mlm'] = record
        env['run_analysis_set'](synthetic(), ['x', 'm'], ['y'], ['c'],
                                mlm_maxiter_by_predictor={'x': 2000})
        self.assertEqual(calls[0][1]['maxiter'], 2000)
        self.assertNotIn('maxiter', calls[1][1])

    def test_invalid_fit_annotation_preserves_statistics(self):
        env = functions_only('scripts/analysis/06_sensitivity.py')
        rows = pd.DataFrame([
            dict(predictor='x', moderator='m', outcome='y', n=120, beta_interaction=.2,
                 se_interaction=np.nan, p_interaction=np.nan,
                 interaction_ci_status='invalid_variance', predictor_ci_status='invalid_variance',
                 moderator_ci_status='ok'),
            dict(predictor='x', moderator='m', outcome='z', n=120, beta_interaction=.3,
                 se_interaction=.2, p_interaction=.1,
                 interaction_ci_status='ok', predictor_ci_status='ok', moderator_ci_status='ok')])
        annotated = env['annotate_mlm_inference'](rows)
        pd.testing.assert_frame_equal(annotated[rows.columns], rows)
        self.assertIn('Do not classify', annotated.iloc[0].inference_note)
        self.assertEqual(annotated.iloc[1].inference_note, '')
        with tempfile.TemporaryDirectory() as tmp:
            env['Path'] = Path
            env['save_moderation'](pd.DataFrame(), rows, Path(tmp), 'Synthetic')
            self.assertIn('x × m → y', (Path(tmp)/'fit_notes.md').read_text())
            saved = pd.read_csv(Path(tmp)/'moderation_mlm.csv')
            self.assertTrue(pd.isna(saved.iloc[0].p_interaction))

    def test_negative_condition_export_and_failure_guard(self):
        env = functions_only('scripts/analysis/05_sensitivity.py')
        result = pd.DataFrame([{'predictor': 'l_amyg-ant_vmPFC_neg', 'ci_status': 'ok'}])
        options = []
        def run(*args, **kw):
            options.append(kw)
            return pd.DataFrame(), pd.DataFrame(), result.copy()
        saved = []
        def save(corr, ols, mlm, directory, **kw):
            directory.mkdir(parents=True, exist_ok=True)
            saved.append(mlm)
        env.update(run_analysis_set=run, save_results=save)
        sample = pd.DataFrame({'l_amyg-ant_vmPFC_neg': [1., 2.]})
        with tempfile.TemporaryDirectory() as tmp:
            env['BASE_DIR'] = Path(tmp)
            env['run_negative_condition'](sample, [])
            self.assertEqual(options[0]['mlm_maxiter_by_predictor'], {'l_amyg-ant_vmPFC_neg': 2000})
            self.assertIn('maxiter=2000', (Path(tmp)/'sensitivity_neg_condition/fit_notes.md').read_text())
            result.loc[0, 'ci_status'] = 'nonconverged'
            with self.assertRaises(RuntimeError):
                env['run_negative_condition'](sample, [])
            self.assertEqual(len(saved), 1)

    def test_targeted_retry_skips_invalid_fits_only(self):
        env = functions_only('scripts/analysis/analysis_utils.py')
        calls = []
        class Result:
            use_t = False
            fe_params = pd.Series({'x': .2})
            params = fe_params
            bse_fe = pd.Series({'x': .1})
            bse = bse_fe
            nobs, ngroups, llf = 120, 60, -50.
            cov_re = pd.DataFrame([[.1]])
            def cov_params(self):
                return pd.DataFrame([[.01]], index=['x'], columns=['x'])
            def conf_int(self, alpha):
                return pd.DataFrame([[.004, .396]], index=['x'])
        class Model:
            def fit(self, **kw):
                calls.append(kw)
                result = Result()
                result.converged = kw['method'] != 'lbfgs'
                return result
        env['smf'] = SimpleNamespace(mixedlm=lambda *a, **kw: Model())
        row = env['run_mlm'](synthetic(), 'x', 'y', ['c'], maxiter=2000)
        self.assertEqual([c['method'] for c in calls], ['lbfgs', 'powell'])
        self.assertEqual(row['optimizer'], 'powell')
        self.assertEqual(row['fit_attempts'], 'lbfgs:nonconverged; powell:ok')
        calls.clear()
        legacy = env['run_mlm'](synthetic(), 'x', 'y', ['c'])
        self.assertEqual(len(calls), 1)
        self.assertFalse(legacy['converged'])
        self.assertNotIn('fit_attempts', legacy)

    def test_errors_recorded_and_sample_guard(self):
        env = functions_only('scripts/analysis/analysis_utils.py')
        model = D['capture_model'](env['run_mlm'], synthetic(), 'x', 'y', ['c'])
        output = D['fit_diagnostics'](model, 'synthetic', ['x'], methods=('nonexistent_optimizer',))
        self.assertEqual(output.iloc[0].fit_status, 'error')
        self.assertNotIn('error_message', output.columns)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'aggregate.csv'
            pd.DataFrame([dict(predictor='x', outcome='y', n=120)]).to_csv(target, index=False)
            D['validate_sample'](model, target, 'x', 'y')
            pd.DataFrame([dict(predictor='x', outcome='y', n=119)]).to_csv(target, index=False)
            with self.assertRaises(ValueError):
                D['validate_sample'](model, target, 'x', 'y')


if __name__ == '__main__':
    unittest.main()
