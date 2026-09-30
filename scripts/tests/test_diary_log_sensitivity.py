"""Synthetic checks of sensitivity routing; no analysis imports or data IO."""
import ast
from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def constants(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


class DiaryLogSensitivityTests(unittest.TestCase):
    def test_all_four_scripts_include_log_outcome_and_fc_direction(self):
        for prefix in ['', 'MR1_validation/']:
            for number in ['02', '04']:
                with self.subTest(cohort=prefix, analysis=number):
                    c = constants(ROOT / prefix / f'scripts/analysis/{number}_sensitivity.py')
                    self.assertEqual(c.get('OUTCOMES_DIARY', c.get('DIARY_OUTCOMES')),
                                     ['PA_score', 'NA_score', 'NA_score_log'])
                    if number == '04':
                        self.assertEqual(c['EXPECTED_DIRECTIONS']['NA_score_log'],
                                         c['EXPECTED_DIRECTIONS']['NA_score'])

    def test_actual_helpers_route_identical_covariates_and_direction(self):
        for prefix in ['', 'MR1_validation/']:
            path = ROOT / prefix / 'scripts/analysis/analysis_utils.py'
            tree = ast.parse(path.read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_analysis_set')
            for one_tailed in [False, True]:
                calls = []

                def correlation(df, pred, out, **kw):
                    calls.append(('correlation', out, [], kw))
                    return {'outcome': out}

                def adjusted(df, pred, out, covs, **kw):
                    calls.append(('adjusted', out, covs, kw))
                    return {'outcome': out}

                def required(df, cols, label):
                    if not set(cols).issubset(df.columns):
                        raise ValueError(label)

                env = dict(pd=pd, DIARY_OUTCOMES=constants(path)['DIARY_OUTCOMES'],
                           run_correlation=correlation, run_ols=adjusted, run_mlm=adjusted,
                           _require_columns=required)
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), env)
                df = pd.DataFrame({c: [1., 2.] for c in
                                   ['predictor', 'age', 'NA_score', 'NA_score_log', 'time_P2_P5', 'n_days_complete']})
                env['run_analysis_set'](df, ['predictor'], ['NA_score', 'NA_score_log'], ['age'],
                                        one_tailed=one_tailed,
                                        expected_directions={'NA_score': -1, 'NA_score_log': -1})
                raw = [(method, covs, kw) for method, out, covs, kw in calls if out == 'NA_score']
                log = [(method, covs, kw) for method, out, covs, kw in calls if out == 'NA_score_log']
                self.assertEqual(raw, log)
                for method, covs, kw in log:
                    if method == 'adjusted':
                        self.assertEqual(covs, ['age', 'time_P2_P5', 'n_days_complete'])
                    self.assertEqual(kw, dict(one_tailed=one_tailed, expected_positive=False))


if __name__ == '__main__':
    unittest.main()
