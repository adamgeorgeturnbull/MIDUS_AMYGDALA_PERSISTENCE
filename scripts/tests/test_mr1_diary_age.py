"""Synthetic checks of actual age code, without importing I/O entry points."""
import ast
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def estimate(frame):
    tree = ast.parse((ROOT / 'MR1_validation/scripts/preprocessing/06_clean_merged_data.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'estimate_diary_age')
    scope = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<age helper>', 'exec'), scope)
    return scope['estimate_diary_age'](frame)


class DiaryAgeTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'is_mker1': [0, 1, 1], 'birth_year': [1955., 1955., 1955.],
                             'StartYear': [2015., 2015., 2014.], 'StartMonth': [9., 9., 9.],
                             'RAACRAGE': [60., 60., 60.], 'RAACIDATE_YR': [2015.] * 3,
                             'RAACIDATE_MO': [3.] * 3})

    def test_cohort_rule_overrides_available_birth_year(self):
        d = self.frame(); before = d.copy(deep=True)
        np.testing.assert_allclose(estimate(d), [60., 60.5, 59.5])
        pd.testing.assert_frame_equal(d, before)

    def test_missing_interview_inputs_do_not_fall_back_to_birth_year(self):
        for col, missing in [('RAACRAGE', 97), ('RAACIDATE_YR', 9998),
                             ('RAACIDATE_MO', 99), ('RAACRAGE', np.nan)]:
            with self.subTest(col=col, missing=missing):
                d = self.frame(); d.loc[1, col] = missing
                self.assertTrue(pd.isna(estimate(d).iloc[1]))
                self.assertEqual(estimate(d).iloc[0], 60.)

    def test_missing_non_milwaukee_birth_year_stays_missing(self):
        d = self.frame(); d.loc[0, 'birth_year'] = np.nan
        self.assertTrue(pd.isna(estimate(d).iloc[0]))

    def test_membership_must_be_known(self):
        for invalid in [np.nan, 2]:
            d = self.frame(); d.loc[1, 'is_mker1'] = invalid
            with self.assertRaises(ValueError): estimate(d)

    def test_matches_actual_m3_milwaukee_formula(self):
        tree = ast.parse((ROOT / 'scripts/preprocessing/06_clean_merged_data.py').read_text())
        assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Tuple)
                                  and any(isinstance(v, ast.Name) and v.id == 'mke2_mask' for v in t.slice.elts)
                                  for t in n.targets))
        d = self.frame()
        mapped = d.rename(columns={'RAACRAGE': 'CACRAGE', 'RAACIDATE_YR': 'CACIDATE_YR',
                                    'RAACIDATE_MO': 'CACIDATE_MO'}).copy()
        mapped['C2PAGE'] = np.nan
        scope = {'df': mapped, 'mke2_mask': d.is_mker1.eq(1)}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<M3 age>', 'exec'), scope)
        np.testing.assert_allclose(estimate(d).loc[d.is_mker1.eq(1)], mapped.loc[d.is_mker1.eq(1), 'C2PAGE'])


if __name__ == '__main__':
    unittest.main()
