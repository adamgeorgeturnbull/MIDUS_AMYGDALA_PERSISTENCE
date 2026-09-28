"""Synthetic tests of the actual MR1 diary deduplication function; no data IO."""
import ast
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[2] / 'MR1_validation/scripts/preprocessing/02_construct_daily_diary_affect.py'
tree = ast.parse(SOURCE.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'deduplicate_diary_days')
namespace = {'np': np, 'pd': pd}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
deduplicate = namespace['deduplicate_diary_days']


def record(day=1, value=1):
    return dict(MIDUSID=101, RA2DDAY=day, RA2DIMON=4, RA2DIYEAR=2012,
                **{f'RA2DC{i}': value for i in range(1, 28)})


class DiaryDuplicatesTests(unittest.TestCase):
    def test_unique_unchanged_and_input_preserved(self):
        frame = pd.DataFrame([record(), record(2)])
        original = frame.copy(deep=True)
        pd.testing.assert_frame_equal(deduplicate(frame), frame)
        pd.testing.assert_frame_equal(frame, original)

    def test_equal_day_weight_and_metadata_difference(self):
        rows = [record(), record(2, 3), record(2, 3)]
        rows[1]['MRID'] = 101
        rows[2]['MRID'] = np.nan
        frame = deduplicate(pd.DataFrame(rows))
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.RA2DC1.mean(), 2)

    def test_matching_missingness_allowed(self):
        row = record()
        row['RA2DC3'] = np.nan
        self.assertEqual(len(deduplicate(pd.DataFrame([row, row]))), 1)

    def test_every_scoring_field_conflict_stops(self):
        for field in [f'RA2DC{i}' for i in range(1, 28)] + ['RA2DIMON', 'RA2DIYEAR']:
            for value in [np.nan, 999]:
                with self.subTest(field=field, value=value):
                    a, b = record(), record()
                    b[field] = value
                    with self.assertRaises(ValueError):
                        deduplicate(pd.DataFrame([a, b]))

    def test_invalid_keys_stop(self):
        for field, values in [('MIDUSID', [np.nan, np.inf, 1.5, 'bad']),
                              ('RA2DDAY', [np.nan, np.inf, 1.5, 'bad', 0, 9])]:
            for value in values:
                row = record()
                row[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    deduplicate(pd.DataFrame([row]))

    def test_missing_column_stops(self):
        with self.assertRaises(ValueError):
            deduplicate(pd.DataFrame([record()]).drop(columns='RA2DC27'))


if __name__ == '__main__':
    unittest.main()
