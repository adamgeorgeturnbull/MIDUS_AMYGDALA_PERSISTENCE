"""Exercise actual PANAS transformation statements with synthetic inputs only.

Do not import preprocessing entry points or execute their file operations.
"""
import ast
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


class PanasLogTransformTests(unittest.TestCase):
    def test_actual_m3_and_mr1_statements_use_plain_log(self):
        for relative, variable in [
            ("scripts/preprocessing/06_clean_merged_data.py", "C5SPGN"),
            ("MR1_validation/scripts/preprocessing/06_clean_merged_data.py", "RA5SPGN"),
        ]:
            with self.subTest(cohort=variable):
                tree = ast.parse((ROOT / relative).read_text())
                assignments = [node for node in ast.walk(tree)
                               if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Subscript)
                                       and isinstance(target.value, ast.Name)
                                       and target.value.id == "df"
                                       and isinstance(target.slice, ast.Constant)
                                       and target.slice.value == variable + "_log"
                                       for target in node.targets)]
                self.assertEqual(len(assignments), 1)
                # Compile only the transformation assignment, never main() or I/O.
                expression = ast.Module(body=assignments, type_ignores=[])
                frame = pd.DataFrame({variable: [1.0, 1.5, math.e, 4.0, 5.0, np.nan]})
                original = frame[variable].copy()
                exec(compile(expression, relative, "exec"), {"np": np, "df": frame})
                expected = [0.0, math.log(1.5), 1.0, math.log(4.0), math.log(5.0), np.nan]
                np.testing.assert_allclose(frame[variable + "_log"], expected, equal_nan=True)
                pd.testing.assert_series_equal(frame[variable], original)
                self.assertEqual(frame[variable].isna().tolist(),
                                 frame[variable + "_log"].isna().tolist())


if __name__ == "__main__":
    unittest.main()
