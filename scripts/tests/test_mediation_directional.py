"""Synthetic-only tests; extract pure functions without importing data loaders."""
import ast
from pathlib import Path
import unittest
import numpy as np
from scipy import stats

SOURCE = Path(__file__).resolve().parents[1] / 'analysis/03b_persistence_age_mediation.py'

def pure_functions(path):
    tree = ast.parse(path.read_text())
    keep = {'wald', 'mc_interval', 'directional_bootstrap_bound'}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    env = {'np': np, 'stats': stats, 'CI_LEVEL': 95., 'N_MC': 200000}
    exec(compile(tree, str(path), 'exec'), env)
    return env

class DirectionalTests(unittest.TestCase):
    def setUp(self):
        self.f = pure_functions(SOURCE)

    def test_normal_tails_and_wrong_direction(self):
        for direction in (-1, 1):
            _, p = self.f['wald'](direction * 1.6448536269514722, 1., direction)
            self.assertAlmostEqual(p, .05)
            _, opposite = self.f['wald'](-direction * 1.6448536269514722, 1., direction)
            self.assertAlmostEqual(opposite, .95)
        self.assertEqual(self.f['wald'](0., 1., -1)[1], .5)
        self.assertTrue(np.isnan(self.f['wald'](1., 0., 1)[1]))
        self.assertGreater(self.f['wald'](9., 1., 1)[1], 0.)

    def test_bootstrap_bound_directions(self):
        values = np.linspace(-1., 19., 2001)
        bound, significant = self.f['directional_bootstrap_bound'](values, 1)
        self.assertAlmostEqual(bound, 0.)
        self.assertFalse(significant)
        self.assertTrue(self.f['directional_bootstrap_bound'](values + .01, 1)[1])
        self.assertTrue(self.f['directional_bootstrap_bound'](-values - .01, -1)[1])
        self.assertFalse(self.f['directional_bootstrap_bound'](values, -1)[1])
        for bad in ([], [np.nan]):
            with self.assertRaises(ValueError):
                self.f['directional_bootstrap_bound'](bad, 1)

    def test_mc_interval_unaffected_by_direction(self):
        args = (-.2, .1, .3, .1)
        lo, hi, p = self.f['mc_interval'](*args, np.random.default_rng(42), -1)
        lo2, hi2, q = self.f['mc_interval'](*args, np.random.default_rng(42), 1)
        self.assertEqual((lo, hi), (lo2, hi2))
        self.assertAlmostEqual(p + q, 1.)
        self.assertLess(p, .05)
        self.assertGreater(q, .95)

if __name__ == '__main__':
    unittest.main()
