"""Admission checks only; synthetic timings are never experimental evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('resident_aggregation', Path(__file__).resolve().parents[1] / 'scripts/aggregate_resident_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeSourceTest(unittest.TestCase):
    def test_frozen_scope_and_finite_times(self):
        methods = sorted(runtime.METHODS)
        summary = {'plan': {'K': 8, 'tasks': 100, 'repeats': 2,
                           'conditions': [1009, 2027, 3037], 'methods': methods,
                           'models_resident_simultaneously': True},
                   'methods': [{'method': m, 'run_time_seconds_per_K8_group': 1.0} for m in methods]}
        rows = [{'generation_seconds': 1.0, 'post_scoring_seconds': .1} for _ in range(3000)]
        runtime.validate_source(summary, rows)
        for field, value in [('K', 1), ('repeats', 1), ('tasks', 12),
                             ('models_resident_simultaneously', False), ('methods', methods[:4])]:
            bad = copy.deepcopy(summary)
            bad['plan'][field] = value
            with self.assertRaises(ValueError):
                runtime.validate_source(bad, rows)
        for value in (float('nan'), float('inf'), -1):
            bad_rows = rows.copy()
            bad_rows[0] = dict(rows[0], generation_seconds=value)
            with self.assertRaises(ValueError):
                runtime.validate_source(summary, bad_rows)
        with self.assertRaises(ValueError):
            runtime.validate_source(summary, rows[:-1])


if __name__ == '__main__':
    unittest.main()
