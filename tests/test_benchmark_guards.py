"""Fast protocol failure tests; no model weights, dataset or GPU required."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_benchmark import CONDITIONS, read_benchmark
from evaluate import global_task_seed, validate_coverage


class BenchmarkGuardsTest(unittest.TestCase):
    def test_frozen_conditions(self):
        self.assertEqual(CONDITIONS, {1009: .8, 2027: 1., 3037: 1.2})
        self.assertEqual(global_task_seed(2027, 5), 5002042)
        with self.assertRaises(ValueError):
            global_task_seed(2027, -1)

    def test_foreign_benchmark_rejected(self):
        path = Mock()
        path.read_bytes.return_value = b'{"id":"foreign","target_structure":"..."}\n'
        for name in ('eterna100v2', 'eterna100', 'rfam27', 'rnasolo764'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'bytes/order'):
                read_benchmark(path, name)

    def test_complete_coverage(self):
        rows = [dict(task_id='one', seed=s, candidate_index=i) for s in CONDITIONS for i in range(8)]
        validate_coverage(rows, ['one'], list(CONDITIONS), 8)
        with self.assertRaisesRegex(RuntimeError, 'candidate coverage'):
            validate_coverage(rows[:-1], ['one'], list(CONDITIONS), 8)
        with self.assertRaisesRegex(RuntimeError, 'candidate coverage'):
            validate_coverage(rows + rows[:1], ['one'], list(CONDITIONS), 8)
        with self.assertRaisesRegex(RuntimeError, 'group mismatch'):
            validate_coverage(rows, ['one', 'missing'], list(CONDITIONS), 8)


if __name__ == '__main__':
    unittest.main()
