import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('inverse_runtime', Path(__file__).resolve().parents[1] / 'scripts/measure_inverse_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class InverseRuntimeTest(unittest.TestCase):
    def test_summary_export_fields_and_smoke_label(self):
        rows = [dict(returned_valid_candidates=7, timeout_candidates=1, run_time_seconds=2.,
                     post_scoring_seconds=.5, end_to_end_seconds=2.5)]
        summary = runtime.summarize(rows, 'a' * 64, smoke=True)
        self.assertEqual(summary['status'], 'complete_smoke')
        self.assertEqual(summary['attempted_candidates'], 8)
        self.assertEqual(summary['failed_candidates'], 1)
        self.assertEqual(summary['plan_sha256'], 'a' * 64)
        self.assertIn('not resident', summary['scope'])

    def test_eight_calls_and_timeout_accounting(self):
        good = Mock(returncode=0)
        good.communicate.return_value = (json.dumps({'sequence': 'ACGU', 'status': 'ok'}), '')
        with patch.object(runtime.subprocess, 'Popen', return_value=good) as launch:
            rows = runtime.generate(0, '....', 1009)
        self.assertEqual(len(rows), 8)
        self.assertEqual(launch.call_count, 8)
        self.assertTrue(all(r['status'] == 'ok' for r in rows))
        failed = Mock()
        failed.communicate.side_effect = [subprocess.TimeoutExpired('child', 45), ('', '')] * 8
        with patch.object(runtime.subprocess, 'Popen', return_value=failed):
            timeouts = runtime.generate(0, '....', 1009)
        self.assertEqual(failed.kill.call_count, 8)
        self.assertTrue(all(r['sequence'] == '' and r['status'] == 'timeout' for r in timeouts))
        self.assertEqual([r['start_sequence_sha256'] for r in rows],
                         [r['start_sequence_sha256'] for r in timeouts])
        bad = Mock(returncode=1)
        bad.communicate.return_value = ('', 'failure')
        with patch.object(runtime.subprocess, 'Popen', return_value=bad), self.assertRaises(RuntimeError):
            runtime.generate(0, '....', 1009)


if __name__ == '__main__':
    unittest.main()
