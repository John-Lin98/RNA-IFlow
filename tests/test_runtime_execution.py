"""CPU-only protocol tests for the resident runtime orchestrator.

These tests intentionally exercise ordering, ledger validation, and failure
handling with synthetic samplers.  They do not load a model and are not model
or scientific-result tests.
"""

from __future__ import annotations

import json
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import measure_resident_runtime as runtime  # noqa: E402


class ResidentRuntimeExecutionTests(unittest.TestCase):
    class _Buffer(io.StringIO):
        def close(self):
            pass

    def _fixture(self):
        tasks = [
            {"id": "short", "target_structure": "....", "_global_index": 0},
            {"id": "long", "target_structure": "........", "_global_index": 1},
        ]
        expected = {}
        samplers = {}
        calls = []
        for method_index, method in enumerate(runtime.METHODS):
            groups = {}
            for task in tasks:
                for seed in runtime.SEEDS:
                    groups[(task["id"], seed)] = [
                        f"{method_index}:{task['id']}:{seed}:{candidate}"
                        for candidate in range(runtime.K)
                    ]
            expected[method] = groups

            def sampler(task, seed, method=method):
                calls.append((method, task["id"], seed))
                return list(expected[method][(task["id"], seed)])

            samplers[method] = sampler
        identities = {method: {"fixture": True} for method in runtime.METHODS}
        plan = {
            "methods": list(runtime.METHODS),
            "tasks": len(tasks),
            "repeats": 2,
        }
        return tasks, expected, samplers, identities, plan, calls

    @staticmethod
    def _clock():
        current = [0]

        def time_ns():
            current[0] += 1_000_000
            return current[0]

        return time_ns

    @staticmethod
    def _score(_sequences, _structure):
        return [{"evaluation_valid": True} for _ in range(runtime.K)]

    def _run(self, **kwargs):
        buffer = self._Buffer()
        with mock.patch.object(Path, "open", return_value=buffer), \
             mock.patch.object(runtime, "sha256", return_value="fixture-ledger-sha"):
            result = runtime.run_timing(output=Path("unused-fixture-output"), **kwargs)
        return result, buffer.getvalue()

    def test_cpu_import_and_cli_help(self):
        imported = __import__("runtime_samplers")
        self.assertTrue(hasattr(imported, "build_samplers"))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "scripts")
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/measure_resident_runtime.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--goforth-root", completed.stdout)

    def test_ordering_and_fresh_ledger(self):
        tasks, expected, samplers, identities, plan, calls = self._fixture()
        synchronizations = []
        result, ledger = self._run(
            tasks=tasks, samplers=samplers, expected=expected, identities=identities,
            plan=plan, score=self._score,
            synchronize=lambda: synchronizations.append(True), time_ns=self._clock())
        rows = [json.loads(line) for line in ledger.splitlines()]

        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(rows), len(tasks) * len(runtime.SEEDS) * 2 * len(runtime.METHODS))
        self.assertEqual(
            calls[: len(runtime.METHODS)],
            [(method, "long", 1009) for method in runtime.METHODS],
        )
        self.assertEqual(
            calls[len(runtime.METHODS) : 2 * len(runtime.METHODS)],
            [(method, "short", 1009) for method in runtime.METHODS],
        )
        self.assertEqual(len(synchronizations), 2 * len(calls))
        self.assertTrue(all(row["generation_seconds"] > 0 for row in rows))

    def test_candidate_identity_mismatch_aborts(self):
        tasks, expected, samplers, identities, plan, _calls = self._fixture()
        original = samplers["RNA-IFlow-RL"]
        invocations = [0]

        def mismatching_sampler(task, seed):
            sequences = original(task, seed)
            invocations[0] += 1
            if invocations[0] > 1:
                sequences[0] = "mismatch"
            return sequences

        samplers["RNA-IFlow-RL"] = mismatching_sampler
        with self.assertRaisesRegex(RuntimeError, "candidate identity mismatch"):
            self._run(tasks=tasks, samplers=samplers, expected=expected, identities=identities,
                      plan=plan, score=self._score, synchronize=lambda: None,
                      time_ns=self._clock())

    def test_scorer_failure_aborts(self):
        tasks, expected, samplers, identities, plan, _calls = self._fixture()

        def failing_score(_sequences, _structure):
            return [{"evaluation_valid": False} for _ in range(runtime.K)]

        with self.assertRaisesRegex(RuntimeError, "scorer failure"):
            self._run(tasks=tasks, samplers=samplers, expected=expected, identities=identities,
                      plan=plan, score=failing_score, synchronize=lambda: None,
                      time_ns=self._clock())

    def test_invalid_scorer_cardinality_aborts(self):
        tasks, expected, samplers, identities, plan, _calls = self._fixture()

        with self.assertRaisesRegex(RuntimeError, "expected 8"):
            self._run(tasks=tasks, samplers=samplers, expected=expected, identities=identities,
                      plan=plan, score=lambda _sequences, _structure: [],
                      synchronize=lambda: None, time_ns=self._clock())


if __name__ == "__main__":
    unittest.main()
