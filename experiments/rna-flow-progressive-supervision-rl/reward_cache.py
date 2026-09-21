"""SQLite-backed ViennaRNA terminal-reward cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import RNA


VIENNA_CONTRACT = {
    "version": RNA.__version__,
    "temperature_c": 37.0,
    "dangles": 2,
    "uniq_ML": 1,
}
RECOVERABLE_EVALUATION_ERROR_CODES = (
    "invalid_target_probability",
    "invalid_ned",
    "invalid_target_energy",
    "missing_mfe_structure",
)
MAXIMUM_INVALID_REWARD_EVALUATION_RATE = 0.001
REWARD_EVALUATION_POLICY = {
    "schema_version": 2,
    "recoverable_error_codes": list(RECOVERABLE_EVALUATION_ERROR_CODES),
    "invalid_candidate_reward": "pessimistic-zero-terminal",
    "maximum_invalid_rate": MAXIMUM_INVALID_REWARD_EVALUATION_RATE,
    "maximum_invalid_cache_key_exposure": 1,
    "all_invalid_groups_allowed": False,
}


class RecoverableCandidateEvaluationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        if code not in RECOVERABLE_EVALUATION_ERROR_CODES:
            raise ValueError(f"unknown recoverable evaluation error code: {code}")
        super().__init__(message)
        self.code = code


def safe_reward_evaluation(evaluator, sequence: str, target_structure: str) -> dict:
    """Convert known per-candidate ViennaRNA failures into explicit zero-reward evidence."""
    try:
        metrics = dict(evaluator(sequence, target_structure))
    except RecoverableCandidateEvaluationError as error:
        return {
            "evaluation_valid": False,
            "evaluation_error_type": type(error).__name__,
            "evaluation_error_code": error.code,
            "evaluation_error": str(error),
            "mfe_structure": "." * len(target_structure),
            "mfe_energy": 0.0,
            "mfe_hit": False,
            "uMFE_hit": False,
            "mfe_degeneracy": 0,
            "pair_f1": 0.0,
            "pair_error": 1.0,
            "target_energy": 0.0,
            "rival_energy_gap": 1.0,
            "rival_margin_credit": 0.0,
            "target_probability": 0.0,
            "target_probability_raw": None,
            "target_probability_clamped": False,
            "NED": 1.0,
        }
    metrics["evaluation_valid"] = True
    metrics["evaluation_error_type"] = None
    metrics["evaluation_error_code"] = None
    metrics["evaluation_error"] = None
    return metrics


def empty_reward_evaluation_coverage() -> dict:
    return {
        "candidate_rows": 0,
        "groups": 0,
        "invalid_candidate_rows": 0,
        "invalid_groups": 0,
        "all_invalid_groups": 0,
        "invalid_cache_key_histogram": {},
        "error_code_histogram": {},
        "invalid_task_histogram": {},
        "invalid_target_length_histogram": {},
        "invalid_update_histogram": {},
    }


def _increment(histogram: dict, key: object, amount: int = 1) -> None:
    text = str(key)
    histogram[text] = int(histogram.get(text, 0)) + amount


def update_reward_evaluation_coverage(
    coverage: dict, evaluation_groups: list[list[dict]], update: int
) -> dict:
    updated = json.loads(json.dumps(coverage))
    for rows in evaluation_groups:
        if not rows:
            raise RuntimeError("reward evaluation group is empty")
        invalid = [row for row in rows if row.get("evaluation_valid") is False]
        if any(row.get("evaluation_valid") not in (True, False) for row in rows):
            raise RuntimeError("reward evaluation validity evidence is absent")
        updated["candidate_rows"] += len(rows)
        updated["groups"] += 1
        updated["invalid_candidate_rows"] += len(invalid)
        updated["invalid_groups"] += int(bool(invalid))
        updated["all_invalid_groups"] += int(len(invalid) == len(rows))
        _increment(updated["invalid_update_histogram"], update, len(invalid))
        for row in invalid:
            error_code = row.get("evaluation_error_code")
            cache_key_value = row.get("reward_cache_key")
            task_id = row.get("task_id")
            target_length = row.get("target_length")
            if (
                error_code not in RECOVERABLE_EVALUATION_ERROR_CODES
                or not isinstance(cache_key_value, str)
                or len(cache_key_value) != 64
                or task_id is None
                or not isinstance(target_length, int)
                or target_length <= 0
            ):
                raise RuntimeError("invalid reward evaluation evidence is incomplete")
            _increment(updated["error_code_histogram"], error_code)
            _increment(updated["invalid_cache_key_histogram"], cache_key_value)
            _increment(updated["invalid_task_histogram"], task_id)
            _increment(updated["invalid_target_length_histogram"], target_length)
    return updated


def finalize_reward_evaluation_coverage(coverage: dict) -> dict:
    finalized = json.loads(json.dumps(coverage))
    rows = finalized["candidate_rows"]
    finalized["invalid_candidate_rate"] = (
        finalized["invalid_candidate_rows"] / rows if rows else 0.0
    )
    finalized["unique_invalid_cache_keys"] = len(
        finalized["invalid_cache_key_histogram"]
    )
    finalized["maximum_invalid_cache_key_exposure"] = max(
        finalized["invalid_cache_key_histogram"].values(), default=0
    )
    finalized["reward_evaluation_policy"] = REWARD_EVALUATION_POLICY
    return finalized


def validate_reward_evaluation_coverage(
    coverage: dict, *, expected_candidate_rows: int, expected_groups: int
) -> None:
    candidate_rows = coverage.get("candidate_rows")
    groups = coverage.get("groups")
    invalid_rows = coverage.get("invalid_candidate_rows")
    invalid_rate = coverage.get("invalid_candidate_rate")
    invalid_key_histogram = coverage.get("invalid_cache_key_histogram")
    histogram_valid = (
        isinstance(invalid_key_histogram, dict)
        and all(
            isinstance(key, str)
            and len(key) == 64
            and isinstance(value, int)
            and value > 0
            for key, value in invalid_key_histogram.items()
        )
    )
    derived_unique_keys = len(invalid_key_histogram) if histogram_valid else -1
    derived_maximum_exposure = (
        max(invalid_key_histogram.values(), default=0) if histogram_valid else -1
    )
    if (
        candidate_rows != expected_candidate_rows
        or groups != expected_groups
        or not isinstance(invalid_rows, int)
        or not 0 <= invalid_rows <= candidate_rows
        or not isinstance(invalid_rate, (int, float))
        or invalid_rate != invalid_rows / candidate_rows
        or invalid_rate > MAXIMUM_INVALID_REWARD_EVALUATION_RATE
        or not isinstance(coverage.get("invalid_groups"), int)
        or not 0 <= coverage["invalid_groups"] <= min(groups, invalid_rows)
        or coverage.get("all_invalid_groups") != 0
        or not histogram_valid
        or derived_maximum_exposure > 1
        or coverage.get("maximum_invalid_cache_key_exposure")
        != derived_maximum_exposure
        or sum(invalid_key_histogram.values()) != invalid_rows
        or coverage.get("unique_invalid_cache_keys") != derived_unique_keys
        or sum(coverage.get("error_code_histogram", {}).values()) != invalid_rows
        or sum(coverage.get("invalid_task_histogram", {}).values()) != invalid_rows
        or sum(coverage.get("invalid_target_length_histogram", {}).values()) != invalid_rows
        or sum(coverage.get("invalid_update_histogram", {}).values()) != invalid_rows
        or coverage.get("reward_evaluation_policy") != REWARD_EVALUATION_POLICY
    ):
        raise RuntimeError("reward evaluation coverage fails the frozen contract")


def validate_reward_evaluation_binding(
    receipt_coverage: dict,
    checkpoint_state: dict,
    *,
    expected_candidate_rows: int,
    expected_groups: int,
) -> None:
    validate_reward_evaluation_coverage(
        receipt_coverage,
        expected_candidate_rows=expected_candidate_rows,
        expected_groups=expected_groups,
    )
    state_coverage = checkpoint_state.get("reward_evaluation_coverage")
    if not isinstance(state_coverage, dict):
        raise RuntimeError("checkpoint reward evaluation coverage is absent")
    finalized_state_coverage = finalize_reward_evaluation_coverage(state_coverage)
    validate_reward_evaluation_coverage(
        finalized_state_coverage,
        expected_candidate_rows=expected_candidate_rows,
        expected_groups=expected_groups,
    )
    if finalized_state_coverage != receipt_coverage:
        raise RuntimeError("receipt reward evaluation coverage does not match checkpoint")


def cache_key(sequence: str, target_structure: str) -> str:
    encoded = json.dumps(
        {
            "sequence": sequence,
            "target_structure": target_structure,
            "vienna": VIENNA_CONTRACT,
            "reward_evaluation_policy": REWARD_EVALUATION_POLICY,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class ViennaRewardCache:
    def __init__(self, path: Path, evaluator) -> None:
        self.path = path
        self.evaluator = evaluator
        self.hits = 0
        self.misses = 0
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                cache_key TEXT PRIMARY KEY,
                sequence TEXT NOT NULL,
                target_structure TEXT NOT NULL,
                vienna_version TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_unix REAL NOT NULL
            )
            """
        )
        self.connection.commit()

    def evaluate(self, sequence: str, target_structure: str) -> dict:
        key = cache_key(sequence, target_structure)
        row = self.connection.execute(
            "SELECT metrics_json FROM evaluations WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is not None:
            self.hits += 1
            return json.loads(row[0])
        metrics = safe_reward_evaluation(self.evaluator, sequence, target_structure)
        encoded = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO evaluations (
                    cache_key, sequence, target_structure, vienna_version, metrics_json, created_unix
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, sequence, target_structure, RNA.__version__, encoded, time.time()),
            )
        self.misses += 1
        return metrics

    def stats(self) -> dict:
        encoded_rows = self.connection.execute("SELECT metrics_json FROM evaluations").fetchall()
        rows = len(encoded_rows)
        invalid_unique_rows = sum(
            json.loads(encoded).get("evaluation_valid", True) is False
            for (encoded,) in encoded_rows
        )
        return {
            "unique_rows": rows,
            "invalid_unique_rows": invalid_unique_rows,
            "invalid_unique_rate": invalid_unique_rows / rows if rows else 0.0,
            "hits": self.hits,
            "misses": self.misses,
            "vienna": VIENNA_CONTRACT,
            "reward_evaluation_policy": REWARD_EVALUATION_POLICY,
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ViennaRewardCache":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
