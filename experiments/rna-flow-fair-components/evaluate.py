"""Resume-safe constrained generation and ViennaRNA 2.7.2 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import RNA
import torch

LEGACY_DIR = Path(__file__).resolve().parents[1] / "dual-prior-rna-flow"
PROGRESSIVE_DIR = Path(__file__).resolve().parents[1] / "rna-flow-progressive-supervision-rl"
sys.path.append(str(LEGACY_DIR))
sys.path.append(str(PROGRESSIVE_DIR))
from constraints import constrained_decode, target_pairs, valid_pair_fraction  # noqa: E402
from flow_core import DirichletConditionalFlow  # noqa: E402
from finalize_supervised_selection import validate_selection  # noqa: E402
from formal_reference_contract import FORMAL_ENDPOINT_CONTRACT  # noqa: E402
from reward_cache import (  # noqa: E402
    RECOVERABLE_EVALUATION_ERROR_CODES,
    RecoverableCandidateEvaluationError,
    safe_reward_evaluation,
    validate_reward_evaluation_binding,
)
from model import FairRNAFlow, STRUCTURE_SYMBOLS, configure_rl_trainable_scope  # noqa: E402


ALLOWED_FORMAL_CANDIDATE_METHODS = {
    "simplex-endpoint-policy-trajectory-grpo",
    "simplex-endpoint-policy-domino-ppo",
    "structured-discrete-mixture-dfm-domino-ppo",
    "simplex-endpoint-policy-sequence-group-fpo",
    "simplex-endpoint-policy-contrastive-task-step",
    "simplex-endpoint-policy-mfe-binary-contrastive-task-step",
    "simplex-endpoint-policy-pass8-frontier-mfe-binary-contrastive-task-step",
    "reward_weighted_fm",
    "terminal_grpo",
    "ce_anchor_only_continuation",
}
AGGREGATE_STATE_SCHEMA_VERSION = 2
AGGREGATE_RECEIPT_SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def contract_sha256(contract: dict) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def candidate_method_identity(candidate: dict | None) -> tuple[str | None, str]:
    if candidate is None:
        return None, "fair-rna-flow-x0"
    method = candidate.get("contract", {}).get("method")
    if method not in ALLOWED_FORMAL_CANDIDATE_METHODS:
        raise RuntimeError("unsupported post-training candidate method")
    return method, f"fair-rna-flow-x0-{method}"


def prepare_evaluation_output(output: Path, resume: bool, contract: dict) -> str:
    digest = contract_sha256(contract)
    record = contract | {"contract_sha256": digest}
    contract_path = output / "evaluation_contract.json"
    if resume:
        if not output.is_dir() or not contract_path.is_file():
            raise FileNotFoundError("resume evaluation contract is absent")
        if json.loads(contract_path.read_text()) != record:
            raise RuntimeError("evaluation resume contract mismatch")
    else:
        output.mkdir(parents=True, exist_ok=False)
        temporary = contract_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, contract_path)
    return digest


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def structure_key(structure: str) -> str:
    return hashlib.sha256(structure.encode()).hexdigest()


def pair_f1(predicted: str, target: str) -> float:
    predicted_pairs = set(target_pairs(predicted))
    target_pair_set = set(target_pairs(target))
    if not predicted_pairs and not target_pair_set:
        return 1.0
    if not predicted_pairs or not target_pair_set:
        return 0.0
    overlap = len(predicted_pairs & target_pair_set)
    return 2 * overlap / (len(predicted_pairs) + len(target_pair_set))


def bounded_probability(value: float, tolerance: float = 0.01) -> tuple[float, bool]:
    """Clamp small partition-function roundoff while rejecting invalid values."""
    if not math.isfinite(value) or value < -tolerance or value > 1 + tolerance:
        raise RecoverableCandidateEvaluationError(
            "invalid_target_probability", f"invalid target probability: {value}"
        )
    bounded = min(max(value, 0.0), 1.0)
    return bounded, bounded != value


def evaluate_candidate(sequence: str, target: str) -> dict:
    md = RNA.md()
    md.temperature = 37.0
    md.dangles = 2
    md.uniq_ML = 1
    compound = RNA.fold_compound(sequence, md)
    mfe_structure, mfe_energy = compound.mfe()
    mfe_structures = {solution.structure for solution in compound.subopt(0)}
    if not mfe_structures:
        raise RecoverableCandidateEvaluationError(
            "missing_mfe_structure", "ViennaRNA returned no MFE structure"
        )
    compound.exp_params_rescale(mfe_energy)
    compound.pf()
    probability_raw = float(compound.pr_structure(target))
    probability, probability_clamped = bounded_probability(probability_raw)
    ned = float(compound.ensemble_defect(target)) / len(target)
    if not math.isfinite(ned) or not 0 <= ned <= 1:
        raise RecoverableCandidateEvaluationError("invalid_ned", f"invalid NED: {ned}")
    score_f1 = pair_f1(mfe_structure, target)
    target_energy = float(compound.eval_structure(target))
    if not math.isfinite(target_energy):
        raise RecoverableCandidateEvaluationError(
            "invalid_target_energy", f"invalid target energy: {target_energy}"
        )
    rival_energy_gap = max(target_energy - float(mfe_energy), 0.0)
    rival_margin_credit = 1.0 / (1.0 + rival_energy_gap)
    return {
        "mfe_structure": mfe_structure,
        "mfe_energy": float(mfe_energy),
        "mfe_hit": target in mfe_structures,
        "uMFE_hit": len(mfe_structures) == 1 and target in mfe_structures,
        "mfe_degeneracy": len(mfe_structures),
        "pair_f1": score_f1,
        "pair_error": 1 - score_f1,
        "target_energy": target_energy,
        "rival_energy_gap": rival_energy_gap,
        "rival_margin_credit": rival_margin_credit,
        "target_probability": probability,
        "target_probability_raw": probability_raw,
        "target_probability_clamped": probability_clamped,
        "NED": ned,
    }


def evaluate_many(
    sequences: list[str], target: str, workers: int, *, safe_recoverable: bool = False,
) -> list[dict]:
    items = [(sequence, target) for sequence in sequences]
    if safe_recoverable:
        evaluator = lambda item: safe_reward_evaluation(evaluate_candidate, *item)
    else:
        evaluator = lambda item: evaluate_candidate(*item)
    if workers == 1:
        return [evaluator(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(evaluator, items))


def structure_inputs(
    model: FairRNAFlow,
    structure: str,
    cached_hidden: torch.Tensor | None,
    candidates: int,
    condition: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    hidden = None
    tokens = torch.tensor(
        [STRUCTURE_SYMBOLS.index(symbol) for symbol in structure], dtype=torch.long, device=device
    ).unsqueeze(0).expand(candidates, -1)
    if cached_hidden is not None:
        hidden = cached_hidden.to(device).float().unsqueeze(0).expand(candidates, -1, -1)
    override = None
    if condition == "zero":
        dim = 64 if model.structure_source == "dotbracket" else hidden.shape[-1]
        override = torch.zeros(candidates, len(structure), dim, device=device)
    elif condition == "shuffle":
        generator = torch.Generator(device=device).manual_seed(9176 + len(structure))
        permutation = torch.randperm(len(structure), generator=generator, device=device)
        if hidden is not None:
            hidden = hidden[:, permutation]
        else:
            tokens = tokens[:, permutation]
    return hidden, tokens, override


class TorchDirichletConditionalFlow:
    """GPU-native equivalent of RNACG's uniform-grid NumPy interpolation."""

    def __init__(self, device: torch.device) -> None:
        reference = DirichletConditionalFlow(alphabet_size=4)
        self.derivatives = torch.from_numpy(reference.beta_cdfs_derivative).to(
            device=device, dtype=torch.float32
        )
        self.alpha_min = float(reference.alphas[0])
        self.alpha_spacing = float(reference.alphas[1] - reference.alphas[0])

    def posterior_to_velocity(
        self, posterior: torch.Tensor, state: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        alpha_value = float(alpha.reshape(-1)[0].item())
        row = int(round((alpha_value - self.alpha_min) / self.alpha_spacing))
        row = min(max(row, 0), self.derivatives.shape[0] - 1)
        probabilities = state.float().clamp(0, 1)
        positions = probabilities * (self.derivatives.shape[1] - 1)
        left = positions.floor().long().clamp(0, self.derivatives.shape[1] - 2)
        fraction = positions - left
        values = self.derivatives[row]
        interpolated = -(values[left] * (1 - fraction) + values[left + 1] * fraction)
        beta = 2.0 / (alpha_value * (alpha_value + 1) * (alpha_value + 2))
        denominator = (1 - probabilities).pow(3) * probabilities.pow(alpha_value - 1)
        ratio = torch.where(
            (probabilities < 1) & (denominator > 0), beta / denominator, torch.zeros_like(denominator)
        )
        coefficient = torch.nan_to_num(interpolated * ratio, nan=0.0, posinf=0.0, neginf=0.0).to(state)
        eye = torch.eye(state.shape[-1], dtype=state.dtype, device=state.device)
        conditional = (eye - state.unsqueeze(-1)) * coefficient.unsqueeze(-2)
        velocity = (posterior.unsqueeze(-2) * conditional).sum(-1)
        return velocity - velocity.mean(dim=-1, keepdim=True)


def deterministic_simplex_project(value: torch.Tensor) -> torch.Tensor:
    """Project RNA probabilities without CUDA's nondeterministic cumsum kernel."""
    if value.shape[-1] != 4:
        raise ValueError("RNA simplex projection requires four channels")
    flat = value.reshape(-1, 4)
    ordered, _ = torch.sort(flat, dim=-1, descending=True)
    prefix = torch.stack(
        (
            ordered[:, 0],
            ordered[:, 0] + ordered[:, 1],
            ordered[:, 0] + ordered[:, 1] + ordered[:, 2],
            ordered[:, 0] + ordered[:, 1] + ordered[:, 2] + ordered[:, 3],
        ),
        dim=-1,
    )
    divisors = torch.tensor((1, 2, 3, 4), dtype=value.dtype, device=value.device)
    thresholds = (prefix - 1) / divisors
    support = (ordered > thresholds).sum(dim=-1, keepdim=True)
    threshold = thresholds.gather(-1, support - 1)
    return torch.clamp(flat - threshold, min=0).view_as(value)


def global_task_seed(seed: int, local_task_index: int, task_index_offset: int = 0) -> int:
    if seed < 0 or local_task_index < 0 or task_index_offset < 0:
        raise ValueError("global task seed inputs must be non-negative")
    return seed + (task_index_offset + local_task_index) * 1_000_003


@torch.inference_mode()
def sample(
    model: FairRNAFlow,
    structure: str,
    cached_hidden: torch.Tensor | None,
    candidates: int,
    steps: int,
    seed: int,
    condition: str,
    device: torch.device,
) -> list[str]:
    generator = torch.Generator(device=device).manual_seed(seed)
    concentrations = torch.ones(candidates, len(structure), 4, device=device)
    state = torch._sample_dirichlet(concentrations, generator=generator)
    mask = torch.ones(candidates, len(structure), device=device)
    hidden, tokens, override = structure_inputs(
        model, structure, cached_hidden, candidates, condition, device
    )
    flow = TorchDirichletConditionalFlow(device)
    schedule = torch.linspace(1.001, 8.0, steps + 1, device=device)
    for index in range(steps):
        alpha = schedule[index].expand(candidates)
        logits = model(
            state,
            alpha,
            mask,
            structure_hidden=hidden,
            structure_tokens=tokens,
            structure_override=override,
        )
        posterior = torch.softmax(logits, dim=-1)
        velocity = flow.posterior_to_velocity(posterior, state, schedule[index])
        state = deterministic_simplex_project(
            state + velocity * (schedule[index + 1] - schedule[index])
        )
    return [constrained_decode(state[index], structure) for index in range(candidates)]


def validate_coverage(rows: list[dict], tasks: list[str], seeds: list[int], candidates: int) -> None:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        groups[(str(row["task_id"]), int(row["seed"]))].append(int(row["candidate_index"]))
    expected = {(str(task), seed) for task in tasks for seed in seeds}
    if set(groups) != expected:
        raise RuntimeError(
            f"coverage group mismatch: missing={sorted(expected - set(groups))[:5]}, "
            f"extra={sorted(set(groups) - expected)[:5]}"
        )
    for key, indices in groups.items():
        if sorted(indices) != list(range(candidates)):
            raise RuntimeError(f"candidate coverage mismatch for {key}: {sorted(indices)}")


def aggregate(rows: list[dict], candidates: int) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_id"]), int(row["seed"]))].append(row)
    task_seed = []
    for (task_id, seed), values in sorted(grouped.items()):
        values.sort(key=lambda row: row["candidate_index"])
        task_seed.append({
            "task_id": task_id,
            "seed": seed,
            "pass_at_1": float(values[0]["uMFE_hit"]),
            f"pass_at_{candidates}": float(any(row["uMFE_hit"] for row in values)),
            "mfe_at_1": float(values[0]["mfe_hit"]),
            f"mfe_at_{candidates}": float(any(row["mfe_hit"] for row in values)),
            "best_pair_f1": max(row["pair_f1"] for row in values),
            "best_target_probability": max(row["target_probability"] for row in values),
            "best_NED": min(row["NED"] for row in values),
            "valid_rate": sum(row["valid"] for row in values) / candidates,
            "valid_pair_rate": sum(row["valid_pair_fraction"] for row in values) / candidates,
            "diversity": len({row["sequence"] for row in values}) / candidates,
        })
    keys = [
        "pass_at_1", f"pass_at_{candidates}", "mfe_at_1", f"mfe_at_{candidates}",
        "best_pair_f1", "best_target_probability", "best_NED", "valid_rate",
        "valid_pair_rate", "diversity",
    ]
    summary = {key: sum(row[key] for row in task_seed) / len(task_seed) for key in keys}
    validity_fields = ["evaluation_valid" in row for row in rows]
    if any(validity_fields) and not all(validity_fields):
        raise RuntimeError("candidate evaluation validity schema is mixed")
    error_code_histogram: dict[str, int] = {}
    bad_count = 0
    for row in rows:
        if "evaluation_valid" not in row:
            continue
        if type(row["evaluation_valid"]) is not bool:
            raise RuntimeError("candidate evaluation validity evidence is invalid")
        if row["evaluation_valid"] is False:
            error_code = row.get("evaluation_error_code")
            if error_code not in RECOVERABLE_EVALUATION_ERROR_CODES:
                raise RuntimeError("recoverable candidate error evidence is invalid")
            bad_count += 1
            error_code_histogram[error_code] = error_code_histogram.get(error_code, 0) + 1
    summary.update({
        "task_seed_groups": len(task_seed),
        "candidate_rows": len(rows),
        "target_probability_clamp_count": sum(
            bool(row.get("target_probability_clamped", False)) for row in rows
        ),
        "bad_count": bad_count,
        "error_code_histogram": error_code_histogram,
        "invalid_candidate_rate": bad_count / len(rows) if rows else 0.0,
    })
    return task_seed, summary


def aggregate_group(values: list[dict], candidates: int) -> dict:
    if len(values) != candidates:
        raise RuntimeError("aggregate-only candidate group coverage mismatch")
    values.sort(key=lambda row: row["candidate_index"])
    return {
        "pass_at_1": float(values[0]["uMFE_hit"]),
        f"pass_at_{candidates}": float(any(row["uMFE_hit"] for row in values)),
        "mfe_at_1": float(values[0]["mfe_hit"]),
        f"mfe_at_{candidates}": float(any(row["mfe_hit"] for row in values)),
        "best_pair_f1": max(row["pair_f1"] for row in values),
        "best_target_probability": max(row["target_probability"] for row in values),
        "best_NED": min(row["NED"] for row in values),
        "valid_rate": sum(row["valid"] for row in values) / candidates,
        "valid_pair_rate": sum(row["valid_pair_fraction"] for row in values) / candidates,
        "diversity": len({row["sequence"] for row in values}) / candidates,
    }


def new_aggregate_state(evaluation_contract_digest: str, keys: tuple[str, ...]) -> dict:
    return {
        "schema_version": AGGREGATE_STATE_SCHEMA_VERSION,
        "status": "running",
        "evaluation_contract_sha256": evaluation_contract_digest,
        "next_task_index": 0,
        "metric_sums": {key: 0.0 for key in keys},
        "task_seed_groups": 0,
        "candidate_rows": 0,
        "target_probability_clamp_count": 0,
        "bad_count": 0,
        "error_code_histogram": {},
        "invalid_candidate_rate": 0.0,
        "runtime_seconds": 0.0,
        "max_cuda_memory_bytes": 0,
    }


def validate_aggregate_state(
    state: dict,
    *,
    evaluation_contract_digest: str,
    keys: tuple[str, ...],
    task_count: int,
    seeds_per_task: int,
    candidates: int,
) -> None:
    expected_fields = {
        "schema_version", "status", "evaluation_contract_sha256", "next_task_index",
        "metric_sums", "task_seed_groups", "candidate_rows",
        "target_probability_clamp_count", "bad_count", "error_code_histogram",
        "invalid_candidate_rate", "runtime_seconds", "max_cuda_memory_bytes",
    }
    if (
        type(state.get("schema_version")) is not int
        or state["schema_version"] != AGGREGATE_STATE_SCHEMA_VERSION
    ):
        raise RuntimeError("aggregate-only resume state schema mismatch")
    if set(state) != expected_fields:
        raise RuntimeError("aggregate-only resume state fields mismatch")
    cursor = state.get("next_task_index")
    if type(cursor) is not int or not 0 <= cursor <= task_count:
        raise RuntimeError("aggregate-only task cursor is outside benchmark")
    expected_status = "complete" if cursor == task_count else "running"
    expected_groups = cursor * seeds_per_task
    expected_rows = expected_groups * candidates
    histogram = state.get("error_code_histogram")
    histogram_valid = (
        isinstance(histogram, dict)
        and all(
            code in RECOVERABLE_EVALUATION_ERROR_CODES
            and type(count) is int
            and count >= 0
            for code, count in histogram.items()
        )
    )
    metric_sums = state.get("metric_sums")
    metric_sums_valid = (
        isinstance(metric_sums, dict)
        and set(metric_sums) == set(keys)
        and all(
            type(value) in (int, float) and math.isfinite(value)
            for value in metric_sums.values()
        )
    )
    bad_count = state.get("bad_count")
    counter_fields = (
        "next_task_index", "task_seed_groups", "candidate_rows",
        "target_probability_clamp_count", "bad_count", "max_cuda_memory_bytes",
    )
    counters_valid = all(type(state.get(field)) is int for field in counter_fields)
    invalid_candidate_rate = state.get("invalid_candidate_rate")
    runtime_seconds = state.get("runtime_seconds")
    expected_invalid_rate = (
        bad_count / expected_rows
        if type(bad_count) is int and expected_rows else 0.0
    )
    if (
        state.get("status") != expected_status
        or state.get("evaluation_contract_sha256") != evaluation_contract_digest
        or not counters_valid
        or state.get("task_seed_groups") != expected_groups
        or state.get("candidate_rows") != expected_rows
        or not metric_sums_valid
        or type(bad_count) is not int
        or not 0 <= bad_count <= expected_rows
        or not histogram_valid
        or sum(histogram.values()) != bad_count
        or type(invalid_candidate_rate) not in (int, float)
        or not math.isfinite(invalid_candidate_rate)
        or invalid_candidate_rate != expected_invalid_rate
        or not 0 <= state["target_probability_clamp_count"] <= expected_rows
        or type(runtime_seconds) not in (int, float)
        or not math.isfinite(runtime_seconds)
        or runtime_seconds < 0
        or state["max_cuda_memory_bytes"] < 0
    ):
        raise RuntimeError("aggregate-only resume state mismatch")


def load_completed_aggregate(
    *, state: dict, summary_path: Path, receipt_path: Path,
    evaluation_contract_digest: str, coverage: dict,
) -> dict:
    if state.get("status") != "complete" or not summary_path.is_file():
        raise RuntimeError("aggregate-only terminal receipt/state mismatch")
    summary = json.loads(summary_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(summary, dict) or not isinstance(receipt, dict):
        raise RuntimeError("aggregate-only terminal receipt failed validation")
    groups = state["task_seed_groups"]
    summary_metrics_valid = (
        groups > 0
        and all(
            summary.get(key) == total / groups
            for key, total in state["metric_sums"].items()
        )
    )
    expected_receipt_fields = {
        "schema_version", "status", "summary_sha256", "evaluation_contract_sha256",
        "coverage", "bad_count", "error_code_histogram", "invalid_candidate_rate",
        "runtime_seconds", "max_cuda_memory_bytes",
    }
    receipt_coverage = receipt.get("coverage")
    coverage_valid = (
        isinstance(receipt_coverage, dict)
        and set(receipt_coverage) == {"tasks", "seeds", "candidates"}
        and all(
            type(receipt_coverage.get(key)) is int and receipt_coverage[key] >= 0
            for key in ("tasks", "seeds", "candidates")
        )
        and receipt_coverage == coverage
    )
    receipt_bad_count = receipt.get("bad_count")
    expected_candidate_rows = (
        receipt_coverage["tasks"]
        * receipt_coverage["seeds"]
        * receipt_coverage["candidates"]
        if coverage_valid else -1
    )
    receipt_histogram = receipt.get("error_code_histogram")
    histogram_valid = (
        isinstance(receipt_histogram, dict)
        and all(
            type(code) is str
            and code in RECOVERABLE_EVALUATION_ERROR_CODES
            and type(count) is int
            and count >= 0
            for code, count in receipt_histogram.items()
        )
    )
    bad_count_valid = (
        type(receipt_bad_count) is int
        and 0 <= receipt_bad_count <= expected_candidate_rows
        and histogram_valid
        and sum(receipt_histogram.values()) == receipt_bad_count
    )
    receipt_invalid_rate = receipt.get("invalid_candidate_rate")
    expected_invalid_rate = (
        receipt_bad_count / expected_candidate_rows
        if bad_count_valid and expected_candidate_rows else 0.0
    )
    invalid_rate_valid = (
        type(receipt_invalid_rate) in (int, float)
        and math.isfinite(receipt_invalid_rate)
        and 0 <= receipt_invalid_rate <= 1
        and receipt_invalid_rate == expected_invalid_rate
        and type(receipt_invalid_rate) is type(state["invalid_candidate_rate"])
    )
    receipt_runtime = receipt.get("runtime_seconds")
    runtime_valid = (
        type(receipt_runtime) in (int, float)
        and math.isfinite(receipt_runtime)
        and receipt_runtime >= 0
        and type(receipt_runtime) is type(state["runtime_seconds"])
    )
    receipt_memory = receipt.get("max_cuda_memory_bytes")
    memory_valid = type(receipt_memory) is int and receipt_memory >= 0
    sha_valid = (
        is_sha256(receipt.get("summary_sha256"))
        and is_sha256(receipt.get("evaluation_contract_sha256"))
    )
    if (
        set(receipt) != expected_receipt_fields
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != AGGREGATE_RECEIPT_SCHEMA_VERSION
        or type(receipt.get("status")) is not str
        or receipt["status"] != "complete"
        or not sha_valid
        or receipt.get("summary_sha256") != sha256(summary_path)
        or receipt.get("evaluation_contract_sha256") != evaluation_contract_digest
        or not coverage_valid
        or not bad_count_valid
        or not invalid_rate_valid
        or not runtime_valid
        or not memory_valid
        or receipt_bad_count != state["bad_count"]
        or receipt_histogram != state["error_code_histogram"]
        or receipt_invalid_rate != state["invalid_candidate_rate"]
        or receipt_runtime != state["runtime_seconds"]
        or receipt_memory != state["max_cuda_memory_bytes"]
        or summary.get("evaluation_contract_sha256") != evaluation_contract_digest
        or summary.get("coverage") != coverage
        or summary.get("task_seed_groups") != state["task_seed_groups"]
        or summary.get("candidate_rows") != state["candidate_rows"]
        or summary.get("target_probability_clamp_count")
        != state["target_probability_clamp_count"]
        or summary.get("bad_count") != state["bad_count"]
        or summary.get("error_code_histogram") != state["error_code_histogram"]
        or summary.get("invalid_candidate_rate") != state["invalid_candidate_rate"]
        or summary.get("runtime_seconds") != state["runtime_seconds"]
        or summary.get("max_cuda_memory_bytes") != state["max_cuda_memory_bytes"]
        or summary.get("provisional_monitor_only") is not True
        or summary.get("target_level_artifacts_written") is not False
        or not summary_metrics_valid
    ):
        raise RuntimeError("aggregate-only terminal receipt failed validation")
    return summary


def provisional_aggregate_evaluation(
    *, args, model, tasks: list[dict], device: torch.device,
    cached_source: bool, evaluation_contract_digest: str,
    source: str, checkpoint: dict, candidate: dict | None,
    candidate_method: str | None, evaluation_method: str,
) -> dict:
    keys = (
        "pass_at_1", f"pass_at_{args.candidates}", "mfe_at_1",
        f"mfe_at_{args.candidates}", "best_pair_f1",
        "best_target_probability", "best_NED", "valid_rate",
        "valid_pair_rate", "diversity",
    )
    state_path = args.output / "aggregate_state.json"
    summary_path = args.output / "summary.json"
    receipt_path = args.output / "receipt.json"
    coverage = {
        "tasks": len(tasks), "seeds": len(args.seeds),
        "candidates": args.candidates,
    }
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError("aggregate-only resume state is absent")
        state = json.loads(state_path.read_text())
    else:
        state = new_aggregate_state(evaluation_contract_digest, keys)
        atomic_json(state_path, state)
    validate_aggregate_state(
        state,
        evaluation_contract_digest=evaluation_contract_digest,
        keys=keys,
        task_count=len(tasks),
        seeds_per_task=len(args.seeds),
        candidates=args.candidates,
    )
    if receipt_path.is_file():
        return load_completed_aggregate(
            state=state, summary_path=summary_path, receipt_path=receipt_path,
            evaluation_contract_digest=evaluation_contract_digest,
            coverage=coverage,
        )
    for task_index in range(state["next_task_index"], len(tasks)):
        task_started = time.time()
        structure = tasks[task_index]["target_structure"]
        cached_hidden = None
        if cached_source:
            payload = torch.load(
                args.structure_cache / f"{structure_key(structure)}.pt",
                map_location="cpu", weights_only=True,
            )
            if payload["target_structure"] != structure:
                raise RuntimeError("evaluation structure cache key/content mismatch")
            cached_hidden = payload["hidden"]
        task_sums = {key: 0.0 for key in keys}
        task_rows = 0
        clamp_count = 0
        task_bad_count = 0
        task_error_code_histogram: dict[str, int] = {}
        for seed in args.seeds:
            sequences = sample(
                model, structure, cached_hidden, args.candidates, args.steps,
                global_task_seed(seed, task_index, getattr(args, "task_index_offset", 0)),
                args.condition, device,
            )
            metrics = evaluate_many(
                sequences, structure, args.eval_workers, safe_recoverable=True
            )
            values = []
            for candidate_index, (sequence, result) in enumerate(zip(sequences, metrics)):
                if result.get("evaluation_valid") not in (True, False):
                    raise RuntimeError("aggregate-only candidate validity evidence is absent")
                if result["evaluation_valid"] is False:
                    error_code = result.get("evaluation_error_code")
                    if error_code not in RECOVERABLE_EVALUATION_ERROR_CODES:
                        raise RuntimeError("aggregate-only recoverable error evidence is invalid")
                    task_bad_count += 1
                    task_error_code_histogram[error_code] = (
                        task_error_code_histogram.get(error_code, 0) + 1
                    )
                pair_validity = valid_pair_fraction(sequence, structure)
                valid = (
                    len(sequence) == len(structure)
                    and not (set(sequence) - set("ACGU"))
                    and pair_validity == 1
                )
                values.append({
                    "candidate_index": candidate_index,
                    "sequence": sequence,
                    "valid": float(valid),
                    "valid_pair_fraction": pair_validity,
                    **result,
                })
                clamp_count += int(bool(result.get("target_probability_clamped", False)))
            group = aggregate_group(values, args.candidates)
            for key in keys:
                task_sums[key] += group[key]
            task_rows += len(values)
        for key in keys:
            state["metric_sums"][key] += task_sums[key]
        state["task_seed_groups"] += len(args.seeds)
        state["candidate_rows"] += task_rows
        state["target_probability_clamp_count"] += clamp_count
        state["bad_count"] += task_bad_count
        for error_code, count in task_error_code_histogram.items():
            state["error_code_histogram"][error_code] = (
                state["error_code_histogram"].get(error_code, 0) + count
            )
        state["invalid_candidate_rate"] = state["bad_count"] / state["candidate_rows"]
        state["runtime_seconds"] += time.time() - task_started
        state["next_task_index"] = task_index + 1
        state["status"] = (
            "complete" if state["next_task_index"] == len(tasks) else "running"
        )
        if device.type == "cuda":
            state["max_cuda_memory_bytes"] = max(
                state["max_cuda_memory_bytes"], torch.cuda.max_memory_allocated(device)
            )
        validate_aggregate_state(
            state,
            evaluation_contract_digest=evaluation_contract_digest,
            keys=keys,
            task_count=len(tasks),
            seeds_per_task=len(args.seeds),
            candidates=args.candidates,
        )
        atomic_json(state_path, state)
        print(json.dumps({
            "tasks_complete": state["next_task_index"],
            "tasks_total": len(tasks),
        }), flush=True)
    expected_groups = len(tasks) * len(args.seeds)
    expected_rows = expected_groups * args.candidates
    validate_aggregate_state(
        state,
        evaluation_contract_digest=evaluation_contract_digest,
        keys=keys,
        task_count=len(tasks),
        seeds_per_task=len(args.seeds),
        candidates=args.candidates,
    )
    if state["task_seed_groups"] != expected_groups or state["candidate_rows"] != expected_rows:
        raise RuntimeError("aggregate-only final coverage mismatch")
    summary = {
        key: state["metric_sums"][key] / expected_groups for key in keys
    }
    summary.update({
        "method": evaluation_method,
        "evaluation_method": evaluation_method,
        "candidate_method": candidate_method,
        "structure_source": source,
        "injection": checkpoint["contract"]["injection"],
        "condition": args.condition,
        "flow_steps": args.steps,
        "seeds": args.seeds,
        "runtime_seconds": state["runtime_seconds"],
        "max_cuda_memory_bytes": state["max_cuda_memory_bytes"],
        "viennarna_version": RNA.__version__,
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_contract_sha256": checkpoint["contract_sha256"],
        "candidate_checkpoint_sha256": (
            sha256(args.candidate_checkpoint) if candidate is not None else None
        ),
        "candidate_contract_sha256": candidate.get("contract_sha256") if candidate else None,
        "benchmark_sha256": sha256(args.benchmark),
        "evaluation_contract_sha256": evaluation_contract_digest,
        "evaluator_sha256": sha256(Path(__file__)),
        "coverage": coverage,
        "task_seed_groups": state["task_seed_groups"],
        "candidate_rows": state["candidate_rows"],
        "target_probability_clamp_count": state["target_probability_clamp_count"],
        "bad_count": state["bad_count"],
        "error_code_histogram": state["error_code_histogram"],
        "invalid_candidate_rate": state["invalid_candidate_rate"],
        "provisional_monitor_only": True,
        "target_level_artifacts_written": False,
    })
    atomic_json(summary_path, summary)
    receipt = {
        "schema_version": AGGREGATE_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "summary_sha256": sha256(summary_path),
        "evaluation_contract_sha256": evaluation_contract_digest,
        "coverage": coverage,
        "bad_count": state["bad_count"],
        "error_code_histogram": state["error_code_histogram"],
        "invalid_candidate_rate": state["invalid_candidate_rate"],
        "runtime_seconds": state["runtime_seconds"],
        "max_cuda_memory_bytes": state["max_cuda_memory_bytes"],
    }
    atomic_json(receipt_path, receipt)
    return summary


def validate_benchmark_route_authorization(
    authorization_path: Path,
    expected_authorization_sha256: str,
    benchmark_path: Path,
    supervised_checkpoint: Path,
    selection_receipt_sha256: str,
    candidate_checkpoint: Path,
    candidate: dict,
    candidate_receipt_sha256: str,
) -> dict:
    """Validate one immutable candidate-specific route-selection authorization."""
    authorization = json.loads(authorization_path.read_text())
    actual_authorization_sha256 = sha256(authorization_path)
    if actual_authorization_sha256 != expected_authorization_sha256:
        raise RuntimeError("Benchmark route authorization hash changed")
    benchmarks = authorization.get("route_selection_benchmarks", {})
    if set(benchmarks) != {"eterna100v2", "rnasolo_clean"}:
        raise RuntimeError("Benchmark route authorization set is not the frozen clean pair")
    matches = [
        name
        for name, row in benchmarks.items()
        if isinstance(row, dict)
        and row.get("sha256") == sha256(benchmark_path)
        and row.get("route_selection_role") == "eligible"
    ]
    candidate_contract = candidate.get("contract", {})
    fixed_hashes = (
        authorization.get("benchmark_pilot_manifest_sha256"),
        authorization.get("benchmark_pilot_state_sha256"),
        authorization.get("formal_plan_sha256"),
    )
    evaluation_protocol = authorization.get("evaluation_protocol")
    if (
        authorization.get("status") != "complete"
        or authorization.get("role")
        != "candidate-specific Benchmark Pilot route-selection authorization"
        or len(matches) != 1
        or any(
            not isinstance(value, str) or len(value) != 64 for value in fixed_hashes
        )
        or authorization.get("supervised_checkpoint_sha256")
        != sha256(supervised_checkpoint)
        or authorization.get("supervised_selection_receipt_sha256")
        != selection_receipt_sha256
        or authorization.get("candidate_method") != candidate_contract.get("method")
        or authorization.get("candidate_checkpoint_sha256")
        != sha256(candidate_checkpoint)
        or authorization.get("candidate_contract_sha256")
        != candidate.get("contract_sha256")
        or authorization.get("candidate_receipt_sha256")
        != candidate_receipt_sha256
        or candidate_contract.get("formal_plan_sha256")
        != authorization.get("formal_plan_sha256")
        or evaluation_protocol != {
            "candidates": 8,
            "flow_steps": 50,
            "decode_seeds": [1009, 2027, 3037],
            "condition": "native",
            "viennarna_version": "2.7.2",
        }
        or authorization.get("fair_evaluator_sha256") != sha256(Path(__file__))
    ):
        raise RuntimeError("candidate-specific Benchmark route authorization is invalid")
    return {
        "benchmark_name": matches[0],
        "route_selection_role": "eligible",
        "authorization_sha256": actual_authorization_sha256,
        "benchmark_pilot_manifest_sha256": authorization[
            "benchmark_pilot_manifest_sha256"
        ],
        "benchmark_pilot_state_sha256": authorization[
            "benchmark_pilot_state_sha256"
        ],
        "formal_plan_sha256": authorization["formal_plan_sha256"],
        "evaluation_protocol": evaluation_protocol,
        "fair_evaluator_sha256": authorization["fair_evaluator_sha256"],
    }


def validate_formal_evaluation_inputs(
    supervised_checkpoint: Path,
    selection_receipt: dict,
    candidate_checkpoint: Path | None = None,
    candidate: dict | None = None,
    candidate_receipt: dict | None = None,
    selection_receipt_sha256: str | None = None,
    expected_reward: str = "terminal",
    route_freeze_receipt: dict | None = None,
    candidate_receipt_sha256: str | None = None,
    route_freeze_receipt_sha256: str | None = None,
    expected_route_freeze_receipt_sha256: str | None = None,
    benchmark_route_authorization: dict | None = None,
) -> None:
    if (
        selection_receipt.get("status") != "complete"
        or selection_receipt.get("checkpoint_sha256") != sha256(supervised_checkpoint)
    ):
        raise RuntimeError("supervised selection receipt does not authorize this checkpoint")
    if candidate_checkpoint is None:
        if candidate is not None or candidate_receipt is not None:
            raise RuntimeError("candidate evidence was supplied without a candidate checkpoint")
        return
    contract = candidate.get("contract") if candidate else None
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() if contract else None
    method = contract.get("method") if contract else None
    common_contract_valid = (
        bool(contract)
        and candidate.get("contract_sha256") == digest
        and method in ALLOWED_FORMAL_CANDIDATE_METHODS
        and contract.get("supervised_checkpoint_sha256")
        == sha256(supervised_checkpoint)
        and (
            selection_receipt_sha256 is None
            or contract.get("supervised_selection_receipt_sha256")
            == selection_receipt_sha256
        )
    )
    if not common_contract_valid:
        raise RuntimeError("formal post-training checkpoint is incompatible with frozen supervision")

    legacy_endpoint_contract = (
        method == "simplex-endpoint-policy-trajectory-grpo"
        and contract.get("branch_update_budget_mode") is None
    )
    if legacy_endpoint_contract:
        expected_contract = FORMAL_ENDPOINT_CONTRACT | {"reward": expected_reward}
        if (
            not contract.get("formal_rl")
            or any(contract.get(key) != value for key, value in expected_contract.items())
        ):
            raise RuntimeError("formal RL checkpoint is incompatible with frozen supervision")
    elif method == "ce_anchor_only_continuation":
        if contract.get("formal_control") is not True:
            raise RuntimeError("formal control checkpoint is incompatible with frozen supervision")
    elif contract.get("formal_rl") is not True:
        raise RuntimeError("formal RL checkpoint is incompatible with frozen supervision")

    route_freeze_authorized = not (
        not route_freeze_receipt
        or not expected_route_freeze_receipt_sha256
        or route_freeze_receipt_sha256 != expected_route_freeze_receipt_sha256
        or route_freeze_receipt.get("status") != "complete"
        or route_freeze_receipt.get("eterna_accessed") is not False
        or route_freeze_receipt.get("selected_method") != method
        or route_freeze_receipt.get("candidate_checkpoint_sha256")
        != sha256(candidate_checkpoint)
        or route_freeze_receipt.get("candidate_contract_sha256") != digest
        or route_freeze_receipt.get("candidate_receipt_sha256")
        != candidate_receipt_sha256
        or route_freeze_receipt.get("supervised_selection_receipt_sha256")
        != selection_receipt_sha256
    )
    if not route_freeze_authorized and benchmark_route_authorization is None:
        raise RuntimeError(
            "formal route freeze or Benchmark Pilot authorization does not authorize this candidate"
        )

    if method == "ce_anchor_only_continuation":
        formally_started = (
            candidate_receipt.get("formal_control_started") is True
            if candidate_receipt else False
        )
        outer_updates = contract.get("outer_updates")
        target_steps = contract.get("target_optimizer_steps")
        complete = (
            type(outer_updates) is int
            and outer_updates > 0
            and type(target_steps) is int
            and target_steps > 0
            and bool(candidate_receipt)
            and type(candidate.get("state", {}).get("next_outer_update")) is int
            and type(candidate.get("state", {}).get("optimizer_steps")) is int
            and type(candidate_receipt.get("outer_updates_complete")) is int
            and type(candidate_receipt.get("optimizer_steps_complete")) is int
            and type(candidate_receipt.get("target_optimizer_steps")) is int
            and candidate.get("state", {}).get("next_outer_update")
            == outer_updates
            and candidate.get("state", {}).get("optimizer_steps")
            == target_steps
            and candidate_receipt.get("outer_updates_complete")
            == outer_updates
            and candidate_receipt.get("optimizer_steps_complete")
            == target_steps
            and candidate_receipt.get("target_optimizer_steps")
            == target_steps
        )
    else:
        formally_started = (
            candidate_receipt.get("formal_rl_started") is True
            if candidate_receipt else False
        )
        updates = contract.get("updates")
        complete = (
            type(updates) is int
            and updates > 0
            and bool(candidate_receipt)
            and type(candidate.get("state", {}).get("next_update")) is int
            and type(candidate_receipt.get("updates_complete")) is int
            and type(candidate_receipt.get("target_updates")) is int
            and candidate.get("state", {}).get("next_update") == updates
            and candidate_receipt.get("updates_complete") == updates
            and candidate_receipt.get("target_updates") == updates
        )
    if (
        not candidate_receipt
        or candidate_receipt.get("status") != "complete"
        or not formally_started
        or not complete
        or candidate_receipt.get("contract_sha256") != digest
        or Path(candidate_receipt.get("last_checkpoint", {}).get("path", "")).resolve()
        != candidate_checkpoint.resolve()
        or candidate_receipt.get("last_checkpoint", {}).get("sha256")
        != sha256(candidate_checkpoint)
    ):
        raise RuntimeError(
            "formal post-training receipt does not authorize this candidate checkpoint"
        )

    if benchmark_route_authorization is not None:
        if candidate_receipt.get("nan_inf_count") != 0:
            raise RuntimeError("Benchmark route candidate did not complete cleanly")
        if method in {"terminal_grpo", "reward_weighted_fm"}:
            try:
                validate_reward_evaluation_binding(
                    candidate_receipt.get("reward_evaluation_coverage", {}),
                    candidate.get("state", {}),
                    expected_candidate_rows=(
                        contract["updates"]
                        * contract["tasks_per_update"]
                        * contract["candidates"]
                    ),
                    expected_groups=contract["updates"] * contract["tasks_per_update"],
                )
            except (KeyError, RuntimeError) as error:
                raise RuntimeError(
                    "Benchmark route candidate reward coverage is invalid"
                ) from error

    if not legacy_endpoint_contract:
        return

    expected_contract = FORMAL_ENDPOINT_CONTRACT | {"reward": expected_reward}
    if (
        candidate_receipt.get("last_checkpoint", {}).get("next_update")
        != expected_contract["updates"]
    ):
        raise RuntimeError("formal RL receipt does not authorize this candidate checkpoint")
    try:
        validate_reward_evaluation_binding(
            candidate_receipt.get("reward_evaluation_coverage", {}),
            candidate.get("state", {}),
            expected_candidate_rows=(
                expected_contract["updates"]
                * expected_contract["tasks_per_update"]
                * expected_contract["candidates"]
            ),
            expected_groups=(
                expected_contract["updates"] * expected_contract["tasks_per_update"]
            ),
        )
    except RuntimeError as error:
        raise RuntimeError(
            "formal RL receipt does not authorize this candidate checkpoint"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--supervised-selection-receipt", type=Path)
    parser.add_argument("--candidate-receipt", type=Path)
    parser.add_argument("--route-freeze-receipt", type=Path)
    parser.add_argument("--expected-route-freeze-receipt-sha256")
    parser.add_argument("--benchmark-route-authorization", type=Path)
    parser.add_argument("--expected-benchmark-route-authorization-sha256")
    parser.add_argument(
        "--expected-rl-reward", choices=["terminal", "pair_credit", "pair_rival"],
        default="terminal",
    )
    parser.add_argument("--formal-benchmark", action="store_true")
    parser.add_argument("--provisional-monitor-only", action="store_true")
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--rnaernie-revision", required=True)
    parser.add_argument("--structure-cache", type=Path)
    parser.add_argument("--structure-cache-manifest", type=Path)
    parser.add_argument("--training-structure-cache-manifest", type=Path)
    parser.add_argument("--condition", choices=["native", "zero", "shuffle"], default="native")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1009, 2027, 3037])
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--task-index-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.formal_benchmark and args.provisional_monitor_only:
        raise ValueError("formal benchmark cannot be marked provisional monitor-only")
    if args.eval_workers <= 0 or args.candidates <= 0 or args.steps <= 0:
        raise ValueError("workers, candidates, and flow steps must be positive")
    if args.task_index_offset < 0:
        raise ValueError("task index offset must be non-negative")
    if args.task_index_offset and not args.provisional_monitor_only:
        raise ValueError("task index offset is restricted to provisional monitor-only evaluation")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    contract = checkpoint.get("contract")
    if not contract or checkpoint.get("contract_sha256") != hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise RuntimeError("checkpoint training contract is absent or corrupt")
    candidate = (
        torch.load(args.candidate_checkpoint, map_location="cpu", weights_only=False)
        if args.candidate_checkpoint else None
    )
    if args.formal_benchmark:
        if args.supervised_selection_receipt is None:
            raise ValueError("formal benchmark requires a supervised selection receipt")
        selection_receipt = validate_selection(args.supervised_selection_receipt)
        candidate_receipt = (
            json.loads(args.candidate_receipt.read_text()) if args.candidate_receipt else None
        )
        route_freeze_receipt = (
            json.loads(args.route_freeze_receipt.read_text())
            if args.route_freeze_receipt else None
        )
        route_authorization_inputs = (
            args.benchmark_route_authorization,
            args.expected_benchmark_route_authorization_sha256,
        )
        if any(route_authorization_inputs) and not all(route_authorization_inputs):
            raise ValueError("Benchmark route authorization requires file and expected hash")
        if args.route_freeze_receipt and args.benchmark_route_authorization:
            raise ValueError("choose route freeze or Benchmark route authorization")
        if args.benchmark_route_authorization and not args.candidate_checkpoint:
            raise ValueError("Benchmark route authorization requires a candidate checkpoint")
        benchmark_route = None
        if args.candidate_checkpoint and args.benchmark_route_authorization:
            benchmark_route = validate_benchmark_route_authorization(
                args.benchmark_route_authorization,
                args.expected_benchmark_route_authorization_sha256,
                args.benchmark,
                args.checkpoint,
                sha256(args.supervised_selection_receipt),
                args.candidate_checkpoint,
                candidate,
                sha256(args.candidate_receipt),
            )
            protocol = benchmark_route["evaluation_protocol"]
            if (
                args.candidates != protocol["candidates"]
                or args.steps != protocol["flow_steps"]
                or args.seeds != protocol["decode_seeds"]
                or args.condition != protocol["condition"]
                or args.limit_tasks is not None
                or RNA.__version__ != protocol["viennarna_version"]
            ):
                raise RuntimeError("Benchmark route evaluation protocol changed")
        validate_formal_evaluation_inputs(
            args.checkpoint,
            selection_receipt,
            args.candidate_checkpoint,
            candidate,
            candidate_receipt,
            sha256(args.supervised_selection_receipt),
            args.expected_rl_reward,
            route_freeze_receipt,
            sha256(args.candidate_receipt) if args.candidate_receipt else None,
            sha256(args.route_freeze_receipt) if args.route_freeze_receipt else None,
            args.expected_route_freeze_receipt_sha256,
            benchmark_route,
        )
    elif (
        args.supervised_selection_receipt
        or args.candidate_receipt
        or args.route_freeze_receipt
        or args.expected_route_freeze_receipt_sha256
        or args.benchmark_route_authorization
        or args.expected_benchmark_route_authorization_sha256
    ):
        raise ValueError("selection and candidate receipts are only valid in formal benchmark mode")
    evaluation_contract = {
        "schema_version": 1,
        "evaluator_sha256": sha256(Path(__file__)),
        "provisional_monitor_only": args.provisional_monitor_only,
        "analysis_boundary": (
            "aggregate-only route monitoring; no per-target inspection or adaptation"
            if args.provisional_monitor_only else None
        ),
        "benchmark_sha256": sha256(args.benchmark),
        "supervised_checkpoint_sha256": sha256(args.checkpoint),
        "supervised_checkpoint_contract_sha256": checkpoint["contract_sha256"],
        "candidate_checkpoint_sha256": (
            sha256(args.candidate_checkpoint) if args.candidate_checkpoint else None
        ),
        "candidate_checkpoint_contract_sha256": (
            candidate.get("contract_sha256") if candidate else None
        ),
        "supervised_selection_receipt_sha256": (
            sha256(args.supervised_selection_receipt)
            if args.supervised_selection_receipt else None
        ),
        "candidate_receipt_sha256": (
            sha256(args.candidate_receipt) if args.candidate_receipt else None
        ),
        "route_freeze_receipt_sha256": (
            sha256(args.route_freeze_receipt) if args.route_freeze_receipt else None
        ),
        "expected_route_freeze_receipt_sha256": (
            args.expected_route_freeze_receipt_sha256
        ),
        "benchmark_route_authorization_sha256": (
            benchmark_route["authorization_sha256"]
            if args.formal_benchmark and benchmark_route else None
        ),
        "benchmark_pilot_manifest_sha256": (
            benchmark_route["benchmark_pilot_manifest_sha256"]
            if args.formal_benchmark and benchmark_route else None
        ),
        "benchmark_pilot_state_sha256": (
            benchmark_route["benchmark_pilot_state_sha256"]
            if args.formal_benchmark and benchmark_route else None
        ),
        "benchmark_pilot_name": (
            benchmark_route["benchmark_name"]
            if args.formal_benchmark and benchmark_route else None
        ),
        "benchmark_route_selection_role": (
            benchmark_route["route_selection_role"]
            if args.formal_benchmark and benchmark_route else None
        ),
        "formal_plan_sha256": (
            benchmark_route["formal_plan_sha256"]
            if args.formal_benchmark and benchmark_route else None
        ),
        "authorized_fair_evaluator_sha256": (
            benchmark_route["fair_evaluator_sha256"]
            if args.formal_benchmark and benchmark_route else None
        ),
        "expected_rl_reward": args.expected_rl_reward,
        "rnaernie_revision": args.rnaernie_revision,
        "rnaernie_weight_sha256": sha256(args.rnaernie / "model.safetensors"),
        "condition": args.condition,
        "candidates": args.candidates,
        "steps": args.steps,
        "seeds": args.seeds,
        "eval_workers": args.eval_workers,
        "limit_tasks": args.limit_tasks,
        "task_index_offset": args.task_index_offset,
        "formal_benchmark": args.formal_benchmark,
        "structure_cache_manifest_sha256": (
            sha256(args.structure_cache_manifest) if args.structure_cache_manifest else None
        ),
        "training_structure_cache_manifest_sha256": (
            sha256(args.training_structure_cache_manifest)
            if args.training_structure_cache_manifest else None
        ),
        "viennarna": {
            "version": RNA.__version__,
            "temperature_c": 37.0,
            "dangles": 2,
            "uniq_ML": 1,
        },
    }
    source = contract["structure_source"]
    cached_source = source.startswith("omnigenome")
    cache_inputs = (
        args.structure_cache,
        args.structure_cache_manifest,
        args.training_structure_cache_manifest,
    )
    if cached_source != all(cache_inputs) or (not cached_source and any(cache_inputs)):
        raise ValueError("cached evaluation requires cache plus training/evaluation manifests")
    if args.rnaernie_revision != contract["rnaernie_revision"]:
        raise RuntimeError("RNAErnie revision does not match training contract")
    if sha256(args.rnaernie / "model.safetensors") != contract["rnaernie_weight_sha256"]:
        raise RuntimeError("RNAErnie weight hash does not match training contract")
    if cached_source:
        training_manifest = json.loads(args.training_structure_cache_manifest.read_text())
        evaluation_manifest = json.loads(args.structure_cache_manifest.read_text())
        if sha256(args.training_structure_cache_manifest) != contract[
            "structure_cache_manifest_sha256"
        ]:
            raise RuntimeError("training cache manifest does not match checkpoint contract")
        identity_keys = ("model_revision", "model_weight_sha256", "hidden_size")
        if any(training_manifest[key] != evaluation_manifest[key] for key in identity_keys):
            raise RuntimeError("training/evaluation structure cache model identity mismatch")
        if evaluation_manifest.get("status") != "complete" or evaluation_manifest.get("bad_count") != 0:
            raise RuntimeError("evaluation structure cache is incomplete")
    if source == "none" and args.condition != "native":
        raise ValueError("unconditioned model has no zero/shuffle control")
    evaluation_contract_digest = prepare_evaluation_output(
        args.output, args.resume, evaluation_contract
    )
    progress_dir = args.output / "task_progress"
    if not args.provisional_monitor_only:
        progress_dir.mkdir(exist_ok=args.resume)
    model = FairRNAFlow(
        args.rnaernie,
        structure_source=source,
        injection=contract["injection"],
        structure_dim=contract.get("structure_dim"),
        lora_rank=contract["lora_rank"],
        lora_alpha=contract["lora_alpha"],
        lora_dropout=contract["lora_dropout"],
        backbone_mode=contract.get("backbone_mode"),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    candidate_method, evaluation_method = candidate_method_identity(candidate)
    if candidate is not None:
        candidate_contract = candidate.get("contract", {})
        candidate_digest = hashlib.sha256(
            json.dumps(candidate_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            candidate.get("contract_sha256") != candidate_digest
            or candidate_contract.get("method") != candidate_method
            or candidate_contract.get("supervised_checkpoint_sha256") != sha256(args.checkpoint)
        ):
            raise RuntimeError("RL candidate checkpoint is incompatible with supervision")
        configure_rl_trainable_scope(
            model, candidate_contract.get("rl_trainable_scope", "inherited")
        )
        model.load_trainable_state_dict(candidate["trainable_model"])
    model.eval()
    tasks = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    if args.limit_tasks is not None:
        tasks = tasks[:args.limit_tasks]
    if args.provisional_monitor_only:
        summary = provisional_aggregate_evaluation(
            args=args, model=model, tasks=tasks, device=device,
            cached_source=cached_source,
            evaluation_contract_digest=evaluation_contract_digest,
            source=source, checkpoint=checkpoint, candidate=candidate,
            candidate_method=candidate_method, evaluation_method=evaluation_method,
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return
    started = time.time()
    all_rows = []
    for task_index, task in enumerate(tasks):
        progress_path = progress_dir / f"task-{task_index:03d}.jsonl"
        if progress_path.is_file():
            task_rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
            validate_coverage(task_rows, [task["id"]], args.seeds, args.candidates)
            all_rows.extend(task_rows)
            continue
        structure = task["target_structure"]
        cached_hidden = None
        if cached_source:
            payload = torch.load(
                args.structure_cache / f"{structure_key(structure)}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if payload["target_structure"] != structure:
                raise RuntimeError("evaluation structure cache key/content mismatch")
            cached_hidden = payload["hidden"]
        task_rows = []
        task_started = time.time()
        for seed in args.seeds:
            sequences = sample(
                model,
                structure,
                cached_hidden,
                args.candidates,
                args.steps,
                global_task_seed(seed, task_index, args.task_index_offset),
                args.condition,
                device,
            )
            metrics = evaluate_many(sequences, structure, args.eval_workers)
            for candidate_index, (sequence, values) in enumerate(zip(sequences, metrics)):
                pair_validity = valid_pair_fraction(sequence, structure)
                valid = len(sequence) == len(structure) and not (set(sequence) - set("ACGU")) and pair_validity == 1
                task_rows.append({
                    "task_id": task["id"],
                    "seed": seed,
                    "candidate_index": candidate_index,
                    "target_structure": structure,
                    "sequence": sequence,
                    "valid": float(valid),
                    "valid_pair_fraction": pair_validity,
                    **values,
                })
        validate_coverage(task_rows, [task["id"]], args.seeds, args.candidates)
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in task_rows))
        os.replace(temporary, progress_path)
        all_rows.extend(task_rows)
        print(json.dumps({
            "tasks_complete": task_index + 1,
            "tasks_total": len(tasks),
            "task_seconds": time.time() - task_started,
        }), flush=True)
    validate_coverage(all_rows, [task["id"] for task in tasks], args.seeds, args.candidates)
    task_seed, summary = aggregate(all_rows, args.candidates)
    summary.update({
        "method": evaluation_method,
        "evaluation_method": evaluation_method,
        "candidate_method": candidate_method,
        "structure_source": source,
        "injection": contract["injection"],
        "condition": args.condition,
        "flow_steps": args.steps,
        "seeds": args.seeds,
        "runtime_seconds": time.time() - started,
        "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "viennarna_version": RNA.__version__,
        "temperature_c": 37.0,
        "dangles": 2,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_contract_sha256": checkpoint["contract_sha256"],
        "candidate_checkpoint": str(args.candidate_checkpoint) if candidate is not None else None,
        "candidate_checkpoint_sha256": (
            sha256(args.candidate_checkpoint) if candidate is not None else None
        ),
        "candidate_contract_sha256": candidate.get("contract_sha256") if candidate else None,
        "supervised_selection_receipt_sha256": (
            sha256(args.supervised_selection_receipt)
            if args.supervised_selection_receipt else None
        ),
        "candidate_receipt_sha256": (
            sha256(args.candidate_receipt) if args.candidate_receipt else None
        ),
        "route_freeze_receipt_sha256": (
            sha256(args.route_freeze_receipt) if args.route_freeze_receipt else None
        ),
        "expected_route_freeze_receipt_sha256": (
            args.expected_route_freeze_receipt_sha256
        ),
        "formal_benchmark": args.formal_benchmark,
        "benchmark_sha256": sha256(args.benchmark),
        "evaluation_contract_sha256": evaluation_contract_digest,
        "evaluator_sha256": sha256(Path(__file__)),
        "structure_cache_manifest_sha256": sha256(args.structure_cache_manifest) if cached_source else None,
        "training_structure_cache_manifest_sha256": (
            sha256(args.training_structure_cache_manifest) if cached_source else None
        ),
        "coverage": {"tasks": len(tasks), "seeds": len(args.seeds), "candidates": args.candidates},
    })
    (args.output / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows)
    )
    (args.output / "task_seed.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in task_seed)
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
