"""Frozen Eterna100-v2 D0 evaluator for structured-discrete DoMinO arms.

Only generation is replaced: ViennaRNA scoring, coverage checks and aggregation
come from the frozen fair evaluator.  This entrypoint is deliberately
fail-closed because it compares checkpoints from four different lineage points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import RNA
import torch

FAIR_DIR = Path(__file__).resolve().parents[1] / "rna-flow-fair-components"
PROGRESSIVE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FAIR_DIR))
sys.path.insert(0, str(PROGRESSIVE_DIR))
from evaluate import (  # noqa: E402
    FairRNAFlow,
    aggregate,
    evaluate_many,
    global_task_seed,
    sha256,
    valid_pair_fraction,
    validate_coverage,
)
from endpoint_policy import (  # noqa: E402
    rollout_discrete_domino_trajectory,
    validate_discrete_domino_trajectory,
)
from model import adapt_loaded_supervised_model_with_lora, configure_rl_trainable_scope  # noqa: E402


METHOD = "structured-discrete-mixture-dfm-domino-ppo"
EXPECTED_EVALUATOR_SHA256 = "3e6d6512d1691238b79a69d3066591b614d40756a0ec42238677b3923afa5b64"
EXPECTED_MODEL_SHA256 = "e8bf2ee1e2d782ac453f6b00779e82b2b54d66ef4d901d29fbeacf3186f182c9"
EXPECTED_ETERNA_SHA256 = "7514e053c8044d2dc96909e40383474375add1e8b926aa19417980a1e37c9412"
SEEDS = [1009, 2027, 3037]
SEED_TEMPERATURE_MAP = {1009: 0.8, 2027: 1.0, 3037: 1.2}
CANDIDATES = 8
TRAJECTORY_STEPS = 8
D3_TASKS_SHA256 = "7c6ee06aaaa24b0b67ce177af90c6eabae920eadbb01d036189afb542055ce61"
D3_TASKS_MANIFEST_SHA256 = "9a282be638934e0588862e852344181a41f579627c33bee50e9516f786648fa8"
C3_OFFICIAL_TASKS_SHA256 = "eb7900b95c193f35bfefd295f571363ce935ebe16eed20737cf5d4110704a993"
C3_OFFICIAL_TASKS_MANIFEST_SHA256 = "97b1e024ee5e8e68279cec9d64c64a9a8168cd99a4f8007a5c78fe102b601370"
COVERAGE_SCALE_ARMS = {
    "c3_official2790_scale": "c3_official2790",
    "c3_official2790_d5_scale": "c3_official2790_d5",
    "t0_flow_yrl_scale": "t0_flow_yrl",
    "t0_flow_yrl_d5_scale": "t0_flow_yrl_d5",
}
COVERAGE_SCALE_UPDATES = {
    "c3_official2790": {1395: {698}, 1744: {1395}, 2093: {1395}},
    "c3_official2790_d5": {
        1744: {1395}, 2093: {1744}, 2442: {2093}, 2790: {2442},
        3139: {2790}, 3488: {3139}, 3837: {3488}, 4185: {3837},
    },
    "t0_flow_yrl": {623: {256}, 1036: {256, 623}, 2072: {1036}, 3108: {2072}},
    "t0_flow_yrl_d5": {
        623: {256}, 1036: {623}, 1554: {1036}, 2072: {1036, 1554},
        2590: {2072}, 3108: {2590}, 3626: {3108}, 4144: {3626},
    },
}
D3_UPDATES = 256
D3_SCOPE = "last_2_backbone_and_head"
D3_GROUP_OBJECTIVES = {
    "d3_t0_u256": "thermodynamic_grpo",
    "d3_t1_u256": "thermo_pkpo",
}
D4_HYBRID_PKPO_OBJECTIVE = "thermo_pkpo_residual"
D4_PKPO_RESIDUAL_COEFFICIENT = 1.0601332682185964
D4_COEFFICIENT = D4_PKPO_RESIDUAL_COEFFICIENT
# U128 is a TRAIN-only monitor label; only the U256 arm is accepted by this
# Eterna entrypoint.  Keeping both labels here makes the lineage namespace
# explicit without permitting an intermediate external evaluation.
D4_GROUP_OBJECTIVES = {
    "d4_hybrid_u128": D4_HYBRID_PKPO_OBJECTIVE,
    "d4_hybrid_u256": D4_HYBRID_PKPO_OBJECTIVE,
}
D4_TERMINAL_ARM = "d4_hybrid_u256"
D4_PKPO_RESIDUAL = {
    "coefficient": D4_PKPO_RESIDUAL_COEFFICIENT,
    "formula": (
        "normalized_group_advantages(rewards) + coefficient * "
        "continuous_maxk_advantages(rewards, k)"
    ),
    "raw_pkpo_normalization": "none",
    "thermodynamic_signal": "dense-centered-normalized-group-advantages",
}
D5_ROLLOUT_REFRESH_MODE = "per_policy_epoch"
D5_STRICT_ON_POLICY_OBJECTIVE = "thermodynamic_grpo"
D5_GROUP_OBJECTIVES = {
    "d5_strict_on_policy_u128": D5_STRICT_ON_POLICY_OBJECTIVE,
    "d5_strict_on_policy_u256": D5_STRICT_ON_POLICY_OBJECTIVE,
}
D5_TERMINAL_ARM = "d5_strict_on_policy_u256"
D5_ROLLOUT_REFRESH = {
    "logical_update_task_cursor_advance": 4,
    "optimizer_steps_per_logical_update": 2,
    "refreshes_per_logical_update": 2,
    "groups_per_logical_update": 8,
    "candidates_per_logical_update": 64,
    "checkpoint_boundary": "after-both-refreshes",
    "resume": "exact-logical-update-boundary-only",
}


def validate_d4_failure_gate_for_d5(path: Path) -> dict:
    """Require the frozen aggregate-only D4 failure before any D5 Eterna run."""
    gate = _read_receipt(path)
    protocol = gate.get("protocol")
    d4 = gate.get("arms", {}).get("D4") if isinstance(gate.get("arms"), dict) else None
    if (
        gate.get("status") != "complete"
        or gate.get("decision") != "NO_METRIC_KEEP"
        or gate.get("winner") != "no_winner"
        or not isinstance(protocol, dict)
        or protocol.get("benchmark") != "Eterna100-v2"
        or protocol.get("candidate_sequences_read") is not False
        or protocol.get("task_seed_read") is not False
        or protocol.get("rldev512_read") is not False
        or protocol.get("hard48_read") is not False
        or protocol.get("long10_read") is not False
        or not isinstance(d4, dict)
        or d4.get("evaluation_arm") != D4_TERMINAL_ARM
        or d4.get("group_objective") != D4_HYBRID_PKPO_OBJECTIVE
        or d4.get("keep") is not False
    ):
        raise RuntimeError("D5 Eterna requires the frozen aggregate-only D4 failure gate")
    return gate
D3_POLICY_OBJECTIVE = {
    "name": "discrete_domino",
    "reward_scope": "complete-terminal-sequence-only",
    "advantage": "group-relative-normalized-terminal-reward-per-task",
    "advantage_application": "trajectory-level-advantage-shared-across-time",
    "loss": "sum-over-discrete-transitions-of-ppo-clipped-surrogate",
    "ratio": "per-discrete-mixture-transition-joint-action-clipped-ratio",
    "transition_policy": "structured-finite-step-mixture-kernel-(1-rho)delta+rho*p_clean",
    "mixture_scheduler": "linear-kappa-exact-finite-step-rho=1/(remaining-steps)",
    "state_space": "RNA-nucleotides-with-six-state-target-pair-units",
    "transition_probability_exact": True,
    "native_pretraining_objective": "dirichlet-simplex-flow-matching-transfer",
    "denoiser_input_bridge": "exact-conditional-Dirichlet-path-mean-from-discrete-state",
    "sequence_likelihood": "not-constructed-not-claimed",
}


def digest_contract(contract: dict) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_receipt(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("checkpoint receipt is not a JSON object")
    return value


def _read_d3_contract(path: Path) -> tuple[dict, str]:
    """Read the immutable D3 contract.json without accepting a wrapper."""
    value = _read_receipt(path)
    recorded_digest = value.pop("contract_sha256", None)
    digest = digest_contract(value)
    if recorded_digest is not None and recorded_digest != digest:
        raise RuntimeError("D3 contract.json digest is invalid")
    return value, digest


def _require_checkpoint_payload(payload: dict, label: str) -> tuple[dict, dict]:
    contract = payload.get("contract")
    state = payload.get("state")
    trainable = payload.get("trainable_model")
    if (
        not isinstance(contract, dict)
        or payload.get("contract_sha256") != digest_contract(contract)
        or not isinstance(state, dict)
        or not isinstance(trainable, dict)
        or not trainable
        or any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
               for name, value in trainable.items())
    ):
        raise RuntimeError(f"{label} checkpoint state keys or contract are malformed")
    return contract, state


def _validate_d4_residual(value: object, label: str) -> dict:
    """Validate the frozen D4 coefficient and optional formula metadata."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} D4 PKPO residual is absent or malformed")
    coefficient = value.get("coefficient")
    if (
        not isinstance(coefficient, (int, float))
        or isinstance(coefficient, bool)
        or not math.isfinite(float(coefficient))
        or float(coefficient) != D4_PKPO_RESIDUAL_COEFFICIENT
    ):
        raise RuntimeError(f"{label} D4 PKPO residual coefficient drifted")
    for key, expected in D4_PKPO_RESIDUAL.items():
        if key in value and value[key] != expected:
            raise RuntimeError(f"{label} D4 PKPO residual field drifted: {key}")
    return value


def _validate_d4_checkpoint_receipt_binding(
    checkpoint: dict, receipt: dict, contract: dict,
) -> None:
    """Require contract/checkpoint/receipt residual objects to be identical."""
    residual = _validate_d4_residual(contract.get("pkpo_residual"), "D4 contract")
    if checkpoint.get("pkpo_residual") != residual:
        raise RuntimeError("D4 checkpoint PKPO residual disagrees with contract")
    d3_receipt = receipt.get("d3")
    if not isinstance(d3_receipt, dict) or d3_receipt.get("pkpo_residual") != residual:
        raise RuntimeError("D4 receipt PKPO residual disagrees with contract")
    if "pkpo_residual" in receipt and receipt["pkpo_residual"] != residual:
        raise RuntimeError("D4 receipt PKPO residual aliases disagree")


def validate_supervised_checkpoint(payload: dict) -> dict:
    """Validate the frozen supervised checkpoint's distinct legacy schema."""
    contract = payload.get("contract")
    trainable = payload.get("trainable_model")
    training_state = payload.get("training_state")
    if (
        not isinstance(contract, dict)
        or payload.get("contract_sha256") != digest_contract(contract)
        or not isinstance(trainable, dict)
        or not trainable
        or any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
               for name, value in trainable.items())
        or not isinstance(training_state, dict)
        or not isinstance(payload.get("optimizer"), dict)
        or not isinstance(payload.get("rng_state"), dict)
    ):
        raise RuntimeError("supervised checkpoint state keys or contract are malformed")
    return contract


def _validate_receipt(
    receipt: dict, checkpoint_path: Path, checkpoint: dict, *,
    expected_update: int, expected_status: str,
) -> None:
    last = receipt.get("last_checkpoint")
    if (
        receipt.get("status") != expected_status
        or receipt.get("formal_rl_started") is not True
        or receipt.get("updates_complete") != expected_update
        or receipt.get("bad_count") != 0
        or receipt.get("nan_inf_count") != 0
        or receipt.get("contract_sha256") != checkpoint["contract_sha256"]
        or not isinstance(last, dict)
        or Path(last.get("path", "")).resolve() != checkpoint_path.resolve()
        or last.get("sha256") != sha256(checkpoint_path)
    ):
        raise RuntimeError("checkpoint receipt is incomplete, foreign, or changed")


def validate_u96(
    checkpoint: dict, receipt: dict, checkpoint_path: Path,
    supervised_sha256: str,
) -> dict:
    """Validate the exact frozen U96 parent used by every non-supervised arm."""
    contract, state = _require_checkpoint_payload(checkpoint, "U96")
    _validate_receipt(receipt, checkpoint_path, checkpoint, expected_update=96, expected_status="complete")
    if (
        contract.get("formal_rl") is not True
        or contract.get("updates") != 96
        or contract.get("trajectory_steps") != TRAJECTORY_STEPS
        or contract.get("tasks_per_update") != 4
        or contract.get("candidates_per_task") != 12
        or contract.get("policy_epochs") != 2
        or contract.get("rl_trainable_scope") != "last_2_backbone_and_head"
        or contract.get("supervised_checkpoint_sha256") != supervised_sha256
        or state.get("next_update") != 96
        or receipt.get("target_updates") != 96
    ):
        raise RuntimeError("U96 checkpoint/receipt does not satisfy the frozen lineage")
    return contract


def validate_c2(
    checkpoint: dict, receipt: dict, checkpoint_path: Path, *,
    supervised_sha256: str, u96_checkpoint_sha256: str, u96_receipt_sha256: str,
    u96_contract_sha256: str,
) -> dict:
    """Validate the U96 -> U128 structured-discrete DoMinO continuation."""
    contract, state = _require_checkpoint_payload(checkpoint, "C2 U128")
    _validate_receipt(receipt, checkpoint_path, checkpoint, expected_update=128, expected_status="complete")
    continuation = contract.get("continuation")
    if (
        contract.get("method") != METHOD
        or contract.get("formal_rl") is not True
        or contract.get("updates") != 128
        or contract.get("branch_update_budget_mode") != "u128_discrete_domino32"
        or contract.get("trajectory_steps") != TRAJECTORY_STEPS
        or contract.get("candidates_per_task") != 12
        or contract.get("policy_epochs") != 2
        or contract.get("rl_trainable_scope") != "last_2_backbone_and_head"
        or contract.get("supervised_checkpoint_sha256") != supervised_sha256
        or state.get("next_update") != 128
        or receipt.get("target_updates") != 128
        or not isinstance(continuation, dict)
        or continuation.get("source_checkpoint_sha256") != u96_checkpoint_sha256
        or continuation.get("source_receipt_sha256") != u96_receipt_sha256
        or continuation.get("source_contract_sha256") != u96_contract_sha256
        or continuation.get("start_update") != 96
        or continuation.get("optimizer_state_restored") is not True
        or continuation.get("task_cursor_and_coverage_restored") is not True
    ):
        raise RuntimeError("C2 U128 checkpoint is not the specified frozen U96 lineage")
    return contract


def validate_c3(
    checkpoint: dict, receipt: dict, checkpoint_path: Path, *,
    supervised_sha256: str, u96_checkpoint_sha256: str, u96_receipt_sha256: str,
    u96_contract_sha256: str, adaptation: str,
) -> dict:
    """Validate a frozen C3 Pass-1 U698 arm and its U96 reinitialization."""
    contract, state = _require_checkpoint_payload(checkpoint, f"C3 {adaptation}")
    _validate_receipt(receipt, checkpoint_path, checkpoint, expected_update=698, expected_status="paused")
    initialization = contract.get("policy_initialization")
    if (
        contract.get("method") != METHOD
        or contract.get("formal_rl") is not True
        or contract.get("updates") != 1395
        or contract.get("trajectory_steps") != TRAJECTORY_STEPS
        or contract.get("candidates_per_task") != CANDIDATES
        or contract.get("policy_epochs") != 2
        or contract.get("rl_trainable_scope") != (
            "last_2_backbone_and_head" if adaptation == "last2" else "adapter_and_head"
        )
        or contract.get("supervised_checkpoint_sha256") != supervised_sha256
        or state.get("next_update") != 698
        or receipt.get("target_updates") != 698
        or receipt.get("pause_after_update") != 698
        or not isinstance(initialization, dict)
        or initialization.get("mode") != "u96-policy-weights-fresh-optimizer-c3-scale"
        or initialization.get("source_checkpoint_sha256") != u96_checkpoint_sha256
        or initialization.get("source_receipt_sha256") != u96_receipt_sha256
        or initialization.get("source_contract_sha256") != u96_contract_sha256
        or initialization.get("source_update") != 96
        or initialization.get("optimizer_state_restored") is not False
        or initialization.get("task_cursor_restored") is not False
        or initialization.get("reference_policy") != "frozen-u96"
        or initialization.get("adaptation") != adaptation
    ):
        raise RuntimeError(f"C3 {adaptation} checkpoint is not the frozen U96 scale lineage")
    if adaptation == "lora_m" and (
        initialization.get("lora_rank") != 89
        or initialization.get("lora_alpha") != 178.0
        or initialization.get("lora_dropout") != 0.0
        or initialization.get("lora_target_profile") != "all_attention_ffn"
    ):
        raise RuntimeError("C3 LoRA-M adaptation contract changed")
    if adaptation == "last2" and any(
        initialization.get(key) != expected
        for key, expected in (("lora_rank", 0), ("lora_alpha", 0.0),
                              ("lora_dropout", 0.0), ("lora_target_profile", None))
    ):
        raise RuntimeError("C3 last2 arm has unexpected LoRA adaptation")
    return contract


def validate_d3_terminal(
    checkpoint: dict, receipt: dict, checkpoint_path: Path, contract: dict,
    contract_sha256: str, *, arm: str, supervised_sha256: str,
    supervised_contract_sha256: str, rnaernie_weight_sha256: str,
    u96_checkpoint_sha256: str, u96_receipt_sha256: str,
    u96_contract_sha256: str,
) -> dict:
    """Fail closed on the one permitted D3 U256 terminal lineage per arm."""
    if arm in D4_GROUP_OBJECTIVES:
        return validate_d4_terminal(
            checkpoint, receipt, checkpoint_path, contract, contract_sha256,
            arm=arm, supervised_sha256=supervised_sha256,
            supervised_contract_sha256=supervised_contract_sha256,
            rnaernie_weight_sha256=rnaernie_weight_sha256,
            u96_checkpoint_sha256=u96_checkpoint_sha256,
            u96_receipt_sha256=u96_receipt_sha256,
            u96_contract_sha256=u96_contract_sha256,
        )
    if arm not in D3_GROUP_OBJECTIVES:
        raise ValueError(f"unsupported D3 terminal arm: {arm}")
    checkpoint_contract, state = _require_checkpoint_payload(checkpoint, "D3 U256")
    if (
        contract_sha256 != digest_contract(contract)
        or checkpoint.get("contract_sha256") != contract_sha256
        or checkpoint_contract != contract
    ):
        raise RuntimeError("D3 checkpoint and immutable contract.json disagree")
    required = {
        "schema_version": 11,
        "method": METHOD,
        "action_policy": "structured-discrete-mixture-next-state",
        "transition": "finite-step-discrete-mixture-kernel",
        "ratio": "per-discrete-mixture-transition-joint-action-clipped-ratio",
        "not_claimed_equivalent_to_gaussian_flow_grpo": True,
        "formal_rl": True,
        "updates": D3_UPDATES,
        "tasks_per_update": 4,
        "candidates_per_task": CANDIDATES,
        "trajectory_steps": TRAJECTORY_STEPS,
        "policy_epochs": 2,
        "reward": "terminal",
        "credit_assignment": "uniform",
        "rl_trainable_scope": D3_SCOPE,
        "seed": 1009,
        "group_objective": D3_GROUP_OBJECTIVES[arm],
        "supervised_checkpoint_sha256": supervised_sha256,
        "supervised_contract_sha256": supervised_contract_sha256,
        "rnaernie_weight_sha256": rnaernie_weight_sha256,
        "tasks_sha256": D3_TASKS_SHA256,
        "tasks_manifest_sha256": D3_TASKS_MANIFEST_SHA256,
    }
    initialization = contract.get("policy_initialization")
    temporal = contract.get("temporal_credit")
    curriculum = contract.get("task_curriculum")
    tv = contract.get("reference_transition_tv")
    pkpo = contract.get("pkpo")
    if (
        any(contract.get(key) != value for key, value in required.items())
        or contract.get("policy_objective") != D3_POLICY_OBJECTIVE
        or not isinstance(temporal, dict) or temporal.get("mode") != "none"
        or not isinstance(curriculum, dict)
        or curriculum.get("mode") != "deterministic_shuffle"
        or curriculum.get("data_scope") != "TRAIN-only"
        or not isinstance(tv, dict)
        or not isinstance(tv.get("coefficient"), (int, float))
        or isinstance(tv.get("coefficient"), bool)
        or not math.isclose(float(tv["coefficient"]), 4.0, rel_tol=0, abs_tol=0)
        or tv.get("distribution") != "structured-clean-categorical-4-or-6-state"
        or tv.get("transition") != "rho-times-posterior-tv"
        or tv.get("reference_detached") is not True
        or not isinstance(pkpo, dict)
        or pkpo.get("algorithm") != "continuous-max-at-k-sloo-minus-one"
        or pkpo.get("schedule") != {"0-84": 8, "85-169": 4, "170-255": 1}
        or pkpo.get("k1") != "leave-one-out-mean-centered-raw-reward"
        or pkpo.get("group_std_normalization") is not False
        or pkpo.get("pass_at_8_bonus") is not False
        or not isinstance(initialization, dict)
        or initialization.get("mode") != "u96-policy-weights-fresh-optimizer-d3-256"
        or initialization.get("source_checkpoint_sha256") != u96_checkpoint_sha256
        or initialization.get("source_receipt_sha256") != u96_receipt_sha256
        or initialization.get("source_contract_sha256") != u96_contract_sha256
        or initialization.get("source_update") != 96
        or initialization.get("optimizer_state_restored") is not False
        or initialization.get("task_cursor_restored") is not False
        or initialization.get("reference_policy") != "frozen-u96"
        or initialization.get("adaptation") != "last2-fresh-init"
        or initialization.get("lora_rank") != 0
        or initialization.get("lora_alpha") != 0.0
        or initialization.get("lora_dropout") != 0.0
        or initialization.get("lora_target_profile") is not None
        or state.get("next_update") != D3_UPDATES
    ):
        raise RuntimeError("D3 U256 checkpoint is not the frozen terminal contract")
    coverage = state.get("reward_evaluation_coverage")
    health = state.get("d3_health")
    if (
        not isinstance(coverage, dict)
        or coverage.get("candidate_rows") != D3_UPDATES * 4 * CANDIDATES
        or coverage.get("groups") != D3_UPDATES * 4
        or coverage.get("invalid_candidate_rows") != 0
        or not isinstance(health, dict)
        or health.get("bad_or_nan_count") != 0
        or health.get("groups") != D3_UPDATES * 4
    ):
        raise RuntimeError("D3 U256 checkpoint reward/health coverage is incomplete")
    last = receipt.get("last_checkpoint")
    d3_receipt = receipt.get("d3")
    if (
        receipt.get("status") != "complete"
        or receipt.get("formal_rl_started") is not True
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("contract_target_updates") != D3_UPDATES
        or receipt.get("updates_complete") != D3_UPDATES
        or receipt.get("target_updates") != D3_UPDATES
        or receipt.get("bad_count") != 0
        or receipt.get("nan_inf_count") != 0
        or not isinstance(last, dict)
        or Path(str(last.get("path", ""))).resolve() != checkpoint_path.resolve()
        or last.get("sha256") != sha256(checkpoint_path)
        or not isinstance(d3_receipt, dict)
        or d3_receipt.get("group_objective") != D3_GROUP_OBJECTIVES[arm]
        or d3_receipt.get("reference_tv_coefficient") != 4.0
        or d3_receipt.get("bad_or_nan_count") != 0
    ):
        raise RuntimeError("D3 U256 terminal receipt is incomplete or foreign")
    return contract


def validate_d4_terminal(
    checkpoint: dict, receipt: dict, checkpoint_path: Path, contract: dict,
    contract_sha256: str, *, arm: str, supervised_sha256: str,
    supervised_contract_sha256: str, rnaernie_weight_sha256: str,
    u96_checkpoint_sha256: str, u96_receipt_sha256: str,
    u96_contract_sha256: str,
) -> dict:
    """Validate the sole permitted D4 terminal lineage before Eterna read."""
    if arm != D4_TERMINAL_ARM:
        raise ValueError(
            "D4 U128 is TRAIN-monitor-only; frozen Eterna requires d4_hybrid_u256"
        )
    checkpoint_contract, state = _require_checkpoint_payload(checkpoint, "D4 U256")
    if (
        contract_sha256 != digest_contract(contract)
        or checkpoint.get("contract_sha256") != contract_sha256
        or checkpoint_contract != contract
    ):
        raise RuntimeError("D4 checkpoint and immutable contract.json disagree")
    required = {
        "schema_version": 11,
        "method": METHOD,
        "action_policy": "structured-discrete-mixture-next-state",
        "transition": "finite-step-discrete-mixture-kernel",
        "ratio": "per-discrete-mixture-transition-joint-action-clipped-ratio",
        "not_claimed_equivalent_to_gaussian_flow_grpo": True,
        "formal_rl": True,
        "updates": D3_UPDATES,
        "tasks_per_update": 4,
        "candidates_per_task": CANDIDATES,
        "trajectory_steps": TRAJECTORY_STEPS,
        "policy_epochs": 2,
        "reward": "terminal",
        "credit_assignment": "uniform",
        "rl_trainable_scope": D3_SCOPE,
        "seed": 1009,
        "group_objective": D4_HYBRID_PKPO_OBJECTIVE,
        "supervised_checkpoint_sha256": supervised_sha256,
        "supervised_contract_sha256": supervised_contract_sha256,
        "rnaernie_weight_sha256": rnaernie_weight_sha256,
        "tasks_sha256": D3_TASKS_SHA256,
        "tasks_manifest_sha256": D3_TASKS_MANIFEST_SHA256,
    }
    initialization = contract.get("policy_initialization")
    temporal = contract.get("temporal_credit")
    curriculum = contract.get("task_curriculum")
    tv = contract.get("reference_transition_tv")
    pkpo = contract.get("pkpo")
    residual = contract.get("pkpo_residual")
    if (
        any(contract.get(key) != value for key, value in required.items())
        or contract.get("policy_objective") != D3_POLICY_OBJECTIVE
        or not isinstance(temporal, dict) or temporal.get("mode") != "none"
        or not isinstance(curriculum, dict)
        or curriculum.get("mode") != "deterministic_shuffle"
        or curriculum.get("data_scope") != "TRAIN-only"
        or not isinstance(tv, dict)
        or not isinstance(tv.get("coefficient"), (int, float))
        or isinstance(tv.get("coefficient"), bool)
        or not math.isclose(float(tv["coefficient"]), 4.0, rel_tol=0, abs_tol=0)
        or tv.get("distribution") != "structured-clean-categorical-4-or-6-state"
        or tv.get("transition") != "rho-times-posterior-tv"
        or tv.get("reference_detached") is not True
        or not isinstance(pkpo, dict)
        or pkpo.get("algorithm") != "continuous-max-at-k-sloo-minus-one"
        or pkpo.get("schedule") != {"0-84": 8, "85-169": 4, "170-255": 1}
        or pkpo.get("k1") != "leave-one-out-mean-centered-raw-reward"
        or pkpo.get("group_std_normalization") is not False
        or pkpo.get("pass_at_8_bonus") is not False
        or not isinstance(initialization, dict)
        or initialization.get("mode") != "u96-policy-weights-fresh-optimizer-d3-256"
        or initialization.get("source_checkpoint_sha256") != u96_checkpoint_sha256
        or initialization.get("source_receipt_sha256") != u96_receipt_sha256
        or initialization.get("source_contract_sha256") != u96_contract_sha256
        or initialization.get("source_update") != 96
        or initialization.get("optimizer_state_restored") is not False
        or initialization.get("task_cursor_restored") is not False
        or initialization.get("reference_policy") != "frozen-u96"
        or initialization.get("adaptation") != "last2-fresh-init"
        or initialization.get("lora_rank") != 0
        or initialization.get("lora_alpha") != 0.0
        or initialization.get("lora_dropout") != 0.0
        or initialization.get("lora_target_profile") is not None
        or state.get("next_update") != D3_UPDATES
    ):
        raise RuntimeError("D4 U256 checkpoint is not the frozen terminal contract")
    _validate_d4_residual(residual, "D4 contract")
    coverage = state.get("reward_evaluation_coverage")
    health = state.get("d3_health")
    if (
        not isinstance(coverage, dict)
        or coverage.get("candidate_rows") != D3_UPDATES * 4 * CANDIDATES
        or coverage.get("groups") != D3_UPDATES * 4
        or coverage.get("invalid_candidate_rows") != 0
        or not isinstance(health, dict)
        or health.get("bad_or_nan_count") != 0
        or health.get("groups") != D3_UPDATES * 4
    ):
        raise RuntimeError("D4 U256 checkpoint reward/health coverage is incomplete")
    last = receipt.get("last_checkpoint")
    d3_receipt = receipt.get("d3")
    if (
        receipt.get("status") != "complete"
        or receipt.get("formal_rl_started") is not True
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("contract_target_updates") != D3_UPDATES
        or receipt.get("updates_complete") != D3_UPDATES
        or receipt.get("target_updates") != D3_UPDATES
        or receipt.get("bad_count") != 0
        or receipt.get("nan_inf_count") != 0
        or not isinstance(last, dict)
        or Path(str(last.get("path", ""))).resolve() != checkpoint_path.resolve()
        or last.get("sha256") != sha256(checkpoint_path)
        or not isinstance(d3_receipt, dict)
        or d3_receipt.get("group_objective") != D4_HYBRID_PKPO_OBJECTIVE
        or d3_receipt.get("reference_tv_coefficient") != 4.0
        or d3_receipt.get("bad_or_nan_count") != 0
    ):
        raise RuntimeError("D4 U256 terminal receipt is incomplete or foreign")
    _validate_d4_checkpoint_receipt_binding(checkpoint, receipt, contract)
    return contract


def _validate_d5_runtime_state(checkpoint: dict, state: dict, update: int) -> None:
    coverage = state.get("reward_evaluation_coverage")
    histogram = state.get("temporal_credit_step_histogram")
    health = state.get("d3_health")
    d5 = checkpoint.get("d5")
    optimizer = checkpoint.get("optimizer")
    rng = checkpoint.get("rng_state_by_rank")
    if (
        not isinstance(coverage, dict)
        or coverage.get("candidate_rows") != update * 64
        or coverage.get("groups") != update * 8
        or coverage.get("invalid_candidate_rows") != 0
        or histogram != {str(step): update * 8 for step in range(TRAJECTORY_STEPS)}
        or not isinstance(health, dict)
        or health.get("bad_or_nan_count") != 0
        or health.get("groups") != update * 8
        or not isinstance(d5, dict)
        or d5.get("logical_update_boundary") != update
        or d5.get("task_cursor") != update * 4
        or d5.get("optimizer_steps") != update * 2
        or d5.get("coverage") != coverage
        or d5.get("temporal_credit_step_histogram") != histogram
        or not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("state"), dict)
        or not optimizer["state"]
        or not isinstance(rng, list)
        or len(rng) != 2
        or any(not isinstance(value, dict) for value in rng)
    ):
        raise RuntimeError("D5 checkpoint strict on-policy coverage/RNG2 boundary is incomplete")
    for value in optimizer["state"].values():
        step = value.get("step") if isinstance(value, dict) else None
        if isinstance(step, torch.Tensor):
            step = step.item() if step.numel() == 1 else None
        if step != update * 2:
            raise RuntimeError("D5 checkpoint AdamW step is not 2*updates")


def validate_d5_terminal(
    checkpoint: dict, receipt: dict, checkpoint_path: Path, contract: dict,
    contract_sha256: str, *, arm: str, supervised_sha256: str,
    supervised_contract_sha256: str, rnaernie_weight_sha256: str,
    u96_checkpoint_sha256: str, u96_receipt_sha256: str,
    u96_contract_sha256: str,
) -> dict:
    """Validate D5's sole terminal U256 strict-on-policy lineage."""
    if arm != D5_TERMINAL_ARM:
        raise ValueError("D5 U128 is TRAIN-monitor-only; frozen Eterna requires d5_strict_on_policy_u256")
    checkpoint_contract, state = _require_checkpoint_payload(checkpoint, "D5 U256")
    if (
        contract_sha256 != digest_contract(contract)
        or checkpoint.get("contract_sha256") != contract_sha256
        or checkpoint_contract != contract
    ):
        raise RuntimeError("D5 checkpoint and immutable contract.json disagree")
    required = {
        "schema_version": 11, "method": METHOD,
        "action_policy": "structured-discrete-mixture-next-state",
        "transition": "finite-step-discrete-mixture-kernel",
        "ratio": "per-discrete-mixture-transition-joint-action-clipped-ratio",
        "not_claimed_equivalent_to_gaussian_flow_grpo": True,
        "formal_rl": True, "updates": D3_UPDATES, "tasks_per_update": 4,
        "candidates_per_task": CANDIDATES, "trajectory_steps": TRAJECTORY_STEPS,
        "policy_epochs": 2, "reward": "terminal", "credit_assignment": "uniform",
        "rl_trainable_scope": D3_SCOPE, "seed": 1009,
        "group_objective": D5_STRICT_ON_POLICY_OBJECTIVE,
        "supervised_checkpoint_sha256": supervised_sha256,
        "supervised_contract_sha256": supervised_contract_sha256,
        "rnaernie_weight_sha256": rnaernie_weight_sha256,
        "tasks_sha256": D3_TASKS_SHA256,
        "tasks_manifest_sha256": D3_TASKS_MANIFEST_SHA256,
        "rollout_refresh_mode": D5_ROLLOUT_REFRESH_MODE,
        "rollout_refresh": D5_ROLLOUT_REFRESH,
    }
    initialization = contract.get("policy_initialization")
    temporal = contract.get("temporal_credit")
    curriculum = contract.get("task_curriculum")
    tv = contract.get("reference_transition_tv")
    pkpo = contract.get("pkpo")
    if (
        any(contract.get(key) != value for key, value in required.items())
        or contract.get("policy_objective") != D3_POLICY_OBJECTIVE
        or "pkpo_residual" in contract or "pkpo_residual" in checkpoint
        or not isinstance(temporal, dict) or temporal.get("mode") != "none"
        or not isinstance(curriculum, dict) or curriculum.get("mode") != "deterministic_shuffle"
        or curriculum.get("data_scope") != "TRAIN-only"
        or not isinstance(tv, dict) or tv.get("coefficient") != 4.0
        or tv.get("distribution") != "structured-clean-categorical-4-or-6-state"
        or tv.get("transition") != "rho-times-posterior-tv" or tv.get("reference_detached") is not True
        or not isinstance(pkpo, dict) or pkpo.get("schedule") != {"0-84": 8, "85-169": 4, "170-255": 1}
        or not isinstance(initialization, dict)
        or initialization.get("mode") != "u96-policy-weights-fresh-optimizer-d3-256"
        or initialization.get("source_checkpoint_sha256") != u96_checkpoint_sha256
        or initialization.get("source_receipt_sha256") != u96_receipt_sha256
        or initialization.get("source_contract_sha256") != u96_contract_sha256
        or initialization.get("source_update") != 96
        or initialization.get("optimizer_state_restored") is not False
        or initialization.get("task_cursor_restored") is not False
        or initialization.get("reference_policy") != "frozen-u96"
        or initialization.get("adaptation") != "last2-fresh-init"
        or initialization.get("lora_rank") != 0
        or initialization.get("lora_alpha") != 0.0
        or initialization.get("lora_dropout") != 0.0
        or initialization.get("lora_target_profile") is not None
        or contract.get("distributed", {}).get("world_size") != 2
        or state.get("next_update") != D3_UPDATES
    ):
        raise RuntimeError("D5 U256 checkpoint is not the frozen strict-on-policy terminal contract")
    _validate_d5_runtime_state(checkpoint, state, D3_UPDATES)
    last = receipt.get("last_checkpoint")
    d3 = receipt.get("d3")
    d5 = receipt.get("d5")
    if (
        receipt.get("status") != "complete" or receipt.get("formal_rl_started") is not True
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("contract_target_updates") != D3_UPDATES
        or receipt.get("updates_complete") != D3_UPDATES or receipt.get("target_updates") != D3_UPDATES
        or receipt.get("bad_count") != 0 or receipt.get("nan_inf_count") != 0
        or not isinstance(last, dict) or Path(str(last.get("path", ""))).resolve() != checkpoint_path.resolve()
        or last.get("sha256") != sha256(checkpoint_path)
        or not isinstance(d3, dict) or d3.get("group_objective") != D5_STRICT_ON_POLICY_OBJECTIVE
        or d3.get("reference_tv_coefficient") != 4.0 or d3.get("bad_or_nan_count") != 0
        or not isinstance(d5, dict) or d5.get("rollout_refresh_mode") != D5_ROLLOUT_REFRESH_MODE
        or d5.get("rollout_refresh") != D5_ROLLOUT_REFRESH
        or d5.get("task_cursor_advance_per_logical_update") != 4
        or d5.get("optimizer_steps_per_logical_update") != 2
        or d5.get("groups_per_logical_update") != 8
        or d5.get("candidates_per_logical_update") != 64
        or not isinstance(d5.get("coverage"), dict)
        or any(
            d5["coverage"].get(key) != value
            for key, value in state["reward_evaluation_coverage"].items()
        )
        or d5["coverage"].get("invalid_candidate_rate") != 0.0
        or d5["coverage"].get("unique_invalid_cache_keys") != 0
        or d5.get("temporal_credit_step_histogram") != state["temporal_credit_step_histogram"]
        or d5.get("resume") != "exact-logical-update-boundary-only"
    ):
        raise RuntimeError("D5 U256 terminal receipt is incomplete or foreign")
    return contract


def validate_coverage_scale_terminal(
    checkpoint: dict,
    receipt: dict,
    checkpoint_path: Path,
    contract: dict,
    contract_sha256: str,
    *,
    kind: str,
    source_checkpoint_path: Path,
    source_receipt_path: Path,
    supervised_sha256: str,
    supervised_contract_sha256: str,
    rnaernie_weight_sha256: str,
    u96_checkpoint_sha256: str,
    u96_receipt_sha256: str,
    u96_contract_sha256: str,
) -> dict:
    """Validate one fresh-output C3/T0 coverage-scaling terminal boundary."""
    checkpoint_contract, state = _require_checkpoint_payload(checkpoint, "coverage scale terminal")
    if (
        contract_sha256 != digest_contract(contract)
        or checkpoint.get("contract_sha256") != contract_sha256
        or checkpoint_contract != contract
    ):
        raise RuntimeError("coverage scale checkpoint and contract disagree")
    target_update = state.get("next_update")
    if kind not in COVERAGE_SCALE_UPDATES or target_update not in COVERAGE_SCALE_UPDATES[kind]:
        raise RuntimeError("coverage scale target update is not preregistered")
    continuation = contract.get("continuation")
    source_update = continuation.get("start_update") if isinstance(continuation, dict) else None
    if source_update not in COVERAGE_SCALE_UPDATES[kind][target_update]:
        raise RuntimeError("coverage scale source update is not preregistered for this target")
    source_checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    source_receipt = _read_receipt(source_receipt_path)
    source_state = source_checkpoint.get("state") if isinstance(source_checkpoint, dict) else None
    source_contract = source_checkpoint.get("contract") if isinstance(source_checkpoint, dict) else None
    if (
        not isinstance(continuation, dict)
        or continuation.get("mode") != "coverage-scale-exact-state-v1"
        or continuation.get("kind") != kind
        or continuation.get("start_update") != source_update
        or continuation.get("source_checkpoint_sha256") != sha256(source_checkpoint_path)
        or continuation.get("source_receipt_sha256") != sha256(source_receipt_path)
        or not isinstance(source_state, dict)
        or source_state.get("next_update") != source_update
        or not isinstance(source_contract, dict)
        or continuation.get("source_contract_sha256") != source_checkpoint.get("contract_sha256")
        or source_receipt.get("updates_complete") != source_update
        or source_receipt.get("bad_count") != 0
        or source_receipt.get("nan_inf_count") != 0
        or continuation.get("optimizer_state_restored") is not True
        or continuation.get("rng_state_by_rank_restored") is not True
        or continuation.get("task_cursor_and_coverage_restored") is not True
    ):
        raise RuntimeError("coverage scale source lineage is incomplete or foreign")
    common_required = {
        "schema_version": 11,
        "method": METHOD,
        "action_policy": "structured-discrete-mixture-next-state",
        "transition": "finite-step-discrete-mixture-kernel",
        "ratio": "per-discrete-mixture-transition-joint-action-clipped-ratio",
        "not_claimed_equivalent_to_gaussian_flow_grpo": True,
        "formal_rl": True,
        "tasks_per_update": 4,
        "candidates_per_task": CANDIDATES,
        "trajectory_steps": TRAJECTORY_STEPS,
        "policy_epochs": 2,
        "reward": "terminal",
        "credit_assignment": "uniform",
        "rl_trainable_scope": D3_SCOPE,
        "seed": 1009,
        "supervised_checkpoint_sha256": supervised_sha256,
        "supervised_contract_sha256": supervised_contract_sha256,
        "rnaernie_weight_sha256": rnaernie_weight_sha256,
    }
    if any(contract.get(key) != value for key, value in common_required.items()):
        raise RuntimeError("coverage scale common scientific contract drifted")
    initialization = contract.get("policy_initialization")
    temporal = contract.get("temporal_credit")
    curriculum = contract.get("task_curriculum")
    stopped_sync = (
        kind == "t0_flow_yrl"
        and target_update == 623
        and receipt.get("status") == "stopped_by_user_for_d5_scaling_comparison"
        and contract.get("updates") == 1036
    )
    if (
        (contract.get("updates") != target_update and not stopped_sync)
        or contract.get("policy_objective") != D3_POLICY_OBJECTIVE
        or not isinstance(temporal, dict)
        or temporal.get("mode") != "none"
        or not isinstance(curriculum, dict)
        or curriculum.get("mode") != "deterministic_shuffle"
        or curriculum.get("data_scope") != "TRAIN-only"
        or not isinstance(initialization, dict)
        or initialization.get("source_checkpoint_sha256") != u96_checkpoint_sha256
        or initialization.get("source_receipt_sha256") != u96_receipt_sha256
        or initialization.get("source_contract_sha256") != u96_contract_sha256
        or initialization.get("source_update") != 96
        or initialization.get("reference_policy") != "frozen-u96"
    ):
        raise RuntimeError("coverage scale initialization/temporal contract drifted")
    if kind in {"c3_official2790", "c3_official2790_d5"}:
        pool_size = 2790
        expected_tasks_sha = C3_OFFICIAL_TASKS_SHA256
        expected_manifest_sha = C3_OFFICIAL_TASKS_MANIFEST_SHA256
        strict_on_policy = kind == "c3_official2790_d5"
        if (
            contract.get("tasks_sha256") != expected_tasks_sha
            or contract.get("tasks_manifest_sha256") != expected_manifest_sha
            or contract.get("group_objective") is not None
            or contract.get("reference_transition_tv") is not None
            or initialization.get("mode") != "u96-policy-weights-fresh-optimizer-c3-scale"
            or initialization.get("adaptation") != "last2"
            or (
                contract.get("rollout_refresh_mode") != D5_ROLLOUT_REFRESH_MODE
                if strict_on_policy else contract.get("rollout_refresh_mode") is not None
            )
            or (
                contract.get("rollout_refresh") != D5_ROLLOUT_REFRESH
                if strict_on_policy else contract.get("rollout_refresh") is not None
            )
        ):
            raise RuntimeError("C3 official2790 scale scientific contract drifted")
    else:
        pool_size = 4143
        expected_tasks_sha = D3_TASKS_SHA256
        expected_manifest_sha = D3_TASKS_MANIFEST_SHA256
        tv = contract.get("reference_transition_tv")
        strict_on_policy = kind == "t0_flow_yrl_d5"
        if (
            contract.get("tasks_sha256") != expected_tasks_sha
            or contract.get("tasks_manifest_sha256") != expected_manifest_sha
            or contract.get("group_objective") != "thermodynamic_grpo"
            or not isinstance(tv, dict)
            or float(tv.get("coefficient", float("nan"))) != 4.0
            or tv.get("reference_detached") is not True
            or initialization.get("mode") != "u96-policy-weights-fresh-optimizer-d3-256"
            or initialization.get("adaptation") != "last2-fresh-init"
            or (
                contract.get("rollout_refresh_mode") != D5_ROLLOUT_REFRESH_MODE
                if strict_on_policy else contract.get("rollout_refresh_mode") is not None
            )
            or (
                contract.get("rollout_refresh") != D5_ROLLOUT_REFRESH
                if strict_on_policy else contract.get("rollout_refresh") is not None
            )
        ):
            raise RuntimeError("T0 Flow-YRL scale scientific contract drifted")
    task_positions = target_update * 4
    if kind == "c3_official2790_d5":
        expected_groups = 1395 * 4 + (target_update - 1395) * 8
    else:
        groups_per_task = 2 if kind == "t0_flow_yrl_d5" else 1
        expected_groups = task_positions * groups_per_task
    coverage = state.get("reward_evaluation_coverage")
    if (
        state.get("task_epoch") != task_positions // pool_size
        or state.get("task_cursor") != task_positions % pool_size
        or not isinstance(coverage, dict)
        or coverage.get("groups") != expected_groups
        or coverage.get("candidate_rows") != expected_groups * CANDIDATES
        or coverage.get("invalid_candidate_rows") != 0
        or state.get("temporal_credit_step_histogram")
        != {str(step): expected_groups for step in range(TRAJECTORY_STEPS)}
    ):
        raise RuntimeError("coverage scale task/trajectory coverage is incomplete")
    if kind in {"t0_flow_yrl_d5", "c3_official2790_d5"}:
        d5 = checkpoint.get("d5")
        if (
            not isinstance(d5, dict)
            or d5.get("logical_update_boundary") != target_update
            or d5.get("task_cursor") != task_positions % pool_size
            or d5.get("optimizer_steps") != target_update * 2
            or d5.get("coverage") != coverage
            or d5.get("temporal_credit_step_histogram")
            != state.get("temporal_credit_step_histogram")
        ):
            raise RuntimeError("D5 coverage scale checkpoint boundary is incomplete")
    last = receipt.get("last_checkpoint")
    complete_terminal = (
        receipt.get("status") == "complete"
        and receipt.get("contract_target_updates") == target_update
    )
    if (
        (not complete_terminal and not stopped_sync)
        or receipt.get("formal_rl_started") is not True
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("updates_complete") != target_update
        or receipt.get("target_updates") != target_update
        or receipt.get("bad_count") != 0
        or receipt.get("nan_inf_count") != 0
        or not isinstance(last, dict)
        or Path(str(last.get("path", ""))).resolve() != checkpoint_path.resolve()
        or last.get("sha256") != sha256(checkpoint_path)
    ):
        raise RuntimeError("coverage scale terminal receipt is incomplete or foreign")
    return contract


def load_model_for_arm(
    *, supervised: dict, supervised_contract: dict, u96: dict,
    candidate: dict | None, adaptation: str | None, rnaernie: Path,
    device: torch.device,
) -> FairRNAFlow:
    """Reconstruct a model in the frozen per-arm load order.

    ``load_trainable_state_dict`` is intentionally used at every stage: it
    checks the expected state-key set against the currently configured model.
    """
    model = FairRNAFlow(
        rnaernie,
        structure_source=supervised_contract["structure_source"],
        injection=supervised_contract["injection"],
        structure_dim=supervised_contract.get("structure_dim"),
        lora_rank=supervised_contract["lora_rank"],
        lora_alpha=supervised_contract["lora_alpha"],
        lora_dropout=supervised_contract["lora_dropout"],
        backbone_mode=supervised_contract.get("backbone_mode"),
    ).to(device)
    model.load_trainable_state_dict(supervised["trainable_model"])
    if adaptation is None:
        return model
    configure_rl_trainable_scope(model, "last_2_backbone_and_head")
    if adaptation == "u96":
        model.load_trainable_state_dict(u96["trainable_model"])
        return model
    if adaptation == "c2":
        if candidate is None:
            raise RuntimeError("c2 candidate is absent")
        model.load_trainable_state_dict(candidate["trainable_model"])
        return model
    if adaptation == "d3":
        if candidate is None:
            raise RuntimeError("d3 candidate is absent")
        model.load_trainable_state_dict(u96["trainable_model"])
        model.load_trainable_state_dict(candidate["trainable_model"])
        return model
    model.load_trainable_state_dict(u96["trainable_model"])
    if adaptation == "last2":
        if candidate is None:
            raise RuntimeError("C3 last2 candidate is absent")
        model.load_trainable_state_dict(candidate["trainable_model"])
        return model
    if adaptation == "lora_m":
        if candidate is None:
            raise RuntimeError("C3 LoRA-M candidate is absent")
        adapt_loaded_supervised_model_with_lora(
            model, rank=89, alpha=178.0, dropout=0.0,
            target_profile="all_attention_ffn",
        )
        model.to(device)
        configure_rl_trainable_scope(model, "adapter_and_head")
        model.load_trainable_state_dict(candidate["trainable_model"])
        return model
    raise RuntimeError(f"unknown D0 arm adaptation: {adaptation}")


def evaluate_arm(
    *, name: str, model: FairRNAFlow, tasks: list[dict], output: Path,
    device: torch.device, workers: int, source_hashes: dict[str, str],
    arm_contract: dict, evaluation_role: str = "formal",
) -> dict:
    """Evaluate one arm with the frozen scorer and D0 trajectory sampler."""
    if evaluation_role not in {"formal", "preflight"}:
        raise ValueError("D0 evaluation role is invalid")
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "task_progress"
    progress.mkdir()
    started = time.time()
    rows: list[dict] = []
    replay_checks = 0
    legal_checks = 0
    model.eval()
    for task_index, task in enumerate(tasks):
        structure = task["target_structure"]
        task_rows: list[dict] = []
        for seed in SEEDS:
            temperature = SEED_TEMPERATURE_MAP[seed]
            temperatures = torch.full((CANDIDATES,), temperature, dtype=torch.float32, device=device)
            trajectory = rollout_discrete_domino_trajectory(
                model, structure, CANDIDATES, TRAJECTORY_STEPS,
                global_task_seed(seed, task_index), device, temperatures,
            )
            validation = validate_discrete_domino_trajectory(
                trajectory, device, expected_structure=structure,
                expected_seed=global_task_seed(seed, task_index),
                expected_candidates=CANDIDATES, expected_steps=TRAJECTORY_STEPS,
                expected_temperatures=temperatures,
            )
            if validation.get("replay_exact") is not True or validation.get("structured_pair_actions_legal") is not True:
                raise RuntimeError("D0 trajectory replay/legal validation failed")
            replay_checks += 1
            legal_checks += 1
            metrics = evaluate_many(trajectory["final_sequences"], structure, workers)
            for candidate_index, (sequence, values) in enumerate(zip(trajectory["final_sequences"], metrics, strict=True)):
                pair_validity = valid_pair_fraction(sequence, structure)
                valid = len(sequence) == len(structure) and not (set(sequence) - set("ACGU")) and pair_validity == 1
                task_rows.append({
                    "task_id": task["id"], "seed": seed, "candidate_index": candidate_index,
                    "sequence": sequence, "valid": float(valid),
                    "valid_pair_fraction": pair_validity, **values,
                })
        validate_coverage(task_rows, [task["id"]], SEEDS, CANDIDATES)
        (progress / f"task-{task_index:03d}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in task_rows)
        )
        rows.extend(task_rows)
        print(json.dumps({"arm": name, "tasks_complete": task_index + 1, "tasks_total": len(tasks)}), flush=True)
    validate_coverage(rows, [task["id"] for task in tasks], SEEDS, CANDIDATES)
    task_seed, summary = aggregate(rows, CANDIDATES)
    bad_count = sum(row.get("evaluation_valid") is False for row in rows)
    runtime_seconds = time.time() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    sampler = {
        "schema_version": 3,
        "policy": "structured-discrete-mixture-dfm-domino-v1",
        "trajectory_steps": TRAJECTORY_STEPS,
        "candidates": CANDIDATES,
        "kernel": "(1-rho)delta+rho*p_clean",
        "seed_temperature_map": {str(seed): temperature for seed, temperature in SEED_TEMPERATURE_MAP.items()},
    }
    coverage = {"tasks": len(tasks), "benchmark_tasks": 100,
                "seeds": len(SEEDS), "candidates": CANDIDATES}
    summary.update({
        "status": "complete" if evaluation_role == "formal" else "preflight_complete",
        "evaluation_role": evaluation_role, "formal_evaluation": evaluation_role == "formal",
        "arm": name, "method": arm_contract.get("method", "supervised"),
        "evaluation_sampler": sampler, "benchmark": "Eterna100-v2",
        "benchmark_sha256": EXPECTED_ETERNA_SHA256,
        "coverage": coverage,
        "bad_count": bad_count, "runtime_seconds": runtime_seconds,
        "max_cuda_memory_bytes": peak_memory, "source_sha256": source_hashes,
        "core_evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "core_model_sha256": EXPECTED_MODEL_SHA256, "adapter_sha256": sha256(Path(__file__)),
        "viennarna_version": RNA.__version__, "replay_validation_count": replay_checks,
        "legal_pair_validation_count": legal_checks,
    })
    continuation = arm_contract.get("continuation")
    if isinstance(continuation, dict) and continuation.get("mode") == "coverage-scale-exact-state-v1":
        summary["coverage_scale"] = {
            "kind": continuation.get("kind"),
            "start_update": continuation.get("start_update"),
            "source_checkpoint_sha256": continuation.get("source_checkpoint_sha256"),
            "source_receipt_sha256": continuation.get("source_receipt_sha256"),
            "source_contract_sha256": continuation.get("source_contract_sha256"),
        }
    if arm_contract.get("group_objective") == D4_HYBRID_PKPO_OBJECTIVE:
        summary["group_objective"] = D4_HYBRID_PKPO_OBJECTIVE
        summary["pkpo_residual"] = arm_contract["pkpo_residual"]
    elif arm_contract.get("rollout_refresh_mode") == D5_ROLLOUT_REFRESH_MODE:
        summary["group_objective"] = D5_STRICT_ON_POLICY_OBJECTIVE
        summary["rollout_refresh_mode"] = D5_ROLLOUT_REFRESH_MODE
        summary["rollout_refresh"] = D5_ROLLOUT_REFRESH
    (output / "task_seed.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in task_seed))
    atomic_json(output / "summary.json", summary)
    evaluation_receipt = {
        "schema_version": 1,
        "status": (
            "complete" if bad_count == 0 else "complete_with_bad_candidates"
        ) if evaluation_role == "formal" else (
            "preflight_complete" if bad_count == 0 else "preflight_complete_with_bad_candidates"
        ),
        "evaluation_role": evaluation_role,
        "formal_evaluation": evaluation_role == "formal",
        "arm": name, "method": arm_contract.get("method", "supervised"),
        "sampler": sampler, "source_sha256": source_hashes,
        "benchmark": {"name": "Eterna100-v2", "sha256": EXPECTED_ETERNA_SHA256},
        "code_identity": {"fair_evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
                          "fair_model_sha256": EXPECTED_MODEL_SHA256,
                          "adapter_sha256": sha256(Path(__file__))},
        "viennarna": {"version": RNA.__version__, "temperature_c": 37.0,
                        "dangles": 2, "uniq_ML": 1},
        "replay": {"validated": True, "count": replay_checks},
        "legal_pair_actions": {"validated": True, "count": legal_checks},
        "coverage": coverage,
        "bad_count": bad_count, "runtime_seconds": runtime_seconds,
        "max_cuda_memory_bytes": peak_memory,
    }
    if isinstance(continuation, dict) and continuation.get("mode") == "coverage-scale-exact-state-v1":
        evaluation_receipt["coverage_scale"] = summary["coverage_scale"]
    if arm_contract.get("group_objective") == D4_HYBRID_PKPO_OBJECTIVE:
        # Expose the exact contract residual in the aggregate receipt so the
        # D4 gate can cross-check training receipt, terminal checkpoint and
        # evaluator provenance without opening task-level outputs.
        evaluation_receipt["group_objective"] = D4_HYBRID_PKPO_OBJECTIVE
        evaluation_receipt["pkpo_residual"] = arm_contract["pkpo_residual"]
    elif arm_contract.get("rollout_refresh_mode") == D5_ROLLOUT_REFRESH_MODE:
        # D5's aggregate provenance is intentionally sufficient for the gate;
        # neither this adapter nor the gate opens per-target Eterna outputs.
        evaluation_receipt["group_objective"] = D5_STRICT_ON_POLICY_OBJECTIVE
        evaluation_receipt["rollout_refresh_mode"] = D5_ROLLOUT_REFRESH_MODE
        evaluation_receipt["rollout_refresh"] = D5_ROLLOUT_REFRESH
    atomic_json(output / "receipt.json", evaluation_receipt)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", required=True,
        choices=[
            "u96", "c2_u128", "c3_last2_u698", "c3_lora_u698",
            *D3_GROUP_OBJECTIVES, *D4_GROUP_OBJECTIVES, *D5_GROUP_OBJECTIVES,
            *COVERAGE_SCALE_ARMS,
        ],
        help="Exactly one frozen arm; formal arms require distinct fresh outputs.",
    )
    parser.add_argument(
        "--evaluation-role", choices=["formal", "preflight"], default="formal",
        help="Preflight is non-formal and may evaluate only 1--2 benchmark tasks.",
    )
    parser.add_argument("--preflight-tasks", type=int)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--supervised-checkpoint", type=Path, required=True)
    parser.add_argument("--u96-checkpoint", type=Path, required=True)
    parser.add_argument("--u96-receipt", type=Path, required=True)
    parser.add_argument("--c2-checkpoint", type=Path)
    parser.add_argument("--c2-receipt", type=Path)
    parser.add_argument("--c3-last2-checkpoint", type=Path)
    parser.add_argument("--c3-last2-receipt", type=Path)
    parser.add_argument("--c3-lora-checkpoint", type=Path)
    parser.add_argument("--c3-lora-receipt", type=Path)
    parser.add_argument("--d3-terminal-checkpoint", type=Path)
    parser.add_argument("--d3-terminal-receipt", type=Path)
    parser.add_argument("--d3-contract", type=Path)
    parser.add_argument("--d4-terminal-checkpoint", type=Path)
    parser.add_argument("--d4-terminal-receipt", type=Path)
    parser.add_argument("--d4-contract", type=Path)
    parser.add_argument("--d5-terminal-checkpoint", type=Path)
    parser.add_argument("--d5-terminal-receipt", type=Path)
    parser.add_argument("--d5-contract", type=Path)
    parser.add_argument("--d4-failure-gate", type=Path)
    parser.add_argument("--scale-checkpoint", type=Path)
    parser.add_argument("--scale-receipt", type=Path)
    parser.add_argument("--scale-contract", type=Path)
    parser.add_argument("--scale-source-checkpoint", type=Path)
    parser.add_argument("--scale-source-receipt", type=Path)
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--rnaernie-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-workers", type=int, default=1)
    args = parser.parse_args()

    if args.arm == "d4_hybrid_u128":
        raise ValueError(
            "D4 U128 is TRAIN-monitor-only; frozen Eterna requires d4_hybrid_u256"
        )
    if args.arm == "d5_strict_on_policy_u128":
        raise ValueError(
            "D5 U128 is TRAIN-monitor-only; frozen Eterna requires d5_strict_on_policy_u256"
        )

    if sha256(FAIR_DIR / "evaluate.py") != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("D0 frozen evaluator bytes changed")
    if sha256(FAIR_DIR / "model.py") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("D0 frozen model bytes changed")
    if sha256(args.benchmark) != EXPECTED_ETERNA_SHA256:
        raise RuntimeError("D0 benchmark is not frozen Eterna100-v2")
    if args.output.exists():
        raise FileExistsError("D0 evaluation output must be fresh")
    if args.eval_workers <= 0:
        raise ValueError("eval workers must be positive")
    if args.evaluation_role == "formal" and args.preflight_tasks is not None:
        raise ValueError("formal D0 evaluation cannot use --preflight-tasks")
    if args.evaluation_role == "preflight" and args.preflight_tasks not in {1, 2}:
        raise ValueError("D0 preflight requires --preflight-tasks 1 or 2")
    if args.arm in D4_GROUP_OBJECTIVES and args.evaluation_role != "formal":
        raise ValueError("D4 Eterna evaluation is terminal-only and cannot run preflight")
    if args.arm in D5_GROUP_OBJECTIVES and args.evaluation_role != "formal":
        raise ValueError("D5 Eterna evaluation is terminal-only and cannot run preflight")

    supervised = torch.load(args.supervised_checkpoint, map_location="cpu", weights_only=False)
    supervised_contract = validate_supervised_checkpoint(supervised)
    if (
        args.rnaernie_revision != supervised_contract.get("rnaernie_revision")
        or sha256(args.rnaernie / "model.safetensors") != supervised_contract.get("rnaernie_weight_sha256")
    ):
        raise RuntimeError("D0 supervised checkpoint/backbone identity mismatch")
    supervised_sha256 = sha256(args.supervised_checkpoint)
    u96 = torch.load(args.u96_checkpoint, map_location="cpu", weights_only=False)
    u96_receipt = _read_receipt(args.u96_receipt)
    u96_contract = validate_u96(u96, u96_receipt, args.u96_checkpoint, supervised_sha256)
    u96_checkpoint_sha256 = sha256(args.u96_checkpoint)
    u96_receipt_sha256 = sha256(args.u96_receipt)
    u96_contract_sha256 = u96["contract_sha256"]

    all_tasks = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    if len(all_tasks) != 100:
        raise RuntimeError("D0 Eterna100-v2 task coverage changed")
    tasks = all_tasks if args.evaluation_role == "formal" else all_tasks[:args.preflight_tasks]
    device = torch.device(args.device)
    source_paths: dict[str, Path] = {
        "benchmark": args.benchmark, "supervised_checkpoint": args.supervised_checkpoint,
        "u96_checkpoint": args.u96_checkpoint, "u96_receipt": args.u96_receipt,
        "rnaernie_model": args.rnaernie / "model.safetensors",
        "fair_evaluate": FAIR_DIR / "evaluate.py", "fair_model": FAIR_DIR / "model.py",
        "endpoint_policy": PROGRESSIVE_DIR / "endpoint_policy.py", "adapter": Path(__file__),
    }
    candidate = None
    adaptation = "u96"
    arm_contract = u96_contract
    if args.arm == "c2_u128":
        if args.c2_checkpoint is None or args.c2_receipt is None:
            raise ValueError("C2 D0 evaluation requires --c2-checkpoint and --c2-receipt")
        candidate = torch.load(args.c2_checkpoint, map_location="cpu", weights_only=False)
        arm_contract = validate_c2(candidate, _read_receipt(args.c2_receipt), args.c2_checkpoint,
                                   supervised_sha256=supervised_sha256,
                                   u96_checkpoint_sha256=u96_checkpoint_sha256,
                                   u96_receipt_sha256=u96_receipt_sha256,
                                   u96_contract_sha256=u96_contract_sha256)
        adaptation = "c2"
        source_paths.update({"c2_checkpoint": args.c2_checkpoint, "c2_receipt": args.c2_receipt})
    elif args.arm == "c3_last2_u698":
        if args.c3_last2_checkpoint is None or args.c3_last2_receipt is None:
            raise ValueError("C3 last2 D0 evaluation requires its checkpoint and receipt")
        candidate = torch.load(args.c3_last2_checkpoint, map_location="cpu", weights_only=False)
        arm_contract = validate_c3(candidate, _read_receipt(args.c3_last2_receipt), args.c3_last2_checkpoint,
                                   supervised_sha256=supervised_sha256,
                                   u96_checkpoint_sha256=u96_checkpoint_sha256,
                                   u96_receipt_sha256=u96_receipt_sha256,
                                   u96_contract_sha256=u96_contract_sha256, adaptation="last2")
        adaptation = "last2"
        source_paths.update({"c3_last2_checkpoint": args.c3_last2_checkpoint,
                             "c3_last2_receipt": args.c3_last2_receipt})
    elif args.arm == "c3_lora_u698":
        if args.c3_lora_checkpoint is None or args.c3_lora_receipt is None:
            raise ValueError("C3 LoRA-M D0 evaluation requires its checkpoint and receipt")
        candidate = torch.load(args.c3_lora_checkpoint, map_location="cpu", weights_only=False)
        arm_contract = validate_c3(candidate, _read_receipt(args.c3_lora_receipt), args.c3_lora_checkpoint,
                                   supervised_sha256=supervised_sha256,
                                   u96_checkpoint_sha256=u96_checkpoint_sha256,
                                   u96_receipt_sha256=u96_receipt_sha256,
                                   u96_contract_sha256=u96_contract_sha256, adaptation="lora_m")
        adaptation = "lora_m"
        source_paths.update({"c3_lora_checkpoint": args.c3_lora_checkpoint,
                             "c3_lora_receipt": args.c3_lora_receipt})
    elif args.arm in D3_GROUP_OBJECTIVES:
        if (
            args.d3_terminal_checkpoint is None
            or args.d3_terminal_receipt is None
            or args.d3_contract is None
        ):
            raise ValueError(
                "D3 terminal D0 evaluation requires checkpoint, receipt, and immutable contract"
            )
        candidate = torch.load(args.d3_terminal_checkpoint, map_location="cpu", weights_only=False)
        d3_contract, d3_contract_sha256 = _read_d3_contract(args.d3_contract)
        arm_contract = validate_d3_terminal(
            candidate, _read_receipt(args.d3_terminal_receipt), args.d3_terminal_checkpoint,
            d3_contract, d3_contract_sha256, arm=args.arm,
            supervised_sha256=supervised_sha256,
            supervised_contract_sha256=supervised["contract_sha256"],
            rnaernie_weight_sha256=sha256(args.rnaernie / "model.safetensors"),
            u96_checkpoint_sha256=u96_checkpoint_sha256,
            u96_receipt_sha256=u96_receipt_sha256,
            u96_contract_sha256=u96_contract_sha256,
        )
        adaptation = "d3"
        source_paths.update({
            "d3_terminal_checkpoint": args.d3_terminal_checkpoint,
            "d3_terminal_receipt": args.d3_terminal_receipt,
            "d3_contract": args.d3_contract,
        })
    elif args.arm in D4_GROUP_OBJECTIVES:
        d4_checkpoint = args.d4_terminal_checkpoint or args.d3_terminal_checkpoint
        d4_receipt = args.d4_terminal_receipt or args.d3_terminal_receipt
        d4_contract_path = args.d4_contract or args.d3_contract
        if d4_checkpoint is None or d4_receipt is None or d4_contract_path is None:
            raise ValueError(
                "D4 terminal D0 evaluation requires checkpoint, receipt, and immutable contract"
            )
        candidate = torch.load(d4_checkpoint, map_location="cpu", weights_only=False)
        d4_contract, d4_contract_sha256 = _read_d3_contract(d4_contract_path)
        arm_contract = validate_d4_terminal(
            candidate, _read_receipt(d4_receipt), d4_checkpoint,
            d4_contract, d4_contract_sha256, arm=args.arm,
            supervised_sha256=supervised_sha256,
            supervised_contract_sha256=supervised["contract_sha256"],
            rnaernie_weight_sha256=sha256(args.rnaernie / "model.safetensors"),
            u96_checkpoint_sha256=u96_checkpoint_sha256,
            u96_receipt_sha256=u96_receipt_sha256,
            u96_contract_sha256=u96_contract_sha256,
        )
        adaptation = "d3"
        source_paths.update({
            "d4_terminal_checkpoint": d4_checkpoint,
            "d4_terminal_receipt": d4_receipt,
            "d4_contract": d4_contract_path,
        })
    elif args.arm in D5_GROUP_OBJECTIVES:
        d5_checkpoint = args.d5_terminal_checkpoint or args.d3_terminal_checkpoint
        d5_receipt = args.d5_terminal_receipt or args.d3_terminal_receipt
        d5_contract_path = args.d5_contract or args.d3_contract
        if d5_checkpoint is None or d5_receipt is None or d5_contract_path is None:
            raise ValueError(
                "D5 terminal D0 evaluation requires checkpoint, receipt, and immutable contract"
            )
        if args.d4_failure_gate is None:
            raise ValueError("D5 terminal Eterna requires --d4-failure-gate")
        validate_d4_failure_gate_for_d5(args.d4_failure_gate)
        candidate = torch.load(d5_checkpoint, map_location="cpu", weights_only=False)
        d5_contract, d5_contract_sha256 = _read_d3_contract(d5_contract_path)
        arm_contract = validate_d5_terminal(
            candidate, _read_receipt(d5_receipt), d5_checkpoint,
            d5_contract, d5_contract_sha256, arm=args.arm,
            supervised_sha256=supervised_sha256,
            supervised_contract_sha256=supervised["contract_sha256"],
            rnaernie_weight_sha256=sha256(args.rnaernie / "model.safetensors"),
            u96_checkpoint_sha256=u96_checkpoint_sha256,
            u96_receipt_sha256=u96_receipt_sha256,
            u96_contract_sha256=u96_contract_sha256,
        )
        adaptation = "d3"
        source_paths.update({
            "d5_terminal_checkpoint": d5_checkpoint,
            "d5_terminal_receipt": d5_receipt,
            "d5_contract": d5_contract_path,
            "d4_terminal_gate": args.d4_failure_gate,
        })
    elif args.arm in COVERAGE_SCALE_ARMS:
        scale_paths = (
            args.scale_checkpoint,
            args.scale_receipt,
            args.scale_contract,
            args.scale_source_checkpoint,
            args.scale_source_receipt,
        )
        if any(path is None for path in scale_paths):
            raise ValueError(
                "coverage scale evaluation requires terminal/source checkpoint+receipt and contract"
            )
        candidate = torch.load(args.scale_checkpoint, map_location="cpu", weights_only=False)
        scale_contract, scale_contract_sha256 = _read_d3_contract(args.scale_contract)
        scale_kind = COVERAGE_SCALE_ARMS[args.arm]
        arm_contract = validate_coverage_scale_terminal(
            candidate,
            _read_receipt(args.scale_receipt),
            args.scale_checkpoint,
            scale_contract,
            scale_contract_sha256,
            kind=scale_kind,
            source_checkpoint_path=args.scale_source_checkpoint,
            source_receipt_path=args.scale_source_receipt,
            supervised_sha256=supervised_sha256,
            supervised_contract_sha256=supervised["contract_sha256"],
            rnaernie_weight_sha256=sha256(args.rnaernie / "model.safetensors"),
            u96_checkpoint_sha256=u96_checkpoint_sha256,
            u96_receipt_sha256=u96_receipt_sha256,
            u96_contract_sha256=u96_contract_sha256,
        )
        adaptation = (
            "last2"
            if scale_kind in {"c3_official2790", "c3_official2790_d5"}
            else "d3"
        )
        source_paths.update({
            "scale_checkpoint": args.scale_checkpoint,
            "scale_receipt": args.scale_receipt,
            "scale_contract": args.scale_contract,
            "scale_source_checkpoint": args.scale_source_checkpoint,
            "scale_source_receipt": args.scale_source_receipt,
        })
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    model = load_model_for_arm(supervised=supervised, supervised_contract=supervised_contract,
                               u96=u96, candidate=candidate, adaptation=adaptation,
                               rnaernie=args.rnaernie, device=device)
    summary = evaluate_arm(name=args.arm, model=model, tasks=tasks, output=args.output,
                           device=device, workers=args.eval_workers,
                           source_hashes=source_hashes, arm_contract=arm_contract,
                           evaluation_role=args.evaluation_role)
    print(json.dumps({"status": summary["status"], "arm": args.arm,
                      "bad_count": summary["bad_count"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
