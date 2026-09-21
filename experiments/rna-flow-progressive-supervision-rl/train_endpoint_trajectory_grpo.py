"""Exact-resume Simplex Endpoint-Policy trajectory GRPO."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import RNA
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

FAIR_DIR = Path(__file__).resolve().parents[1] / "rna-flow-fair-components"
LEGACY_DIR = Path(__file__).resolve().parents[1] / "dual-prior-rna-flow"
sys.path.insert(0, str(LEGACY_DIR))
sys.path.insert(0, str(FAIR_DIR))
from constraints import constrained_decode  # noqa: E402
from evaluate import evaluate_candidate, pair_f1, sha256  # noqa: E402
from model import configure_rl_trainable_scope  # noqa: E402
try:
    from model import adapt_loaded_supervised_model_with_lora  # noqa: E402
except ImportError:  # Minimal model stubs used by DDP lifecycle tests.
    adapt_loaded_supervised_model_with_lora = None
from train import (  # noqa: E402
    FlowDataset,
    capture_rng_state,
    collate,
    restore_rng_state,
)

from endpoint_policy import (  # noqa: E402
    endpoint_policy_entropy,
    endpoint_policy_kl,
    factorized_trajectory_grpo_loss,
    recompute_trajectory_unit_log_probabilities,
    rollout_endpoint_trajectory,
    structure_error_credit_weights,
    temporal_credit_selection,
    validate_endpoint_trajectory,
)
from pkpo import continuous_maxk_advantages, pkpo_schedule_k  # noqa: E402
try:
    from endpoint_policy import (  # noqa: E402
        endpoint_units, structured_discrete_reference_transition_tv,
    )
except ImportError:  # Minimal endpoint-policy stubs used by DDP lifecycle tests.
    def structured_discrete_reference_transition_tv(*args, **kwargs):
        raise RuntimeError("D3 reference-transition TV requires the concrete endpoint policy")

    def endpoint_units(structure: str):
        if not isinstance(structure, str) or set(structure) - set("()."):
            raise ValueError("invalid dot-bracket structure")
        return tuple((index,) for index in range(len(structure)))
try:
    from endpoint_policy import (
        aggregate_step_local_values,
        domino_endpoint_ppo_loss,
        recompute_discrete_domino_unit_log_probabilities,
        rollout_discrete_domino_trajectory,
        sequence_group_fpo_loss,
        temporal_credit_step_weights,
        validate_discrete_domino_trajectory,
    )
except ImportError:  # Minimal endpoint_policy stubs used by DDP lifecycle tests.
    def aggregate_step_local_values(values, step_weights=None):
        if step_weights is None:
            return values.mean()
        weights = torch.as_tensor(step_weights, dtype=values.dtype, device=values.device)
        return (values * weights[None, :]).sum(dim=1).mean()

    def temporal_credit_step_weights(
        mode, *, selected_steps, importance_weight,
    ):
        if mode == "all_steps":
            return None
        if isinstance(importance_weight, (int, float)):
            return (float(importance_weight),)
        return tuple(float(value) for value in importance_weight)

    def sequence_group_fpo_loss(new, old, advantages, clip_ratio):
        return factorized_trajectory_grpo_loss(new, old, advantages, clip_ratio)

    def domino_endpoint_ppo_loss(new, old, advantages, clip_ratio, *, normalize_time=False):
        loss, diagnostics = factorized_trajectory_grpo_loss(
            new, old, advantages, clip_ratio
        )
        if not normalize_time:
            loss = loss * new.shape[1]
        return loss, diagnostics | {
            "objective": "domino-endpoint-ppo",
            "time_aggregation": "mean" if normalize_time else "sum",
            "native_discrete_flow_transition_exact": False,
        }

    def recompute_discrete_domino_unit_log_probabilities(model, trajectory, device):
        return recompute_trajectory_unit_log_probabilities(model, trajectory, device)

    def rollout_discrete_domino_trajectory(
        model, structure, candidates, steps, seed, device, temperature=1.0,
    ):
        return rollout_endpoint_trajectory(
            model, structure, candidates, steps, seed, device, temperature
        )

    def validate_discrete_domino_trajectory(*args, **kwargs):
        return validate_endpoint_trajectory(*args, **kwargs)
from finalize_supervised_selection import validate_authorized_checkpoint  # noqa: E402
from reward_cache import (  # noqa: E402
    MAXIMUM_INVALID_REWARD_EVALUATION_RATE,
    REWARD_EVALUATION_POLICY,
    ViennaRewardCache,
    cache_key,
    empty_reward_evaluation_coverage,
    finalize_reward_evaluation_coverage,
    safe_reward_evaluation,
    update_reward_evaluation_coverage,
    validate_reward_evaluation_coverage,
)
from vienna_target_contract import TARGET_LENGTH_BINS, validate_target_contract  # noqa: E402
from rl_primitives import (  # noqa: E402
    normalized_group_advantages,
    official_terminal_reward,
    pair_credit_terminal_reward,
    pair_rival_terminal_reward,
    stepwise_pair_advantages,
)
from run_rl_objective_debug import atomic_json, contract_sha256, load_model  # noqa: E402
from train_terminal_grpo import (  # noqa: E402
    append_jsonl,
    atomic_jsonl,
    reconcile_history,
    task_batch_indices,
)
from endpoint_trajectory_ddp import (  # noqa: E402
    CONTRASTIVE_TASK_STEP_METHOD,
    CONTRASTIVE_TASK_STEP_RATIO,
    DOMINO_ENDPOINT_PPO_METHOD,
    DOMINO_ENDPOINT_PPO_RATIO,
    DISCRETE_DOMINO_METHOD,
    DISCRETE_DOMINO_RATIO,
    SEQUENCE_GROUP_FPO_METHOD,
    SEQUENCE_GROUP_FPO_RATIO,
    branch_rank_rng_state_from_checkpoint,
    branch_resume_implementation_changes,
    checkpoint_file_sha256,
    contrastive_task_step_loss,
    contrastive_task_step_objective_contract,
    discrete_domino_objective_contract,
    domino_endpoint_ppo_objective_contract,
    distributed_context,
    gather_objects,
    gather_rank_rng_states,
    global_sum,
    global_task_mean_scale,
    global_max_memory,
    global_token_mean_scale,
    keyed_dirichlet_path,
    local_supervised_indices,
    load_rank_local_resume_checkpoint,
    optimizer_parameter_manifest_from_model,
    optimizer_parameter_manifest_sha256,
    primary_call,
    rank_local_call,
    rank_rng_state_from_checkpoint,
    rank_task_offsets,
    require_even_global_batches,
    restore_branch_optimizer_state,
    validate_branch_resume_checkpoint,
    validate_branch_resume_contract,
    validate_branch_resume_preflight_receipt,
    validate_branch_resume_review_receipt,
    validate_branch_resume_stage_arguments,
    validate_rank_rng_states,
    validated_correctness_tiers,
    sequence_group_fpo_objective_contract,
)


IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(__file__).with_name("endpoint_policy.py"),
    Path(__file__).with_name("endpoint_trajectory_ddp.py"),
    Path(__file__).with_name("finalize_supervised_selection.py"),
    Path(__file__).with_name("reward_cache.py"),
    Path(__file__).with_name("rl_primitives.py"),
    Path(__file__).with_name("run_rl_objective_debug.py"),
    Path(__file__).with_name("train_terminal_grpo.py"),
    Path(__file__).with_name("vienna_target_contract.py"),
    LEGACY_DIR / "constraints.py",
    FAIR_DIR / "evaluate.py",
    FAIR_DIR / "model.py",
    FAIR_DIR / "train.py",
)


def implementation_manifest() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[2]
    return {
        str(path.resolve().relative_to(repository_root)): sha256(path)
        for path in IMPLEMENTATION_FILES
    }


def require_clean_formal_source(repository_root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root, check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise RuntimeError("formal trajectory RL requires a clean source worktree")


def atomic_checkpoint(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


ONLINE_CURRICULUM_ALGORITHM = "terminal-reward-moments-weighted-top-k-v1"
ONLINE_CURRICULUM_MIN_EXPLORATION_PROBABILITY = 0.1

D3_TASKS_SHA256 = "7c6ee06aaaa24b0b67ce177af90c6eabae920eadbb01d036189afb542055ce61"
D3_TASKS_MANIFEST_SHA256 = "9a282be638934e0588862e852344181a41f579627c33bee50e9516f786648fa8"
D3_MONITOR_TASKS = 48
D3_SELECTED_ROWS = 4143
D3_SELECTED_LENGTH_BINS = {
    "000-064": 1101, "065-128": 1667, "129-256": 802, "257-510": 573,
}
D3_INVALID_POLICY = {
    "reject_task_if_any_candidate_invalid": True,
    "required_evaluation_valid": True, "required_evaluation_error": None,
    "required_target_probability_raw": "finite and present",
    "safe_zero_fallback_allowed": False, "invalid_candidate_count": 13,
    "invalid_task_count": 4,
}
D4_HYBRID_PKPO_OBJECTIVE = "thermo_pkpo_residual"
D5_ROLLOUT_REFRESH_MODE = "per_policy_epoch"


def d5_rollout_refresh_contract() -> dict:
    """Immutable strict-on-policy geometry for one logical D5 update."""
    return {
        "logical_update_task_cursor_advance": 4,
        "optimizer_steps_per_logical_update": 2,
        "refreshes_per_logical_update": 2,
        "groups_per_logical_update": 8,
        "candidates_per_logical_update": 64,
        "checkpoint_boundary": "after-both-refreshes",
        "resume": "exact-logical-update-boundary-only",
    }


def validate_d5_logical_update_geometry(
    refreshes: list[dict], policy_epoch_diagnostics: list[dict], *,
    tasks_per_update: int, candidates: int,
) -> None:
    """Fail closed unless both D5 behavior snapshots form one full logical update."""
    if (
        len(refreshes) != 2
        or len(policy_epoch_diagnostics) != 2
        or tasks_per_update != 4
        or candidates != 8
    ):
        raise RuntimeError("D5 logical update geometry is incomplete")
    paths: set[str] = set()
    task_ids_by_refresh: list[list[str]] = []
    for refresh_index, (refresh, diagnostics) in enumerate(
        zip(refreshes, policy_epoch_diagnostics)
    ):
        rollouts = refresh.get("rollouts")
        rewards = refresh.get("rewards")
        if (
            refresh.get("refresh_index") != refresh_index
            or not isinstance(rollouts, list)
            or not isinstance(rewards, list)
            or len(rollouts) != tasks_per_update
            or len(rewards) != tasks_per_update * candidates
            or any(not isinstance(rollout, dict) for rollout in rollouts)
            or [int(rollout.get("task_offset", -1)) for rollout in rollouts]
            != list(range(tasks_per_update))
            or any(rollout.get("refresh_index") != refresh_index for rollout in rollouts)
            or float(diagnostics.get("ratio_max_abs_error", float("inf"))) > 1e-5
        ):
            raise RuntimeError("D5 logical update refresh diagnostics are malformed")
        for rollout in rollouts:
            for key in ("trajectory", "evaluations"):
                path = rollout.get(key)
                if not isinstance(path, str) or path in paths:
                    raise RuntimeError("D5 refresh artifact paths are not independent")
                paths.add(path)
        task_ids = [rollout.get("task_id") for rollout in rollouts]
        if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
            raise RuntimeError("D5 refresh task identities are malformed")
        task_ids_by_refresh.append(task_ids)
    if task_ids_by_refresh[0] != task_ids_by_refresh[1]:
        raise RuntimeError("D5 refreshes do not retain the same logical tasks")


def d5_group_weighted_drift_totals(
    policy_epoch_diagnostics: list[dict], *, groups_per_refresh: int,
) -> dict[str, float]:
    """Turn both refresh-level means into group-weighted terminal sums."""
    keys = ("posterior_tv", "transition_tv", "reference_tv_contribution")
    if len(policy_epoch_diagnostics) != 2 or groups_per_refresh <= 0:
        raise RuntimeError("D5 drift diagnostics do not cover both refreshes")
    totals = {}
    for key in keys:
        values = [float(item.get(key, float("nan"))) for item in policy_epoch_diagnostics]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"D5 non-finite refresh drift diagnostic: {key}")
        totals[key] = sum(value * groups_per_refresh for value in values)
    return totals


def d3_group_advantages(
    rewards: torch.Tensor,
    group_objective: str,
    pkpo_residual_coefficient: float | None,
    *,
    update: int,
) -> dict[str, torch.Tensor | int | None | bool]:
    """Build the frozen D3/D4 terminal group advantages without mutating rewards.

    D4 deliberately retains the dense, centered thermodynamic GRPO signal and
    adds an *unnormalized* continuous max@k residual.  Keeping this construction
    here makes the sampled reward-to-advantage contract independently testable.
    """
    thermo, thermo_effective = normalized_group_advantages(rewards)
    if group_objective == "thermodynamic_grpo":
        return {
            "thermo": thermo, "raw_pkpo": None, "combined": thermo,
            "pkpo_k": None, "effective": thermo_effective,
        }
    if group_objective == "thermo_pkpo":
        pkpo_k = d3_pkpo_k(update)
        raw_pkpo = continuous_maxk_advantages(rewards, pkpo_k)
        return {
            "thermo": thermo, "raw_pkpo": raw_pkpo, "combined": raw_pkpo,
            "pkpo_k": pkpo_k,
            "effective": bool(float(raw_pkpo.detach().abs().max()) > 1e-6),
        }
    if group_objective != D4_HYBRID_PKPO_OBJECTIVE:
        raise ValueError("unknown D3/D4 group objective")
    if (
        not isinstance(pkpo_residual_coefficient, (int, float))
        or isinstance(pkpo_residual_coefficient, bool)
        or not math.isfinite(float(pkpo_residual_coefficient))
        or float(pkpo_residual_coefficient) <= 0
    ):
        raise ValueError("D4 PKPO residual coefficient must be finite and positive")
    pkpo_k = d3_pkpo_k(update)
    raw_pkpo = continuous_maxk_advantages(rewards, pkpo_k)
    combined = thermo + float(pkpo_residual_coefficient) * raw_pkpo
    if not torch.isfinite(combined).all():
        raise RuntimeError("D4 combined advantages are non-finite")
    return {
        "thermo": thermo, "raw_pkpo": raw_pkpo, "combined": combined,
        "pkpo_k": pkpo_k,
        "effective": bool(float(combined.detach().abs().max()) > 1e-6),
    }


def advantage_summary(advantages: torch.Tensor) -> dict[str, float]:
    """Finite population statistics retained in D4 history/checkpoint receipts."""
    if not isinstance(advantages, torch.Tensor) or advantages.ndim != 1:
        raise ValueError("advantage diagnostics require a one-dimensional tensor")
    if advantages.numel() == 0 or not torch.isfinite(advantages).all():
        raise RuntimeError("advantage diagnostics are non-finite")
    detached = advantages.detach()
    return {
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }


def d3_pkpo_k(update: int) -> int:
    """Public trainer-level schedule alias for checkpoints and tests."""
    return pkpo_schedule_k(update)


def d3_train_monitor_ids(tasks: list[dict]) -> list[str]:
    """Freeze a TRAIN-only health monitor without consulting external sets."""
    identifiers = [str(task["id"]) for task in tasks]
    if len(identifiers) != len(set(identifiers)) or len(identifiers) < D3_MONITOR_TASKS:
        raise RuntimeError("D3 monitor requires at least 48 unique TRAIN task IDs")
    return sorted(identifiers, key=lambda value: hashlib.sha256(value.encode()).hexdigest())[
        :D3_MONITOR_TASKS
    ]


def validate_d3_flow_yrl_tasks_manifest(
    path: Path, tasks_path: Path, tasks: list[dict],
) -> dict:
    """Validate the formal D1 Flow-YRL selector artifact, not the legacy schema."""
    manifest = json.loads(path.read_text())
    selection = manifest.get("selection") if isinstance(manifest, dict) else None
    outputs = manifest.get("outputs") if isinstance(manifest, dict) else None
    flow_tasks = outputs.get("flow_yrl_tasks") if isinstance(outputs, dict) else None
    metric = manifest.get("metric_contract") if isinstance(manifest, dict) else None
    invalid = manifest.get("invalid_policy") if isinstance(manifest, dict) else None
    overlap = manifest.get("overlap") if isinstance(manifest, dict) else None
    expected_metric = {
        "aon": "mean(target_probability)",
        "nsd": "population_std(target_probability, ddof=0) / AoN",
        "aon_threshold": 0.1, "nsd_threshold": 0.5, "aon_inclusive": True,
        "nsd_strict": True, "candidates_per_task": 8,
    }
    if (
        sha256(tasks_path) != D3_TASKS_SHA256
        or sha256(path) != D3_TASKS_MANIFEST_SHA256
        or not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("role") != "TRAIN-only formal D1 Flow-YRL selector"
        or not isinstance(selection, dict)
        or selection.get("selected_rows") != D3_SELECTED_ROWS
        or selection.get("selected_length_bins") != D3_SELECTED_LENGTH_BINS
        or not isinstance(flow_tasks, dict)
        or flow_tasks.get("rows") != D3_SELECTED_ROWS
        or flow_tasks.get("sha256") != D3_TASKS_SHA256
        or metric != expected_metric
        or invalid != D3_INVALID_POLICY
        or not isinstance(overlap, dict)
        or set(overlap) != {"eterna100", "eterna100v2", "rldev512", "rnasolo764", "hard48", "long10"}
        or len(tasks) != D3_SELECTED_ROWS
    ):
        raise RuntimeError("D3 Flow-YRL task manifest is incomplete or drifted")
    for name in ("eterna100", "eterna100v2", "rldev512", "rnasolo764"):
        evidence = overlap[name]
        if (
            not isinstance(evidence, dict)
            or evidence.get("content_read") is not True
            or evidence.get("exact_structure_overlap_unique") != 0
            or evidence.get("exact_structure_overlap_tasks") != 0
        ):
            raise RuntimeError("D3 Flow-YRL external overlap evidence drifted")
    for name, rows in (("hard48", 48), ("long10", 10)):
        evidence = overlap[name]
        if (
            not isinstance(evidence, dict)
            or evidence.get("content_read") is not False
            or evidence.get("path_only_excluded") is not True
            or evidence.get("rows") != rows
        ):
            raise RuntimeError("D3 Flow-YRL path-only exclusion evidence drifted")
    identifiers: set[str] = set()
    structures: set[str] = set()
    for row in tasks:
        if set(row) != {"id", "target_structure"}:
            raise RuntimeError("D3 Flow-YRL task row schema drifted")
        identifier, structure = row["id"], row["target_structure"]
        if not isinstance(identifier, str) or not identifier or not isinstance(structure, str) or not structure:
            raise RuntimeError("D3 Flow-YRL task identity is invalid")
        # ``endpoint_units`` validates balanced dot-bracket syntax and rejects
        # unsupported symbols, without reading any external benchmark content.
        endpoint_units(structure)
        if identifier in identifiers or structure in structures:
            raise RuntimeError("D3 Flow-YRL task IDs or structures are not unique")
        identifiers.add(identifier)
        structures.add(structure)
    source_binding = manifest.get("source", {}).get("source_contract_binding")
    source_revision = (
        source_binding.get("repository_revision")
        if isinstance(source_binding, dict) else None
    )
    if not isinstance(source_revision, str) or not source_revision:
        raise RuntimeError("D3 Flow-YRL source revision is absent or drifted")
    return manifest | {"validated_source_revision": source_revision}


def validate_d3_formal_arguments(args) -> None:
    """Fail closed on the fixed D3-256 formal/preflight launch geometry."""
    d3_init = bool(args.d3_policy_init_checkpoint or args.d3_policy_init_receipt)
    c3_init = bool(
        getattr(args, "c3_policy_init_checkpoint", None)
        or getattr(args, "c3_policy_init_receipt", None)
    )
    pkpo_residual_coefficient = getattr(args, "pkpo_residual_coefficient", None)
    rollout_refresh_mode = getattr(args, "rollout_refresh_mode", None)
    scale_resume_kind = getattr(args, "scale_resume_kind", None)
    c3_d5_scale = c3_init and scale_resume_kind == "c3_official2790_d5"
    if bool(args.d3_policy_init_checkpoint) != bool(args.d3_policy_init_receipt):
        raise ValueError("D3 policy initialization requires checkpoint and receipt together")
    if not d3_init:
        if (
            args.group_objective is not None
            or args.reference_tv_coefficient is not None
            or pkpo_residual_coefficient is not None
            or (rollout_refresh_mode is not None and not c3_d5_scale)
        ):
            raise ValueError("D3/D4 group objective/transition TV/PKPO residual require a frozen initializer")
        if c3_d5_scale and rollout_refresh_mode != D5_ROLLOUT_REFRESH_MODE:
            raise ValueError("C3+D5 scaling requires strict per-policy-epoch rollout refresh")
        return
    residual_objective = args.group_objective == D4_HYBRID_PKPO_OBJECTIVE
    d5_strict_on_policy = rollout_refresh_mode == D5_ROLLOUT_REFRESH_MODE
    t0_scale_resume = scale_resume_kind in {"t0_flow_yrl", "t0_flow_yrl_d5"}
    residual_coefficient_valid = (
        isinstance(pkpo_residual_coefficient, (int, float))
        and not isinstance(pkpo_residual_coefficient, bool)
        and math.isfinite(float(pkpo_residual_coefficient))
        and float(pkpo_residual_coefficient) > 0
    )
    if (
        args.policy_objective != "discrete_domino"
        or args.group_objective not in {
            "thermodynamic_grpo", "thermo_pkpo", D4_HYBRID_PKPO_OBJECTIVE,
        }
        or (residual_objective and not residual_coefficient_valid)
        or (not residual_objective and pkpo_residual_coefficient is not None)
        or rollout_refresh_mode not in {None, D5_ROLLOUT_REFRESH_MODE}
        or (d5_strict_on_policy and args.group_objective != "thermodynamic_grpo")
        or args.formal_rl is not True
        or (
            args.updates != 256
            and not (t0_scale_resume and args.updates in {623, 1036, 1554, 2072, 2590, 3108, 3626, 4144})
        )
        or args.tasks_per_update != 4
        or args.candidates != 8
        or args.trajectory_steps != 8
        or args.policy_epochs != 2
        or args.temporal_credit != "all_steps"
        or args.reward != "terminal"
        or args.credit_assignment != "uniform"
        or args.task_curriculum != "deterministic_shuffle"
        or args.seed != 1009
        or args.rl_trainable_scope != "last_2_backbone_and_head"
        or args.gradient_checkpointing
        or not math.isclose(args.learning_rate, 5e-6, rel_tol=0, abs_tol=1e-15)
        or not math.isclose(args.kl_coefficient, 0.01, rel_tol=0, abs_tol=1e-15)
        or not math.isclose(args.ce_coefficient, 0.1, rel_tol=0, abs_tol=1e-15)
        or args.reference_tv_coefficient is None
        or not math.isfinite(args.reference_tv_coefficient)
        or args.reference_tv_coefficient < 0
        or (args.reference_tv_coefficient == 0 and args.device.startswith("cuda"))
        or (args.preflight_updates is not None and args.preflight_updates != 1)
    ):
        raise ValueError("D3-256 formal/preflight contract is invalid")


def validate_d3_resume_payload(
    payload: dict, expected_contract: dict, expected_contract_sha256: str, world_size: int,
) -> None:
    """Strict D3 counterpart to legacy U128 resume plumbing.

    D3 is U0..U256 with eight candidates, so the fixed C1 U96..U128 helper
    cannot be reused without weakening either contract.  This validator binds
    the immutable D3 contract, AdamW coverage/state, rank RNG container and
    deterministic task cursor before any model state is restored.
    """
    contract = payload.get("contract") if isinstance(payload, dict) else None
    state = payload.get("state") if isinstance(payload, dict) else None
    optimizer = payload.get("optimizer") if isinstance(payload, dict) else None
    next_update = state.get("next_update") if isinstance(state, dict) else None
    coverage = state.get("reward_evaluation_coverage") if isinstance(state, dict) else None
    histogram = state.get("temporal_credit_step_histogram") if isinstance(state, dict) else None
    d5_strict_on_policy = contract.get("rollout_refresh_mode") == D5_ROLLOUT_REFRESH_MODE
    groups_per_update = 8 if d5_strict_on_policy else 4
    if (
        contract != expected_contract
        or payload.get("contract_sha256") != expected_contract_sha256
        or not isinstance(state, dict)
        or not isinstance(next_update, int)
        or not 0 < next_update <= 256
        or state.get("task_epoch") is None
        or state.get("task_cursor") != next_update * 4
        or not isinstance(coverage, dict)
        or coverage.get("candidate_rows") != next_update * groups_per_update * 8
        or coverage.get("groups") != next_update * groups_per_update
        or coverage.get("invalid_candidate_rows") != 0
        or coverage.get("invalid_update_histogram") != {str(update): 0 for update in range(next_update)}
        or histogram != {str(step): next_update * groups_per_update for step in range(8)}
        or not isinstance(state.get("d3_health"), dict)
        or not isinstance(payload.get("trainable_model"), dict)
        or not payload["trainable_model"]
        or not isinstance(payload.get("optimizer_parameter_manifest"), list)
        or not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("state"), dict)
        or not isinstance(optimizer.get("param_groups"), list)
        or len(payload.get("rng_state_by_rank", [])) != world_size
        or contract.get("distributed", {}).get("world_size") != world_size
        or contract.get("group_objective") not in {
            "thermodynamic_grpo", "thermo_pkpo", D4_HYBRID_PKPO_OBJECTIVE,
        }
        or contract.get("reference_transition_tv", {}).get("reference_detached") is not True
    ):
        raise RuntimeError("D3 resume checkpoint is incomplete or foreign")
    if d5_strict_on_policy:
        d5_checkpoint = payload.get("d5")
        if (
            contract.get("group_objective") != "thermodynamic_grpo"
            or contract.get("policy_epochs") != 2
            or contract.get("rollout_refresh") != d5_rollout_refresh_contract()
            or not isinstance(d5_checkpoint, dict)
            or d5_checkpoint.get("logical_update_boundary") != next_update
            or d5_checkpoint.get("task_cursor") != next_update * 4
            or d5_checkpoint.get("optimizer_steps") != next_update * 2
            or d5_checkpoint.get("coverage") != coverage
            or d5_checkpoint.get("temporal_credit_step_histogram") != histogram
        ):
            raise RuntimeError("D5 resume checkpoint strict on-policy contract is malformed")
        validate_rank_rng_states(payload.get("rng_state_by_rank"), world_size, require_full_state=True)
    elif (
        "rollout_refresh_mode" in contract or "rollout_refresh" in contract
        or "d5" in payload
    ):
        raise RuntimeError("D3 resume checkpoint unexpectedly contains a rollout refresh mode")
    residual = contract.get("pkpo_residual")
    if contract.get("group_objective") == D4_HYBRID_PKPO_OBJECTIVE:
        coefficient = residual.get("coefficient") if isinstance(residual, dict) else None
        if (
            not isinstance(coefficient, (int, float))
            or isinstance(coefficient, bool)
            or not math.isfinite(float(coefficient))
            or float(coefficient) <= 0
            or payload.get("pkpo_residual") != residual
        ):
            raise RuntimeError("D4 resume checkpoint PKPO residual contract is malformed")
    elif residual is not None or "pkpo_residual" in payload:
        raise RuntimeError("D3 resume checkpoint unexpectedly contains a PKPO residual")
    parameter_ids = [parameter for group in optimizer["param_groups"] for parameter in group.get("params", [])]
    if len(parameter_ids) != len(payload["optimizer_parameter_manifest"]) or set(parameter_ids) != set(optimizer["state"]):
        raise RuntimeError("D3 resume AdamW parameter coverage is incomplete")
    expected_step = next_update * int(contract["policy_epochs"])
    for parameter_id in parameter_ids:
        member = optimizer["state"].get(parameter_id)
        step = member.get("step") if isinstance(member, dict) else None
        if isinstance(step, torch.Tensor):
            step = step.item() if step.numel() == 1 else None
        if (
            step != expected_step
            or not isinstance(member, dict)
            or any(
                not isinstance(member.get(name), torch.Tensor)
                or not torch.isfinite(member[name]).all()
                for name in ("exp_avg", "exp_avg_sq")
            )
        ):
            raise RuntimeError("D3 resume AdamW state is malformed")


def validate_scale_resume_source(
    payload: dict,
    receipt: dict,
    checkpoint: Path,
    kind: str,
    target_updates: int,
    world_size: int,
) -> dict:
    """Validate a frozen C3/T0 boundary for fresh-output exact scaling continuation."""
    contract = payload.get("contract") if isinstance(payload, dict) else None
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(contract, dict) or not isinstance(state, dict):
        raise RuntimeError("scale-resume source checkpoint is malformed")
    source_hash = contract_sha256(contract)
    next_update = state.get("next_update")
    if (
        payload.get("contract_sha256") != source_hash
        or receipt.get("contract_sha256") != source_hash
        or receipt.get("scientific_contract_sha256") != source_hash
        or receipt.get("world_size") != world_size
        or receipt.get("updates_complete") != next_update
        or receipt.get("bad_count") != 0
        or receipt.get("nan_inf_count") != 0
        or not isinstance(receipt.get("last_checkpoint"), dict)
        or receipt["last_checkpoint"].get("next_update") != next_update
        or receipt["last_checkpoint"].get("sha256") != sha256(checkpoint)
        or target_updates <= int(next_update or -1)
    ):
        raise RuntimeError("scale-resume source receipt/checkpoint binding is invalid")
    validate_rank_rng_states(
        payload.get("rng_state_by_rank"), world_size, require_full_state=True
    )
    if kind in {"t0_flow_yrl", "t0_flow_yrl_d5"}:
        strict_on_policy = kind == "t0_flow_yrl_d5"
        coverage = state.get("reward_evaluation_coverage")
        histogram = state.get("temporal_credit_step_histogram")
        task_positions = int(next_update or 0) * 4
        groups_per_update = 8 if strict_on_policy else 4
        optimizer = payload.get("optimizer")
        source_boundary_valid = (
            isinstance(next_update, int)
            and next_update in {256, 623, 1036, 1554, 2072, 2590, 3108, 3626}
        )
        if (
            not source_boundary_valid
            or contract.get("method") != DISCRETE_DOMINO_METHOD
            or contract.get("group_objective") != "thermodynamic_grpo"
            or contract.get("tasks_per_update") != 4
            or contract.get("candidates_per_task") != 8
            or contract.get("policy_epochs") != 2
            or contract.get("reward") != "terminal"
            or contract.get("credit_assignment") != "uniform"
            or contract.get("rl_trainable_scope") != "last_2_backbone_and_head"
            or contract.get("reference_transition_tv", {}).get("coefficient") != 4.0
            or contract.get("reference_transition_tv", {}).get("reference_detached") is not True
            or (
                contract.get("rollout_refresh_mode") != D5_ROLLOUT_REFRESH_MODE
                if strict_on_policy else contract.get("rollout_refresh_mode") is not None
            )
            or (
                contract.get("rollout_refresh") != d5_rollout_refresh_contract()
                if strict_on_policy else contract.get("rollout_refresh") is not None
            )
            or state.get("task_epoch") != task_positions // 4143
            or state.get("task_cursor") != task_positions % 4143
            or not isinstance(coverage, dict)
            or coverage.get("candidate_rows") != next_update * groups_per_update * 8
            or coverage.get("groups") != next_update * groups_per_update
            or coverage.get("invalid_candidate_rows") != 0
            or histogram != {
                str(step): next_update * groups_per_update for step in range(8)
            }
            or not isinstance(state.get("d3_health"), dict)
            or not isinstance(payload.get("trainable_model"), dict)
            or not payload["trainable_model"]
            or not isinstance(payload.get("optimizer_parameter_manifest"), list)
            or not isinstance(optimizer, dict)
            or not isinstance(optimizer.get("state"), dict)
            or not isinstance(optimizer.get("param_groups"), list)
        ):
            raise RuntimeError("T0 Flow-YRL scale-resume source is incomplete or foreign")
        d5_checkpoint = payload.get("d5")
        if strict_on_policy:
            if (
                not isinstance(d5_checkpoint, dict)
                or d5_checkpoint.get("logical_update_boundary") != next_update
                or d5_checkpoint.get("task_cursor") != task_positions % 4143
                or d5_checkpoint.get("optimizer_steps") != next_update * 2
                or d5_checkpoint.get("coverage") != coverage
                or d5_checkpoint.get("temporal_credit_step_histogram") != histogram
            ):
                raise RuntimeError("D5 Flow-YRL scale-resume boundary is incomplete or foreign")
        elif d5_checkpoint is not None:
            raise RuntimeError("plain T0 Flow-YRL scale-resume unexpectedly contains D5 state")
        parameter_ids = [
            parameter for group in optimizer["param_groups"] for parameter in group.get("params", [])
        ]
        if len(parameter_ids) != len(payload["optimizer_parameter_manifest"]) or set(parameter_ids) != set(optimizer["state"]):
            raise RuntimeError("T0 Flow-YRL scale-resume AdamW coverage is incomplete")
        for parameter_id in parameter_ids:
            member = optimizer["state"][parameter_id]
            step = member.get("step") if isinstance(member, dict) else None
            if isinstance(step, torch.Tensor):
                step = step.item() if step.numel() == 1 else None
            if step != next_update * 2:
                raise RuntimeError("T0 Flow-YRL scale-resume AdamW step is inconsistent")
    elif kind in {"c3_official2790", "c3_official2790_d5"}:
        strict_requested = kind == "c3_official2790_d5"
        initialization = contract.get("policy_initialization")
        coverage = state.get("reward_evaluation_coverage")
        histogram = state.get("temporal_credit_step_histogram")
        task_positions = int(next_update or 0) * 4
        source_strict = contract.get("rollout_refresh_mode") == D5_ROLLOUT_REFRESH_MODE
        source_boundary_valid = (
            next_update in {698, 1395}
            if not strict_requested
            else next_update in {1395, 1744, 2093, 2442, 2790, 3139, 3488, 3837}
        )
        expected_source_contract_updates = (
            1395 if not strict_requested else int(next_update or -1)
        )
        if strict_requested and next_update == 1395:
            expected_source_contract_updates = 1395
        expected_groups = (
            task_positions
            if not source_strict
            else 1395 * 4 + (int(next_update) - 1395) * 8
        )
        if (
            not source_boundary_valid
            or contract.get("method") != DISCRETE_DOMINO_METHOD
            or contract.get("updates") != expected_source_contract_updates
            or contract.get("tasks_per_update") != 4
            or contract.get("candidates_per_task") != 8
            or contract.get("policy_epochs") != 2
            or contract.get("reward") != "terminal"
            or contract.get("credit_assignment") != "uniform"
            or contract.get("rl_trainable_scope") != "last_2_backbone_and_head"
            or not isinstance(initialization, dict)
            or initialization.get("mode") != "u96-policy-weights-fresh-optimizer-c3-scale"
            or initialization.get("adaptation") != "last2"
            or state.get("task_epoch") != task_positions // 2790
            or state.get("task_cursor") != task_positions % 2790
            or not isinstance(coverage, dict)
            or coverage.get("candidate_rows") != expected_groups * 8
            or coverage.get("groups") != expected_groups
            or coverage.get("invalid_candidate_rows") != 0
            or histogram != {str(step): expected_groups for step in range(8)}
            or (
                strict_requested
                and next_update != 1395
                and (
                    not source_strict
                    or contract.get("rollout_refresh") != d5_rollout_refresh_contract()
                )
            )
            or (
                strict_requested
                and next_update == 1395
                and (source_strict or contract.get("rollout_refresh") is not None)
            )
        ):
            raise RuntimeError("C3 official2790 scale-resume source is incomplete or foreign")
        d5_checkpoint = payload.get("d5")
        if strict_requested and next_update != 1395:
            if (
                not isinstance(d5_checkpoint, dict)
                or d5_checkpoint.get("logical_update_boundary") != next_update
                or d5_checkpoint.get("task_cursor") != task_positions % 2790
                or d5_checkpoint.get("optimizer_steps") != next_update * 2
                or d5_checkpoint.get("coverage") != coverage
                or d5_checkpoint.get("temporal_credit_step_histogram") != histogram
            ):
                raise RuntimeError("C3+D5 scale-resume boundary is incomplete or foreign")
        elif d5_checkpoint is not None:
            raise RuntimeError("plain C3 scale-resume unexpectedly contains D5 state")
        optimizer = payload.get("optimizer")
        if (
            not isinstance(payload.get("trainable_model"), dict)
            or not payload["trainable_model"]
            or not isinstance(payload.get("optimizer_parameter_manifest"), list)
            or not isinstance(optimizer, dict)
            or not isinstance(optimizer.get("state"), dict)
            or not isinstance(optimizer.get("param_groups"), list)
        ):
            raise RuntimeError("C3 official2790 scale-resume optimizer state is incomplete")
        parameter_ids = [
            parameter for group in optimizer["param_groups"] for parameter in group.get("params", [])
        ]
        if len(parameter_ids) != len(payload["optimizer_parameter_manifest"]) or set(parameter_ids) != set(optimizer["state"]):
            raise RuntimeError("C3 official2790 scale-resume AdamW coverage is incomplete")
        for parameter_id in parameter_ids:
            member = optimizer["state"][parameter_id]
            step = member.get("step") if isinstance(member, dict) else None
            if isinstance(step, torch.Tensor):
                step = step.item() if step.numel() == 1 else None
            if step != next_update * 2:
                raise RuntimeError("C3 official2790 scale-resume AdamW step is inconsistent")
    else:
        raise ValueError("unknown scale-resume kind")
    return {
        "kind": kind,
        "checkpoint_sha256": sha256(checkpoint),
        "contract_sha256": source_hash,
        "source_contract": contract,
        "start_update": next_update,
        "world_size": world_size,
    }


def validate_scale_resume_requested_contract(source: dict, requested: dict) -> None:
    """Allow only execution-lineage/update-budget changes across a scaling boundary."""
    execution_only = {
        "updates",
        "continuation",
        "repository_revision",
        "implementation_manifest",
        "script_sha256",
        "endpoint_trajectory_ddp_sha256",
        "endpoint_policy_sha256",
        "evaluator_sha256",
    }
    # Historical contracts sometimes omitted optional top-level fields that newer
    # code serializes explicitly as null. Treat only missing-vs-null as equivalent;
    # every non-null scientific field must still match exactly.
    source_science = {
        key: value
        for key, value in source.items()
        if key not in execution_only and value is not None
    }
    requested_science = {
        key: value
        for key, value in requested.items()
        if key not in execution_only and value is not None
    }
    if source_science != requested_science:
        changed = sorted(
            key
            for key in set(source_science) | set(requested_science)
            if source_science.get(key) != requested_science.get(key)
        )
        c3_d5_transition = (
            requested.get("continuation", {}).get("kind") == "c3_official2790_d5"
            and requested.get("rollout_refresh_mode") == D5_ROLLOUT_REFRESH_MODE
            and requested.get("rollout_refresh") == d5_rollout_refresh_contract()
            and source.get("rollout_refresh_mode") is None
            and source.get("rollout_refresh") is None
            and changed == ["rollout_refresh", "rollout_refresh_mode"]
        )
        if not c3_d5_transition:
            raise RuntimeError(
                f"coverage scale-resume scientific contract drifted: {changed}"
            )


def unscaled_gradient_l2(objective: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    """Read a diagnostic gradient norm without adding or changing gradients."""
    gradients = torch.autograd.grad(
        objective, parameters, retain_graph=True, allow_unused=True,
    )
    square_sum = sum(
        (gradient.detach().float().square().sum() for gradient in gradients if gradient is not None),
        torch.zeros((), device=objective.device),
    )
    value = float(square_sum.sqrt().cpu())
    if not math.isfinite(value):
        raise RuntimeError("D3 calibration gradient is non-finite")
    return value


def unscaled_gradient_cosine(
    left: torch.Tensor, right: torch.Tensor, parameters: list[torch.nn.Parameter],
) -> float:
    """Cosine of two diagnostic gradients without changing accumulated gradients."""
    left_gradients = torch.autograd.grad(
        left, parameters, retain_graph=True, allow_unused=True,
    )
    right_gradients = torch.autograd.grad(
        right, parameters, retain_graph=True, allow_unused=True,
    )
    dot = torch.zeros((), device=left.device)
    left_square_sum = torch.zeros((), device=left.device)
    right_square_sum = torch.zeros((), device=left.device)
    for left_gradient, right_gradient in zip(left_gradients, right_gradients):
        if left_gradient is not None:
            left_square_sum += left_gradient.detach().float().square().sum()
        if right_gradient is not None:
            right_square_sum += right_gradient.detach().float().square().sum()
        if left_gradient is not None and right_gradient is not None:
            dot += (left_gradient.detach().float() * right_gradient.detach().float()).sum()
    denominator = left_square_sum.sqrt() * right_square_sum.sqrt()
    value = 0.0 if float(denominator.cpu()) == 0.0 else float((dot / denominator).cpu())
    if not math.isfinite(value):
        raise RuntimeError("D4 hybrid gradient cosine is non-finite")
    return value


def initialize_task_curriculum_state(task_count: int) -> dict:
    if task_count <= 0:
        raise ValueError("task curriculum requires at least one task")
    return {
        "algorithm": ONLINE_CURRICULUM_ALGORITHM,
        "observation_count": [0] * task_count,
        "reward_sum": [0.0] * task_count,
        "reward_square_sum": [0.0] * task_count,
        "adaptive_updates": 0,
    }


def _deterministic_unit_interval(seed: int, update: int, task_index: int) -> float:
    digest = hashlib.sha256(
        f"online-task-curriculum-v1:{seed}:{update}:{task_index}".encode()
    ).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 1) / ((1 << 64) + 1)


def online_task_probabilities(
    curriculum_state: dict, minimum_exploration_probability: float,
) -> list[float]:
    counts = curriculum_state["observation_count"]
    reward_sums = curriculum_state["reward_sum"]
    reward_square_sums = curriculum_state["reward_square_sum"]
    task_count = len(counts)
    if (
        task_count == 0
        or len(reward_sums) != task_count
        or len(reward_square_sums) != task_count
        or any(int(count) <= 0 for count in counts)
    ):
        raise RuntimeError("online task probabilities require complete TRAIN coverage")
    scores = []
    for count, reward_sum, reward_square_sum in zip(
        counts, reward_sums, reward_square_sums
    ):
        mean = float(reward_sum) / int(count)
        variance = max(float(reward_square_sum) / int(count) - mean * mean, 0.0)
        # Mid-reward groups and empirically variable groups expose learnable signal.
        scores.append(max(4.0 * mean * (1.0 - mean), 0.0) + math.sqrt(variance))
    score_sum = sum(scores)
    exploitation = (
        [score / score_sum for score in scores]
        if score_sum > 0 else [1.0 / task_count] * task_count
    )
    exploration = minimum_exploration_probability / task_count
    return [
        exploration + (1.0 - minimum_exploration_probability) * probability
        for probability in exploitation
    ]


def select_task_curriculum_batch(
    *, mode: str, task_count: int, task_epoch: int, task_cursor: int,
    batch_size: int, seed: int, update: int, curriculum_state: dict | None,
    minimum_exploration_probability: float,
) -> tuple[list[int], int, int, dict]:
    if mode == "deterministic_shuffle":
        indices, next_epoch, next_cursor = task_batch_indices(
            task_count, task_epoch, task_cursor, batch_size, seed
        )
        return indices, next_epoch, next_cursor, {"phase": "deterministic_shuffle"}
    if mode != "online_learnability" or curriculum_state is None:
        raise ValueError("unsupported or missing online task curriculum state")
    counts = curriculum_state["observation_count"]
    if len(counts) != task_count:
        raise RuntimeError("online task curriculum state has the wrong task count")
    if any(int(count) == 0 for count in counts):
        indices, next_epoch, next_cursor = task_batch_indices(
            task_count, task_epoch, task_cursor, batch_size, seed
        )
        return indices, next_epoch, next_cursor, {"phase": "full_train_coverage"}
    probabilities = online_task_probabilities(
        curriculum_state, minimum_exploration_probability
    )
    keys = [
        (-math.log(_deterministic_unit_interval(seed, update, index)) / probability, index)
        for index, probability in enumerate(probabilities)
    ]
    indices = [index for _, index in sorted(keys)[:batch_size]]
    return indices, task_epoch, task_cursor, {
        "phase": "weighted_top_k",
        "minimum_task_probability": min(probabilities),
        "maximum_task_probability": max(probabilities),
    }


def update_task_curriculum_statistics(
    curriculum_state: dict, task_groups: list[dict], *, adaptive: bool,
) -> dict:
    updated = {
        "algorithm": curriculum_state["algorithm"],
        "observation_count": list(curriculum_state["observation_count"]),
        "reward_sum": list(curriculum_state["reward_sum"]),
        "reward_square_sum": list(curriculum_state["reward_square_sum"]),
        "adaptive_updates": int(curriculum_state["adaptive_updates"]) + int(adaptive),
    }
    for group in task_groups:
        index = int(group["task_index"])
        updated["observation_count"][index] += int(group["reward_count"])
        updated["reward_sum"][index] += float(group["reward_sum"])
        updated["reward_square_sum"][index] += float(group["reward_square_sum"])
    return updated


def task_curriculum_diagnostics(curriculum_state: dict) -> dict:
    counts = [int(value) for value in curriculum_state["observation_count"]]
    observed = [index for index, count in enumerate(counts) if count > 0]
    means = [
        float(curriculum_state["reward_sum"][index]) / counts[index]
        for index in observed
    ]
    variances = [
        max(
            float(curriculum_state["reward_square_sum"][index]) / counts[index]
            - means[position] ** 2,
            0.0,
        )
        for position, index in enumerate(observed)
    ]
    return {
        "algorithm": curriculum_state["algorithm"],
        "tasks_observed": len(observed),
        "task_count": len(counts),
        "coverage_fraction": len(observed) / len(counts),
        "minimum_observations_per_task": min(counts),
        "maximum_observations_per_task": max(counts),
        "observed_reward_mean_minimum": min(means) if means else None,
        "observed_reward_mean_maximum": max(means) if means else None,
        "observed_reward_variance_mean": (
            sum(variances) / len(variances) if variances else None
        ),
        "adaptive_updates": int(curriculum_state["adaptive_updates"]),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reward_function(name: str):
    if name == "terminal":
        return official_terminal_reward
    if name == "pair_credit":
        return pair_credit_terminal_reward
    if name == "pair_rival":
        return pair_rival_terminal_reward
    raise ValueError(f"unknown endpoint trajectory reward: {name}")


def uses_structure_credit(credit_assignment: str, strength: float) -> bool:
    return credit_assignment in {"structure_error", "negative_error"} and strength > 0


def validate_completed_r4_endpoint_parent(
    checkpoint_path: Path,
    receipt_path: Path,
    *,
    supervised_checkpoint_sha256: str,
    supervised_contract_sha256: str,
    rnaernie_revision: str,
    rl_trainable_scope: str,
) -> dict:
    """Validate a completed endpoint run before forking its trainable state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict):
        raise RuntimeError("parent endpoint receipt is not a JSON object")
    contract_record_path = receipt_path.parent / "contract.json"
    if not contract_record_path.is_file():
        raise RuntimeError("parent endpoint receipt has no sibling contract.json")
    contract_record = json.loads(contract_record_path.read_text())
    if not isinstance(contract_record, dict):
        raise RuntimeError("parent endpoint contract is not a JSON object")
    contract = checkpoint.get("contract")
    contract_hash = checkpoint.get("contract_sha256")
    state = checkpoint.get("state")
    stored_contract = {
        key: value for key, value in contract_record.items() if key != "contract_sha256"
    }
    last_checkpoint = receipt.get("last_checkpoint")
    if (
        not isinstance(contract, dict)
        or not isinstance(checkpoint.get("trainable_model"), dict)
        or not isinstance(state, dict)
        or contract_hash != contract_sha256(contract)
        or contract_record.get("contract_sha256") != contract_hash
        or stored_contract != contract
        or contract.get("method") != "simplex-endpoint-policy-trajectory-grpo"
        or contract.get("formal_rl") is not True
        or contract.get("trajectory_steps") != 8
        or not isinstance(contract.get("parent_chain", []), list)
        or contract.get("supervised_checkpoint_sha256") != supervised_checkpoint_sha256
        or contract.get("supervised_contract_sha256") != supervised_contract_sha256
        or contract.get("rnaernie_revision") != rnaernie_revision
        or contract.get("rl_trainable_scope") != rl_trainable_scope
        or receipt.get("status") != "complete"
        or receipt.get("formal_rl_started") is not True
        or receipt.get("contract_sha256") != contract_hash
        or not isinstance(last_checkpoint, dict)
        or Path(last_checkpoint.get("path", "")).resolve() != checkpoint_path.resolve()
        or last_checkpoint.get("sha256") != sha256(checkpoint_path)
        or state.get("next_update") != contract.get("updates")
        or receipt.get("updates_complete") != contract.get("updates")
        or receipt.get("target_updates") != contract.get("updates")
    ):
        raise RuntimeError("parent R4 endpoint checkpoint/receipt/contract is incomplete or invalid")
    return {
        "checkpoint_sha256": sha256(checkpoint_path),
        "receipt_sha256": sha256(receipt_path),
        "contract_sha256": contract_hash,
        "parent_chain": list(contract.get("parent_chain", [])),
        "updates_complete": contract["updates"],
        "temporal_credit_mode": (
            contract.get("temporal_credit", {}).get("mode")
            if isinstance(contract.get("temporal_credit"), dict) else None
        ),
    }


def temporal_credit_advantages(
    unit_advantages: torch.Tensor,
    selected_steps: tuple[int, ...],
) -> torch.Tensor:
    """Slice only step-indexed advantages; scalar and unit credit spans all steps."""
    if unit_advantages.ndim != 3:
        return unit_advantages
    return unit_advantages[:, list(selected_steps)]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_exact_float(value: object, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isclose(float(value), expected, rel_tol=0, abs_tol=1e-15)
    )


def load_contrastive_in_place_contract(output: Path) -> dict:
    """Read and validate the immutable contract before any in-place resume work."""
    path = output / "contract.json"
    if not path.is_file():
        raise FileNotFoundError("contrastive in-place resume output contract is absent")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise RuntimeError("contrastive in-place resume contract is not a JSON object")
    embedded_hash = record.pop("contract_sha256", None)
    continuation = record.get("continuation")
    source_hash_fields = (
        "source_checkpoint_sha256",
        "source_receipt_sha256",
        "source_contract_sha256",
        "resume_plumbing_review_sha256",
        "optimizer_parameter_manifest_sha256",
    )
    implementation_changes = (
        continuation.get("allowed_implementation_changes")
        if isinstance(continuation, dict) else None
    )
    if (
        not _is_sha256(embedded_hash)
        or contract_sha256(record) != embedded_hash
        or record.get("schema_version") != 8
        or record.get("method") != CONTRASTIVE_TASK_STEP_METHOD
        or record.get("ratio") != CONTRASTIVE_TASK_STEP_RATIO
        or record.get("policy_objective")
        != contrastive_task_step_objective_contract()
        or record.get("formal_rl") is not True
        or record.get("updates") != 144
        or not _is_exact_float(record.get("learning_rate"), 5e-6)
        or not _is_exact_float(record.get("kl_coefficient"), 0.01)
        or record.get("distributed", {}).get("world_size") != 4
        or record.get("temporal_credit", {}).get("mode") != "causal_is_single"
        or not isinstance(continuation, dict)
        or continuation.get("mode") != "completed-branch-exact-state-v1"
        or continuation.get("start_update") != 96
        or continuation.get("policy_objective") != "contrastive_task_step"
        or any(not _is_sha256(continuation.get(key)) for key in source_hash_fields)
        or not isinstance(implementation_changes, dict)
        or set(implementation_changes) != {
            str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2])),
            str(
                Path(__file__).with_name("endpoint_trajectory_ddp.py")
                .resolve().relative_to(Path(__file__).resolve().parents[2])
            ),
        }
        or any(
            not isinstance(change, dict)
            or set(change) != {"source_sha256", "requested_sha256"}
            or not _is_sha256(change["source_sha256"])
            or not _is_sha256(change["requested_sha256"])
            for change in implementation_changes.values()
        )
        or continuation.get("optimizer_state_restored") is not True
        or continuation.get("rng_state_by_rank_restored") is not True
        or continuation.get("task_cursor_and_coverage_restored") is not True
        or continuation.get("learning_rate_mode") != "half"
        or not _is_exact_float(continuation.get("source_learning_rate"), 1e-5)
        or not _is_exact_float(continuation.get("requested_learning_rate"), 5e-6)
        or continuation.get("kl_coefficient_mode") != "fixed"
        or not _is_exact_float(continuation.get("source_kl_coefficient"), 0.01)
        or not _is_exact_float(continuation.get("requested_kl_coefficient"), 0.01)
        or continuation.get("scheduler") != "none"
    ):
        raise RuntimeError("contrastive in-place resume contract is incomplete or invalid")
    return {
        "contract": record,
        "contract_sha256": embedded_hash,
        "contract_path": path,
        "contract_file_sha256": sha256(path),
    }


def load_sequence_group_fpo_in_place_contract(output: Path) -> dict:
    """Load the immutable fresh-B1 contract before an exact in-place resume."""
    path = output / "contract.json"
    if not path.is_file():
        raise FileNotFoundError("sequence-group FPO resume output contract is absent")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise RuntimeError("sequence-group FPO resume contract is not a JSON object")
    digest = record.pop("contract_sha256", None)
    if (
        not _is_sha256(digest)
        or contract_sha256(record) != digest
        or record.get("method") != SEQUENCE_GROUP_FPO_METHOD
        or record.get("ratio") != SEQUENCE_GROUP_FPO_RATIO
        or record.get("policy_objective") != sequence_group_fpo_objective_contract()
        or record.get("branch_update_budget_mode") != "u128_matched32"
    ):
        raise RuntimeError("sequence-group FPO resume contract is incomplete or foreign")
    return {"contract": record, "contract_sha256": digest}


def validate_in_place_requested_contract(requested: dict, existing: dict) -> None:
    """Require reconstruction from current args/source to equal the frozen contract."""
    if (
        requested != existing.get("contract")
        or contract_sha256(requested) != existing.get("contract_sha256")
    ):
        raise RuntimeError("contrastive in-place resume requested contract changed")


def contrastive_branch_gate(
    continuation: dict, preflight_receipt_sha256: str,
) -> dict:
    """Rebuild the non-contract provenance gate from immutable identities."""
    gate = {
        "schema_version": 1,
        "approval_scope": "formal-after-exact-preflight",
        "source_contract_sha256": continuation.get("source_contract_sha256"),
        "source_checkpoint_sha256": continuation.get("source_checkpoint_sha256"),
        "source_receipt_sha256": continuation.get("source_receipt_sha256"),
        "review_receipt_sha256": continuation.get("resume_plumbing_review_sha256"),
        "preflight_receipt_sha256": preflight_receipt_sha256,
    }
    if any(not _is_sha256(value) for key, value in gate.items() if key.endswith("sha256")):
        raise RuntimeError("contrastive branch gate provenance is incomplete")
    return gate


def load_contrastive_branch_gate(
    output: Path, continuation: dict, preflight_receipt_sha256: str,
) -> dict:
    """Require an in-place resume to preserve the first formal launch gate."""
    path = output / "branch_gate.json"
    if not path.is_file():
        raise FileNotFoundError("contrastive in-place resume branch gate is absent")
    gate = json.loads(path.read_text())
    expected = contrastive_branch_gate(continuation, preflight_receipt_sha256)
    if gate != expected:
        raise RuntimeError("contrastive in-place resume branch gate changed")
    return gate


def validate_policy_objective_arguments(
    args, branch_resume_enabled: bool, in_place_contract: dict | None = None,
) -> None:
    """Fail closed for the frozen continuation and fresh Stage-B B1 contract."""
    if args.policy_objective == "normalized_grpo":
        return
    if args.policy_objective == "sequence_group_fpo":
        fresh_branch = branch_resume_enabled and args.resume is None
        in_place_resume = args.resume is not None and not branch_resume_enabled
        exact = (
            (fresh_branch or in_place_resume)
            and args.formal_rl is True
            and args.distributed is True
            and args.updates == 128
            and (
                (fresh_branch
                 and args.branch_update_budget_mode == "u128_matched32"
                 and args.branch_learning_rate_mode == "half"
                 and args.branch_kl_coefficient_mode == "fixed")
                or (in_place_resume
                    and args.branch_update_budget_mode == "u192_legacy"
                    and args.branch_learning_rate_mode is None
                    and args.branch_kl_coefficient_mode is None)
            )
            and args.seed == 1009
            and args.tasks_per_update == 4
            and args.candidates == 12
            and args.trajectory_steps == 8
            and args.temporal_credit == "all_steps"
            and args.task_curriculum == "deterministic_shuffle"
            and args.policy_epochs == 2
            and args.reward == "terminal"
            and args.credit_assignment == "uniform"
            and math.isclose(args.learning_rate, 5e-6, rel_tol=0, abs_tol=1e-15)
            and math.isclose(args.kl_coefficient, 0.01, rel_tol=0, abs_tol=1e-15)
            and math.isclose(args.ce_coefficient, 0.1, rel_tol=0, abs_tol=1e-15)
            and args.rl_trainable_scope == "last_2_backbone_and_head"
            and args.causal_step_policy is None
            and args.expected_causal_step_policy_sha256 is None
            and (args.preflight_updates is None or (fresh_branch and args.preflight_updates == 1))
        )
        if not exact:
            raise ValueError(
                "sequence_group_fpo requires the exact U96-to-U128 Stage-B B0-matched contract"
            )
        return
    if args.policy_objective == "domino_endpoint_ppo":
        fresh_branch = branch_resume_enabled and args.resume is None
        exact = (
            fresh_branch
            and args.formal_rl is True
            and args.distributed is True
            and args.updates == 128
            and args.branch_update_budget_mode == "u128_domino32"
            and args.branch_learning_rate_mode == "half"
            and args.branch_kl_coefficient_mode == "fixed"
            and args.seed == 1009
            and args.tasks_per_update == 4
            and args.candidates == 12
            and args.trajectory_steps == 8
            and args.temporal_credit == "all_steps"
            and args.task_curriculum == "deterministic_shuffle"
            and args.policy_epochs == 2
            and args.reward == "terminal"
            and args.credit_assignment == "uniform"
            and math.isclose(args.learning_rate, 5e-6, rel_tol=0, abs_tol=1e-15)
            and math.isclose(args.kl_coefficient, 0.01, rel_tol=0, abs_tol=1e-15)
            and math.isclose(args.ce_coefficient, 0.1, rel_tol=0, abs_tol=1e-15)
            and args.rl_trainable_scope == "last_2_backbone_and_head"
            and args.causal_step_policy is None
            and args.expected_causal_step_policy_sha256 is None
            and (args.preflight_updates is None or args.preflight_updates == 1)
        )
        if not exact:
            raise ValueError(
                "domino_endpoint_ppo requires the scoped U96-to-U128 endpoint-policy contract"
            )
        return
    if args.policy_objective == "discrete_domino":
        fresh_branch = branch_resume_enabled and args.resume is None
        exact = (
            fresh_branch
            and args.formal_rl is True
            and args.distributed is True
            and args.updates == 128
            and args.branch_update_budget_mode == "u128_discrete_domino32"
            and args.branch_learning_rate_mode == "half"
            and args.branch_kl_coefficient_mode == "fixed"
            and args.seed == 1009
            and args.tasks_per_update == 4
            and args.candidates == 12
            and args.trajectory_steps == 8
            and args.temporal_credit == "all_steps"
            and args.task_curriculum == "deterministic_shuffle"
            and args.policy_epochs == 2
            and args.reward == "terminal"
            and args.credit_assignment == "uniform"
            and math.isclose(args.learning_rate, 5e-6, rel_tol=0, abs_tol=1e-15)
            and math.isclose(args.kl_coefficient, 0.01, rel_tol=0, abs_tol=1e-15)
            and math.isclose(args.ce_coefficient, 0.1, rel_tol=0, abs_tol=1e-15)
            and args.rl_trainable_scope == "last_2_backbone_and_head"
            and args.causal_step_policy is None
            and args.expected_causal_step_policy_sha256 is None
            and (args.preflight_updates is None or args.preflight_updates == 1)
        )
        if not exact:
            raise ValueError(
                "discrete_domino requires the scoped U96-to-U128 structured discrete contract"
            )
        return
    if args.policy_objective != "contrastive_task_step":
        raise ValueError("unknown endpoint trajectory policy objective")
    in_place_resume = args.resume is not None
    source_mode = (
        branch_resume_enabled
        and not in_place_resume
        and in_place_contract is None
        and args.branch_learning_rate_mode == "half"
        and args.branch_kl_coefficient_mode == "fixed"
    ) or (
        in_place_resume
        and not branch_resume_enabled
        and in_place_contract is not None
        and args.branch_learning_rate_mode is None
        and args.branch_kl_coefficient_mode is None
        and args.preflight_updates is None
    )
    exact = (
        source_mode
        and args.formal_rl is True
        and args.distributed is True
        and args.updates == 144
        and args.tasks_per_update == 4
        and args.candidates == 12
        and args.trajectory_steps == 8
        and args.temporal_credit == "causal_is_single"
        and args.task_curriculum == "deterministic_shuffle"
        and args.policy_epochs == 2
        and args.reward == "terminal"
        and args.credit_assignment == "uniform"
        and math.isclose(args.learning_rate, 5e-6, rel_tol=0, abs_tol=1e-15)
        and math.isclose(args.kl_coefficient, 0.01, rel_tol=0, abs_tol=1e-15)
    )
    if not exact:
        raise ValueError(
            "contrastive_task_step requires the frozen U96 causal branch contract"
        )


def initialize_contrastive_resume_state(
    state: dict, *, legacy_u96_branch: bool, in_place_resume: bool,
    tasks_per_update: int = 4, candidates: int = 12,
) -> None:
    if tasks_per_update != 4 or candidates != 12:
        raise RuntimeError("contrastive resume state geometry changed")
    fields = {
        "contrastive_pair_count": 0,
        "contrastive_effective_pair_groups": 0,
        "contrastive_ineffective_groups": 0,
        "contrastive_mixed_correctness_groups": 0,
        "contrastive_tier_candidate_counts": {"0": 0, "2": 0, "3": 0},
        "contrastive_behavior_delta_max_abs_error": 0.0,
        "contrastive_policy_gradient_finite": True,
        "contrastive_policy_gradient_nonzero_groups": 0,
    }
    present = {key for key in fields if key in state}
    if legacy_u96_branch:
        if present or state.get("next_update") != 96:
            raise RuntimeError("legacy U96 state has unexpected contrastive fields")
        state.update(fields)
        return
    if in_place_resume:
        if present != set(fields):
            raise RuntimeError("contrastive in-place resume state is incomplete")
        if (
            any(
                type(state[key]) is not int or state[key] < 0
                for key in (
                    "contrastive_pair_count",
                    "contrastive_effective_pair_groups",
                    "contrastive_ineffective_groups",
                    "contrastive_mixed_correctness_groups",
                    "contrastive_policy_gradient_nonzero_groups",
                )
            )
            or not isinstance(state["contrastive_tier_candidate_counts"], dict)
            or set(state["contrastive_tier_candidate_counts"]) != {"0", "2", "3"}
            or any(
                type(value) is not int or value < 0
                for value in state["contrastive_tier_candidate_counts"].values()
            )
            or not isinstance(
                state["contrastive_behavior_delta_max_abs_error"], (int, float)
            )
            or not math.isfinite(
                float(state["contrastive_behavior_delta_max_abs_error"])
            )
            or state["contrastive_policy_gradient_finite"] is not True
        ):
            raise RuntimeError("contrastive in-place resume state is malformed")
        next_update = state.get("next_update")
        if type(next_update) is not int or not 96 <= next_update <= 144:
            raise RuntimeError("contrastive in-place resume update is outside U96..U144")
        expected_groups = (next_update - 96) * tasks_per_update
        effective = state["contrastive_effective_pair_groups"]
        ineffective = state["contrastive_ineffective_groups"]
        mixed = state["contrastive_mixed_correctness_groups"]
        nonzero = state["contrastive_policy_gradient_nonzero_groups"]
        pair_count = state["contrastive_pair_count"]
        if (
            effective + ineffective != expected_groups
            or mixed != effective
            or nonzero != effective
            or sum(state["contrastive_tier_candidate_counts"].values())
            != expected_groups * candidates
            or pair_count < effective
            or pair_count > expected_groups * candidates * (candidates - 1) // 2
            or float(state["contrastive_behavior_delta_max_abs_error"]) > 1e-5
        ):
            raise RuntimeError("contrastive in-place resume accounting is inconsistent")
        return
    raise RuntimeError("contrastive state must originate from exact U96 branch resume")


def load_causal_step_policy(path: Path, expected_sha256: str, parent_identity: dict) -> dict:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    policy = json.loads(payload)
    tickets = policy.get("ticket_counts")
    if (
        digest != expected_sha256
        or policy.get("schema_version") != 1
        or policy.get("status") != "frozen"
        or policy.get("method") != "causal-is-single"
        or policy.get("bad_count") != 0
        or policy.get("trajectory_steps") != 8
        or not isinstance(tickets, list)
        or len(tickets) != 8
        or any(type(value) is not int or value <= 0 for value in tickets)
        or policy.get("ticket_total") != sum(tickets)
        or policy.get("objective") != "uniform-eight-step-expectation"
        or policy.get("importance_weight") != "1/(8*q_s)"
        or policy.get("r4_checkpoint_sha256") != parent_identity["checkpoint_sha256"]
        or policy.get("r4_receipt_sha256") != parent_identity["receipt_sha256"]
        or policy.get("r4_contract_sha256") != parent_identity["contract_sha256"]
    ):
        raise RuntimeError("causal step policy is incomplete or invalid")
    return policy | {"sha256": digest}


def backward_staged_anchor(policy_objective, anchor_closure, anchor_coefficient: float):
    """Backpropagate equivalent summed objectives without co-resident forward graphs."""
    if not torch.isfinite(policy_objective):
        raise RuntimeError("non-finite endpoint trajectory policy objective")
    policy_objective.backward()
    anchor_loss, anchor_diagnostics = anchor_closure()
    if not torch.isfinite(anchor_loss):
        raise RuntimeError("non-finite endpoint trajectory anchor loss")
    if anchor_coefficient:
        (anchor_coefficient * anchor_loss).backward()
    total_loss = policy_objective.detach() + anchor_coefficient * anchor_loss.detach()
    return anchor_loss, anchor_diagnostics, total_loss


def backward_group_member(objective: torch.Tensor, group_size: int) -> None:
    """Accumulate the exact gradient of a group mean without retaining sibling graphs."""
    if group_size <= 0 or not torch.isfinite(objective):
        raise RuntimeError("invalid taskwise trajectory objective")
    (objective / group_size).backward()


def merge_policy_epoch_diagnostics(
    diagnostics_by_rank: list[list[dict]], local_tasks_per_rank: int,
) -> list[dict]:
    """Restore global task-weighted diagnostics from equally sharded DDP ranks."""
    if not diagnostics_by_rank or any(len(rows) != len(diagnostics_by_rank[0]) for rows in diagnostics_by_rank):
        raise RuntimeError("distributed policy diagnostics are incomplete")
    merged = []
    weighted_fields = (
        "policy_loss", "total_loss", "reference_kl_before_update", "entropy_before_update",
        "importance_weighted_policy_loss", "importance_weighted_policy_objective",
        "importance_weighted_reference_kl", "importance_weighted_entropy",
        "posterior_tv", "transition_tv", "reference_tv_coefficient",
        "reference_tv_contribution", "pkpo_advantage_mean", "pkpo_advantage_std",
        "pkpo_advantage_min", "pkpo_advantage_max",
        "calibration_policy_gradient_l2", "calibration_transition_tv_gradient_l2",
        "calibration_reference_kl_gradient_l2", "calibration_ce_gradient_l2",
        "ce_anchor_loss", "ce_accuracy", "ce_anchor_alpha", "clip_fraction", "mean_ratio",
    )
    for policy_epoch in range(len(diagnostics_by_rank[0])):
        rows = [items[policy_epoch] for items in diagnostics_by_rank]
        if any(int(row["policy_epoch"]) != policy_epoch for row in rows):
            raise RuntimeError("distributed policy epoch diagnostics are misaligned")
        record = {"policy_epoch": policy_epoch}
        for field in weighted_fields:
            record[field] = sum(float(row[field]) for row in rows) / len(rows)
        pkpo_values = {row.get("pkpo_k") for row in rows}
        if len(pkpo_values) != 1:
            raise RuntimeError("distributed D3 PKPO schedule disagrees across ranks")
        record["pkpo_k"] = pkpo_values.pop()
        hybrid_fields = (
            "thermo_advantage_mean", "thermo_advantage_std", "thermo_advantage_min",
            "thermo_advantage_max", "raw_pkpo_advantage_mean", "raw_pkpo_advantage_std",
            "raw_pkpo_advantage_min", "raw_pkpo_advantage_max", "combined_advantage_mean",
            "combined_advantage_std", "combined_advantage_min", "combined_advantage_max",
            "calibration_thermo_policy_gradient_l2",
            "calibration_raw_pkpo_policy_gradient_l2",
            "calibration_combined_policy_gradient_l2",
            "calibration_thermo_raw_pkpo_gradient_cosine",
        )
        hybrid_fields_present = [all(field in row for field in hybrid_fields) for row in rows]
        if any(hybrid_fields_present):
            if not all(hybrid_fields_present):
                raise RuntimeError("distributed D4 hybrid diagnostics are incomplete")
            for field in hybrid_fields:
                record[field] = sum(float(row[field]) for row in rows) / len(rows)
        record["gradient_norm_before_clip"] = max(
            float(row["gradient_norm_before_clip"]) for row in rows
        )
        record["effective_groups"] = sum(int(row["effective_groups"]) for row in rows)
        record["minimum_ratio"] = min(float(row["minimum_ratio"]) for row in rows)
        record["maximum_ratio"] = max(float(row["maximum_ratio"]) for row in rows)
        record["ratio_max_abs_error"] = max(
            float(row["ratio_max_abs_error"]) for row in rows
        )
        contrastive_fields_present = ["pair_count" in row for row in rows]
        if any(contrastive_fields_present):
            if not all(contrastive_fields_present):
                raise RuntimeError(
                    "distributed contrastive policy diagnostics are incomplete"
                )
            record.update({
                "pair_count": sum(int(row["pair_count"]) for row in rows),
                "effective_pair_groups": sum(
                    int(row["effective_pair_groups"]) for row in rows
                ),
                "ineffective_groups": sum(
                    int(row["ineffective_groups"]) for row in rows
                ),
                "mixed_correctness_groups": sum(
                    int(row["mixed_correctness_groups"]) for row in rows
                ),
                "tier_candidate_counts": {
                    str(tier): sum(
                        int(row["tier_candidate_counts"][str(tier)])
                        for row in rows
                    )
                    for tier in (0, 2, 3)
                },
                "behavior_delta_max_abs_error": max(
                    float(row["behavior_delta_max_abs_error"]) for row in rows
                ),
                "policy_gradient_finite": all(
                    row["policy_gradient_finite"] is True for row in rows
                ),
                "policy_gradient_nonzero_groups": sum(
                    int(row["policy_gradient_nonzero_groups"]) for row in rows
                ),
            })
        global_tokens = {int(row["ce_anchor_global_tokens"]) for row in rows}
        if len(global_tokens) != 1:
            raise RuntimeError("distributed CE diagnostics disagree on global token count")
        record["ce_anchor_global_tokens"] = global_tokens.pop()
        record["ce_anchor_local_tokens"] = sum(
            int(row["ce_anchor_local_tokens"]) for row in rows
        )
        merged.append(record)
    if local_tasks_per_rank <= 0:
        raise RuntimeError("distributed local task count must be positive")
    return merged


def endpoint_ce_anchor_loss(
    model: torch.nn.Module,
    global_batch: dict,
    *,
    local_indices: list[int],
    seed: int,
    update: int,
    policy_epoch: int,
    context,
) -> tuple[torch.Tensor, dict]:
    """Compute a supervised anchor whose DDP gradient is the global token mean."""
    device = context.device
    state, alpha = keyed_dirichlet_path(
        global_batch["clean"], seed, update, policy_epoch, device
    )
    index = torch.tensor(local_indices, dtype=torch.long, device=device)
    clean = global_batch["clean"].to(device).index_select(0, index)
    mask = global_batch["mask"].to(device).index_select(0, index)
    structure_tokens = global_batch["structure_tokens"].to(device).index_select(0, index)
    structure_hidden = global_batch["structure_hidden"]
    if structure_hidden is not None:
        structure_hidden = structure_hidden.to(device).index_select(0, index)
    prediction = model(
        state.index_select(0, index),
        alpha.index_select(0, index),
        mask,
        structure_hidden=structure_hidden,
        structure_tokens=structure_tokens,
    )
    token_loss = F.cross_entropy(prediction.transpose(1, 2), clean, reduction="none")
    local_token_loss_sum = (token_loss * mask).sum()
    local_tokens = mask.sum()
    global_tokens = global_sum(context, local_tokens)
    local_loss = local_token_loss_sum / local_tokens
    global_loss = global_sum(context, local_token_loss_sum) / global_tokens
    local_correct = ((prediction.argmax(-1) == clean) * mask.bool()).sum()
    global_accuracy = global_sum(context, local_correct) / global_tokens
    scale = global_token_mean_scale(local_tokens, global_tokens, context.world_size)
    return local_loss * scale, {
        "global_loss": float(global_loss.cpu()),
        "accuracy": float(global_accuracy.cpu()),
        "alpha": float(
            (global_sum(context, alpha.index_select(0, index).sum()) / alpha.shape[0]).cpu()
        ),
        "local_tokens": int(local_tokens.detach().cpu()),
        "global_tokens": int(global_tokens.detach().cpu()),
    }


def enable_deterministic_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Checkpoint the BERT encoder without enabling training-time dropout."""
    encoder = getattr(getattr(model.rnaernie, "encoder", None), "layer", None)
    encoder_module = getattr(model.rnaernie, "encoder", None)
    if encoder is None or encoder_module is None:
        raise RuntimeError("RNAErnie does not expose a checkpointable encoder")
    model.rnaernie.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Transformers gates checkpointing on Encoder.training. Set only this module's
    # flag so child Dropout modules remain in deterministic evaluation mode.
    encoder_module.training = True


def validate_tasks_manifest(path: Path, tasks_path: Path, task_count: int) -> dict:
    manifest = json.loads(path.read_text())
    tasks = [
        json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()
    ]
    selection = manifest.get("selection", {})
    source = manifest.get("source", {})
    overlap = manifest.get("overlap", {})
    actual_length_bins = dict(Counter(task.get("length_bin") for task in tasks))
    expected_per_bin = selection.get("per_length_bin")
    if (
        manifest.get("status") != "complete"
        or manifest.get("bad_count") != 0
        or "TRAIN-only" not in manifest.get("role", "")
        or selection.get("rows") != task_count
        or selection.get("sha256") != sha256(tasks_path)
        or overlap.get("validation") != 0
        or overlap.get("eterna100v2") != 0
        or not source.get("revision")
        or not source.get("sha256")
        or selection.get("rows_per_length_bin") != actual_length_bins
        or manifest.get("length_bins") != actual_length_bins
        or set(actual_length_bins) != set(TARGET_LENGTH_BINS)
        or (
            expected_per_bin is not None
            and (
                not isinstance(expected_per_bin, int)
                or any(count != expected_per_bin for count in actual_length_bins.values())
            )
        )
    ):
        raise RuntimeError("trajectory task manifest fails the TRAIN-only/overlap/hash gate")
    if len(tasks) != task_count:
        raise RuntimeError("trajectory task file row count mismatch")
    validate_target_contract(tasks, manifest.get("vienna_target_representability", {}))
    return manifest


def validate_evaluations(
    evaluations: list[dict], sequences: list[str], structure: str, task_id: str, reward_fn,
) -> None:
    if (
        len(evaluations) != len(sequences)
        or [row.get("candidate_index") for row in evaluations] != list(range(len(sequences)))
        or [row.get("sequence") for row in evaluations] != sequences
    ):
        raise RuntimeError("trajectory evaluation coverage mismatch")
    for row, sequence in zip(evaluations, sequences):
        metrics = safe_reward_evaluation(evaluate_candidate, sequence, structure)
        expected = {
            "task_id": task_id,
            "target_length": len(structure),
            "reward_cache_key": cache_key(sequence, structure),
            "reward": reward_fn(metrics),
            **metrics,
        }
        for key, value in expected.items():
            actual = row.get(key)
            if isinstance(value, float):
                if actual is None or not math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f"persisted trajectory evaluation mismatch for {key}")
            elif actual != value:
                raise RuntimeError(f"persisted trajectory evaluation mismatch for {key}")


def terminal_evaluation_correctness_tiers(
    evaluations: list[dict], device: torch.device,
) -> torch.Tensor:
    """Read only validated terminal MFE/uMFE booleans; ignore continuous metrics."""
    if not evaluations or any(
        row.get("evaluation_valid") is not True
        or row.get("evaluation_error_type") is not None
        or row.get("evaluation_error_code") is not None
        or type(row.get("mfe_hit")) is not bool
        or type(row.get("uMFE_hit")) is not bool
        for row in evaluations
    ):
        raise RuntimeError(
            "contrastive correctness requires valid error-free terminal evaluations"
        )
    return validated_correctness_tiers(
        torch.tensor(
            [row["mfe_hit"] for row in evaluations], dtype=torch.bool, device=device
        ),
        torch.tensor(
            [row["uMFE_hit"] for row in evaluations], dtype=torch.bool, device=device
        ),
    )


def provisional_structure_metrics(sequence: str, target: str) -> dict:
    md = RNA.md()
    md.temperature = 37.0
    md.dangles = 2
    md.uniq_ML = 1
    structure, energy = RNA.fold_compound(sequence, md).mfe()
    return {
        "mfe_structure": structure,
        "mfe_energy": float(energy),
        "pair_f1": pair_f1(structure, target),
    }


def build_stepwise_evaluations(trajectory: dict, target: str) -> list[dict]:
    states = trajectory["states"]
    rows = []
    for candidate_index in range(states.shape[0]):
        for state_index in range(states.shape[1]):
            sequence = constrained_decode(states[candidate_index, state_index], target)
            rows.append({
                "candidate_index": candidate_index,
                "state_index": state_index,
                "sequence": sequence,
                **provisional_structure_metrics(sequence, target),
            })
    return rows


def validate_stepwise_evaluations(
    rows: list[dict], trajectory: dict, target: str,
) -> torch.Tensor:
    candidates = int(trajectory["states"].shape[0])
    states = int(trajectory["states"].shape[1])
    expected_keys = [(candidate, state) for candidate in range(candidates) for state in range(states)]
    if [(row.get("candidate_index"), row.get("state_index")) for row in rows] != expected_keys:
        raise RuntimeError("stepwise evaluation coverage mismatch")
    recomputed = build_stepwise_evaluations(trajectory, target)
    for actual, expected in zip(rows, recomputed):
        if actual.keys() != expected.keys():
            raise RuntimeError("stepwise evaluation fields mismatch")
        for key, value in expected.items():
            observed = actual[key]
            if isinstance(value, float):
                if not math.isclose(float(observed), value, rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(f"stepwise evaluation mismatch for {key}")
            elif observed != value:
                raise RuntimeError(f"stepwise evaluation mismatch for {key}")
    return torch.tensor(
        [[row["pair_f1"] for row in rows if row["candidate_index"] == candidate]
         for candidate in range(candidates)],
        dtype=torch.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--tasks-manifest", type=Path, required=True)
    parser.add_argument("--supervised-train", type=Path, required=True)
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--rnaernie-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--seed", type=int, default=9176)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--tasks-per-update", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--trajectory-steps", type=int, default=8)
    parser.add_argument(
        "--temporal-credit",
        choices=[
            "all_steps", "uniform_single", "informative_early_single",
            "causal_is_single", "causal_window_2",
        ],
        default="all_steps",
    )
    parser.add_argument(
        "--policy-objective",
        choices=[
            "normalized_grpo", "contrastive_task_step", "sequence_group_fpo",
            "domino_endpoint_ppo", "discrete_domino",
        ],
        default="normalized_grpo",
    )
    parser.add_argument("--causal-step-policy", type=Path)
    parser.add_argument("--expected-causal-step-policy-sha256")
    parser.add_argument(
        "--task-curriculum",
        choices=["deterministic_shuffle", "online_learnability"],
        default="deterministic_shuffle",
    )
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    parser.add_argument(
        "--reward", choices=["terminal", "pair_credit", "pair_rival"], default="terminal"
    )
    parser.add_argument(
        "--credit-assignment",
        choices=["uniform", "structure_error", "negative_error", "stepwise_pair"],
        default="uniform",
    )
    parser.add_argument("--structure-credit-strength", type=float, default=1.0)
    parser.add_argument("--supervised-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--kl-coefficient", type=float, default=0.01)
    parser.add_argument("--ce-coefficient", type=float, default=0.1)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument(
        "--rl-trainable-scope",
        choices=[
            "inherited", "adapter_and_head", "last_2_backbone_and_head",
            "full_backbone_and_head",
        ],
        default="inherited",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--scale-resume-checkpoint", type=Path)
    parser.add_argument("--scale-resume-receipt", type=Path)
    parser.add_argument(
        "--scale-resume-kind",
        choices=[
            "c3_official2790", "c3_official2790_d5",
            "t0_flow_yrl", "t0_flow_yrl_d5",
        ],
        help="Fresh-output exact continuation from a frozen scaling boundary.",
    )
    parser.add_argument("--c3-policy-init-checkpoint", type=Path)
    parser.add_argument("--c3-policy-init-receipt", type=Path)
    parser.add_argument("--d3-policy-init-checkpoint", type=Path)
    parser.add_argument("--d3-policy-init-receipt", type=Path)
    parser.add_argument(
        "--group-objective", choices=[
            "thermodynamic_grpo", "thermo_pkpo", D4_HYBRID_PKPO_OBJECTIVE,
        ],
        help="Frozen D3 group-relative objective; only valid with structured discrete DoMinO.",
    )
    parser.add_argument(
        "--pkpo-residual-coefficient", type=float,
        help=(
            "D4 only: finite positive multiplier for raw continuous max@k PKPO added "
            "to dense centered thermodynamic advantages."
        ),
    )
    parser.add_argument(
        "--rollout-refresh-mode", choices=[D5_ROLLOUT_REFRESH_MODE],
        help=(
            "D5 only: strict on-policy refresh after each policy epoch; each "
            "logical update retains one four-task cursor advance."
        ),
    )
    parser.add_argument(
        "--reference-tv-coefficient", type=float,
        help="Explicit TV-inspired reference-transition coefficient for D3 (CPU unit runs may use 0).",
    )
    parser.add_argument(
        "--c3-adaptation", choices=["last2", "lora_m"],
        help="C3 scale-up parameterization from the frozen U96 policy weights.",
    )
    parser.add_argument("--c3-lora-rank", type=int, default=89)
    parser.add_argument("--c3-lora-alpha", type=float, default=178.0)
    parser.add_argument("--c3-lora-dropout", type=float, default=0.0)
    parser.add_argument("--pause-after-update", type=int)
    parser.add_argument("--branch-resume-checkpoint", type=Path)
    parser.add_argument("--branch-resume-receipt", type=Path)
    parser.add_argument("--branch-resume-review-receipt", type=Path)
    parser.add_argument("--branch-preflight-receipt", type=Path)
    parser.add_argument(
        "--branch-learning-rate-mode", choices=["fixed", "half"],
    )
    parser.add_argument(
        "--branch-kl-coefficient-mode", choices=["fixed", "tenfold_pilot48"],
    )
    parser.add_argument(
        "--branch-update-budget-mode",
        choices=[
            "u192_legacy", "u144_matched48", "u128_matched32", "u128_domino32",
            "u128_discrete_domino32",
        ],
        default="u192_legacy",
        help=(
            "Reviewed branch purpose: u192_legacy preserves historical U96->U192; "
            "u144_matched48 is required for causal-is-single/window-2 matched U96->U144; "
            "u128_matched32 is the frozen Stage-B B0/B1 U96->U128 comparison; "
            "u128_domino32 is the scoped DoMinO-style endpoint-policy U96->U128 pilot; "
            "u128_discrete_domino32 is the structured discrete-mixture DoMinO U96->U128 pilot."
        ),
    )
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--parent-receipt", type=Path)
    parser.add_argument("--immutable-checkpoint-every", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--preflight-updates", type=int)
    parser.add_argument("--formal-rl", action="store_true")
    parser.add_argument("--supervised-selection-receipt", type=Path)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.tasks_per_update,
        args.candidates,
        args.trajectory_steps,
        args.policy_epochs,
        args.supervised_batch_size,
        args.learning_rate,
        args.immutable_checkpoint_every,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("trajectory GRPO sizes and learning rate must be positive")
    if args.policy_epochs < 2:
        raise ValueError("trajectory GRPO requires at least two policy epochs per old rollout")
    if (
        args.weight_decay < 0
        or not 0 < args.clip_ratio < 1
        or args.kl_coefficient < 0
        or args.ce_coefficient < 0
        or args.entropy_coefficient < 0
        or not 0 <= args.structure_credit_strength <= 1
        or any(value <= 0 or not math.isfinite(value) for value in args.temperatures)
    ):
        raise ValueError("trajectory GRPO optimizer/policy coefficients are invalid")
    if args.preflight_updates is not None and args.preflight_updates <= 0:
        raise ValueError("preflight updates must be positive")
    if args.pause_after_update is not None and not 0 < args.pause_after_update <= args.updates:
        raise ValueError("pause-after-update must be within the frozen update budget")
    c3_policy_init = bool(args.c3_policy_init_checkpoint or args.c3_policy_init_receipt)
    d3_policy_init = bool(args.d3_policy_init_checkpoint or args.d3_policy_init_receipt)
    scale_resume = bool(
        args.scale_resume_checkpoint or args.scale_resume_receipt or args.scale_resume_kind
    )
    if scale_resume and not all(
        (args.scale_resume_checkpoint, args.scale_resume_receipt, args.scale_resume_kind)
    ):
        raise ValueError("scale resume requires checkpoint, receipt, and kind together")
    if scale_resume and (args.resume or args.branch_resume_checkpoint):
        raise ValueError("scale resume is mutually exclusive with in-place/branch resume")
    if (
        scale_resume
        and args.scale_resume_kind in {"c3_official2790", "c3_official2790_d5"}
        and not c3_policy_init
    ):
        raise ValueError("C3 scale resume requires the frozen C3 U96 policy initializer")
    if (
        scale_resume
        and args.scale_resume_kind in {"t0_flow_yrl", "t0_flow_yrl_d5"}
        and not d3_policy_init
    ):
        raise ValueError("T0 scale resume requires the frozen D3 U96 policy initializer")
    d5_strict_on_policy = args.rollout_refresh_mode == D5_ROLLOUT_REFRESH_MODE
    if c3_policy_init and d3_policy_init:
        raise ValueError("C3 and D3 policy initializers are mutually exclusive")
    validate_d3_formal_arguments(args)
    if bool(args.c3_policy_init_checkpoint) != bool(args.c3_policy_init_receipt):
        raise ValueError("C3 policy initialization requires checkpoint and receipt together")
    if c3_policy_init:
        c3_d5_scale = scale_resume and args.scale_resume_kind == "c3_official2790_d5"
        expected_scope = (
            "last_2_backbone_and_head" if args.c3_adaptation == "last2"
            else "adapter_and_head" if args.c3_adaptation == "lora_m" else None
        )
        if (
            args.policy_objective != "discrete_domino"
            or args.formal_rl is not True
            or args.distributed is not True
            or (
                args.updates != 1395
                and not (
                    scale_resume
                    and (
                        (
                            args.scale_resume_kind == "c3_official2790"
                            and args.updates in {1744, 2093}
                        )
                        or (
                            args.scale_resume_kind == "c3_official2790_d5"
                            and args.updates in {1744, 2093, 2442, 2790, 3139, 3488, 3837, 4185}
                        )
                    )
                )
            )
            or args.tasks_per_update != 4
            or args.candidates != 8
            or args.trajectory_steps != 8
            or args.policy_epochs != 2
            or args.reward != "terminal"
            or args.credit_assignment != "uniform"
            or args.task_curriculum != "deterministic_shuffle"
            or args.gradient_checkpointing
            or expected_scope is None
            or args.rl_trainable_scope != expected_scope
            or not math.isclose(args.learning_rate, 5e-6, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(args.kl_coefficient, 0.01, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(args.ce_coefficient, 0.1, rel_tol=0, abs_tol=1e-15)
            or args.c3_lora_rank <= 0
            or args.c3_lora_alpha <= 0
            or args.c3_lora_dropout < 0
            or (c3_d5_scale and args.c3_adaptation != "last2")
            or (c3_d5_scale and not d5_strict_on_policy)
            or (not c3_d5_scale and d5_strict_on_policy)
            or args.pause_after_update not in {None, 698}
        ):
            raise ValueError("C3 Discrete-DoMinO scale contract is invalid")
    elif args.c3_adaptation:
        raise ValueError("C3 adaptation controls require a frozen C3 policy initializer")
    if (
        args.pause_after_update is not None
        and not c3_policy_init
        and not (
            scale_resume
            and d3_policy_init
            and args.scale_resume_kind in {"t0_flow_yrl", "t0_flow_yrl_d5"}
        )
    ):
        raise ValueError("pause controls require a frozen scaling continuation")
    if bool(args.parent_checkpoint) != bool(args.parent_receipt):
        raise ValueError("parent checkpoint and parent receipt must be provided together")
    branch_arguments = (
        args.branch_resume_checkpoint,
        args.branch_resume_receipt,
        args.branch_resume_review_receipt,
        args.branch_learning_rate_mode,
        args.branch_kl_coefficient_mode,
    )
    if any(branch_arguments) and not all(branch_arguments):
        raise ValueError(
            "branch resume requires checkpoint, source receipt, review receipt, and learning-rate mode"
        )
    if args.resume and any(branch_arguments):
        raise ValueError("in-place resume and branch resume are mutually exclusive")
    if any(branch_arguments) and not args.parent_checkpoint:
        raise ValueError("branch resume requires the original completed R4 parent")
    if args.branch_update_budget_mode != "u192_legacy" and not any(branch_arguments):
        raise ValueError(
            "a non-legacy branch update budget requires an explicit branch resume"
        )
    contrastive_in_place_contract = (
        load_contrastive_in_place_contract(args.output)
        if args.resume and args.policy_objective == "contrastive_task_step" else None
    )
    sequence_group_fpo_in_place_contract = (
        load_sequence_group_fpo_in_place_contract(args.output)
        if args.resume and args.policy_objective == "sequence_group_fpo" else None
    )
    validate_branch_resume_stage_arguments(
        all(branch_arguments), args.preflight_updates, args.branch_preflight_receipt,
        contrastive_in_place_resume=(
            args.resume is not None
            and args.policy_objective == "contrastive_task_step"
        ),
    )
    if not c3_policy_init and not d3_policy_init:
        validate_policy_objective_arguments(
            args, all(branch_arguments), contrastive_in_place_contract
        )
    branch_target_updates = (
        128 if args.branch_update_budget_mode in {
            "u128_matched32", "u128_domino32", "u128_discrete_domino32"
        } else
        144 if args.branch_update_budget_mode == "u144_matched48" else
        144
        if args.policy_objective == "contrastive_task_step"
        or args.branch_kl_coefficient_mode == "tenfold_pilot48"
        else 192
    )
    if any(branch_arguments) and (
        args.formal_rl is not True
        or args.updates != branch_target_updates
        or (args.preflight_updates is not None and args.preflight_updates != 1)
    ):
        raise ValueError(
            "branch resume must use its reviewed update budget; preflight may run exactly one update"
        )
    if args.branch_update_budget_mode == "u144_matched48" and any(branch_arguments):
        if (
            args.temporal_credit not in {"causal_is_single", "causal_window_2"}
            or args.policy_objective != "normalized_grpo"
            or args.branch_learning_rate_mode != "half"
            or args.branch_kl_coefficient_mode != "fixed"
            or args.task_curriculum != "deterministic_shuffle"
            or args.tasks_per_update != 4
            or args.candidates != 12
            or args.trajectory_steps != 8
            or args.policy_epochs != 2
            or args.reward != "terminal"
            or args.credit_assignment != "uniform"
        ):
            raise ValueError(
                "u144_matched48 requires the frozen causal normalized-GRPO branch contract"
            )
    if args.branch_update_budget_mode == "u128_domino32" and any(branch_arguments):
        if (
            args.policy_objective != "domino_endpoint_ppo"
            or args.temporal_credit != "all_steps"
            or args.branch_learning_rate_mode != "half"
            or args.branch_kl_coefficient_mode != "fixed"
            or args.seed != 1009
            or args.task_curriculum != "deterministic_shuffle"
            or args.tasks_per_update != 4
            or args.candidates != 12
            or args.trajectory_steps != 8
            or args.policy_epochs != 2
            or args.reward != "terminal"
            or args.credit_assignment != "uniform"
            or not math.isclose(args.learning_rate, 5e-6, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(args.kl_coefficient, 0.01, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(args.ce_coefficient, 0.1, rel_tol=0, abs_tol=1e-15)
            or args.rl_trainable_scope != "last_2_backbone_and_head"
        ):
            raise ValueError("u128_domino32 requires the scoped DoMinO endpoint contract")
    if args.branch_update_budget_mode == "u128_discrete_domino32" and any(branch_arguments):
        if (
            args.policy_objective != "discrete_domino"
            or args.temporal_credit != "all_steps"
            or args.branch_learning_rate_mode != "half"
            or args.branch_kl_coefficient_mode != "fixed"
            or args.seed != 1009
            or args.task_curriculum != "deterministic_shuffle"
            or args.tasks_per_update != 4
            or args.candidates != 12
            or args.trajectory_steps != 8
            or args.policy_epochs != 2
            or args.reward != "terminal"
            or args.credit_assignment != "uniform"
            or not math.isclose(args.learning_rate, 5e-6, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(args.kl_coefficient, 0.01, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(args.ce_coefficient, 0.1, rel_tol=0, abs_tol=1e-15)
            or args.rl_trainable_scope != "last_2_backbone_and_head"
        ):
            raise ValueError(
                "u128_discrete_domino32 requires the structured Discrete-DoMinO contract"
            )
    if args.branch_update_budget_mode == "u128_matched32" and any(branch_arguments):
        b0 = args.policy_objective == "normalized_grpo"
        b1 = args.policy_objective == "sequence_group_fpo"
        if (
            not (b0 or b1)
            or args.branch_learning_rate_mode != "half"
            or args.branch_kl_coefficient_mode != "fixed"
            or args.seed != 1009
            or args.task_curriculum != "deterministic_shuffle"
            or args.tasks_per_update != 4
            or args.candidates != 12
            or args.trajectory_steps != 8
            or args.policy_epochs != 2
            or args.reward != "terminal"
            or args.credit_assignment != "uniform"
            or not math.isclose(args.learning_rate, 5e-6, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(args.kl_coefficient, 0.01, rel_tol=0, abs_tol=1e-15)
            or not math.isclose(args.ce_coefficient, 0.1, rel_tol=0, abs_tol=1e-15)
            or args.rl_trainable_scope != "last_2_backbone_and_head"
            or (b0 and args.temporal_credit != "causal_is_single")
            or (b1 and args.temporal_credit != "all_steps")
        ):
            raise ValueError("u128_matched32 requires the frozen Stage-B B0/B1 contract")
    if args.resume and args.parent_checkpoint and args.resume.resolve() == args.parent_checkpoint.resolve():
        raise ValueError("parent checkpoint initializes a fork and cannot be used as --resume")
    if args.temporal_credit != "all_steps" and args.trajectory_steps != 8:
        raise ValueError("single-step temporal credit requires --trajectory-steps 8")
    causal_arguments = bool(args.causal_step_policy) and bool(
        args.expected_causal_step_policy_sha256
    )
    causal_mode = args.temporal_credit in {"causal_is_single", "causal_window_2"}
    if causal_mode != causal_arguments:
        raise ValueError(
            "causal temporal credit requires a policy and expected SHA; other modes forbid them"
        )
    if bool(args.causal_step_policy) != bool(args.expected_causal_step_policy_sha256):
        raise ValueError("causal step policy path and expected SHA must be provided together")
    if causal_mode and not args.parent_checkpoint:
        raise ValueError("causal temporal credit requires a completed R4 parent fork")
    if args.task_curriculum == "online_learnability" and (
        args.temporal_credit != "uniform_single"
        or args.reward != "terminal"
        or args.parent_checkpoint is None
    ):
        raise ValueError(
            "online learnability requires terminal reward, uniform_single temporal credit, "
            "and a completed parent fork"
        )
    branch_formal_preflight = bool(args.branch_resume_checkpoint) and (
        args.formal_rl is True and args.preflight_updates == 1
    )
    c3_formal_preflight = c3_policy_init and (
        args.formal_rl is True and args.preflight_updates == 1
    )
    d3_formal_preflight = d3_policy_init and (
        args.formal_rl is True and args.preflight_updates == 1
    )
    if (
        args.formal_rl == (args.preflight_updates is not None)
        and not branch_formal_preflight
        and not c3_formal_preflight
        and not d3_formal_preflight
    ):
        raise ValueError("choose formal RL or an explicitly bounded preflight")
    if args.formal_rl and args.rl_trainable_scope not in {
        "adapter_and_head", "last_2_backbone_and_head", "full_backbone_and_head",
        "inherited",
    }:
        raise ValueError("formal trajectory RL trainable scope is unsupported")
    repository_root = Path(__file__).resolve().parents[2]
    if args.formal_rl:
        require_clean_formal_source(repository_root)

    context = distributed_context(args.distributed, args.device)
    rank, world_size = context.rank, context.world_size
    if args.policy_objective == "contrastive_task_step" and world_size != 4:
        raise ValueError("contrastive_task_step requires the frozen world size 4")
    if args.policy_objective in {"domino_endpoint_ppo", "discrete_domino"} and world_size != 2:
        raise ValueError("C1 DoMinO pilots require the approved DDP2 world size")
    local_tasks_per_rank, local_supervised_batch_size = require_even_global_batches(
        args.tasks_per_update, args.supervised_batch_size, world_size
    )

    supervised = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
    supervised_contract = supervised.get("contract")
    allowed_methods = {
        "pre-backbone-lora-dotbracket-x0-flow",
        "pre-backbone-full-dotbracket-x0-flow",
    }
    if (
        not supervised_contract
        or supervised.get("contract_sha256") != contract_sha256(supervised_contract)
        or supervised_contract.get("method") not in allowed_methods
        or supervised_contract.get("rnaernie_revision") != args.rnaernie_revision
        or supervised_contract.get("rnaernie_weight_sha256")
        != sha256(args.rnaernie / "model.safetensors")
    ):
        raise RuntimeError("unsupported or corrupt supervised trajectory initialization")
    if args.formal_rl:
        if args.supervised_selection_receipt is None:
            raise ValueError("formal trajectory RL requires a supervised selection receipt")
        validate_authorized_checkpoint(
            args.supervised_selection_receipt, args.initialize_from
        )

    parent_identity = None
    if args.parent_checkpoint:
        parent_identity = primary_call(
            context,
            "parent R4 endpoint checkpoint validation",
            lambda: validate_completed_r4_endpoint_parent(
                args.parent_checkpoint,
                args.parent_receipt,
                supervised_checkpoint_sha256=sha256(args.initialize_from),
                supervised_contract_sha256=supervised["contract_sha256"],
                rnaernie_revision=args.rnaernie_revision,
                rl_trainable_scope=args.rl_trainable_scope,
            ),
        )
        if (
            args.task_curriculum == "online_learnability"
            and parent_identity["temporal_credit_mode"] != "uniform_single"
        ):
            raise RuntimeError("online learnability parent must use uniform_single temporal credit")
    causal_step_policy = (
        primary_call(
            context,
            "causal step policy validation",
            lambda: load_causal_step_policy(
                args.causal_step_policy,
                args.expected_causal_step_policy_sha256,
                parent_identity,
            ),
        )
        if args.causal_step_policy else None
    )
    branch_resume_identity = None
    if args.branch_resume_checkpoint:
        def inspect_branch_resume_source() -> dict:
            payload = torch.load(
                args.branch_resume_checkpoint, map_location="cpu", weights_only=False
            )
            receipt = json.loads(args.branch_resume_receipt.read_text())
            identity = validate_branch_resume_checkpoint(
                payload, receipt, args.branch_resume_checkpoint, world_size
            )
            return identity | {
                "checkpoint_sha256": sha256(args.branch_resume_checkpoint),
                "receipt_sha256": sha256(args.branch_resume_receipt),
            }

        branch_resume_identity = primary_call(
            context, "branch-resume source validation", inspect_branch_resume_source
        )

    c3_policy_init_identity = None
    if c3_policy_init:
        def inspect_c3_policy_initializer() -> dict:
            payload = torch.load(
                args.c3_policy_init_checkpoint, map_location="cpu", weights_only=False
            )
            receipt = json.loads(args.c3_policy_init_receipt.read_text())
            identity = validate_branch_resume_checkpoint(
                payload, receipt, args.c3_policy_init_checkpoint, world_size
            )
            if identity["start_update"] != 96:
                raise RuntimeError("C3 policy initializer must be the frozen U96 checkpoint")
            return identity | {
                "checkpoint_sha256": sha256(args.c3_policy_init_checkpoint),
                "receipt_sha256": sha256(args.c3_policy_init_receipt),
            }

        c3_policy_init_identity = primary_call(
            context, "C3 U96 policy initializer validation", inspect_c3_policy_initializer
        )

    d3_policy_init_identity = None
    if d3_policy_init:
        def inspect_d3_policy_initializer() -> dict:
            payload = torch.load(
                args.d3_policy_init_checkpoint, map_location="cpu", weights_only=False
            )
            receipt = json.loads(args.d3_policy_init_receipt.read_text())
            identity = validate_branch_resume_checkpoint(
                payload, receipt, args.d3_policy_init_checkpoint, world_size
            )
            # ``validate_branch_resume_checkpoint`` binds checkpoint SHA,
            # receipt SHA, full AdamW state, rank RNG, cursor and U96 lineage.
            if identity["start_update"] != 96:
                raise RuntimeError("D3 policy initializer must be the frozen U96 checkpoint")
            return identity | {
                "checkpoint_sha256": sha256(args.d3_policy_init_checkpoint),
                "receipt_sha256": sha256(args.d3_policy_init_receipt),
            }

        d3_policy_init_identity = primary_call(
            context, "D3 U96 policy initializer validation", inspect_d3_policy_initializer
        )

    scale_resume_identity = None
    if scale_resume:
        def inspect_scale_resume_source() -> dict:
            payload = torch.load(
                args.scale_resume_checkpoint, map_location="cpu", weights_only=False
            )
            receipt = json.loads(args.scale_resume_receipt.read_text())
            identity = validate_scale_resume_source(
                payload,
                receipt,
                args.scale_resume_checkpoint,
                args.scale_resume_kind,
                args.updates,
                world_size,
            )
            return identity | {
                "receipt_sha256": sha256(args.scale_resume_receipt),
            }

        scale_resume_identity = primary_call(
            context, "coverage scale-resume source validation", inspect_scale_resume_source
        )

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    tasks_manifest = (
        validate_d3_flow_yrl_tasks_manifest(args.tasks_manifest, args.tasks, tasks)
        if d3_policy_init else validate_tasks_manifest(args.tasks_manifest, args.tasks, len(tasks))
    )
    supervised_data = FlowDataset(args.supervised_train, "dotbracket", None)
    if args.tasks_per_update > len(tasks) or args.supervised_batch_size > len(supervised_data):
        raise ValueError("trajectory or supervised batch exceeds its dataset")
    task_ids = [str(task["id"]) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("trajectory task identifiers are not unique")
    d3_monitor_ids = d3_train_monitor_ids(tasks) if d3_policy_init else None

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    current_implementation_manifest = implementation_manifest()
    branch_resume_review_sha256 = None
    branch_preflight_receipt_sha256 = None
    formal_branch_gate = None
    allowed_implementation_changes = None
    if branch_resume_identity is not None:
        requested_implementation_contract = {
            "implementation_manifest": current_implementation_manifest,
            "repository_revision": revision,
            "branch_update_budget_mode": args.branch_update_budget_mode,
            "script_sha256": sha256(Path(__file__)),
            "endpoint_policy_sha256": sha256(
                Path(__file__).with_name("endpoint_policy.py")
            ),
            "endpoint_trajectory_ddp_sha256": sha256(
                Path(__file__).with_name("endpoint_trajectory_ddp.py")
            ),
        }
        allowed_implementation_changes = branch_resume_implementation_changes(
            branch_resume_identity["contract"], requested_implementation_contract
        )

        def inspect_branch_resume_review() -> str:
            review = json.loads(args.branch_resume_review_receipt.read_text())
            validate_branch_resume_review_receipt(
                review,
                branch_resume_identity["contract"],
                requested_implementation_contract,
                branch_resume_identity["contract_sha256"],
                branch_resume_identity["checkpoint_sha256"],
                branch_resume_identity["receipt_sha256"],
            )
            return sha256(args.branch_resume_review_receipt)

        branch_resume_review_sha256 = primary_call(
            context, "branch-resume independent review validation", inspect_branch_resume_review
        )
    contract = {
        "schema_version": (
            11 if args.policy_objective == "discrete_domino"
            else 10 if args.policy_objective == "domino_endpoint_ppo"
            else 9 if args.policy_objective == "sequence_group_fpo"
            else 8 if args.policy_objective == "contrastive_task_step"
            else 7 if branch_resume_identity is not None else 6
        ),
        "method": (
            DISCRETE_DOMINO_METHOD
            if args.policy_objective == "discrete_domino"
            else DOMINO_ENDPOINT_PPO_METHOD
            if args.policy_objective == "domino_endpoint_ppo"
            else SEQUENCE_GROUP_FPO_METHOD
            if args.policy_objective == "sequence_group_fpo"
            else CONTRASTIVE_TASK_STEP_METHOD
            if args.policy_objective == "contrastive_task_step"
            else "simplex-endpoint-policy-trajectory-grpo"
        ),
        "action_policy": (
            "structured-discrete-mixture-next-state"
            if args.policy_objective == "discrete_domino"
            else "categorical-single-paired-categorical-clean-endpoint"
        ),
        "transition": (
            "finite-step-discrete-mixture-kernel"
            if args.policy_objective == "discrete_domino"
            else "sampled-endpoint-dirichlet-conditional-velocity"
        ),
        "ratio": (
            DISCRETE_DOMINO_RATIO
            if args.policy_objective == "discrete_domino"
            else DOMINO_ENDPOINT_PPO_RATIO
            if args.policy_objective == "domino_endpoint_ppo"
            else SEQUENCE_GROUP_FPO_RATIO
            if args.policy_objective == "sequence_group_fpo"
            else CONTRASTIVE_TASK_STEP_RATIO
            if args.policy_objective == "contrastive_task_step"
            else "per-flow-step-joint-action-clipped-surrogate"
        ),
        "not_claimed_equivalent_to_gaussian_flow_grpo": True,
        "formal_rl": args.formal_rl,
        "reward": args.reward,
        "credit_assignment": args.credit_assignment,
        "structure_credit_strength": args.structure_credit_strength,
        "rl_trainable_scope": args.rl_trainable_scope,
        "supervised_checkpoint_sha256": sha256(args.initialize_from),
        "supervised_contract_sha256": supervised["contract_sha256"],
        "supervised_selection_receipt_sha256": (
            sha256(args.supervised_selection_receipt) if args.supervised_selection_receipt else None
        ),
        "tasks_sha256": sha256(args.tasks),
        "tasks_manifest_sha256": sha256(args.tasks_manifest),
        "tasks_source_revision": (
            tasks_manifest["validated_source_revision"]
            if d3_policy_init else tasks_manifest["source"]["revision"]
        ),
        "supervised_train_sha256": sha256(args.supervised_train),
        "rnaernie_revision": args.rnaernie_revision,
        "rnaernie_weight_sha256": supervised_contract["rnaernie_weight_sha256"],
        "seed": args.seed,
        "updates": args.updates,
        "branch_update_budget_mode": (
            args.branch_update_budget_mode if branch_resume_identity is not None else None
        ),
        "tasks_per_update": args.tasks_per_update,
        "global_tasks_per_update": args.tasks_per_update,
        "local_tasks_per_rank": local_tasks_per_rank,
        "candidates": args.candidates,
        "candidates_per_task": args.candidates,
        "global_candidate_rollouts_per_update": args.tasks_per_update * args.candidates,
        "trajectory_steps": args.trajectory_steps,
        "temporal_credit": {
            "mode": "none" if args.policy_objective in {
                "sequence_group_fpo", "domino_endpoint_ppo", "discrete_domino"
            } else args.temporal_credit,
            "selection": "all-structured-discrete-dfm-transitions"
            if args.policy_objective == "discrete_domino"
            else "all-legal-flow-matching-steps-without-temporal-credit"
            if args.policy_objective in {"sequence_group_fpo", "domino_endpoint_ppo"}
            else "all-rollout-steps" if args.temporal_credit == "all_steps"
            else (
                "sha256(endpoint-causal-is-temporal-credit-v1:policy_sha256:seed:update:global_task_offset)"
                if args.temporal_credit in {"causal_is_single", "causal_window_2"}
                else "sha256(endpoint-temporal-credit-v1:seed:update:global_task_offset)"
            ),
            "causal_window_size": (
                None if args.temporal_credit not in {"causal_is_single", "causal_window_2"}
                else 1 if args.temporal_credit == "causal_is_single" else 2
            ),
            "single_step_population": (
                None if args.temporal_credit == "all_steps"
                else 4 if args.temporal_credit == "informative_early_single"
                else 8
            ),
            "causal_step_policy_sha256": (
                None if causal_step_policy is None else causal_step_policy["sha256"]
            ),
            "causal_step_ticket_counts": (
                None if causal_step_policy is None else causal_step_policy["ticket_counts"]
            ),
            "source_aggregate_sha256": (
                None if causal_step_policy is None
                else causal_step_policy["source_aggregate_sha256"]
            ),
            "probability_by_step": (
                None if causal_step_policy is None
                else causal_step_policy["probability_by_step"]
            ),
            "importance_weight": (
                None if causal_step_policy is None
                else "1/(8*p_t)" if args.temporal_credit == "causal_window_2"
                else "1/(8*q_s)"
            ),
            "importance_weight_scope": (
                None if causal_step_policy is None else "policy+KL-entropy; CE excluded"
            ),
        },
        "task_curriculum": {
            "mode": args.task_curriculum,
            "algorithm": (
                "deterministic-epoch-shuffle"
                if args.task_curriculum == "deterministic_shuffle"
                else ONLINE_CURRICULUM_ALGORITHM
            ),
            "warmup": (
                None if args.task_curriculum == "deterministic_shuffle"
                else "one-complete-deterministic-shuffle-pass-over-TRAIN-only-tasks"
            ),
            "statistics": (
                None if args.task_curriculum == "deterministic_shuffle"
                else "checkpointed-terminal-reward-count-sum-square-sum"
            ),
            "selection": (
                None if args.task_curriculum == "deterministic_shuffle"
                else "deterministic-weighted-top-k-without-replacement"
            ),
            "selection_key": (
                None if args.task_curriculum == "deterministic_shuffle"
                else "sha256(online-task-curriculum-v1:seed:update:task-index)"
            ),
            "minimum_exploration_probability": (
                None if args.task_curriculum == "deterministic_shuffle"
                else ONLINE_CURRICULUM_MIN_EXPLORATION_PROBABILITY
            ),
            "data_scope": "TRAIN-only",
        },
        "parent_chain": (
            [] if parent_identity is None
            else parent_identity["parent_chain"] + [{
                "checkpoint_sha256": parent_identity["checkpoint_sha256"],
                "receipt_sha256": parent_identity["receipt_sha256"],
                "contract_sha256": parent_identity["contract_sha256"],
                "updates_complete": parent_identity["updates_complete"],
            }]
        ),
        "parent_checkpoint_sha256": (
            None if parent_identity is None else parent_identity["checkpoint_sha256"]
        ),
        "parent_receipt_sha256": (
            None if parent_identity is None else parent_identity["receipt_sha256"]
        ),
        "parent_contract_sha256": (
            None if parent_identity is None else parent_identity["contract_sha256"]
        ),
        "policy_initialization": (
            None if c3_policy_init_identity is None and d3_policy_init_identity is None else {
                "mode": (
                    "u96-policy-weights-fresh-optimizer-c3-scale"
                    if c3_policy_init_identity is not None
                    else "u96-policy-weights-fresh-optimizer-d3-256"
                ),
                "source_checkpoint_sha256": (
                    c3_policy_init_identity or d3_policy_init_identity
                )["checkpoint_sha256"],
                "source_receipt_sha256": (
                    c3_policy_init_identity or d3_policy_init_identity
                )["receipt_sha256"],
                "source_contract_sha256": (
                    c3_policy_init_identity or d3_policy_init_identity
                )["contract_sha256"],
                "source_update": 96,
                "optimizer_state_restored": False,
                "task_cursor_restored": False,
                "reference_policy": "frozen-u96",
                "adaptation": (
                    args.c3_adaptation if c3_policy_init_identity is not None
                    else "last2-fresh-init"
                ),
                "lora_rank": args.c3_lora_rank if c3_policy_init_identity is not None and args.c3_adaptation == "lora_m" else 0,
                "lora_alpha": args.c3_lora_alpha if c3_policy_init_identity is not None and args.c3_adaptation == "lora_m" else 0.0,
                "lora_dropout": args.c3_lora_dropout if c3_policy_init_identity is not None and args.c3_adaptation == "lora_m" else 0.0,
                "lora_target_profile": "all_attention_ffn" if c3_policy_init_identity is not None and args.c3_adaptation == "lora_m" else None,
            }
        ),
        "continuation": (
            {
                "mode": "coverage-scale-exact-state-v1",
                "kind": scale_resume_identity["kind"],
                "source_checkpoint_sha256": scale_resume_identity["checkpoint_sha256"],
                "source_receipt_sha256": scale_resume_identity["receipt_sha256"],
                "source_contract_sha256": scale_resume_identity["contract_sha256"],
                "source_repository_revision": scale_resume_identity["source_contract"].get("repository_revision"),
                "source_target_updates": scale_resume_identity["source_contract"].get("updates"),
                "start_update": scale_resume_identity["start_update"],
                "optimizer_state_restored": True,
                "rng_state_by_rank_restored": True,
                "source_world_size": scale_resume_identity["world_size"],
                "target_world_size": world_size,
                "task_cursor_and_coverage_restored": True,
                "scheduler": "none",
            }
            if scale_resume_identity is not None else
            copy.deepcopy(contrastive_in_place_contract["contract"]["continuation"])
            if contrastive_in_place_contract is not None else
            None if branch_resume_identity is None else {
                "mode": "completed-branch-exact-state-v1",
                "source_checkpoint_sha256": branch_resume_identity[
                    "checkpoint_sha256"
                ],
                "source_receipt_sha256": branch_resume_identity["receipt_sha256"],
                "source_contract_sha256": branch_resume_identity["contract_sha256"],
                "start_update": branch_resume_identity["start_update"],
                "allowed_implementation_changes": allowed_implementation_changes,
                "resume_plumbing_review_sha256": branch_resume_review_sha256,
                "optimizer_state_restored": True,
                "rng_state_by_rank_restored": (
                    branch_resume_identity["source_world_size"]
                    == branch_resume_identity["target_world_size"]
                ),
                "rng_migration": branch_resume_identity["rng_migration"],
                "source_world_size": branch_resume_identity["source_world_size"],
                "target_world_size": branch_resume_identity["target_world_size"],
                "task_cursor_and_coverage_restored": True,
                "optimizer_parameter_manifest_sha256": branch_resume_identity[
                    "parameter_manifest_sha256"
                ],
                "learning_rate_mode": args.branch_learning_rate_mode,
                "source_learning_rate": branch_resume_identity["source_learning_rate"],
                "requested_learning_rate": args.learning_rate,
                "kl_coefficient_mode": args.branch_kl_coefficient_mode,
                "source_kl_coefficient": branch_resume_identity["contract"][
                    "kl_coefficient"
                ],
                "requested_kl_coefficient": args.kl_coefficient,
                "scheduler": "none",
            }
        ),
        "posthoc_exploratory": not d3_policy_init,
        "policy_epochs": args.policy_epochs,
        "temperatures": args.temperatures,
        "supervised_batch_size": args.supervised_batch_size,
        "global_supervised_batch_size": args.supervised_batch_size,
        "local_supervised_batch_size": local_supervised_batch_size,
        "distributed": {
            "enabled": args.distributed,
            "world_size": world_size,
            "backend": context.backend,
            "task_assignment": "contiguous-global-task-offsets",
            "supervised_assignment": "contiguous-global-batch-offsets",
            "policy_loss_normalization": "global-task-mean-via-ddp-average",
            "ce_loss_normalization": "global-valid-token-mean-via-ddp-average",
            "supervised_path_randomness": "keyed-global-row-dirichlet(seed,update,policy_epoch,global_row)",
            "find_unused_parameters": True if args.distributed else None,
        },
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "clip_ratio": args.clip_ratio,
        "gradient_clip_norm": 1.0,
        "kl_coefficient": args.kl_coefficient,
        "ce_coefficient": args.ce_coefficient,
        "entropy_coefficient": args.entropy_coefficient,
        "group_objective": args.group_objective,
        "pkpo": (
            None if not d3_policy_init else {
                "algorithm": "continuous-max-at-k-sloo-minus-one",
                "schedule": {"0-84": 8, "85-169": 4, "170-255": 1},
                "k1": "leave-one-out-mean-centered-raw-reward",
                "group_std_normalization": False,
                "pass_at_8_bonus": False,
            }
        ),
        "reference_transition_tv": (
            None if not d3_policy_init else {
                "coefficient": args.reference_tv_coefficient,
                "distribution": "structured-clean-categorical-4-or-6-state",
                "transition": "rho-times-posterior-tv",
                "reference_detached": True,
                "claim": "TV-inspired/reference-transition; not native DFM theory",
            }
        ),
        "train_health_monitor": (
            None if d3_monitor_ids is None else {
                "scope": "TRAIN-only; health monitoring only; no external selection",
                "ids": d3_monitor_ids,
                "ids_sha256": hashlib.sha256(
                    json.dumps(d3_monitor_ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        ),
        "immutable_checkpoint_every": args.immutable_checkpoint_every,
        "gradient_checkpointing": args.gradient_checkpointing,
        "group_backward": "taskwise-exact-mean-gradient",
        "repository_revision": revision,
        "implementation_manifest": current_implementation_manifest,
        "script_sha256": sha256(Path(__file__)),
        "endpoint_trajectory_ddp_sha256": sha256(
            Path(__file__).with_name("endpoint_trajectory_ddp.py")
        ),
        "endpoint_policy_sha256": sha256(Path(__file__).with_name("endpoint_policy.py")),
        "evaluator_sha256": sha256(FAIR_DIR / "evaluate.py"),
        "reward_evaluation_policy": REWARD_EVALUATION_POLICY,
        "maximum_invalid_reward_evaluation_rate": MAXIMUM_INVALID_REWARD_EVALUATION_RATE,
        "viennarna": {
            "version": "2.7.2",
            "temperature_c": 37.0,
            "dangles": 2,
            "uniq_ML": 1,
            "target_probability_roundoff_tolerance": 0.01,
        },
    }
    # Do not add a null field to historical T0/T1 contracts: exact in-place
    # resume must keep their frozen hashes byte-for-byte compatible.
    if args.group_objective == D4_HYBRID_PKPO_OBJECTIVE:
        contract["pkpo_residual"] = {
            "coefficient": args.pkpo_residual_coefficient,
            "formula": "normalized_group_advantages(rewards) + coefficient * continuous_maxk_advantages(rewards, k)",
            "raw_pkpo_normalization": "none",
            "thermodynamic_signal": "dense-centered-normalized-group-advantages",
        }
    if d5_strict_on_policy:
        contract["rollout_refresh_mode"] = D5_ROLLOUT_REFRESH_MODE
        contract["rollout_refresh"] = d5_rollout_refresh_contract()
    if args.policy_objective == "contrastive_task_step":
        contract["policy_objective"] = contrastive_task_step_objective_contract()
        contract["continuation"]["policy_objective"] = "contrastive_task_step"
    elif args.policy_objective == "sequence_group_fpo":
        contract["policy_objective"] = sequence_group_fpo_objective_contract()
        if branch_resume_identity is not None:
            contract["continuation"]["policy_objective"] = "sequence_group_fpo"
    elif args.policy_objective == "domino_endpoint_ppo":
        contract["policy_objective"] = domino_endpoint_ppo_objective_contract()
        if branch_resume_identity is not None:
            contract["continuation"]["policy_objective"] = "domino_endpoint_ppo"
    elif args.policy_objective == "discrete_domino":
        contract["policy_objective"] = discrete_domino_objective_contract()
        if branch_resume_identity is not None:
            contract["continuation"]["policy_objective"] = "discrete_domino"
    if scale_resume_identity is not None:
        validate_scale_resume_requested_contract(
            scale_resume_identity["source_contract"], contract
        )
    if sequence_group_fpo_in_place_contract is not None:
        # The resume contract is immutable; generic resume validation below
        # additionally rejects a foreign checkpoint before model/AdamW/RNG4
        # restoration.
        contract = sequence_group_fpo_in_place_contract["contract"]
    if contrastive_in_place_contract is not None:
        validate_in_place_requested_contract(contract, contrastive_in_place_contract)
    contract_hash = contract_sha256(contract)
    if branch_resume_identity is not None:
        validate_branch_resume_contract(
            branch_resume_identity["contract"],
            contract,
            args.branch_learning_rate_mode,
            args.branch_kl_coefficient_mode,
        )
        if args.preflight_updates is None:
            def inspect_branch_preflight_receipt() -> str:
                preflight_receipt = json.loads(args.branch_preflight_receipt.read_text())
                validate_branch_resume_preflight_receipt(
                    preflight_receipt,
                    args.branch_preflight_receipt,
                    contract,
                    contract_hash,
                    branch_resume_identity["contract_sha256"],
                    branch_resume_identity["checkpoint_sha256"],
                    branch_resume_identity["receipt_sha256"],
                    branch_resume_review_sha256,
                    args.output,
                )
                return sha256(args.branch_preflight_receipt)

            branch_preflight_receipt_sha256 = primary_call(
                context,
                "branch-resume exact preflight validation",
                inspect_branch_preflight_receipt,
            )
            if args.policy_objective == "contrastive_task_step":
                formal_branch_gate = contrastive_branch_gate(
                    contract["continuation"], branch_preflight_receipt_sha256
                )
    elif contrastive_in_place_contract is not None:
        def inspect_in_place_preflight_and_gate() -> tuple[str, dict]:
            preflight_receipt = json.loads(args.branch_preflight_receipt.read_text())
            continuation = contract["continuation"]
            validate_branch_resume_preflight_receipt(
                preflight_receipt,
                args.branch_preflight_receipt,
                contract,
                contract_hash,
                continuation["source_contract_sha256"],
                continuation["source_checkpoint_sha256"],
                continuation["source_receipt_sha256"],
                continuation["resume_plumbing_review_sha256"],
                args.output,
            )
            digest = sha256(args.branch_preflight_receipt)
            return digest, load_contrastive_branch_gate(
                args.output, continuation, digest
            )

        branch_preflight_receipt_sha256, formal_branch_gate = primary_call(
            context,
            "contrastive in-place preflight and branch-gate validation",
            inspect_in_place_preflight_and_gate,
        )
    branch_resume_payload = None
    scale_resume_payload = None
    c3_policy_init_payload = None
    d3_policy_init_payload = None
    if c3_policy_init_identity is not None:
        def load_c3_policy_init_payload() -> dict:
            payload = torch.load(
                args.c3_policy_init_checkpoint, map_location="cpu", weights_only=False
            )
            if (
                sha256(args.c3_policy_init_checkpoint)
                != c3_policy_init_identity["checkpoint_sha256"]
                or sha256(args.c3_policy_init_receipt)
                != c3_policy_init_identity["receipt_sha256"]
                or payload.get("contract_sha256")
                != c3_policy_init_identity["contract_sha256"]
            ):
                raise RuntimeError("C3 U96 policy initializer changed after validation")
            return payload

        c3_policy_init_payload = rank_local_call(
            context, "C3 U96 policy initializer load", load_c3_policy_init_payload
        )
    if d3_policy_init_identity is not None:
        def load_d3_policy_init_payload() -> dict:
            payload = torch.load(
                args.d3_policy_init_checkpoint, map_location="cpu", weights_only=False
            )
            if (
                sha256(args.d3_policy_init_checkpoint)
                != d3_policy_init_identity["checkpoint_sha256"]
                or sha256(args.d3_policy_init_receipt)
                != d3_policy_init_identity["receipt_sha256"]
                or payload.get("contract_sha256")
                != d3_policy_init_identity["contract_sha256"]
            ):
                raise RuntimeError("D3 U96 policy initializer changed after validation")
            return payload

        d3_policy_init_payload = rank_local_call(
            context, "D3 U96 policy initializer load", load_d3_policy_init_payload
        )
    if branch_resume_identity is not None:
        def load_branch_resume_payload() -> dict:
            payload = torch.load(
                args.branch_resume_checkpoint, map_location="cpu", weights_only=False
            )
            receipt = json.loads(args.branch_resume_receipt.read_text())
            identity = validate_branch_resume_checkpoint(
                payload, receipt, args.branch_resume_checkpoint, world_size
            )
            if (
                checkpoint_file_sha256(args.branch_resume_checkpoint)
                != branch_resume_identity["checkpoint_sha256"]
                or sha256(args.branch_resume_receipt)
                != branch_resume_identity["receipt_sha256"]
                or identity["contract_sha256"]
                != branch_resume_identity["contract_sha256"]
                or identity["parameter_manifest_sha256"]
                != branch_resume_identity["parameter_manifest_sha256"]
                or sha256(args.branch_resume_review_receipt)
                != branch_resume_review_sha256
            ):
                raise RuntimeError("branch-resume source changed after validation")
            return payload

        branch_resume_payload = rank_local_call(
            context, "branch-resume checkpoint load", load_branch_resume_payload
        )
    if scale_resume_identity is not None:
        def load_scale_resume_payload() -> dict:
            payload = torch.load(
                args.scale_resume_checkpoint, map_location="cpu", weights_only=False
            )
            receipt = json.loads(args.scale_resume_receipt.read_text())
            identity = validate_scale_resume_source(
                payload,
                receipt,
                args.scale_resume_checkpoint,
                args.scale_resume_kind,
                args.updates,
                world_size,
            )
            if (
                identity["checkpoint_sha256"] != scale_resume_identity["checkpoint_sha256"]
                or identity["contract_sha256"] != scale_resume_identity["contract_sha256"]
                or sha256(args.scale_resume_receipt) != scale_resume_identity["receipt_sha256"]
            ):
                raise RuntimeError("coverage scale-resume source changed after validation")
            return payload

        scale_resume_payload = rank_local_call(
            context, "coverage scale-resume checkpoint load", load_scale_resume_payload
        )
    if args.resume:
        def validate_resume_output() -> None:
            if not args.output.is_dir():
                raise FileNotFoundError("trajectory resume output directory is absent")

        primary_call(context, "resume output validation", validate_resume_output)
        if d3_policy_init:
            def inspect_d3_resume() -> dict:
                payload = torch.load(args.resume, map_location="cpu", weights_only=False)
                validate_d3_resume_payload(payload, contract, contract_hash, world_size)
                return {"sha256": sha256(args.resume), "state": payload["state"]}

            d3_resume_metadata = primary_call(context, "D3 resume validation", inspect_d3_resume)

            def load_d3_resume() -> dict:
                payload = torch.load(args.resume, map_location="cpu", weights_only=False)
                validate_d3_resume_payload(payload, contract, contract_hash, world_size)
                if sha256(args.resume) != d3_resume_metadata["sha256"]:
                    raise RuntimeError("D3 resume checkpoint changed after validation")
                return payload

            resume_payload = rank_local_call(context, "D3 resume checkpoint load", load_d3_resume)
        else:
            resume_payload, _ = load_rank_local_resume_checkpoint(
                context, args.resume, contract, contract_hash
            )
    else:
        def initialize_output() -> None:
            args.output.mkdir(parents=True, exist_ok=False)
            atomic_json(args.output / "contract.json", contract | {"contract_sha256": contract_hash})
            if formal_branch_gate is not None:
                atomic_json(args.output / "branch_gate.json", formal_branch_gate)

        primary_call(context, "output initialization", initialize_output)
        resume_payload = None
    if branch_resume_payload is not None:
        resume_payload = branch_resume_payload
    if scale_resume_payload is not None:
        resume_payload = scale_resume_payload
    parent_payload = None
    if parent_identity is not None:
        def load_parent_payload() -> dict:
            payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
            if sha256(args.parent_checkpoint) != parent_identity["checkpoint_sha256"]:
                raise RuntimeError("parent checkpoint changed after validation")
            if sha256(args.parent_receipt) != parent_identity["receipt_sha256"]:
                raise RuntimeError("parent receipt changed after validation")
            if (
                payload.get("contract_sha256") != parent_identity["contract_sha256"]
                or not isinstance(payload.get("trainable_model"), dict)
            ):
                raise RuntimeError("parent checkpoint payload changed after validation")
            return payload

        parent_payload = rank_local_call(
            context, "parent R4 endpoint checkpoint load", load_parent_payload
        )
    rollout_dir = args.output / "rollouts"
    checkpoint_dir = args.output / "checkpoints"
    def initialize_directories() -> None:
        rollout_dir.mkdir(exist_ok=bool(args.resume))
        checkpoint_dir.mkdir(exist_ok=bool(args.resume))

    primary_call(context, "output directory initialization", initialize_directories)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_everything(args.seed + rank * 100_003)
    device = context.device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    current = load_model(supervised, args.rnaernie, device)
    reference = load_model(supervised, args.rnaernie, device)
    if c3_policy_init_payload is not None:
        # Reconstruct the exact U96 policy weights first.  Both C3 arms reset
        # AdamW/data cursor from this common policy state; only parameterization differs.
        configure_rl_trainable_scope(current, "last_2_backbone_and_head")
        configure_rl_trainable_scope(reference, "last_2_backbone_and_head")
        current.load_trainable_state_dict(c3_policy_init_payload["trainable_model"])
        reference.load_trainable_state_dict(c3_policy_init_payload["trainable_model"])
        if args.c3_adaptation == "lora_m":
            if adapt_loaded_supervised_model_with_lora is None:
                raise RuntimeError("LoRA-M helper is unavailable in this model implementation")
            adapt_loaded_supervised_model_with_lora(
                current,
                rank=args.c3_lora_rank,
                alpha=args.c3_lora_alpha,
                dropout=args.c3_lora_dropout,
                target_profile="all_attention_ffn",
            )
            # LoRA A/B modules are created after the base model was moved; move
            # the newly injected modules onto the active rank device before DDP.
            current.to(device)
            configure_rl_trainable_scope(current, "adapter_and_head")
        elif args.c3_adaptation != "last2":
            raise RuntimeError("unknown C3 adaptation after policy reconstruction")
    elif d3_policy_init_payload is not None:
        # D3 is a fresh branch at U96: policy/reference weights match U96 but
        # AdamW, task cursor and RNG are newly initialized below.
        configure_rl_trainable_scope(current, "last_2_backbone_and_head")
        configure_rl_trainable_scope(reference, "last_2_backbone_and_head")
        current.load_trainable_state_dict(d3_policy_init_payload["trainable_model"])
        reference.load_trainable_state_dict(d3_policy_init_payload["trainable_model"])
    else:
        configure_rl_trainable_scope(current, args.rl_trainable_scope)
        configure_rl_trainable_scope(reference, args.rl_trainable_scope)
        if parent_payload is not None:
            current.load_trainable_state_dict(parent_payload["trainable_model"])
            reference.load_trainable_state_dict(parent_payload["trainable_model"])
    if args.gradient_checkpointing:
        enable_deterministic_gradient_checkpointing(current)
    reference.requires_grad_(False)
    reference.eval()
    if resume_payload:
        current.load_trainable_state_dict(resume_payload["trainable_model"])
    trainable = [parameter for parameter in current.parameters() if parameter.requires_grad]
    optimizer_parameter_manifest = optimizer_parameter_manifest_from_model(current)
    if (
        branch_resume_identity is not None
        and optimizer_parameter_manifest != branch_resume_identity["parameter_manifest"]
    ):
        raise RuntimeError("branch-resume model/optimizer parameter order changed")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay, foreach=False
    )
    if resume_payload:
        if branch_resume_identity is not None:
            restore_branch_optimizer_state(
                optimizer, resume_payload["optimizer"], args.learning_rate
            )
        else:
            optimizer.load_state_dict(resume_payload["optimizer"])
        state = copy.deepcopy(resume_payload["state"])
        restore_rng_state(
            branch_rank_rng_state_from_checkpoint(resume_payload, context)
            if branch_resume_identity is not None
            else rank_rng_state_from_checkpoint(resume_payload, context)
        )
    else:
        state = {
            "next_update": 0,
            "task_epoch": 0,
            "task_cursor": 0,
            "reward_evaluation_coverage": empty_reward_evaluation_coverage(),
            "temporal_credit_step_histogram": {
                str(step): 0 for step in range(args.trajectory_steps)
            },
        }
        if d3_policy_init:
            state["d3_health"] = {
                "groups": 0,
                "pkpo_k_histogram": {"8": 0, "4": 0, "1": 0},
                "posterior_tv_sum": 0.0,
                "transition_tv_sum": 0.0,
                "transition_tv_contribution_sum": 0.0,
                "bad_or_nan_count": 0,
                "monitor_encountered_ids": [],
            }
            if args.group_objective == D4_HYBRID_PKPO_OBJECTIVE:
                state["d3_health"]["last_d4_advantage_diagnostics"] = None
        if args.task_curriculum == "online_learnability":
            state["task_curriculum"] = initialize_task_curriculum_state(len(tasks))
    if args.policy_objective == "contrastive_task_step":
        initialize_contrastive_resume_state(
            state,
            legacy_u96_branch=branch_resume_payload is not None,
            in_place_resume=args.resume is not None,
            tasks_per_update=args.tasks_per_update,
            candidates=args.candidates,
        )
    elif args.policy_objective in {
        "sequence_group_fpo", "domino_endpoint_ppo", "discrete_domino"
    } and branch_resume_payload is not None:
        # The source U96 causal histogram is provenance for B0, not temporal
        # credit for B1.  Start B1's all-step coverage counter at its exact
        # U96 boundary while preserving model/AdamW/RNG/cursor/coverage state.
        if state.get("next_update") != 96:
            raise RuntimeError("all-step endpoint-policy branch must start from exact U96")
        state["temporal_credit_step_histogram"] = {
            str(step): 0 for step in range(args.trajectory_steps)
        }
    history_path = args.output / "train_history.jsonl"
    def reconcile_output_history() -> None:
        reconcile_history(
            history_path,
            resume_payload if (args.resume or scale_resume_payload is not None) else None,
        )

    primary_call(context, "history reconciliation", reconcile_output_history)
    training_model = (
        DistributedDataParallel(
            current,
            device_ids=[context.local_rank] if device.type == "cuda" else None,
            output_device=context.local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        if world_size > 1 else current
    )
    target_update_bounds = [args.updates]
    if args.preflight_updates is not None:
        target_update_bounds.append(state["next_update"] + args.preflight_updates)
    if args.pause_after_update is not None:
        target_update_bounds.append(args.pause_after_update)
    target_updates = min(target_update_bounds)
    if target_updates < state["next_update"]:
        raise RuntimeError("execution bound is behind the restored checkpoint")
    reward_fn = reward_function(args.reward)
    started = time.time()
    initial_next_update = int(state["next_update"])
    initial_reward_evaluation_coverage = copy.deepcopy(
        state["reward_evaluation_coverage"]
    )
    maximum_memory = 0

    cache_path = (
        args.output / "reward_cache.sqlite"
        if context.is_primary else args.output / f"reward_cache.rank-{rank:03d}.sqlite"
    )
    cache = rank_local_call(
        context, "reward cache initialization", lambda: ViennaRewardCache(cache_path, evaluate_candidate)
    )
    try:
        while state["next_update"] < target_updates:
            update = int(state["next_update"])
            indices, next_epoch, next_cursor, curriculum_selection = (
                select_task_curriculum_batch(
                    mode=args.task_curriculum,
                    task_count=len(tasks),
                    task_epoch=int(state["task_epoch"]),
                    task_cursor=int(state["task_cursor"]),
                    batch_size=args.tasks_per_update,
                    seed=args.seed,
                    update=update,
                    curriculum_state=state.get("task_curriculum"),
                    minimum_exploration_probability=(
                        ONLINE_CURRICULUM_MIN_EXPLORATION_PROBABILITY
                    ),
                )
            )
            local_offsets = rank_task_offsets(args.tasks_per_update, world_size, rank)
            selected_tasks = [
                (offset, indices[offset], tasks[indices[offset]]) for offset in local_offsets
            ]
            task_infos = []
            all_rewards = []
            all_sequences = []
            rollout_hashes = []
            for task_offset, task_index, task in selected_tasks:
                def prepare_rank_local_task():
                    stem = (
                        f"update-{update:06d}-refresh-00-task-{task_offset:03d}"
                        if d5_strict_on_policy
                        else f"update-{update:06d}-task-{task_offset:03d}"
                    )
                    trajectory_path = rollout_dir / f"{stem}.pt"
                    evaluation_path = rollout_dir / f"{stem}-evaluations.jsonl"
                    stepwise_path = rollout_dir / f"{stem}-stepwise.jsonl"
                    trajectory_seed = args.seed + update * 1_000_003 + task_offset * 10_007
                    if args.policy_objective in {
                        "sequence_group_fpo", "domino_endpoint_ppo", "discrete_domino"
                    }:
                        # B1 has no temporal sampler, window, or IS weight:
                        # every legal Flow-Matching step belongs to the same
                        # completed sequence's terminal-reward objective.
                        selected_steps = tuple(range(args.trajectory_steps))
                        temporal_credit_importance_weight = 1.0
                        temporal_credit_weights = None
                    else:
                        selected_steps, temporal_credit_importance_weight = temporal_credit_selection(
                            args.temporal_credit,
                            seed=args.seed,
                            update=update,
                            global_task_offset=task_offset,
                            trajectory_steps=args.trajectory_steps,
                            causal_ticket_counts=(
                                None if causal_step_policy is None
                                else causal_step_policy["ticket_counts"]
                            ),
                            causal_policy_sha256=(
                                None if causal_step_policy is None
                                else causal_step_policy["sha256"]
                            ),
                        )
                        temporal_credit_weights = temporal_credit_step_weights(
                            args.temporal_credit,
                            selected_steps=selected_steps,
                            importance_weight=temporal_credit_importance_weight,
                        )
                    temperature_value = args.temperatures[
                        (update * args.tasks_per_update + task_offset) % len(args.temperatures)
                    ]
                    temperatures = torch.full(
                        (args.candidates,), temperature_value, dtype=torch.float32, device=device
                    )
                    if trajectory_path.is_file():
                        trajectory = torch.load(
                            trajectory_path, map_location="cpu", weights_only=False
                        )
                    else:
                        rollout_fn = (
                            rollout_discrete_domino_trajectory
                            if args.policy_objective == "discrete_domino"
                            else rollout_endpoint_trajectory
                        )
                        trajectory = rollout_fn(
                            current, task["target_structure"], args.candidates,
                            args.trajectory_steps, trajectory_seed,
                            device, temperatures,
                        )
                        temporary = trajectory_path.with_suffix(".tmp")
                        torch.save(trajectory, temporary)
                        os.replace(temporary, trajectory_path)
                    validation_fn = (
                        validate_discrete_domino_trajectory
                        if args.policy_objective == "discrete_domino"
                        else validate_endpoint_trajectory
                    )
                    trajectory_validation = validation_fn(
                        trajectory, device,
                        expected_structure=task["target_structure"],
                        expected_seed=trajectory_seed,
                        expected_candidates=args.candidates,
                        expected_steps=args.trajectory_steps,
                        expected_temperatures=temperatures,
                    )
                    if evaluation_path.is_file():
                        evaluations = [
                            json.loads(line) for line in evaluation_path.read_text().splitlines()
                            if line.strip()
                        ]
                    else:
                        evaluations = []
                        for candidate_index, sequence in enumerate(trajectory["final_sequences"]):
                            metrics = cache.evaluate(sequence, task["target_structure"])
                            evaluations.append({
                                "task_id": str(task["id"]),
                                "target_length": len(task["target_structure"]),
                                "reward_cache_key": cache_key(
                                    sequence, task["target_structure"]
                                ),
                                "candidate_index": candidate_index,
                                "sequence": sequence,
                                "reward": reward_fn(metrics),
                                **metrics,
                            })
                        atomic_jsonl(evaluation_path, evaluations)
                    validate_evaluations(
                        evaluations, trajectory["final_sequences"], task["target_structure"],
                        str(task["id"]), reward_fn,
                    )
                    nonuniform_credit = (
                        args.credit_assignment == "stepwise_pair"
                        or uses_structure_credit(
                            args.credit_assignment, args.structure_credit_strength
                        )
                    )
                    if nonuniform_credit and any(
                        row["evaluation_valid"] is False for row in evaluations
                    ):
                        raise RuntimeError(
                            "invalid reward evaluations are prohibited for non-uniform credit"
                        )
                    rewards = torch.tensor(
                        [row["reward"] for row in evaluations], dtype=torch.float32, device=device
                    )
                    correctness_tiers = (
                        terminal_evaluation_correctness_tiers(evaluations, device)
                        if args.policy_objective == "contrastive_task_step" else None
                    )
                    if d3_policy_init:
                        d3_advantages = d3_group_advantages(
                            rewards,
                            args.group_objective,
                            args.pkpo_residual_coefficient,
                            update=update,
                        )
                        # Preserve the frozen T0/T1 behavior: T0 consumes the
                        # thermodynamic signal while T1 consumes raw PKPO.
                        # D4 alone consumes their explicit residual sum.
                        advantages = d3_advantages["combined"]
                        effective = bool(d3_advantages["effective"])
                        pkpo_k = d3_advantages["pkpo_k"]
                    else:
                        advantages, effective = normalized_group_advantages(rewards)
                        pkpo_k = None
                        d3_advantages = {
                            "thermo": advantages, "raw_pkpo": None,
                            "combined": advantages,
                        }
                    stepwise_effective_steps = None
                    if args.credit_assignment == "stepwise_pair":
                        if stepwise_path.is_file():
                            stepwise_rows = [
                                json.loads(line) for line in stepwise_path.read_text().splitlines()
                                if line.strip()
                            ]
                        else:
                            stepwise_rows = build_stepwise_evaluations(
                                trajectory, task["target_structure"]
                            )
                            atomic_jsonl(stepwise_path, stepwise_rows)
                        provisional_scores = validate_stepwise_evaluations(
                            stepwise_rows, trajectory, task["target_structure"]
                        ).to(device)
                        step_advantages, stepwise_effective_steps = stepwise_pair_advantages(
                            rewards, provisional_scores
                        )
                        unit_advantages = step_advantages[:, :, None].expand(
                            -1, -1, trajectory["old_unit_log_probabilities"].shape[-1]
                        )
                    elif uses_structure_credit(
                        args.credit_assignment, args.structure_credit_strength
                    ):
                        unit_advantages = torch.stack([
                            structure_error_credit_weights(
                                row["mfe_structure"], task["target_structure"],
                                float(advantage.item()),
                                positive_uniform=args.credit_assignment == "negative_error",
                                strength=args.structure_credit_strength,
                            ) * advantage.cpu()
                            for row, advantage in zip(evaluations, advantages)
                        ]).to(device)
                    else:
                        unit_advantages = advantages
                    with torch.no_grad():
                        if args.policy_objective == "discrete_domino":
                            _, reference_logits = recompute_discrete_domino_unit_log_probabilities(
                                reference, trajectory, device
                            )
                        elif args.temporal_credit == "all_steps":
                            _, reference_logits = recompute_trajectory_unit_log_probabilities(
                                reference, trajectory, device
                            )
                        else:
                            _, reference_logits = recompute_trajectory_unit_log_probabilities(
                                reference, trajectory, device, selected_steps=selected_steps
                            )
                    rollout_hash = {
                        **(
                            {"refresh_index": 0, "task_id": str(task["id"])}
                            if d5_strict_on_policy else {}
                        ),
                        "task_offset": task_offset,
                        "trajectory": str(trajectory_path),
                        "trajectory_sha256": sha256(trajectory_path),
                        "evaluations": str(evaluation_path),
                        "evaluations_sha256": sha256(evaluation_path),
                        "stepwise_evaluations": (
                            str(stepwise_path) if args.credit_assignment == "stepwise_pair" else None
                        ),
                        "stepwise_evaluations_sha256": (
                            sha256(stepwise_path) if args.credit_assignment == "stepwise_pair" else None
                        ),
                        "stepwise_effective_steps": stepwise_effective_steps,
                        "temporal_credit_mode": contract["temporal_credit"]["mode"],
                        "temporal_credit_selected_steps": list(selected_steps),
                        "temporal_credit_importance_weight": temporal_credit_importance_weight,
                        "temporal_credit_step_weights": (
                            None if temporal_credit_weights is None
                            else list(temporal_credit_weights)
                        ),
                        "temperature": temperature_value,
                        **trajectory_validation,
                    }
                    return {
                        "task_index": task_index,
                        "task": task,
                        "evaluations": evaluations,
                        "trajectory": trajectory,
                        "advantages": advantages,
                        "unit_advantages": unit_advantages,
                        "temporal_credit_selected_steps": selected_steps,
                        "temporal_credit_importance_weight": temporal_credit_importance_weight,
                        "temporal_credit_step_weights": temporal_credit_weights,
                        "effective": effective,
                        "pkpo_k": pkpo_k,
                        "thermo_advantages": d3_advantages["thermo"],
                        "raw_pkpo_advantages": d3_advantages["raw_pkpo"],
                        "combined_advantages": d3_advantages["combined"],
                        "stepwise_effective_steps": stepwise_effective_steps,
                        "reference_logits": reference_logits,
                        **(
                            {"correctness_tiers": correctness_tiers}
                            if args.policy_objective == "contrastive_task_step" else {}
                        ),
                    }, rewards, list(trajectory["final_sequences"]), rollout_hash

                info, rewards, sequences, rollout_hash = rank_local_call(
                    context, f"rollout preparation task {task_offset}", prepare_rank_local_task
                )
                task_infos.append(info)
                all_rewards.append(rewards)
                all_sequences.extend(sequences)
                rollout_hashes.append(rollout_hash)

            # D5 is deliberately narrow: its second refresh has the same four
            # task identities but is sampled only after the first optimizer
            # step.  Keeping this isolated preserves the legacy rollout path
            # and its frozen filenames/contracts exactly.
            def prepare_d5_second_refresh():
                refreshed_infos = []
                refreshed_rewards = []
                refreshed_sequences = []
                refreshed_rollouts = []
                for task_offset, task_index, task in selected_tasks:
                    def prepare_rank_local_task():
                        stem = f"update-{update:06d}-refresh-01-task-{task_offset:03d}"
                        trajectory_path = rollout_dir / f"{stem}.pt"
                        evaluation_path = rollout_dir / f"{stem}-evaluations.jsonl"
                        trajectory_seed = (
                            args.seed
                            + update * 1_000_003
                            + task_offset * 10_007
                            + 500_009
                        )
                        selected_steps = tuple(range(args.trajectory_steps))
                        temporal_credit_importance_weight = 1.0
                        temperatures = torch.full(
                            (args.candidates,),
                            args.temperatures[
                                (update * args.tasks_per_update + task_offset)
                                % len(args.temperatures)
                            ],
                            dtype=torch.float32,
                            device=device,
                        )
                        if trajectory_path.is_file():
                            trajectory = torch.load(
                                trajectory_path, map_location="cpu", weights_only=False
                            )
                        else:
                            trajectory = rollout_discrete_domino_trajectory(
                                current, task["target_structure"], args.candidates,
                                args.trajectory_steps, trajectory_seed, device, temperatures,
                            )
                            temporary = trajectory_path.with_suffix(".tmp")
                            torch.save(trajectory, temporary)
                            os.replace(temporary, trajectory_path)
                        trajectory_validation = validate_discrete_domino_trajectory(
                            trajectory, device,
                            expected_structure=task["target_structure"],
                            expected_seed=trajectory_seed,
                            expected_candidates=args.candidates,
                            expected_steps=args.trajectory_steps,
                            expected_temperatures=temperatures,
                        )
                        if evaluation_path.is_file():
                            evaluations = [
                                json.loads(line)
                                for line in evaluation_path.read_text().splitlines()
                                if line.strip()
                            ]
                        else:
                            evaluations = []
                            for candidate_index, sequence in enumerate(
                                trajectory["final_sequences"]
                            ):
                                metrics = cache.evaluate(sequence, task["target_structure"])
                                evaluations.append({
                                    "task_id": str(task["id"]),
                                    "target_length": len(task["target_structure"]),
                                    "reward_cache_key": cache_key(
                                        sequence, task["target_structure"]
                                    ),
                                    "candidate_index": candidate_index,
                                    "sequence": sequence,
                                    "reward": reward_fn(metrics),
                                    **metrics,
                                })
                            atomic_jsonl(evaluation_path, evaluations)
                        validate_evaluations(
                            evaluations, trajectory["final_sequences"],
                            task["target_structure"], str(task["id"]), reward_fn,
                        )
                        rewards = torch.tensor(
                            [row["reward"] for row in evaluations],
                            dtype=torch.float32, device=device,
                        )
                        advantages, effective = normalized_group_advantages(rewards)
                        with torch.no_grad():
                            _, reference_logits = (
                                recompute_discrete_domino_unit_log_probabilities(
                                    reference, trajectory, device
                                )
                            )
                        rollout_hash = {
                            "refresh_index": 1,
                            "task_id": str(task["id"]),
                            "task_offset": task_offset,
                            "trajectory": str(trajectory_path),
                            "trajectory_sha256": sha256(trajectory_path),
                            "evaluations": str(evaluation_path),
                            "evaluations_sha256": sha256(evaluation_path),
                            "stepwise_evaluations": None,
                            "stepwise_evaluations_sha256": None,
                            "stepwise_effective_steps": None,
                            "temporal_credit_mode": contract["temporal_credit"]["mode"],
                            "temporal_credit_selected_steps": list(selected_steps),
                            "temporal_credit_importance_weight": 1.0,
                            "temporal_credit_step_weights": None,
                            "temperature": float(temperatures[0].item()),
                            **trajectory_validation,
                        }
                        return {
                            "task_index": task_index,
                            "task": task,
                            "evaluations": evaluations,
                            "trajectory": trajectory,
                            "advantages": advantages,
                            "unit_advantages": advantages,
                            "temporal_credit_selected_steps": selected_steps,
                            "temporal_credit_importance_weight": temporal_credit_importance_weight,
                            "temporal_credit_step_weights": None,
                            "effective": effective,
                            "pkpo_k": None,
                            "thermo_advantages": advantages,
                            "raw_pkpo_advantages": None,
                            "combined_advantages": advantages,
                            "stepwise_effective_steps": None,
                            "reference_logits": reference_logits,
                        }, rewards, list(trajectory["final_sequences"]), rollout_hash

                    info, rewards, sequences, rollout_hash = rank_local_call(
                        context,
                        f"D5 refresh-01 rollout preparation task {task_offset}",
                        prepare_rank_local_task,
                    )
                    refreshed_infos.append(info)
                    refreshed_rewards.append(rewards)
                    refreshed_sequences.extend(sequences)
                    refreshed_rollouts.append(rollout_hash)
                return (
                    refreshed_infos, refreshed_rewards, refreshed_sequences,
                    refreshed_rollouts,
                )

            d5_refresh_data = []
            if d5_strict_on_policy:
                d5_refresh_data.append((
                    task_infos, all_rewards, all_sequences, rollout_hashes,
                ))

            def prepare_supervised_batch():
                supervised_rows = [
                    supervised_data[
                        (update * args.supervised_batch_size + index) % len(supervised_data)
                    ]
                    for index in range(args.supervised_batch_size)
                ]
                return collate(supervised_rows)

            supervised_batch = rank_local_call(
                context, "supervised batch preparation", prepare_supervised_batch
            )
            policy_epoch_diagnostics = []
            for policy_epoch in range(args.policy_epochs):
                if d5_strict_on_policy and policy_epoch == 1:
                    task_infos, all_rewards, all_sequences, rollout_hashes = (
                        prepare_d5_second_refresh()
                    )
                    d5_refresh_data.append((
                        task_infos, all_rewards, all_sequences, rollout_hashes,
                    ))
                task_losses = []
                task_kls = []
                task_entropies = []
                weighted_task_losses = []
                weighted_task_kls = []
                weighted_task_entropies = []
                task_posterior_tvs = []
                task_transition_tvs = []
                task_transition_tv_contributions = []
                task_pkpo_advantages = []
                task_thermo_advantages = []
                task_raw_pkpo_advantages = []
                task_combined_advantages = []
                calibration_gradients = []
                hybrid_calibration_gradients = []
                group_diagnostics = []
                ratio_errors = []
                optimizer.zero_grad(set_to_none=True)
                for info in task_infos:
                    task = info["task"]
                    trajectory = info["trajectory"]
                    selected_steps = info["temporal_credit_selected_steps"]
                    temporal_step_weights = info["temporal_credit_step_weights"]
                    if args.policy_objective == "discrete_domino":
                        new_unit_log_probs, current_logits = (
                            recompute_discrete_domino_unit_log_probabilities(
                                training_model, trajectory, device
                            )
                        )
                        old_unit_log_probs = trajectory["old_unit_log_probabilities"].to(device)
                    elif args.temporal_credit == "all_steps":
                        new_unit_log_probs, current_logits = (
                            recompute_trajectory_unit_log_probabilities(
                                training_model, trajectory, device
                            )
                        )
                        old_unit_log_probs = trajectory["old_unit_log_probabilities"].to(device)
                    else:
                        new_unit_log_probs, current_logits = (
                            recompute_trajectory_unit_log_probabilities(
                                training_model, trajectory, device,
                                selected_steps=selected_steps,
                            )
                        )
                        old_unit_log_probs = trajectory["old_unit_log_probabilities"].to(device)[
                            :, list(selected_steps)
                        ]
                    ratio_error = float(
                        (torch.exp(new_unit_log_probs.detach() - old_unit_log_probs) - 1)
                        .abs().max().item()
                    )
                    if (policy_epoch == 0 or d5_strict_on_policy) and ratio_error > 1e-5:
                        raise RuntimeError(
                            f"behavior/current action-factor ratio mismatch: {ratio_error}"
                        )
                    if args.policy_objective == "contrastive_task_step":
                        loss, diagnostics = contrastive_task_step_loss(
                            new_unit_log_probs,
                            old_unit_log_probs,
                            info["correctness_tiers"],
                        )
                        if policy_epoch == 0 and diagnostics[
                            "pair_delta_max_abs"
                        ] > 1e-5:
                            raise RuntimeError(
                                "contrastive behavior pair-delta mismatch: "
                                f"{diagnostics['pair_delta_max_abs']}"
                            )
                        diagnostics["behavior_delta_max_abs_error"] = (
                            diagnostics["pair_delta_max_abs"]
                            if policy_epoch == 0 else 0.0
                        )
                        diagnostics["effective"] = diagnostics[
                            "effective_pair_group"
                        ]
                    elif args.policy_objective == "sequence_group_fpo":
                        # The terminal group advantage belongs to the complete
                        # sampled sequence.  The loss helper broadcasts it to
                        # every legal Flow-Matching step without constructing a
                        # full-sequence likelihood or temporal reward signal.
                        loss, diagnostics = sequence_group_fpo_loss(
                            new_unit_log_probs,
                            old_unit_log_probs,
                            info["advantages"],
                            args.clip_ratio,
                        )
                        diagnostics["effective"] = info["effective"]
                    elif args.policy_objective in {"domino_endpoint_ppo", "discrete_domino"}:
                        # DoMinO-PPO: each saved stochastic endpoint sampler
                        # transition receives its own old/new joint-action
                        # ratio.  The terminal group-relative advantage is
                        # trajectory-level and is shared across time, exactly as
                        # permitted in the terminal-reward formulation.  The
                        # current backbone remains a Dirichlet-simplex flow, so
                        # this is an endpoint-policy MDP rather than a claim of
                        # native discrete-flow transition likelihood.
                        loss, diagnostics = domino_endpoint_ppo_loss(
                            new_unit_log_probs,
                            old_unit_log_probs,
                            info["advantages"],
                            args.clip_ratio,
                            normalize_time=False,
                        )
                        diagnostics["effective"] = info["effective"]
                    else:
                        factorized_kwargs = (
                            {} if temporal_step_weights is None
                            else {"step_weights": temporal_step_weights}
                        )
                        loss, diagnostics = factorized_trajectory_grpo_loss(
                            new_unit_log_probs, old_unit_log_probs,
                            temporal_credit_advantages(
                                info["unit_advantages"], selected_steps
                            ), args.clip_ratio, **factorized_kwargs,
                        )
                        diagnostics["effective"] = info["effective"]
                    task_losses.append(loss)
                    group_diagnostics.append(diagnostics)
                    ratio_errors.append(ratio_error)
                    temperatures = trajectory["temperatures"].to(device)
                    step_kls = []
                    step_entropies = []
                    for selected_index in range(len(selected_steps)):
                        step_kls.append(endpoint_policy_kl(
                            current_logits[:, selected_index],
                            info["reference_logits"][:, selected_index],
                            task["target_structure"], temperatures,
                        ))
                        step_entropies.append(endpoint_policy_entropy(
                            current_logits[:, selected_index], task["target_structure"], temperatures
                        ))
                    task_kl = aggregate_step_local_values(
                        torch.stack(step_kls, dim=1), temporal_step_weights
                    )
                    task_entropy = aggregate_step_local_values(
                        torch.stack(step_entropies, dim=1), temporal_step_weights
                    )
                    if d3_policy_init:
                        tv_by_step = [
                            structured_discrete_reference_transition_tv(
                                current_logits[:, selected_index],
                                info["reference_logits"][:, selected_index],
                                trajectory["states"][:, step].to(device),
                                task["target_structure"], temperatures,
                                float(trajectory["replacement_probabilities"][step]),
                            )
                            for selected_index, step in enumerate(selected_steps)
                        ]
                        posterior_tv = aggregate_step_local_values(torch.stack([
                            item["posterior_tv"].mean(dim=-1) for item in tv_by_step
                        ], dim=1), temporal_step_weights)
                        transition_tv = aggregate_step_local_values(torch.stack([
                            item["transition_tv"].mean(dim=-1) for item in tv_by_step
                        ], dim=1), temporal_step_weights)
                        transition_tv_contribution = (
                            args.reference_tv_coefficient * transition_tv
                        )
                        if not torch.isfinite(transition_tv_contribution):
                            raise RuntimeError("non-finite D3 reference-transition TV contribution")
                    else:
                        posterior_tv = torch.zeros((), device=device)
                        transition_tv = torch.zeros((), device=device)
                        transition_tv_contribution = torch.zeros((), device=device)
                    task_objective = (
                        loss
                        + args.kl_coefficient * task_kl
                        - args.entropy_coefficient * task_entropy
                        + transition_tv_contribution
                    )
                    importance_weight = info["temporal_credit_importance_weight"]
                    if temporal_step_weights is None:
                        # Legacy all-steps path.  Causal single/window paths
                        # already carry explicit per-step HT weights.
                        weighted_task_losses.append(loss.detach() * importance_weight)
                        weighted_task_kls.append(task_kl.detach() * importance_weight)
                        weighted_task_entropies.append(task_entropy.detach() * importance_weight)
                        task_objective = task_objective * importance_weight
                    elif args.policy_objective == "contrastive_task_step":
                        # The reviewed contrastive U96 continuation keeps its
                        # scalar causal-is-single loss implementation.  KL and
                        # entropy are already aggregated through the explicit
                        # one-step weight above, while the contrastive loss
                        # still needs the legacy scalar multiplier.
                        weighted_task_losses.append(loss.detach() * importance_weight)
                        weighted_task_kls.append(task_kl.detach())
                        weighted_task_entropies.append(task_entropy.detach())
                        task_objective = (
                            loss * importance_weight
                            + args.kl_coefficient * task_kl
                            - args.entropy_coefficient * task_entropy
                        )
                    else:
                        weighted_task_losses.append(loss.detach())
                        weighted_task_kls.append(task_kl.detach())
                        weighted_task_entropies.append(task_entropy.detach())
                    if not torch.isfinite(task_objective):
                        raise RuntimeError("non-finite endpoint trajectory policy objective")
                    if d3_policy_init and args.preflight_updates == 1 and policy_epoch == 1:
                        calibration_gradients.append({
                            "policy": unscaled_gradient_l2(loss, trainable),
                            "transition_tv": unscaled_gradient_l2(transition_tv, trainable),
                            "reference_kl": unscaled_gradient_l2(task_kl, trainable),
                        })
                        if args.group_objective == D4_HYBRID_PKPO_OBJECTIVE:
                            thermo_loss, _ = domino_endpoint_ppo_loss(
                                new_unit_log_probs, old_unit_log_probs,
                                info["thermo_advantages"], args.clip_ratio,
                                normalize_time=False,
                            )
                            raw_pkpo_loss, _ = domino_endpoint_ppo_loss(
                                new_unit_log_probs, old_unit_log_probs,
                                info["raw_pkpo_advantages"], args.clip_ratio,
                                normalize_time=False,
                            )
                            hybrid_calibration_gradients.append({
                                "thermo_policy": unscaled_gradient_l2(thermo_loss, trainable),
                                "raw_pkpo_policy": unscaled_gradient_l2(raw_pkpo_loss, trainable),
                                "combined_policy": unscaled_gradient_l2(loss, trainable),
                                "thermo_raw_pkpo_cosine": unscaled_gradient_cosine(
                                    thermo_loss, raw_pkpo_loss, trainable,
                                ),
                            })
                    # DDP averages rank gradients; this turns the local task sum
                    # into the exact global task mean.
                    (task_objective * global_task_mean_scale(
                        args.tasks_per_update, world_size
                    )).backward()
                    task_kls.append(task_kl.detach())
                    task_entropies.append(task_entropy.detach())
                    task_posterior_tvs.append(posterior_tv.detach())
                    task_transition_tvs.append(transition_tv.detach())
                    task_transition_tv_contributions.append(
                        transition_tv_contribution.detach()
                    )
                    task_pkpo_advantages.append(info["advantages"].detach())
                    if args.group_objective == D4_HYBRID_PKPO_OBJECTIVE:
                        task_thermo_advantages.append(info["thermo_advantages"].detach())
                        task_raw_pkpo_advantages.append(info["raw_pkpo_advantages"].detach())
                        task_combined_advantages.append(info["combined_advantages"].detach())
                    task_losses[-1] = task_losses[-1].detach()
                policy_loss = torch.stack(task_losses).mean()
                reference_kl = torch.stack(task_kls).mean()
                entropy = torch.stack(task_entropies).mean()
                posterior_tv = torch.stack(task_posterior_tvs).mean()
                transition_tv = torch.stack(task_transition_tvs).mean()
                transition_tv_contribution = torch.stack(task_transition_tv_contributions).mean()
                policy_objective = (
                    policy_loss + args.kl_coefficient * reference_kl
                    - args.entropy_coefficient * entropy
                    + transition_tv_contribution
                )
                importance_weighted_policy_loss = torch.stack(weighted_task_losses).mean()
                importance_weighted_reference_kl = torch.stack(weighted_task_kls).mean()
                importance_weighted_entropy = torch.stack(weighted_task_entropies).mean()
                importance_weighted_policy_objective = (
                    importance_weighted_policy_loss
                    + args.kl_coefficient * importance_weighted_reference_kl
                    - args.entropy_coefficient * importance_weighted_entropy
                    + transition_tv_contribution
                )
                scaled_ce_loss, ce_diagnostics = endpoint_ce_anchor_loss(
                    training_model,
                    supervised_batch,
                    local_indices=local_supervised_indices(
                        args.supervised_batch_size, world_size, rank
                    ),
                    seed=args.seed,
                    update=update,
                    policy_epoch=policy_epoch,
                    context=context,
                )
                if not torch.isfinite(scaled_ce_loss):
                    raise RuntimeError("non-finite endpoint trajectory anchor loss")
                calibration_ce_gradient = (
                    unscaled_gradient_l2(scaled_ce_loss, trainable)
                    if d3_policy_init and args.preflight_updates == 1 and policy_epoch == 1
                    else 0.0
                )
                if args.ce_coefficient:
                    (args.ce_coefficient * scaled_ce_loss).backward()
                total_loss = (
                    importance_weighted_policy_objective
                    + args.ce_coefficient * ce_diagnostics["global_loss"]
                )
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                    raise RuntimeError("invalid endpoint trajectory gradient norm")
                optimizer.step()
                policy_epoch_record = {
                    "policy_epoch": policy_epoch,
                    "policy_loss": float(policy_loss.detach().item()),
                    "importance_weighted_policy_loss": float(
                        importance_weighted_policy_loss.item()
                    ),
                    "importance_weighted_policy_objective": float(
                        importance_weighted_policy_objective.item()
                    ),
                    "importance_weighted_reference_kl": float(
                        importance_weighted_reference_kl.item()
                    ),
                    "importance_weighted_entropy": float(
                        importance_weighted_entropy.item()
                    ),
                    "total_loss": float(total_loss.detach().item()),
                    "reference_kl_before_update": float(reference_kl.detach().item()),
                    "entropy_before_update": float(entropy.detach().item()),
                    "posterior_tv": float(posterior_tv.detach().item()),
                    "transition_tv": float(transition_tv.detach().item()),
                    "reference_tv_coefficient": (
                        args.reference_tv_coefficient if d3_policy_init else 0.0
                    ),
                    "reference_tv_contribution": float(
                        transition_tv_contribution.detach().item()
                    ),
                    # T0 thermodynamic scaling does not optimize PKPO; this field is
                    # diagnostic only. Preserve the original D3 schedule through U255
                    # and keep its terminal k=1 label for later coverage-scaling logs.
                    "pkpo_k": d3_pkpo_k(min(update, 255)) if d3_policy_init else None,
                    "pkpo_advantage_mean": float(
                        torch.cat(task_pkpo_advantages).mean().detach().item()
                    ),
                    "pkpo_advantage_std": float(
                        torch.cat(task_pkpo_advantages).std(unbiased=False).detach().item()
                    ),
                    "pkpo_advantage_min": float(torch.cat(task_pkpo_advantages).min().detach().item()),
                    "pkpo_advantage_max": float(torch.cat(task_pkpo_advantages).max().detach().item()),
                    "calibration_policy_gradient_l2": (
                        sum(row["policy"] for row in calibration_gradients) / len(calibration_gradients)
                        if calibration_gradients else 0.0
                    ),
                    "calibration_transition_tv_gradient_l2": (
                        sum(row["transition_tv"] for row in calibration_gradients) / len(calibration_gradients)
                        if calibration_gradients else 0.0
                    ),
                    "calibration_reference_kl_gradient_l2": (
                        sum(row["reference_kl"] for row in calibration_gradients) / len(calibration_gradients)
                        if calibration_gradients else 0.0
                    ),
                    "calibration_ce_gradient_l2": calibration_ce_gradient,
                    "ce_anchor_loss": ce_diagnostics["global_loss"],
                    "ce_accuracy": ce_diagnostics["accuracy"],
                    "ce_anchor_alpha": ce_diagnostics["alpha"],
                    "ce_anchor_local_tokens": ce_diagnostics["local_tokens"],
                    "ce_anchor_global_tokens": ce_diagnostics["global_tokens"],
                    "gradient_norm_before_clip": gradient_norm,
                    "effective_groups": sum(
                        bool(item["effective"]) for item in group_diagnostics
                    ),
                    "clip_fraction": sum(
                        item["clip_fraction"] for item in group_diagnostics
                    ) / len(group_diagnostics),
                    "mean_ratio": sum(
                        item["mean_ratio"] for item in group_diagnostics
                    ) / len(group_diagnostics),
                    "minimum_ratio": min(
                        item["minimum_ratio"] for item in group_diagnostics
                    ),
                    "maximum_ratio": max(
                        item["maximum_ratio"] for item in group_diagnostics
                    ),
                    "ratio_max_abs_error": max(ratio_errors),
                }
                if args.group_objective == D4_HYBRID_PKPO_OBJECTIVE:
                    d4_advantage_summaries = {
                        "thermo": advantage_summary(torch.cat(task_thermo_advantages)),
                        "raw_pkpo": advantage_summary(torch.cat(task_raw_pkpo_advantages)),
                        "combined": advantage_summary(torch.cat(task_combined_advantages)),
                    }
                    policy_epoch_record.update({
                        f"{name}_advantage_{stat}": value
                        for name, summary in d4_advantage_summaries.items()
                        for stat, value in summary.items()
                    })
                    policy_epoch_record.update({
                        "calibration_thermo_policy_gradient_l2": (
                            sum(row["thermo_policy"] for row in hybrid_calibration_gradients)
                            / len(hybrid_calibration_gradients)
                            if hybrid_calibration_gradients else 0.0
                        ),
                        "calibration_raw_pkpo_policy_gradient_l2": (
                            sum(row["raw_pkpo_policy"] for row in hybrid_calibration_gradients)
                            / len(hybrid_calibration_gradients)
                            if hybrid_calibration_gradients else 0.0
                        ),
                        "calibration_combined_policy_gradient_l2": (
                            sum(row["combined_policy"] for row in hybrid_calibration_gradients)
                            / len(hybrid_calibration_gradients)
                            if hybrid_calibration_gradients else 0.0
                        ),
                        "calibration_thermo_raw_pkpo_gradient_cosine": (
                            sum(row["thermo_raw_pkpo_cosine"] for row in hybrid_calibration_gradients)
                            / len(hybrid_calibration_gradients)
                            if hybrid_calibration_gradients else 0.0
                        ),
                    })
                if args.policy_objective == "contrastive_task_step":
                    policy_epoch_record.update({
                        "pair_count": sum(
                            int(item["pair_count"]) for item in group_diagnostics
                        ),
                        "effective_pair_groups": sum(
                            bool(item["effective_pair_group"])
                            for item in group_diagnostics
                        ),
                        "ineffective_groups": sum(
                            bool(item["ineffective_group"])
                            for item in group_diagnostics
                        ),
                        "mixed_correctness_groups": sum(
                            bool(item["mixed_correctness_group"])
                            for item in group_diagnostics
                        ),
                        "tier_candidate_counts": {
                            str(tier): sum(
                                int(item["tier_candidate_counts"][str(tier)])
                                for item in group_diagnostics
                            )
                            for tier in (0, 2, 3)
                        },
                        "behavior_delta_max_abs_error": max(
                            float(item["behavior_delta_max_abs_error"])
                            for item in group_diagnostics
                        ),
                        "policy_gradient_finite": all(
                            bool(item["policy_gradient_finite"])
                            for item in group_diagnostics
                        ),
                        "policy_gradient_nonzero_groups": sum(
                            bool(item["policy_gradient_nonzero"])
                            for item in group_diagnostics
                        ),
                    })
                policy_epoch_diagnostics.append(policy_epoch_record)
            if device.type == "cuda":
                maximum_memory = max(maximum_memory, torch.cuda.max_memory_allocated(device))
            maximum_memory = global_max_memory(context, maximum_memory)
            rank_update = {
                "evaluations": [info["evaluations"] for info in task_infos],
                "rewards": [float(value) for rewards in all_rewards for value in rewards.cpu()],
                "sequences": all_sequences,
                "rollouts": rollout_hashes,
                "task_groups": [
                    {
                        "task_index": int(info["task_index"]),
                        "reward_count": int(rewards.numel()),
                        "reward_sum": float(rewards.sum().item()),
                        "reward_square_sum": float(rewards.square().sum().item()),
                    }
                    for info, rewards in zip(task_infos, all_rewards)
                ],
                "monitor_encountered_ids": [
                    str(info["task"]["id"]) for info in task_infos
                    if str(info["task"]["id"]) in d3_monitor_ids
                ] if d3_policy_init else [],
                "stepwise_effective_steps": sum(
                    int(info["stepwise_effective_steps"] or 0) for info in task_infos
                ),
                "policy_epoch_diagnostics": policy_epoch_diagnostics,
            }
            if d5_strict_on_policy:
                if len(d5_refresh_data) != 2:
                    raise RuntimeError("D5 logical update did not complete both policy refreshes")
                rank_update["d5_refreshes"] = [
                    {
                        "refresh_index": refresh_index,
                        "evaluations": [info["evaluations"] for info in refresh_infos],
                        "rewards": [
                            float(value)
                            for rewards in refresh_rewards
                            for value in rewards.cpu()
                        ],
                        "sequences": refresh_sequences,
                        "rollouts": refresh_rollouts,
                        "task_groups": [
                            {
                                "task_index": int(info["task_index"]),
                                "reward_count": int(rewards.numel()),
                                "reward_sum": float(rewards.sum().item()),
                                "reward_square_sum": float(rewards.square().sum().item()),
                            }
                            for info, rewards in zip(refresh_infos, refresh_rewards)
                        ],
                        "monitor_encountered_ids": (
                            [] if d3_monitor_ids is None else [
                                str(info["task"]["id"]) for info in refresh_infos
                                if str(info["task"]["id"]) in d3_monitor_ids
                            ]
                        ),
                        "stepwise_effective_steps": sum(
                            int(info["stepwise_effective_steps"] or 0)
                            for info in refresh_infos
                        ),
                    }
                    for refresh_index, (
                        refresh_infos, refresh_rewards, refresh_sequences, refresh_rollouts,
                    ) in enumerate(d5_refresh_data)
                ]
            rank_updates = gather_objects(context, rank_update)
            def assemble_update_record():
                ordered_rollouts = sorted(
                    (rollout for rank_value in rank_updates for rollout in rank_value["rollouts"]),
                    key=lambda rollout: int(rollout["task_offset"]),
                )
                if [int(rollout["task_offset"]) for rollout in ordered_rollouts] != list(
                    range(args.tasks_per_update)
                ):
                    raise RuntimeError("distributed trajectory task assignment has duplicate or missing tasks")
                d5_refreshes = None
                all_rollouts_for_coverage = ordered_rollouts
                if d5_strict_on_policy:
                    d5_refreshes = []
                    for refresh_index in range(2):
                        refresh_rows = [
                            refresh
                            for rank_value in rank_updates
                            for refresh in rank_value.get("d5_refreshes", [])
                            if refresh.get("refresh_index") == refresh_index
                        ]
                        refresh_rollouts = sorted(
                            (rollout for refresh in refresh_rows for rollout in refresh["rollouts"]),
                            key=lambda rollout: int(rollout["task_offset"]),
                        )
                        if (
                            len(refresh_rows) != world_size
                            or [int(rollout["task_offset"]) for rollout in refresh_rollouts]
                            != list(range(args.tasks_per_update))
                            or any(rollout.get("refresh_index") != refresh_index for rollout in refresh_rollouts)
                        ):
                            raise RuntimeError("D5 refresh rollouts are incomplete, overwritten, or foreign")
                        d5_refreshes.append({
                            "refresh_index": refresh_index,
                            "rollouts": refresh_rollouts,
                            "evaluations": [
                                evaluations for refresh in refresh_rows
                                for evaluations in refresh["evaluations"]
                            ],
                            "rewards": [
                                value for refresh in refresh_rows for value in refresh["rewards"]
                            ],
                            "sequences": [
                                sequence for refresh in refresh_rows
                                for sequence in refresh["sequences"]
                            ],
                        })
                    all_rollouts_for_coverage = [
                        rollout for refresh in d5_refreshes for rollout in refresh["rollouts"]
                    ]
                    all_evaluations = [
                        evaluations for refresh in d5_refreshes
                        for evaluations in refresh["evaluations"]
                    ]
                else:
                    all_evaluations = [
                        evaluations
                        for rank_value in rank_updates
                        for evaluations in rank_value["evaluations"]
                    ]
                state.update({
                    "next_update": update + 1,
                    "task_epoch": next_epoch,
                    "task_cursor": next_cursor,
                    "reward_evaluation_coverage": update_reward_evaluation_coverage(
                        state["reward_evaluation_coverage"], all_evaluations, update
                    ),
                })
                curriculum_diagnostics = None
                if args.task_curriculum == "online_learnability":
                    task_groups = [
                        group for rank_value in rank_updates
                        for group in rank_value["task_groups"]
                    ]
                    state["task_curriculum"] = update_task_curriculum_statistics(
                        state["task_curriculum"],
                        task_groups,
                        adaptive=curriculum_selection["phase"] == "weighted_top_k",
                    )
                    curriculum_diagnostics = task_curriculum_diagnostics(
                        state["task_curriculum"]
                    ) | {
                        "selection_phase": curriculum_selection["phase"],
                        "minimum_task_probability": curriculum_selection.get(
                            "minimum_task_probability"
                        ),
                        "maximum_task_probability": curriculum_selection.get(
                            "maximum_task_probability"
                        ),
                    }
                histogram = dict(state["temporal_credit_step_histogram"])
                for rollout in all_rollouts_for_coverage:
                    for step in rollout["temporal_credit_selected_steps"]:
                        key = str(step)
                        if key not in histogram:
                            raise RuntimeError("temporal-credit step is outside the contract histogram")
                        histogram[key] += 1
                state["temporal_credit_step_histogram"] = histogram
                coverage_update = finalize_reward_evaluation_coverage(
                    state["reward_evaluation_coverage"]
                )
                reward_values = [
                    value for rank_value in rank_updates for value in rank_value["rewards"]
                ]
                sequence_values = [
                    sequence for rank_value in rank_updates for sequence in rank_value["sequences"]
                ]
                if d5_strict_on_policy:
                    reward_values = [
                        value for refresh in (d5_refreshes or [])
                        for value in refresh["rewards"]
                    ]
                    sequence_values = [
                        sequence for refresh in (d5_refreshes or [])
                        for sequence in refresh["sequences"]
                    ]
                merged_diagnostics = merge_policy_epoch_diagnostics(
                    [rank_value["policy_epoch_diagnostics"] for rank_value in rank_updates],
                    local_tasks_per_rank,
                )
                if d5_strict_on_policy:
                    validate_d5_logical_update_geometry(
                        d5_refreshes or [], merged_diagnostics,
                        tasks_per_update=args.tasks_per_update,
                        candidates=args.candidates,
                    )
                if d3_policy_init:
                    final_diagnostics = merged_diagnostics[-1]
                    health = state["d3_health"]
                    d3_groups_this_update = args.tasks_per_update * (
                        2 if d5_strict_on_policy else 1
                    )
                    health["groups"] += d3_groups_this_update
                    health["pkpo_k_histogram"][str(final_diagnostics["pkpo_k"])] += (
                        d3_groups_this_update
                    )
                    if d5_strict_on_policy:
                        drift_totals = d5_group_weighted_drift_totals(
                            merged_diagnostics,
                            groups_per_refresh=args.tasks_per_update,
                        )
                        health["posterior_tv_sum"] += drift_totals["posterior_tv"]
                        health["transition_tv_sum"] += drift_totals["transition_tv"]
                        health["transition_tv_contribution_sum"] += drift_totals[
                            "reference_tv_contribution"
                        ]
                    else:
                        health["posterior_tv_sum"] += float(final_diagnostics["posterior_tv"])
                        health["transition_tv_sum"] += float(final_diagnostics["transition_tv"])
                        health["transition_tv_contribution_sum"] += float(
                            final_diagnostics["reference_tv_contribution"]
                        )
                    if args.group_objective == D4_HYBRID_PKPO_OBJECTIVE:
                        health["last_d4_advantage_diagnostics"] = {
                            name: {
                                statistic: final_diagnostics[
                                    f"{name}_advantage_{statistic}"
                                ]
                                for statistic in ("mean", "std", "min", "max")
                            }
                            for name in ("thermo", "raw_pkpo", "combined")
                        }
                    if not all(math.isfinite(float(final_diagnostics[key])) for key in (
                        "posterior_tv", "transition_tv", "reference_tv_contribution",
                    )):
                        health["bad_or_nan_count"] += 1
                    health["monitor_encountered_ids"] = sorted(set(
                        health["monitor_encountered_ids"] + [
                            identifier for rank_value in rank_updates
                            for identifier in rank_value["monitor_encountered_ids"]
                        ]
                    ))
                contrastive_update = None
                if args.policy_objective == "contrastive_task_step":
                    contrastive_update = merged_diagnostics[0]
                    invariant_fields = (
                        "pair_count", "effective_pair_groups", "ineffective_groups",
                        "mixed_correctness_groups", "tier_candidate_counts",
                    )
                    if (
                        any(
                            diagnostics[field] != contrastive_update[field]
                            for diagnostics in merged_diagnostics[1:]
                            for field in invariant_fields
                        )
                        or contrastive_update["effective_pair_groups"]
                        + contrastive_update["ineffective_groups"]
                        != args.tasks_per_update
                        or contrastive_update["mixed_correctness_groups"]
                        != contrastive_update["effective_pair_groups"]
                        or contrastive_update["policy_gradient_nonzero_groups"]
                        != contrastive_update["effective_pair_groups"]
                        or sum(contrastive_update["tier_candidate_counts"].values())
                        != args.tasks_per_update * args.candidates
                        or contrastive_update["behavior_delta_max_abs_error"] > 1e-5
                        or not all(
                            diagnostics["policy_gradient_finite"] is True
                            for diagnostics in merged_diagnostics
                        )
                    ):
                        raise RuntimeError(
                            "contrastive task-step update diagnostics violate the contract"
                        )
                    state["contrastive_pair_count"] += int(
                        contrastive_update["pair_count"]
                    )
                    state["contrastive_effective_pair_groups"] += int(
                        contrastive_update["effective_pair_groups"]
                    )
                    state["contrastive_ineffective_groups"] += int(
                        contrastive_update["ineffective_groups"]
                    )
                    state["contrastive_mixed_correctness_groups"] += int(
                        contrastive_update["mixed_correctness_groups"]
                    )
                    state["contrastive_policy_gradient_nonzero_groups"] += int(
                        contrastive_update["policy_gradient_nonzero_groups"]
                    )
                    for tier in (0, 2, 3):
                        state["contrastive_tier_candidate_counts"][str(tier)] += int(
                            contrastive_update["tier_candidate_counts"][str(tier)]
                        )
                    state["contrastive_behavior_delta_max_abs_error"] = max(
                        float(state["contrastive_behavior_delta_max_abs_error"]),
                        float(contrastive_update["behavior_delta_max_abs_error"]),
                    )
                    state["contrastive_policy_gradient_finite"] = bool(
                        state["contrastive_policy_gradient_finite"]
                        and all(
                            diagnostics["policy_gradient_finite"] is True
                            for diagnostics in merged_diagnostics
                        )
                    )
                record = {
                    "update": update,
                    "world_size": world_size,
                    "global_tasks_per_update": args.tasks_per_update,
                    "local_tasks_per_rank": local_tasks_per_rank,
                    "candidates_per_task": args.candidates,
                    "global_supervised_batch_size": args.supervised_batch_size,
                    "local_supervised_batch_size": local_supervised_batch_size,
                    "policy_epochs": args.policy_epochs,
                    "temporal_credit": {
                        "mode": contract["temporal_credit"]["mode"],
                        "selected_steps_by_task": [
                            {
                                "task_offset": rollout["task_offset"],
                                "steps": rollout["temporal_credit_selected_steps"],
                                "importance_weight": rollout[
                                    "temporal_credit_importance_weight"
                                ],
                            }
                            for rollout in (
                                all_rollouts_for_coverage
                                if d5_strict_on_policy else ordered_rollouts
                            )
                        ],
                        "step_histogram": histogram,
                    },
                    "task_curriculum": (
                        {"mode": "deterministic_shuffle"}
                        if curriculum_diagnostics is None else curriculum_diagnostics
                    ),
                    "policy_epoch_diagnostics": merged_diagnostics,
                    "reward_mean": sum(reward_values) / len(reward_values),
                    "reward_min": min(reward_values),
                    "reward_max": max(reward_values),
                    "effective_groups": (
                        sum(item["effective_groups"] for item in merged_diagnostics)
                        if d5_strict_on_policy
                        else merged_diagnostics[0]["effective_groups"]
                    ),
                    "stepwise_effective_steps": sum(
                        int(rank_value["stepwise_effective_steps"]) for rank_value in rank_updates
                    ),
                    "clip_fraction_after_first_update": merged_diagnostics[-1]["clip_fraction"],
                    "mean_ratio_after_first_update": merged_diagnostics[-1]["mean_ratio"],
                    "ratio_max_abs_error_before_update": (
                        max(item["ratio_max_abs_error"] for item in merged_diagnostics)
                        if d5_strict_on_policy
                        else merged_diagnostics[0]["ratio_max_abs_error"]
                    ),
                    "diversity": len(set(sequence_values)) / len(sequence_values),
                    "rollouts": (
                        all_rollouts_for_coverage
                        if d5_strict_on_policy else ordered_rollouts
                    ),
                    "max_cuda_memory_bytes": maximum_memory,
                    "reward_evaluation_invalid_rows": coverage_update[
                        "invalid_candidate_rows"
                    ],
                    "reward_evaluation_invalid_rate": coverage_update[
                        "invalid_candidate_rate"
                    ],
                }
                if d5_strict_on_policy:
                    if d5_refreshes is None or len(d5_refreshes) != 2:
                        raise RuntimeError("D5 refresh diagnostics are incomplete")
                    record["d5"] = {
                        "rollout_refresh_mode": D5_ROLLOUT_REFRESH_MODE,
                        "task_cursor_advance": args.tasks_per_update,
                        "optimizer_steps": len(merged_diagnostics),
                        "groups": len(all_evaluations),
                        "candidates": sum(
                            len(evaluations) for evaluations in all_evaluations
                        ),
                        "refreshes": [
                            {
                                "refresh_index": refresh["refresh_index"],
                                "reward_mean": (
                                    sum(refresh["rewards"]) / len(refresh["rewards"])
                                ),
                                "reward_min": min(refresh["rewards"]),
                                "reward_max": max(refresh["rewards"]),
                                "diversity": (
                                    len(set(refresh["sequences"]))
                                    / len(refresh["sequences"])
                                ),
                                "reference_kl_before_update": diagnostics[
                                    "reference_kl_before_update"
                                ],
                                "posterior_tv": diagnostics["posterior_tv"],
                                "transition_tv": diagnostics["transition_tv"],
                                "ratio_max_abs_error_before_update": diagnostics[
                                    "ratio_max_abs_error"
                                ],
                                "rollouts": refresh["rollouts"],
                            }
                            for refresh, diagnostics in zip(
                                d5_refreshes, merged_diagnostics
                            )
                        ],
                    }
                if contrastive_update is not None:
                    record["contrastive_task_step"] = {
                        "pair_count": contrastive_update["pair_count"],
                        "effective_pair_groups": contrastive_update[
                            "effective_pair_groups"
                        ],
                        "ineffective_groups": contrastive_update[
                            "ineffective_groups"
                        ],
                        "mixed_correctness_groups": contrastive_update[
                            "mixed_correctness_groups"
                        ],
                        "tier_candidate_counts": contrastive_update[
                            "tier_candidate_counts"
                        ],
                        "behavior_delta_max_abs_error": contrastive_update[
                            "behavior_delta_max_abs_error"
                        ],
                        "policy_gradient_finite": all(
                            diagnostics["policy_gradient_finite"] is True
                            for diagnostics in merged_diagnostics
                        ),
                        "policy_gradient_nonzero_groups": contrastive_update[
                            "policy_gradient_nonzero_groups"
                        ],
                    }
                if d3_policy_init:
                    record["d3"] = {
                        "group_objective": args.group_objective,
                        "pkpo_k": merged_diagnostics[-1]["pkpo_k"],
                        "pkpo_advantage": {
                            key: merged_diagnostics[-1][f"pkpo_advantage_{key}"]
                            for key in ("mean", "std", "min", "max")
                        },
                        "posterior_tv": merged_diagnostics[-1]["posterior_tv"],
                        "transition_tv": merged_diagnostics[-1]["transition_tv"],
                        "reference_tv_coefficient": args.reference_tv_coefficient,
                        "reference_tv_contribution": merged_diagnostics[-1][
                            "reference_tv_contribution"
                        ],
                        "reference_kl": merged_diagnostics[-1]["reference_kl_before_update"],
                        "bad_or_nan_count": state["d3_health"]["bad_or_nan_count"],
                        "monitor_ids_sha256": contract["train_health_monitor"]["ids_sha256"],
                        "preflight_gradient_calibration": (
                            None if args.preflight_updates != 1 else {
                                key: merged_diagnostics[-1][f"calibration_{key}_gradient_l2"]
                                for key in ("policy", "transition_tv", "reference_kl", "ce")
                            }
                        ),
                    }
                    if args.group_objective == D4_HYBRID_PKPO_OBJECTIVE:
                        record["d3"]["pkpo_residual"] = contract["pkpo_residual"]
                        record["d3"]["advantages"] = health[
                            "last_d4_advantage_diagnostics"
                        ]
                        record["d3"]["preflight_hybrid_policy_gradient_calibration"] = (
                            None if args.preflight_updates != 1 else {
                                "thermo_policy_gradient_l2": merged_diagnostics[-1][
                                    "calibration_thermo_policy_gradient_l2"
                                ],
                                "raw_pkpo_policy_gradient_l2": merged_diagnostics[-1][
                                    "calibration_raw_pkpo_policy_gradient_l2"
                                ],
                                "combined_policy_gradient_l2": merged_diagnostics[-1][
                                    "calibration_combined_policy_gradient_l2"
                                ],
                                "thermo_raw_pkpo_gradient_cosine": merged_diagnostics[-1][
                                    "calibration_thermo_raw_pkpo_gradient_cosine"
                                ],
                            }
                        )
                return state, record

            state, record = primary_call(
                context, "update aggregation and validation", assemble_update_record
            )
            rng_states = gather_rank_rng_states(context, capture_rng_state())
            def save_update_checkpoint() -> None:
                checkpoint_payload = {
                    "contract": contract,
                    "contract_sha256": contract_hash,
                    "state": state,
                    "trainable_model": current.trainable_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimizer_parameter_manifest": optimizer_parameter_manifest,
                    "rng_state": rng_states[0],
                    "rng_state_by_rank": rng_states,
                    "last_record": record,
                    **(
                        {"pkpo_residual": contract["pkpo_residual"]}
                        if args.group_objective == D4_HYBRID_PKPO_OBJECTIVE else {}
                    ),
                    **(
                        {"d5": {
                            "logical_update_boundary": state["next_update"],
                            "task_cursor": state["task_cursor"],
                            "optimizer_steps": state["next_update"] * args.policy_epochs,
                            "coverage": state["reward_evaluation_coverage"],
                            "temporal_credit_step_histogram": state[
                                "temporal_credit_step_histogram"
                            ],
                        }}
                        if d5_strict_on_policy else {}
                    ),
                }
                checkpoint_boundary = (
                    state["next_update"] % args.immutable_checkpoint_every == 0
                    or state["next_update"] == target_updates
                )
                checkpoint_path = checkpoint_dir / (
                    f"checkpoint-update-{update + 1:06d}.pt"
                    if checkpoint_boundary else "checkpoint-latest.pt"
                )
                if checkpoint_boundary and checkpoint_path.exists():
                    raise FileExistsError(
                        f"immutable trajectory checkpoint exists: {checkpoint_path}"
                    )
                atomic_checkpoint(checkpoint_path, checkpoint_payload)
                append_jsonl(history_path, record)
                atomic_json(args.output / "last.json", {
                    "path": str(checkpoint_path),
                    "sha256": sha256(checkpoint_path),
                    "next_update": state["next_update"],
                })
                print(json.dumps(record, sort_keys=True), flush=True)

            primary_call(context, "checkpoint and history write", save_update_checkpoint)
        cache_stats = rank_local_call(context, "reward cache statistics", cache.stats)
    finally:
        # This is coordinated before cache statistics are gathered, so a rank-local
        # close error cannot leave peers blocked in the following object collective.
        rank_local_call(context, "reward cache close", cache.close)

    cache_stats_by_rank = gather_objects(context, cache_stats)
    def validate_and_write_receipt() -> None:
        reward_evaluation_coverage = finalize_reward_evaluation_coverage(
            state["reward_evaluation_coverage"]
        )
        groups_per_logical_update = 8 if d5_strict_on_policy else 4
        updates_this_run = int(state["next_update"]) - initial_next_update
        expected_groups = (
            int(initial_reward_evaluation_coverage["groups"])
            + updates_this_run * groups_per_logical_update
        )
        expected_candidate_rows = (
            int(initial_reward_evaluation_coverage["candidate_rows"])
            + updates_this_run * groups_per_logical_update * args.candidates
        )
        validate_reward_evaluation_coverage(
            reward_evaluation_coverage,
            expected_candidate_rows=expected_candidate_rows,
            expected_groups=expected_groups,
        )
        stepwise_files = sorted(rollout_dir.glob("*-stepwise.jsonl"))
        stepwise_records = sum(
            len([line for line in path.read_text().splitlines() if line.strip()])
            for path in stepwise_files
        )
        runtime_seconds = time.time() - started
        receipt = {
            "schema_version": 1,
            "status": (
                "preflight_complete" if args.preflight_updates is not None
                else "paused" if target_updates < args.updates
                else "complete"
            ),
            "formal_rl_started": args.formal_rl and args.preflight_updates is None,
            "updates_complete": state["next_update"],
            "target_updates": target_updates,
            "contract_target_updates": args.updates,
            "pause_after_update": args.pause_after_update,
            "start_update": initial_next_update,
            "updates_this_run": updates_this_run,
            "contract_sha256": contract_hash,
            "scientific_contract_sha256": contract_hash,
            "contract_file": {
                "path": str((args.output / "contract.json").resolve()),
                "sha256": sha256(args.output / "contract.json"),
            },
            "output_path": str(args.output.resolve()),
            "world_size": world_size,
            "global_tasks_per_update": args.tasks_per_update,
            "local_tasks_per_rank": local_tasks_per_rank,
            "candidates_per_task": args.candidates,
            "global_supervised_batch_size": args.supervised_batch_size,
            "local_supervised_batch_size": local_supervised_batch_size,
            "last_checkpoint": json.loads((args.output / "last.json").read_text()),
            "reward_cache": cache_stats_by_rank[0],
            "reward_cache_by_rank": cache_stats_by_rank,
            "reward_evaluation_coverage": reward_evaluation_coverage,
            "bad_count": reward_evaluation_coverage["invalid_candidate_rows"],
            "temporal_credit": {
                "mode": contract["temporal_credit"]["mode"],
                "step_histogram": state["temporal_credit_step_histogram"],
                "causal_step_policy_sha256": contract["temporal_credit"][
                    "causal_step_policy_sha256"
                ],
                "causal_step_ticket_counts": contract["temporal_credit"][
                    "causal_step_ticket_counts"
                ],
                "source_aggregate_sha256": contract["temporal_credit"][
                    "source_aggregate_sha256"
                ],
                "importance_weight": contract["temporal_credit"]["importance_weight"],
                "importance_weight_scope": contract["temporal_credit"][
                    "importance_weight_scope"
                ],
            },
            "task_curriculum": (
                {"mode": "deterministic_shuffle"}
                if args.task_curriculum == "deterministic_shuffle"
                else {"mode": "online_learnability"}
                | task_curriculum_diagnostics(state["task_curriculum"])
            ),
            "parent_chain": contract["parent_chain"],
            "parent_checkpoint_sha256": contract["parent_checkpoint_sha256"],
            "parent_receipt_sha256": contract["parent_receipt_sha256"],
            "parent_contract_sha256": contract["parent_contract_sha256"],
            "policy_initialization": contract["policy_initialization"],
            "continuation": contract["continuation"],
            "branch_gate": (
                formal_branch_gate
                if formal_branch_gate is not None else
                None if branch_resume_identity is None else {
                    "approval_scope": "formal-after-exact-preflight",
                    "source_contract_sha256": branch_resume_identity[
                        "contract_sha256"
                    ],
                    "source_checkpoint_sha256": branch_resume_identity[
                        "checkpoint_sha256"
                    ],
                    "source_receipt_sha256": branch_resume_identity["receipt_sha256"],
                    "review_receipt_sha256": branch_resume_review_sha256,
                    "preflight_receipt_sha256": branch_preflight_receipt_sha256,
                }
            ),
            "branch_gate_file": (
                None if formal_branch_gate is None else {
                    "path": str((args.output / "branch_gate.json").resolve()),
                    "sha256": sha256(args.output / "branch_gate.json"),
                }
            ),
            "posthoc_exploratory": contract["posthoc_exploratory"],
            "stepwise_mfe_evaluation": {
                "files": len(stepwise_files),
                "persisted_records": stepwise_records,
                "verification_recomputations": stepwise_records,
                "minimum_mfe_fold_calls": 2 * stepwise_records,
            },
            "runtime_seconds": runtime_seconds,
            "samples_per_second": (
                updates_this_run * groups_per_logical_update * args.candidates
                / max(runtime_seconds, 1e-9)
            ),
            "max_cuda_memory_bytes": maximum_memory,
            "nan_inf_count": 0,
            "trainable_parameters": sum(
                parameter.numel() for parameter in current.parameters() if parameter.requires_grad
            ),
            "backbone_trainable_parameters": sum(
                parameter.numel()
                for parameter in current.rnaernie.parameters()
                if parameter.requires_grad
            ),
        }
        if args.policy_objective == "contrastive_task_step":
            contrastive_groups = (
                int(state["contrastive_effective_pair_groups"])
                + int(state["contrastive_ineffective_groups"])
            )
            expected_contrastive_groups = (
                int(state["next_update"]) - 96
            ) * args.tasks_per_update
            contrastive_receipt = {
                "objective": contrastive_task_step_objective_contract(),
                "pair_count": int(state["contrastive_pair_count"]),
                "effective_pair_groups": int(
                    state["contrastive_effective_pair_groups"]
                ),
                "ineffective_groups": int(state["contrastive_ineffective_groups"]),
                "mixed_correctness_groups": int(
                    state["contrastive_mixed_correctness_groups"]
                ),
                "tier_candidate_counts": dict(
                    state["contrastive_tier_candidate_counts"]
                ),
                "behavior_delta_max_abs_error": float(
                    state["contrastive_behavior_delta_max_abs_error"]
                ),
                "policy_gradient_finite": state[
                    "contrastive_policy_gradient_finite"
                ],
                "policy_gradient_nonzero_groups": int(
                    state["contrastive_policy_gradient_nonzero_groups"]
                ),
                "expected_groups_since_u96": expected_contrastive_groups,
            }
            if (
                contrastive_groups != expected_contrastive_groups
                or contrastive_receipt["mixed_correctness_groups"]
                != contrastive_receipt["effective_pair_groups"]
                or contrastive_receipt["policy_gradient_nonzero_groups"]
                != contrastive_receipt["effective_pair_groups"]
                or sum(contrastive_receipt["tier_candidate_counts"].values())
                != expected_contrastive_groups * args.candidates
                or contrastive_receipt["pair_count"] < 0
                or contrastive_receipt["behavior_delta_max_abs_error"] > 1e-5
                or contrastive_receipt["policy_gradient_finite"] is not True
            ):
                raise RuntimeError(
                    "contrastive task-step receipt accounting is incomplete"
                )
            receipt["contrastive_task_step"] = contrastive_receipt
        if d3_policy_init:
            health = state["d3_health"]
            if health["bad_or_nan_count"] != 0:
                raise RuntimeError("D3 receipt has non-finite transition-TV diagnostics")
            if args.formal_rl and args.preflight_updates is None and not health["monitor_encountered_ids"]:
                raise RuntimeError("D3 formal run has zero TRAIN health-monitor task encounters")
            receipt["d3"] = {
                "group_objective": args.group_objective,
                "pkpo_schedule": contract["pkpo"]["schedule"],
                "pkpo_k_histogram": health["pkpo_k_histogram"],
                "posterior_tv_mean": health["posterior_tv_sum"] / max(health["groups"], 1),
                "transition_tv_mean": health["transition_tv_sum"] / max(health["groups"], 1),
                "reference_tv_coefficient": args.reference_tv_coefficient,
                "reference_tv_contribution_mean": (
                    health["transition_tv_contribution_sum"] / max(health["groups"], 1)
                ),
                "reference_kl": "per-policy-epoch diagnostics in train_history.jsonl",
                "bad_or_nan_count": health["bad_or_nan_count"],
                "monitor": contract["train_health_monitor"],
                "monitor_coverage": {
                    "encountered_ids": health["monitor_encountered_ids"],
                    "encountered_count": len(health["monitor_encountered_ids"]),
                },
            }
            if args.group_objective == D4_HYBRID_PKPO_OBJECTIVE:
                receipt["d3"]["pkpo_residual"] = contract["pkpo_residual"]
                receipt["d3"]["last_advantage_diagnostics"] = health[
                    "last_d4_advantage_diagnostics"
                ]
        if d5_strict_on_policy:
            receipt["d5"] = {
                "rollout_refresh_mode": contract["rollout_refresh_mode"],
                "rollout_refresh": contract["rollout_refresh"],
                "task_cursor_advance_per_logical_update": 4,
                "optimizer_steps_per_logical_update": 2,
                "groups_per_logical_update": 8,
                "candidates_per_logical_update": 64,
                "coverage": reward_evaluation_coverage,
                "temporal_credit_step_histogram": state[
                    "temporal_credit_step_histogram"
                ],
                "resume": "exact-logical-update-boundary-only",
            }
        atomic_json(args.output / "receipt.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    primary_call(context, "receipt validation and write", validate_and_write_receipt)
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
