"""Distributed execution primitives for endpoint-trajectory GRPO."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist


CONTRASTIVE_TASK_STEP_METHOD = "simplex-endpoint-policy-contrastive-task-step"
CONTRASTIVE_TASK_STEP_RATIO = (
    "selected-flow-step-discordant-candidate-pair-log-probability-delta"
)
SEQUENCE_GROUP_FPO_METHOD = "simplex-endpoint-policy-sequence-group-fpo"
SEQUENCE_GROUP_FPO_RATIO = "per-flow-step-joint-action-clipped-surrogate"
DOMINO_ENDPOINT_PPO_METHOD = "simplex-endpoint-policy-domino-ppo"
DOMINO_ENDPOINT_PPO_RATIO = "per-endpoint-step-joint-action-clipped-ratio"
DISCRETE_DOMINO_METHOD = "structured-discrete-mixture-dfm-domino-ppo"
DISCRETE_DOMINO_RATIO = "per-discrete-mixture-transition-joint-action-clipped-ratio"


def discrete_domino_objective_contract() -> dict:
    """Bind the structured discrete DFM inner-MDP used for the C1-B pilot."""
    return {
        "name": "discrete_domino",
        "reward_scope": "complete-terminal-sequence-only",
        "advantage": "group-relative-normalized-terminal-reward-per-task",
        "advantage_application": "trajectory-level-advantage-shared-across-time",
        "loss": "sum-over-discrete-transitions-of-ppo-clipped-surrogate",
        "ratio": DISCRETE_DOMINO_RATIO,
        "transition_policy": "structured-finite-step-mixture-kernel-(1-rho)delta+rho*p_clean",
        "mixture_scheduler": "linear-kappa-exact-finite-step-rho=1/(remaining-steps)",
        "state_space": "RNA-nucleotides-with-six-state-target-pair-units",
        "transition_probability_exact": True,
        "native_pretraining_objective": "dirichlet-simplex-flow-matching-transfer",
        "denoiser_input_bridge": "exact-conditional-Dirichlet-path-mean-from-discrete-state",
        "sequence_likelihood": "not-constructed-not-claimed",
    }


def domino_endpoint_ppo_objective_contract() -> dict:
    """Bind the DoMinO-style endpoint MDP without claiming native DFM exactness."""
    return {
        "name": "domino_endpoint_ppo",
        "reward_scope": "complete-terminal-sequence-only",
        "advantage": "group-relative-normalized-terminal-reward-per-task",
        "advantage_application": "trajectory-level-advantage-shared-across-time",
        "loss": "sum-over-endpoint-transitions-of-ppo-clipped-surrogate",
        "ratio": DOMINO_ENDPOINT_PPO_RATIO,
        "transition_policy": "stochastic-clean-endpoint-action-then-deterministic-simplex-step",
        "native_discrete_flow_transition_exact": False,
        "native_model_family": "dirichlet-simplex-flow-matching",
        "sequence_likelihood": "not-constructed-not-claimed",
    }


def sequence_group_fpo_objective_contract() -> dict:
    """Return the scoped B1 objective identity without a sequence-likelihood claim."""
    return {
        "name": "sequence_group_fpo",
        "reward_scope": "complete-terminal-sequence-only",
        "advantage": "group-relative-normalized-terminal-reward-per-task",
        "advantage_application": "same-sequence-advantage-at-every-legal-flow-step",
        "loss": "advantage-weighted-flow-matching-step-local-clipped-surrogate",
        "ratio": SEQUENCE_GROUP_FPO_RATIO,
        "temporal_credit": "forbidden",
        "per_step_reward": "forbidden",
        "sequence_likelihood": "not-constructed-not-claimed",
    }


def contrastive_task_step_objective_contract() -> dict:
    """Return the exact single-variable objective identity for U96 continuations."""
    return {
        "name": "contrastive_task_step",
        "pair_label": "MFE-primary-uMFE-secondary-correctness-tier",
        "correctness_tier": "2*int(mfe_hit)+int(uMFE_hit)",
        "legal_correctness_tiers": [0, 2, 3],
        "invalid_correctness_state": "uMFE-hit-without-MFE-hit-fail-closed",
        "invalid_evaluation": "fail-closed-never-treated-as-negative",
        "pair_population": "all-candidate-pairs-with-tier_w>tier_l",
        "pair_weighting": "uniform-mean-within-task-step-then-global-task-mean-after-causal-IS",
        "orientation_rule": "one-winner-to-loser-orientation-per-unordered-discordant-pair",
        "step_scope": "formal-causal_is_single-selected-step-per-task",
        "candidate_log_probability": "mean-over-selected-step-endpoint-action-units",
        "loss": "softplus(-((new_w-new_l)-(old_w-old_l)))",
        "tie_handling": "same-tier-excluded-target_probability-never-breaks-ties",
        "no_pair_policy_loss": "auditable-differentiable-zero",
        "forbidden_pair_labels": "Pair-F1-rival-structure-error-negative-error-benchmark-GT",
        "data_scope": "TRAIN-only-existing-terminal-evaluation-correctness-fields",
    }


def validated_correctness_tiers(
    mfe_hits: torch.Tensor, umfe_hits: torch.Tensor,
) -> torch.Tensor:
    """Construct legal 0/2/3 correctness tiers from terminal evaluation booleans."""
    if (
        mfe_hits.ndim != 1
        or umfe_hits.shape != mfe_hits.shape
        or mfe_hits.dtype != torch.bool
        or umfe_hits.dtype != torch.bool
        or mfe_hits.numel() < 2
    ):
        raise ValueError("contrastive correctness hit tensors are invalid")
    if bool((umfe_hits & ~mfe_hits).any().item()):
        raise RuntimeError("uMFE hit without MFE hit is an illegal correctness state")
    return 2 * mfe_hits.to(torch.long) + umfe_hits.to(torch.long)


def contrastive_task_step_loss(
    new_unit_log_probabilities: torch.Tensor,
    old_unit_log_probabilities: torch.Tensor,
    correctness_tiers: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Mean strict-correctness-tier pair loss on one selected Flow step.

    Each candidate's selected-step log probability is the mean over its endpoint
    action units. Tied correctness tiers never create a pair. A group without a strict pair
    returns a graph-connected zero so distributed backward remains auditable.
    """
    if (
        new_unit_log_probabilities.ndim != 3
        or new_unit_log_probabilities.shape[1] != 1
        or not new_unit_log_probabilities.requires_grad
        or old_unit_log_probabilities.shape != new_unit_log_probabilities.shape
        or correctness_tiers.ndim != 1
        or correctness_tiers.shape[0] != new_unit_log_probabilities.shape[0]
        or correctness_tiers.shape[0] < 2
        or correctness_tiers.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
        or not torch.isfinite(new_unit_log_probabilities).all()
        or not torch.isfinite(old_unit_log_probabilities).all()
        or not bool(
            ((correctness_tiers == 0) | (correctness_tiers == 2) | (correctness_tiers == 3))
            .all().item()
        )
    ):
        raise ValueError("contrastive task-step tensor contract is invalid")
    new_scores = new_unit_log_probabilities[:, 0].mean(dim=-1)
    old_scores = old_unit_log_probabilities[:, 0].mean(dim=-1)
    winner_indices, loser_indices = torch.where(
        correctness_tiers[:, None] > correctness_tiers[None, :]
    )
    pair_count = int(winner_indices.numel())
    if pair_count:
        pair_deltas = (
            (new_scores[winner_indices] - new_scores[loser_indices])
            - (old_scores[winner_indices] - old_scores[loser_indices])
        )
        loss = F.softplus(-pair_deltas).mean()
        pair_delta_max_abs = float(pair_deltas.detach().abs().max().item())
    else:
        loss = new_scores.sum() * 0.0
        pair_delta_max_abs = 0.0
    policy_gradient = torch.autograd.grad(
        loss,
        new_unit_log_probabilities,
        retain_graph=True,
        allow_unused=False,
    )[0]
    gradient_finite = bool(torch.isfinite(policy_gradient).all().item())
    gradient_nonzero = bool((policy_gradient.detach().abs() > 0).any().item())
    if not torch.isfinite(loss) or not gradient_finite:
        raise RuntimeError("contrastive task-step loss or policy gradient is non-finite")
    if pair_count and not gradient_nonzero:
        raise RuntimeError("effective contrastive task-step group has zero policy gradient")
    if not pair_count and gradient_nonzero:
        raise RuntimeError("ineffective contrastive task-step group has nonzero policy gradient")
    joint_log_ratios = new_scores.detach() - old_scores.detach()
    bounded = torch.clamp(joint_log_ratios, min=-20.0, max=20.0)
    ratios = torch.exp(bounded)
    tier_candidate_counts = {
        str(tier): int((correctness_tiers == tier).sum().item())
        for tier in (0, 2, 3)
    }
    return loss, {
        "pair_count": pair_count,
        "effective_pair_group": bool(pair_count),
        "ineffective_group": not bool(pair_count),
        "pair_delta_max_abs": pair_delta_max_abs,
        "policy_gradient_finite": gradient_finite,
        "policy_gradient_nonzero": gradient_nonzero,
        "tier_candidate_counts": tier_candidate_counts,
        "mixed_correctness_group": bool(pair_count),
        "mean_ratio": float(ratios.mean().item()),
        "minimum_ratio": float(ratios.min().item()),
        "maximum_ratio": float(ratios.max().item()),
        "clip_fraction": 0.0,
        "ratio_unit": "mean-selected-flow-step-action-unit",
        "pair_weighting": "uniform-mean-within-task-step-then-global-task-mean-after-causal-IS",
        "orientation_rule": "one-winner-to-loser-orientation-per-unordered-discordant-pair",
    }


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str | None

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def distributed_context(enabled: bool, requested_device: str) -> DistributedContext:
    """Initialize torchrun state, using Gloo only for explicit CPU preflights."""
    if not enabled:
        return DistributedContext(0, 1, 0, torch.device(requested_device), None)
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if any(name not in os.environ for name in required):
        raise RuntimeError("distributed trajectory training must be launched with torchrun")
    requested = torch.device(requested_device)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}") if requested.type == "cuda" else requested
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return DistributedContext(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def require_even_global_batches(
    tasks_per_update: int, supervised_batch_size: int, world_size: int,
) -> tuple[int, int]:
    if world_size <= 0:
        raise ValueError("world size must be positive")
    if tasks_per_update % world_size:
        raise ValueError("global tasks-per-update must divide evenly across ranks")
    if supervised_batch_size % world_size:
        raise ValueError("global supervised-batch-size must divide evenly across ranks")
    return tasks_per_update // world_size, supervised_batch_size // world_size


def rank_task_offsets(tasks_per_update: int, world_size: int, rank: int) -> list[int]:
    if world_size <= 0 or tasks_per_update <= 0 or tasks_per_update % world_size:
        raise ValueError("global tasks-per-update must divide evenly across ranks")
    if not 0 <= rank < world_size:
        raise ValueError("rank is outside world size")
    local_tasks = tasks_per_update // world_size
    start = rank * local_tasks
    return list(range(start, start + local_tasks))


def local_supervised_indices(
    supervised_batch_size: int, world_size: int, rank: int,
) -> list[int]:
    if world_size <= 0 or supervised_batch_size <= 0 or supervised_batch_size % world_size:
        raise ValueError("global supervised-batch-size must divide evenly across ranks")
    if not 0 <= rank < world_size:
        raise ValueError("rank is outside world size")
    local_batch = supervised_batch_size // world_size
    start = rank * local_batch
    return list(range(start, start + local_batch))


def global_task_mean_scale(global_tasks: int, world_size: int) -> float:
    if global_tasks <= 0 or world_size <= 0 or global_tasks % world_size:
        raise ValueError("global task mean requires an even positive task partition")
    return world_size / global_tasks


def global_token_mean_scale(
    local_tokens: float | torch.Tensor,
    global_tokens: float | torch.Tensor,
    world_size: int,
) -> float | torch.Tensor:
    if world_size <= 0 or float(local_tokens) <= 0 or float(global_tokens) <= 0:
        raise ValueError("global token mean requires positive token counts")
    return world_size * local_tokens / global_tokens


def keyed_dirichlet_path(
    clean: torch.Tensor,
    seed: int,
    update: int,
    policy_epoch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the complete supervised path from a global batch identity."""
    if clean.ndim != 2 or clean.shape[0] <= 0:
        raise ValueError("keyed Dirichlet path requires a non-empty [batch, length] tensor")
    clean = clean.to(device)
    clean_onehot = F.one_hot(clean, num_classes=4).float()
    states = []
    alphas = []
    for global_row in range(clean.shape[0]):
        generator = torch.Generator(device=device).manual_seed(
            json_keyed_seed(seed, update, policy_epoch, global_row)
        )
        alpha = 1 + torch.rand((), device=device, generator=generator) * 7
        concentration = torch.ones_like(clean_onehot[global_row])
        concentration = concentration + clean_onehot[global_row] * (alpha - 1)
        states.append(torch._sample_dirichlet(concentration, generator=generator))
        alphas.append(alpha)
    return torch.stack(states), torch.stack(alphas)


def json_keyed_seed(seed: int, update: int, policy_epoch: int, global_row: int) -> int:
    if seed < 0 or update < 0 or policy_epoch < 0 or global_row < 0:
        raise ValueError("keyed supervised path identifiers must be non-negative")
    encoded = f"endpoint-ce-v1:{seed}:{update}:{policy_epoch}:{global_row}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def validate_rank_rng_states(
    states: object,
    world_size: int,
    require_full_state: bool = False,
) -> list[dict]:
    if (
        not isinstance(states, list)
        or len(states) != world_size
        or any(not isinstance(state, dict) for state in states)
    ):
        raise RuntimeError("resume checkpoint lacks matching per-rank RNG state")
    if not require_full_state:
        return states
    for rank, state in enumerate(states):
        if set(state) != {"python", "numpy", "torch", "cuda"}:
            raise RuntimeError(
                f"resume checkpoint rank {rank} RNG state has missing or extra fields"
            )
        python_state = state["python"]
        numpy_state = state["numpy"]
        torch_state = state["torch"]
        cuda_states = state["cuda"]
        if (
            not isinstance(python_state, tuple)
            or len(python_state) != 3
            or not isinstance(numpy_state, tuple)
            or len(numpy_state) != 5
            or not isinstance(torch_state, torch.Tensor)
            or torch_state.dtype != torch.uint8
            or torch_state.numel() == 0
            or not isinstance(cuda_states, list)
            or len(cuda_states) != world_size
            or any(
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.uint8
                or value.numel() == 0
                for value in cuda_states
            )
        ):
            raise RuntimeError(f"resume checkpoint rank {rank} RNG state is malformed")
    return states


def rank_rng_state_from_checkpoint(payload: dict, context: DistributedContext) -> dict:
    states = payload.get("rng_state_by_rank")
    if states is None and context.world_size == 1:
        states = [payload.get("rng_state")]
    return validate_rank_rng_states(states, context.world_size)[context.rank]


def branch_rank_rng_state_from_checkpoint(payload: dict, context: DistributedContext) -> dict:
    """Restore branch RNG, allowing the reviewed DDP4->DDP2 C1 migration.

    Global task selection, rollout generators and supervised Dirichlet draws are
    keyed by global identifiers in the trainer.  For the world-size migration we
    retain source rank 0/1 CPU RNG states and truncate each rank's CUDA-state list
    to the two visible devices.  This is deterministic and identical across the
    two C1 arms, but is explicitly not claimed as byte-identical DDP4 continuation.
    """
    source_world_size = payload.get("contract", {}).get("distributed", {}).get("world_size")
    if source_world_size == context.world_size:
        return rank_rng_state_from_checkpoint(payload, context)
    if source_world_size != 4 or context.world_size != 2:
        raise RuntimeError("unsupported branch RNG world-size migration")
    states = validate_rank_rng_states(
        payload.get("rng_state_by_rank"), 4, require_full_state=True
    )
    source = states[context.rank]
    return {
        "python": source["python"],
        "numpy": source["numpy"],
        "torch": source["torch"],
        "cuda": list(source["cuda"][:2]),
    }


def checkpoint_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


BRANCH_RESUME_MUTABLE_CONTRACT_FIELDS = frozenset({
    "continuation",
    "endpoint_trajectory_ddp_sha256",
    "endpoint_policy_sha256",
    "branch_update_budget_mode",
    "implementation_manifest",
    "kl_coefficient",
    "learning_rate",
    "method",
    "policy_objective",
    "ratio",
    "repository_revision",
    "schema_version",
    "script_sha256",
    "temporal_credit",
    "updates",
})

BRANCH_RESUME_ALLOWED_IMPLEMENTATION_FILES = frozenset({
    "experiments/rna-flow-progressive-supervision-rl/endpoint_trajectory_ddp.py",
    "experiments/rna-flow-progressive-supervision-rl/train_endpoint_trajectory_grpo.py",
})

# The causal-window continuation changes the sampler/loss implementation in
# endpoint_policy.py in addition to the two existing resume-plumbing files.
# Keep the legacy allowlist unchanged so historical U192/KL10 review receipts
# remain valid, and select this stricter set only when the requested contract
# explicitly opts into causal_window_2.
CAUSAL_WINDOW_2_BRANCH_RESUME_ALLOWED_IMPLEMENTATION_FILES = frozenset({
    *BRANCH_RESUME_ALLOWED_IMPLEMENTATION_FILES,
    "experiments/rna-flow-progressive-supervision-rl/endpoint_policy.py",
})

U128_MATCHED32_BRANCH_RESUME_ALLOWED_IMPLEMENTATION_FILES = frozenset({
    *BRANCH_RESUME_ALLOWED_IMPLEMENTATION_FILES,
    "experiments/rna-flow-progressive-supervision-rl/endpoint_policy.py",
})


def _branch_resume_allowed_implementation_files(
    requested_contract: dict,
    source_contract: dict | None = None,
) -> frozenset[str]:
    temporal = requested_contract.get("temporal_credit")
    budget_mode = requested_contract.get("branch_update_budget_mode")
    causal_matched = budget_mode == "u144_matched48"
    endpoint_changed = (
        source_contract is not None
        and source_contract.get("endpoint_policy_sha256")
        != requested_contract.get("endpoint_policy_sha256")
    )
    if budget_mode in {"u128_matched32", "u128_domino32", "u128_discrete_domino32"}:
        return U128_MATCHED32_BRANCH_RESUME_ALLOWED_IMPLEMENTATION_FILES
    if causal_matched and endpoint_changed:
        return CAUSAL_WINDOW_2_BRANCH_RESUME_ALLOWED_IMPLEMENTATION_FILES
    return BRANCH_RESUME_ALLOWED_IMPLEMENTATION_FILES


def branch_resume_implementation_changes(
    source_contract: dict,
    requested_contract: dict,
) -> dict[str, dict[str, str]]:
    source = source_contract.get("implementation_manifest")
    requested = requested_contract.get("implementation_manifest")
    if (
        not isinstance(source, dict)
        or not isinstance(requested, dict)
        or set(source) != set(requested)
        or any(
            not isinstance(path, str)
            or not isinstance(source_sha, str)
            or not isinstance(requested.get(path), str)
            for path, source_sha in source.items()
        )
    ):
        raise RuntimeError("branch-resume implementation manifest is malformed")
    changed = {
        path: {"source_sha256": source[path], "requested_sha256": requested[path]}
        for path in sorted(source)
        if source[path] != requested[path]
    }
    allowed_files = _branch_resume_allowed_implementation_files(
        requested_contract, source_contract
    )
    if set(changed) != allowed_files:
        raise RuntimeError(
            "branch-resume implementation changes are not the reviewed resume plumbing"
        )
    trainer = "experiments/rna-flow-progressive-supervision-rl/train_endpoint_trajectory_grpo.py"
    helper = "experiments/rna-flow-progressive-supervision-rl/endpoint_trajectory_ddp.py"
    if (
        source_contract.get("script_sha256") != source[trainer]
        or requested_contract.get("script_sha256") != requested[trainer]
        or source_contract.get("endpoint_trajectory_ddp_sha256") != source[helper]
        or requested_contract.get("endpoint_trajectory_ddp_sha256") != requested[helper]
    ):
        raise RuntimeError("branch-resume implementation top-level hashes are inconsistent")
    if "experiments/rna-flow-progressive-supervision-rl/endpoint_policy.py" in allowed_files:
        endpoint_policy = (
            "experiments/rna-flow-progressive-supervision-rl/endpoint_policy.py"
        )
        if (
            source_contract.get("endpoint_policy_sha256") != source[endpoint_policy]
            or requested_contract.get("endpoint_policy_sha256") != requested[endpoint_policy]
        ):
            raise RuntimeError("branch-resume endpoint-policy hashes are inconsistent")
    return changed


def validate_branch_resume_review_receipt(
    receipt: dict,
    source_contract: dict,
    requested_contract: dict,
    source_contract_sha256: str,
    source_checkpoint_sha256: str,
    source_receipt_sha256: str,
) -> dict:
    allowed = sorted(
        _branch_resume_allowed_implementation_files(requested_contract, source_contract)
    )
    manifest = requested_contract.get("implementation_manifest")
    changes = branch_resume_implementation_changes(source_contract, requested_contract)
    reviewed_change_set_sha256 = hashlib.sha256(
        json.dumps(changes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "approved"
        or receipt.get("review_mode") not in {
            "independent-read-only", "lead-local-no-codex"
        }
        or (
            receipt.get("review_mode") == "lead-local-no-codex"
            and (
                requested_contract.get("branch_update_budget_mode")
                not in {"u128_domino32", "u128_discrete_domino32"}
                or receipt.get("codex_or_chatgpt_work_used") is not False
            )
        )
        or receipt.get("approval_scope") != "preflight-only"
        or receipt.get("source_contract_sha256") != source_contract_sha256
        or receipt.get("source_checkpoint_sha256") != source_checkpoint_sha256
        or receipt.get("source_receipt_sha256") != source_receipt_sha256
        or receipt.get("source_repository_revision")
        != source_contract.get("repository_revision")
        or receipt.get("requested_repository_revision")
        != requested_contract.get("repository_revision")
        or receipt.get("allowed_files") != allowed
        or not isinstance(manifest, dict)
        or receipt.get("requested_implementation_sha256")
        != hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        or receipt.get("reviewed_diff_sha256") != reviewed_change_set_sha256
    ):
        raise RuntimeError("branch-resume review receipt is incomplete or mismatched")
    return receipt | {"verified_reviewed_diff_sha256": reviewed_change_set_sha256}


def validate_branch_resume_stage_arguments(
    branch_resume_enabled: bool,
    preflight_updates: int | None,
    branch_preflight_receipt: Path | None,
    contrastive_in_place_resume: bool = False,
) -> None:
    """Make the reviewed one-update preflight impossible to bypass or reuse."""
    if contrastive_in_place_resume:
        if branch_resume_enabled or preflight_updates is not None:
            raise ValueError("contrastive in-place resume cannot be a branch or preflight")
        if branch_preflight_receipt is None:
            raise ValueError(
                "contrastive in-place resume requires its completed fresh preflight receipt"
            )
        return
    if not branch_resume_enabled:
        if branch_preflight_receipt is not None:
            raise ValueError("branch preflight receipt is only valid for branch resume")
        return
    if preflight_updates == 1:
        if branch_preflight_receipt is not None:
            raise ValueError("branch preflight must use a fresh output without a prior receipt")
        return
    if preflight_updates is None:
        if branch_preflight_receipt is None:
            raise ValueError("formal branch resume requires a completed fresh preflight receipt")
        return
    raise ValueError("branch resume preflight must run exactly one update")


def validate_branch_resume_preflight_receipt(
    receipt: dict,
    receipt_path: Path,
    requested_contract: dict,
    requested_contract_sha256: str,
    source_contract_sha256: str,
    source_checkpoint_sha256: str,
    source_receipt_sha256: str,
    review_receipt_sha256: str,
    formal_output: Path,
) -> dict:
    """Authorize formal branch training only through its exact fresh preflight."""
    preflight_output_value = receipt.get("output_path") if isinstance(receipt, dict) else None
    preflight_output = (
        Path(preflight_output_value).resolve()
        if isinstance(preflight_output_value, str) else None
    )
    contract_file = receipt.get("contract_file") if isinstance(receipt, dict) else None
    last_checkpoint = receipt.get("last_checkpoint") if isinstance(receipt, dict) else None
    branch_gate = receipt.get("branch_gate") if isinstance(receipt, dict) else None
    continuation = requested_contract.get("continuation")
    expected_contract_path = (
        preflight_output / "contract.json" if preflight_output is not None else None
    )
    expected_checkpoint_path = (
        preflight_output / "checkpoints" / "checkpoint-update-000097.pt"
        if preflight_output is not None else None
    )
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "preflight_complete"
        or receipt.get("formal_rl_started") is not False
        or receipt.get("start_update") != 96
        or receipt.get("updates_this_run") != 1
        or receipt.get("updates_complete") != 97
        or receipt.get("target_updates") != 97
        or receipt.get("bad_count") != 0
        or receipt.get("nan_inf_count") != 0
        or receipt.get("contract_sha256") != requested_contract_sha256
        or receipt.get("scientific_contract_sha256") != requested_contract_sha256
        or receipt.get("world_size")
        != requested_contract.get("distributed", {}).get("world_size")
        or preflight_output is None
        or receipt_path.resolve() != preflight_output / "receipt.json"
        or formal_output.resolve() == preflight_output
        or not isinstance(contract_file, dict)
        or expected_contract_path is None
        or Path(contract_file.get("path", "")).resolve() != expected_contract_path
        or not expected_contract_path.is_file()
        or contract_file.get("sha256") != checkpoint_file_sha256(expected_contract_path)
        or not isinstance(last_checkpoint, dict)
        or expected_checkpoint_path is None
        or Path(last_checkpoint.get("path", "")).resolve() != expected_checkpoint_path
        or last_checkpoint.get("next_update") != 97
        or not expected_checkpoint_path.is_file()
        or last_checkpoint.get("sha256") != checkpoint_file_sha256(expected_checkpoint_path)
        or not isinstance(branch_gate, dict)
        or branch_gate.get("approval_scope") != "formal-after-exact-preflight"
        or branch_gate.get("source_contract_sha256") != source_contract_sha256
        or branch_gate.get("source_checkpoint_sha256") != source_checkpoint_sha256
        or branch_gate.get("source_receipt_sha256") != source_receipt_sha256
        or branch_gate.get("review_receipt_sha256") != review_receipt_sha256
        or receipt.get("continuation") != continuation
        or not isinstance(continuation, dict)
        or continuation.get("source_contract_sha256") != source_contract_sha256
        or continuation.get("source_checkpoint_sha256") != source_checkpoint_sha256
        or continuation.get("source_receipt_sha256") != source_receipt_sha256
        or continuation.get("resume_plumbing_review_sha256") != review_receipt_sha256
    ):
        raise RuntimeError("branch-resume preflight receipt is incomplete or mismatched")
    contract_record = json.loads(expected_contract_path.read_text())
    embedded_contract_sha256 = contract_record.pop("contract_sha256", None)
    if (
        embedded_contract_sha256 != requested_contract_sha256
        or contract_record != requested_contract
        or hashlib.sha256(
            json.dumps(contract_record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() != requested_contract_sha256
    ):
        raise RuntimeError("branch-resume preflight scientific contract changed")
    coverage = receipt.get("reward_evaluation_coverage")
    if not isinstance(coverage, dict) or coverage.get("invalid_candidate_rows") != 0:
        raise RuntimeError("branch-resume preflight has invalid reward coverage")
    if requested_contract.get("method") == CONTRASTIVE_TASK_STEP_METHOD:
        contrastive = receipt.get("contrastive_task_step")
        if (
            not isinstance(contrastive, dict)
            or contrastive.get("objective")
            != contrastive_task_step_objective_contract()
            or type(contrastive.get("pair_count")) is not int
            or contrastive["pair_count"] <= 0
            or type(contrastive.get("effective_pair_groups")) is not int
            or contrastive["effective_pair_groups"] <= 0
            or type(contrastive.get("ineffective_groups")) is not int
            or contrastive["effective_pair_groups"]
            + contrastive["ineffective_groups"] != 4
            or contrastive.get("mixed_correctness_groups")
            != contrastive["effective_pair_groups"]
            or contrastive.get("policy_gradient_nonzero_groups")
            != contrastive["effective_pair_groups"]
            or not isinstance(contrastive.get("tier_candidate_counts"), dict)
            or set(contrastive["tier_candidate_counts"]) != {"0", "2", "3"}
            or any(
                type(value) is not int or value < 0
                for value in contrastive["tier_candidate_counts"].values()
            )
            or sum(contrastive["tier_candidate_counts"].values()) != 48
            or contrastive.get("expected_groups_since_u96") != 4
            or not isinstance(
                contrastive.get("behavior_delta_max_abs_error"), (int, float)
            )
            or not torch.isfinite(
                torch.tensor(contrastive["behavior_delta_max_abs_error"])
            )
            or contrastive["behavior_delta_max_abs_error"] > 1e-5
            or contrastive.get("policy_gradient_finite") is not True
        ):
            raise RuntimeError(
                "contrastive branch-resume preflight receipt is incomplete"
            )
    return receipt | {
        "verified_preflight_receipt_sha256": checkpoint_file_sha256(receipt_path),
        "verified_preflight_output": str(preflight_output),
    }


def optimizer_parameter_manifest_from_state_dict(state: dict) -> list[dict]:
    if not isinstance(state, dict) or not state:
        raise RuntimeError("branch-resume checkpoint has no trainable model state")
    manifest = []
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError("branch-resume trainable model state is malformed")
        manifest.append({
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        })
    return manifest


def optimizer_parameter_manifest_from_model(model: torch.nn.Module) -> list[dict]:
    manifest = [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
        for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not manifest:
        raise RuntimeError("branch-resume model has no trainable parameters")
    return manifest


def optimizer_parameter_manifest_sha256(manifest: list[dict]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def restore_branch_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state: dict,
    learning_rate: float,
) -> None:
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    if {float(group["lr"]) for group in optimizer.param_groups} != {float(learning_rate)}:
        raise RuntimeError("branch-resume optimizer learning-rate override failed")


def validate_branch_resume_contract(
    source_contract: dict,
    requested_contract: dict,
    learning_rate_mode: str,
    kl_coefficient_mode: str = "fixed",
) -> dict:
    """Allow reviewed continuations, including the scoped U128 DoMinO endpoint pilot."""
    if learning_rate_mode not in {"fixed", "half"}:
        raise ValueError("branch-resume learning-rate mode must be fixed or half")
    if kl_coefficient_mode not in {"fixed", "tenfold_pilot48"}:
        raise ValueError("branch-resume KL mode must be fixed or tenfold_pilot48")
    if not isinstance(source_contract, dict) or not isinstance(requested_contract, dict):
        raise RuntimeError("branch-resume contracts must be JSON objects")
    objective = requested_contract.get("policy_objective")
    contrastive = requested_contract.get("method") == CONTRASTIVE_TASK_STEP_METHOD
    source_temporal = source_contract.get("temporal_credit")
    requested_temporal = requested_contract.get("temporal_credit")
    budget_mode = requested_contract.get("branch_update_budget_mode", "u192_legacy")
    if budget_mode not in {
        "u192_legacy", "u144_matched48", "u128_matched32", "u128_domino32",
        "u128_discrete_domino32",
    }:
        raise RuntimeError("branch-resume update budget purpose is unknown")
    window2 = (
        isinstance(requested_temporal, dict)
        and requested_temporal.get("mode") == "causal_window_2"
    )
    matched_causal = budget_mode == "u144_matched48"
    matched_stage_b = budget_mode == "u128_matched32"
    matched_domino = budget_mode == "u128_domino32"
    matched_discrete_domino = budget_mode == "u128_discrete_domino32"
    if window2 and not matched_causal:
        raise RuntimeError("causal_window_2 requires the explicit u144_matched48 branch purpose")
    expected_updates = (
        128 if matched_stage_b or matched_domino or matched_discrete_domino else
        144 if matched_causal or contrastive or kl_coefficient_mode == "tenfold_pilot48"
        else 192
    )
    if (
        source_contract.get("formal_rl") is not True
        or requested_contract.get("formal_rl") is not True
        or source_contract.get("updates") != 96
        or requested_contract.get("updates") != expected_updates
    ):
        raise RuntimeError("branch-resume update budget violates its reviewed mode")
    if kl_coefficient_mode == "tenfold_pilot48" and learning_rate_mode != "half":
        raise RuntimeError("KL10 pilot must use the frozen half learning rate")
    if matched_causal:
        if (
            not isinstance(source_temporal, dict)
            or source_temporal.get("mode") != "causal_is_single"
            or source_temporal.get("causal_window_size", 1) != 1
            or not isinstance(requested_temporal, dict)
            or requested_temporal.get("mode")
            not in {"causal_is_single", "causal_window_2"}
            or requested_temporal.get("causal_window_size")
            != (2 if window2 else 1)
            or learning_rate_mode != "half"
            or kl_coefficient_mode != "fixed"
            or contrastive
        ):
            raise RuntimeError(
                "u144_matched48 must be the reviewed U96 causal continuation"
            )
    if matched_stage_b:
        sequence_group_fpo = requested_contract.get("method") == SEQUENCE_GROUP_FPO_METHOD
        if (
            not isinstance(source_temporal, dict)
            or source_temporal.get("mode") != "causal_is_single"
            or source_temporal.get("causal_window_size", 1) != 1
            or not isinstance(requested_temporal, dict)
            or learning_rate_mode != "half"
            or kl_coefficient_mode != "fixed"
            or requested_contract.get("updates") != 128
            or requested_contract.get("seed") != 1009
            or requested_contract.get("tasks_per_update") != 4
            or requested_contract.get("candidates_per_task") != 12
            or requested_contract.get("policy_epochs") != 2
            or requested_contract.get("trajectory_steps") != 8
            or requested_contract.get("reward") != "terminal"
            or requested_contract.get("credit_assignment") != "uniform"
            or requested_contract.get("rl_trainable_scope") != "last_2_backbone_and_head"
            or requested_contract.get("ce_coefficient") != 0.1
            or requested_contract.get("task_curriculum", {}).get("mode")
            != "deterministic_shuffle"
            or (sequence_group_fpo and (
                requested_contract.get("ratio") != SEQUENCE_GROUP_FPO_RATIO
                or requested_contract.get("policy_objective")
                != sequence_group_fpo_objective_contract()
                or requested_temporal.get("mode") != "none"
                or requested_temporal.get("selection")
                != "all-legal-flow-matching-steps-without-temporal-credit"
                or any(
                    requested_temporal.get(key) is not None
                    for key in (
                        "causal_window_size", "causal_step_policy_sha256",
                        "causal_step_ticket_counts", "source_aggregate_sha256",
                        "probability_by_step", "importance_weight",
                        "importance_weight_scope",
                    )
                )
            ))
            or (not sequence_group_fpo and (
                requested_contract.get("method") != source_contract.get("method")
                or requested_contract.get("ratio") != source_contract.get("ratio")
                or requested_contract.get("policy_objective")
                != source_contract.get("policy_objective")
                or requested_temporal.get("mode") != "causal_is_single"
            ))
        ):
            raise RuntimeError("u128_matched32 violates the frozen Stage-B B0/B1 contract")
    if matched_domino:
        if (
            not isinstance(source_temporal, dict)
            or source_temporal.get("mode") != "causal_is_single"
            or source_temporal.get("causal_window_size", 1) != 1
            or not isinstance(requested_temporal, dict)
            or requested_temporal.get("mode") != "none"
            or requested_temporal.get("selection")
            != "all-legal-flow-matching-steps-without-temporal-credit"
            or requested_contract.get("method") != DOMINO_ENDPOINT_PPO_METHOD
            or requested_contract.get("ratio") != DOMINO_ENDPOINT_PPO_RATIO
            or requested_contract.get("policy_objective")
            != domino_endpoint_ppo_objective_contract()
            or learning_rate_mode != "half"
            or kl_coefficient_mode != "fixed"
            or requested_contract.get("updates") != 128
            or requested_contract.get("distributed", {}).get("world_size") != 2
            or requested_contract.get("seed") != 1009
            or requested_contract.get("tasks_per_update") != 4
            or requested_contract.get("candidates_per_task") != 12
            or requested_contract.get("policy_epochs") != 2
            or requested_contract.get("trajectory_steps") != 8
            or requested_contract.get("reward") != "terminal"
            or requested_contract.get("credit_assignment") != "uniform"
            or requested_contract.get("rl_trainable_scope") != "last_2_backbone_and_head"
            or requested_contract.get("ce_coefficient") != 0.1
            or requested_contract.get("task_curriculum", {}).get("mode")
            != "deterministic_shuffle"
            or any(
                requested_temporal.get(key) is not None
                for key in (
                    "causal_window_size", "causal_step_policy_sha256",
                    "causal_step_ticket_counts", "source_aggregate_sha256",
                    "probability_by_step", "importance_weight",
                    "importance_weight_scope",
                )
            )
        ):
            raise RuntimeError("u128_domino32 violates the scoped DoMinO endpoint contract")
    if matched_discrete_domino:
        if (
            not isinstance(source_temporal, dict)
            or source_temporal.get("mode") != "causal_is_single"
            or source_temporal.get("causal_window_size", 1) != 1
            or not isinstance(requested_temporal, dict)
            or requested_temporal.get("mode") != "none"
            or requested_temporal.get("selection")
            != "all-structured-discrete-dfm-transitions"
            or requested_contract.get("method") != DISCRETE_DOMINO_METHOD
            or requested_contract.get("ratio") != DISCRETE_DOMINO_RATIO
            or requested_contract.get("policy_objective")
            != discrete_domino_objective_contract()
            or learning_rate_mode != "half"
            or kl_coefficient_mode != "fixed"
            or requested_contract.get("updates") != 128
            or requested_contract.get("distributed", {}).get("world_size") != 2
            or requested_contract.get("seed") != 1009
            or requested_contract.get("tasks_per_update") != 4
            or requested_contract.get("candidates_per_task") != 12
            or requested_contract.get("policy_epochs") != 2
            or requested_contract.get("trajectory_steps") != 8
            or requested_contract.get("reward") != "terminal"
            or requested_contract.get("credit_assignment") != "uniform"
            or requested_contract.get("rl_trainable_scope") != "last_2_backbone_and_head"
            or requested_contract.get("ce_coefficient") != 0.1
            or requested_contract.get("task_curriculum", {}).get("mode")
            != "deterministic_shuffle"
            or any(
                requested_temporal.get(key) is not None
                for key in (
                    "causal_window_size", "causal_step_policy_sha256",
                    "causal_step_ticket_counts", "source_aggregate_sha256",
                    "probability_by_step", "importance_weight",
                    "importance_weight_scope",
                )
            )
        ):
            raise RuntimeError(
                "u128_discrete_domino32 violates the structured Discrete-DoMinO contract"
            )
    if window2:
        if (
            not isinstance(source_temporal, dict)
            or source_temporal.get("mode") != "causal_is_single"
            or source_temporal.get("causal_window_size", 1) != 1
            or not isinstance(requested_temporal, dict)
            or requested_temporal.get("causal_window_size") != 2
        ):
            raise RuntimeError(
                "causal_window_2 branch must be the reviewed U96 causal-is-single continuation"
            )
        # Only the mode/window and its mathematically implied importance label
        # may differ.  Ticket policy, key, probabilities, and all contract
        # metadata remain frozen across the continuation.
        temporal_differences = {
            key for key in set(source_temporal) | set(requested_temporal)
            if key not in {"mode", "causal_window_size", "importance_weight"}
            and source_temporal.get(key) != requested_temporal.get(key)
        }
        if temporal_differences:
            raise RuntimeError(
                "causal_window_2 temporal contract changed frozen metadata: "
                f"{sorted(temporal_differences)}"
            )
    if contrastive:
        if (
            objective != contrastive_task_step_objective_contract()
            or requested_contract.get("ratio") != CONTRASTIVE_TASK_STEP_RATIO
            or requested_contract.get("reward") != "terminal"
            or requested_contract.get("credit_assignment") != "uniform"
            or requested_contract.get("temporal_credit", {}).get("mode")
            != "causal_is_single"
            or requested_contract.get("task_curriculum", {}).get("mode")
            != "deterministic_shuffle"
            or requested_contract.get("updates") != 144
            or learning_rate_mode != "half"
            or kl_coefficient_mode != "fixed"
        ):
            raise RuntimeError(
                "contrastive task-step branch violates its frozen objective contract"
            )
    elif not matched_stage_b and not matched_domino and not matched_discrete_domino and (
        requested_contract.get("method") != source_contract.get("method")
        or requested_contract.get("ratio") != source_contract.get("ratio")
        or objective != source_contract.get("policy_objective")
    ):
        raise RuntimeError("non-contrastive branch changed the frozen policy objective")
    implementation_changes = branch_resume_implementation_changes(
        source_contract, requested_contract
    )
    continuation = requested_contract.get("continuation")
    if (
        not isinstance(continuation, dict)
        or continuation.get("mode") != "completed-branch-exact-state-v1"
        or continuation.get("start_update") != 96
        or continuation.get("allowed_implementation_changes") != implementation_changes
        or continuation.get("policy_objective")
        != (
            "contrastive_task_step" if contrastive
            else "sequence_group_fpo" if requested_contract.get("method") == SEQUENCE_GROUP_FPO_METHOD
            else "domino_endpoint_ppo" if requested_contract.get("method") == DOMINO_ENDPOINT_PPO_METHOD
            else "discrete_domino" if requested_contract.get("method") == DISCRETE_DOMINO_METHOD
            else None
        )
        or not isinstance(continuation.get("resume_plumbing_review_sha256"), str)
        or len(continuation["resume_plumbing_review_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in continuation["resume_plumbing_review_sha256"]
        )
    ):
        raise RuntimeError("branch-resume continuation/review binding is incomplete")
    source_core = {
        key: value for key, value in source_contract.items()
        if key not in BRANCH_RESUME_MUTABLE_CONTRACT_FIELDS
    }
    requested_core = {
        key: value for key, value in requested_contract.items()
        if key not in BRANCH_RESUME_MUTABLE_CONTRACT_FIELDS
    }
    stage_b_mutable = {
        "method", "ratio", "policy_objective", "temporal_credit", "schema_version",
    } if matched_stage_b or matched_domino or matched_discrete_domino else set()
    if matched_domino or matched_discrete_domino:
        # C1 deliberately migrates the source U96 DDP4 checkpoint to DDP2.
        # These per-rank fields are algebraic consequences of preserving the
        # same global task/supervised batch geometry.  Gradient checkpointing
        # is a compute/memory implementation choice and is explicitly tested
        # by C0 before it can be carried into C1.
        stage_b_mutable = set(stage_b_mutable) | {
            "distributed", "local_tasks_per_rank", "local_supervised_batch_size",
            "gradient_checkpointing",
        }
    if matched_discrete_domino:
        # These two fields are the scientific variable under test for the
        # discrete arm: a true structured discrete transition policy replaces
        # the stochastic endpoint action + deterministic simplex transition.
        stage_b_mutable = set(stage_b_mutable) | {"action_policy", "transition"}
    if ({key: value for key, value in source_core.items() if key not in stage_b_mutable}
        != {key: value for key, value in requested_core.items() if key not in stage_b_mutable}):
        differing = sorted({
            key for key in set(source_core) | set(requested_core)
            if key not in stage_b_mutable and source_core.get(key) != requested_core.get(key)
        })
        raise RuntimeError(
            f"branch-resume scientific contract mismatch: {differing}"
        )
    source_updates = source_contract["updates"]
    requested_updates = requested_contract["updates"]
    source_lr = source_contract.get("learning_rate")
    requested_lr = requested_contract.get("learning_rate")
    if not isinstance(source_lr, (int, float)) or not isinstance(requested_lr, (int, float)):
        raise RuntimeError("branch-resume learning rates are missing")
    expected_lr = float(source_lr) * (0.5 if learning_rate_mode == "half" else 1.0)
    if not torch.isclose(
        torch.tensor(float(requested_lr), dtype=torch.float64),
        torch.tensor(expected_lr, dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    ):
        raise RuntimeError("branch-resume requested learning rate violates its mode")
    source_kl = source_contract.get("kl_coefficient")
    requested_kl = requested_contract.get("kl_coefficient")
    if not isinstance(source_kl, (int, float)) or not isinstance(requested_kl, (int, float)):
        raise RuntimeError("branch-resume KL coefficients are missing")
    expected_kl = float(source_kl) * (
        10.0 if kl_coefficient_mode == "tenfold_pilot48" else 1.0
    )
    if not torch.isclose(
        torch.tensor(float(requested_kl), dtype=torch.float64),
        torch.tensor(expected_kl, dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    ):
        raise RuntimeError("branch-resume requested KL coefficient violates its mode")
    return {
        "source_updates": source_updates,
        "target_updates": requested_updates,
        "source_learning_rate": float(source_lr),
        "requested_learning_rate": float(requested_lr),
        "learning_rate_mode": learning_rate_mode,
        "source_kl_coefficient": float(source_kl),
        "requested_kl_coefficient": float(requested_kl),
        "kl_coefficient_mode": kl_coefficient_mode,
        "policy_objective": "contrastive_task_step" if contrastive else "normalized_grpo",
        "allowed_implementation_changes": implementation_changes,
    }


def validate_branch_resume_checkpoint(
    payload: dict,
    receipt: dict,
    checkpoint_path: Path,
    world_size: int,
) -> dict:
    """Validate a terminal formal checkpoint before a fresh-output continuation."""
    contract = payload.get("contract")
    state = payload.get("state")
    contract_hash = payload.get("contract_sha256")
    last_checkpoint = receipt.get("last_checkpoint") if isinstance(receipt, dict) else None
    optimizer = payload.get("optimizer")
    if (
        not isinstance(contract, dict)
        or contract_hash is None
        or not isinstance(state, dict)
        or not isinstance(optimizer, dict)
        or not isinstance(payload.get("trainable_model"), dict)
        or not isinstance(receipt, dict)
        or receipt.get("status") != "complete"
        or receipt.get("formal_rl_started") is not True
        or receipt.get("bad_count") != 0
        or receipt.get("contract_sha256") != contract_hash
        or not isinstance(last_checkpoint, dict)
        or Path(last_checkpoint.get("path", "")).resolve() != checkpoint_path.resolve()
        or last_checkpoint.get("sha256") != checkpoint_file_sha256(checkpoint_path)
        or contract_hash != hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        or state.get("next_update") != contract.get("updates")
        or receipt.get("updates_complete") != contract.get("updates")
        or receipt.get("target_updates") != contract.get("updates")
        or world_size not in {2, 4}
        or contract.get("distributed", {}).get("world_size") != 4
        or contract.get("formal_rl") is not True
        or contract.get("updates") != 96
        or contract.get("tasks_per_update") != 4
        or contract.get("candidates_per_task") != 12
        or contract.get("policy_epochs") != 2
        or contract.get("trajectory_steps") != 8
        or contract.get("temporal_credit", {}).get("mode") != "causal_is_single"
        or contract.get("task_curriculum", {}).get("mode") != "deterministic_shuffle"
    ):
        raise RuntimeError("branch-resume checkpoint/receipt is incomplete or invalid")
    validate_rank_rng_states(
        payload.get("rng_state_by_rank"), 4, require_full_state=True
    )
    expected_groups = contract["updates"] * contract["tasks_per_update"]
    expected_candidates = expected_groups * contract["candidates_per_task"]
    coverage = state.get("reward_evaluation_coverage")
    receipt_coverage = receipt.get("reward_evaluation_coverage")
    temporal_histogram = state.get("temporal_credit_step_histogram")
    receipt_histogram = receipt.get("temporal_credit", {}).get("step_histogram")
    expected_steps = {str(step) for step in range(contract["trajectory_steps"])}
    if (
        state.get("task_epoch") != 0
        or state.get("task_cursor") != expected_groups
        or not isinstance(coverage, dict)
        or not isinstance(receipt_coverage, dict)
        or coverage.get("candidate_rows") != expected_candidates
        or coverage.get("groups") != expected_groups
        or coverage.get("invalid_candidate_rows") != 0
        or any(receipt_coverage.get(key) != value for key, value in coverage.items())
        or receipt_coverage.get("candidate_rows") != expected_candidates
        or receipt_coverage.get("groups") != expected_groups
        or receipt_coverage.get("invalid_candidate_rows") != 0
        or coverage.get("invalid_update_histogram")
        != {str(update): 0 for update in range(contract["updates"])}
        or not isinstance(temporal_histogram, dict)
        or not isinstance(receipt_histogram, dict)
        or set(temporal_histogram) != expected_steps
        or temporal_histogram != receipt_histogram
        or any(type(value) is not int or value < 0 for value in temporal_histogram.values())
        or sum(temporal_histogram.values()) != expected_groups
    ):
        raise RuntimeError("branch-resume task cursor/coverage/histogram is invalid")
    manifest = optimizer_parameter_manifest_from_state_dict(payload["trainable_model"])
    stored_manifest = payload.get("optimizer_parameter_manifest")
    if stored_manifest is not None and stored_manifest != manifest:
        raise RuntimeError("branch-resume optimizer parameter manifest mismatch")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("branch-resume optimizer parameter coverage mismatch")
    parameter_ids = [
        parameter_id
        for group in groups
        for parameter_id in group.get("params", [])
    ]
    if len(parameter_ids) != len(manifest) or len(set(parameter_ids)) != len(parameter_ids):
        raise RuntimeError("branch-resume optimizer parameter coverage mismatch")
    optimizer_states = optimizer.get("state")
    if not isinstance(optimizer_states, dict) or set(optimizer_states) != set(parameter_ids):
        raise RuntimeError("branch-resume optimizer state does not cover every parameter")
    expected_optimizer_step = contract["updates"] * contract["policy_epochs"]
    for parameter_id, parameter_manifest in zip(parameter_ids, manifest, strict=True):
        parameter_state = optimizer_states[parameter_id]
        if not isinstance(parameter_state, dict):
            raise RuntimeError("branch-resume optimizer state is malformed")
        step = parameter_state.get("step")
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise RuntimeError("branch-resume optimizer step is malformed")
            step = step.item()
        if step != expected_optimizer_step:
            raise RuntimeError("branch-resume optimizer step is not 192 for every parameter")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state.get(moment_name)
            if (
                not isinstance(moment, torch.Tensor)
                or list(moment.shape) != parameter_manifest["shape"]
                or not torch.isfinite(moment).all()
            ):
                raise RuntimeError(
                    f"branch-resume optimizer {moment_name} is missing or malformed"
                )
    group_lrs = {float(group.get("lr", float("nan"))) for group in groups}
    if group_lrs != {float(contract.get("learning_rate", float("nan")))}:
        raise RuntimeError("branch-resume optimizer learning rate disagrees with contract")
    group_weight_decays = {
        float(group.get("weight_decay", float("nan"))) for group in groups
    }
    if group_weight_decays != {float(contract.get("weight_decay", float("nan")))}:
        raise RuntimeError("branch-resume optimizer weight decay disagrees with contract")
    return {
        "contract": contract,
        "contract_sha256": contract_hash,
        "start_update": int(state["next_update"]),
        "parameter_manifest": manifest,
        "parameter_manifest_sha256": optimizer_parameter_manifest_sha256(manifest),
        "source_learning_rate": float(contract["learning_rate"]),
        "optimizer_step": expected_optimizer_step,
        "task_epoch": int(state["task_epoch"]),
        "task_cursor": int(state["task_cursor"]),
        "reward_evaluation_coverage": coverage,
        "temporal_credit_step_histogram": temporal_histogram,
        "source_world_size": 4,
        "target_world_size": int(world_size),
        "rng_migration": (
            "exact-rank-state" if world_size == 4
            else "ddp4-to-ddp2-source-rank01-cuda-state-truncation"
        ),
    }


def validate_resume_checkpoint(
    payload: dict,
    expected_contract: dict,
    expected_contract_sha256: str,
    expected_world_size: int,
) -> None:
    distributed = payload.get("contract", {}).get("distributed", {})
    if (
        payload.get("contract") != expected_contract
        or payload.get("contract_sha256") != expected_contract_sha256
        or distributed.get("world_size") != expected_world_size
    ):
        raise RuntimeError("trajectory GRPO resume contract/hash/world-size mismatch")
    if not isinstance(payload.get("state"), dict):
        raise RuntimeError("trajectory GRPO resume checkpoint has no state")
    validate_rank_rng_states(payload.get("rng_state_by_rank"), expected_world_size)
    if expected_contract.get("method") == CONTRASTIVE_TASK_STEP_METHOD:
        validate_contrastive_resume_checkpoint_state(payload, expected_world_size)
    if expected_contract.get("method") == SEQUENCE_GROUP_FPO_METHOD:
        validate_sequence_group_fpo_resume_checkpoint_state(
            payload, expected_world_size
        )
    if expected_contract.get("method") == DOMINO_ENDPOINT_PPO_METHOD:
        validate_domino_endpoint_ppo_resume_checkpoint_state(
            payload, expected_world_size
        )
    if expected_contract.get("method") == DISCRETE_DOMINO_METHOD:
        validate_discrete_domino_resume_checkpoint_state(
            payload, expected_world_size
        )


def validate_discrete_domino_resume_checkpoint_state(
    payload: dict, expected_world_size: int,
) -> None:
    """Reuse the strict all-step U128 state gate for the structured discrete pilot."""
    contract = payload.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("method") != DISCRETE_DOMINO_METHOD
        or contract.get("ratio") != DISCRETE_DOMINO_RATIO
        or contract.get("policy_objective") != discrete_domino_objective_contract()
    ):
        raise RuntimeError("Discrete-DoMinO resume contract is incomplete or foreign")
    normalized = dict(payload)
    normalized["contract"] = dict(contract)
    normalized["contract"]["policy_objective"] = sequence_group_fpo_objective_contract()
    validate_sequence_group_fpo_resume_checkpoint_state(normalized, expected_world_size)


def validate_domino_endpoint_ppo_resume_checkpoint_state(
    payload: dict, expected_world_size: int,
) -> None:
    """Reuse the strict all-step U128 state gate for the DoMinO endpoint pilot."""
    contract = payload.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("method") != DOMINO_ENDPOINT_PPO_METHOD
        or contract.get("ratio") != DOMINO_ENDPOINT_PPO_RATIO
        or contract.get("policy_objective") != domino_endpoint_ppo_objective_contract()
    ):
        raise RuntimeError("DoMinO endpoint resume contract is incomplete or foreign")
    normalized = dict(payload)
    normalized["contract"] = dict(contract)
    normalized["contract"]["policy_objective"] = sequence_group_fpo_objective_contract()
    validate_sequence_group_fpo_resume_checkpoint_state(normalized, expected_world_size)


def validate_sequence_group_fpo_resume_checkpoint_state(
    payload: dict, expected_world_size: int,
) -> None:
    """Fail closed before restoring a B1 model, AdamW, RNG4, cursor, or coverage."""
    contract = payload["contract"]
    state = payload["state"]
    next_update = state.get("next_update")
    coverage = state.get("reward_evaluation_coverage")
    temporal_histogram = state.get("temporal_credit_step_histogram")
    optimizer = payload.get("optimizer")
    trainable_model = payload.get("trainable_model")
    if (
        expected_world_size not in {2, 4}
        or contract.get("distributed", {}).get("world_size") != expected_world_size
        or contract.get("formal_rl") is not True
        # ``updates`` and ``next_update`` are absolute formal update indices.
        # The matched Stage-B budget is the interval U96 -> U128 (32 updates),
        # not a fresh U0 -> U32 run.
        or contract.get("updates") != 128
        or contract.get("tasks_per_update") != 4
        or contract.get("candidates_per_task") != 12
        or contract.get("policy_epochs") != 2
        or contract.get("trajectory_steps") != 8
        or contract.get("reward") != "terminal"
        or contract.get("credit_assignment") != "uniform"
        or contract.get("temporal_credit", {}).get("mode") != "none"
        or contract.get("policy_objective") != sequence_group_fpo_objective_contract()
        or contract.get("continuation", {}).get("start_update") != 96
        or type(next_update) is not int
        or not 96 < next_update <= 128
        or state.get("task_epoch") != 0
        or state.get("task_cursor") != next_update * 4
        or not isinstance(coverage, dict)
        or coverage.get("candidate_rows") != next_update * 4 * 12
        or coverage.get("groups") != next_update * 4
        or coverage.get("invalid_candidate_rows") != 0
        or coverage.get("invalid_update_histogram")
        != {str(update): 0 for update in range(next_update)}
        or not isinstance(temporal_histogram, dict)
        or temporal_histogram != {
            str(step): (next_update - 96) * 4 for step in range(8)
        }
        or not isinstance(trainable_model, dict)
        or not trainable_model
        or not isinstance(optimizer, dict)
        or not isinstance(optimizer.get("state"), dict)
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer["param_groups"]
    ):
        raise RuntimeError("sequence-group FPO resume state is incomplete or foreign")
    validate_rank_rng_states(
        payload.get("rng_state_by_rank"), expected_world_size, require_full_state=True
    )
    manifest = optimizer_parameter_manifest_from_state_dict(trainable_model)
    stored_manifest = payload.get("optimizer_parameter_manifest")
    if stored_manifest != manifest:
        raise RuntimeError("sequence-group FPO resume optimizer manifest mismatch")
    parameter_ids = [
        parameter_id
        for group in optimizer["param_groups"]
        for parameter_id in group.get("params", [])
    ]
    if (
        len(parameter_ids) != len(manifest)
        or len(set(parameter_ids)) != len(parameter_ids)
        or set(optimizer["state"]) != set(parameter_ids)
    ):
        raise RuntimeError("sequence-group FPO resume AdamW coverage is incomplete")
    expected_step = next_update * contract["policy_epochs"]
    for parameter_id, parameter_manifest in zip(parameter_ids, manifest, strict=True):
        parameter_state = optimizer["state"][parameter_id]
        step = parameter_state.get("step") if isinstance(parameter_state, dict) else None
        if isinstance(step, torch.Tensor):
            step = step.item() if step.numel() == 1 else None
        if step != expected_step:
            raise RuntimeError("sequence-group FPO resume AdamW step is inconsistent")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state.get(moment_name)
            if (
                not isinstance(moment, torch.Tensor)
                or list(moment.shape) != parameter_manifest["shape"]
                or not torch.isfinite(moment).all()
            ):
                raise RuntimeError("sequence-group FPO resume AdamW moment is invalid")


def validate_contrastive_resume_checkpoint_state(
    payload: dict, expected_world_size: int,
) -> None:
    """Fail closed on every dynamic invariant before an in-place U97..U144 load."""
    contract = payload["contract"]
    state = payload["state"]
    next_update = state.get("next_update")
    if (
        expected_world_size != 4
        or contract.get("distributed", {}).get("world_size") != 4
        or contract.get("updates") != 144
        or contract.get("tasks_per_update") != 4
        or contract.get("candidates_per_task") != 12
        or contract.get("policy_epochs") != 2
        or contract.get("trajectory_steps") != 8
        or type(next_update) is not int
        or not 96 < next_update <= 144
    ):
        raise RuntimeError("contrastive resume checkpoint update geometry is invalid")
    validate_rank_rng_states(
        payload.get("rng_state_by_rank"), expected_world_size, require_full_state=True
    )
    expected_groups = next_update * 4
    expected_candidates = expected_groups * 12
    coverage = state.get("reward_evaluation_coverage")
    temporal_histogram = state.get("temporal_credit_step_histogram")
    if (
        state.get("task_epoch") != 0
        or state.get("task_cursor") != expected_groups
        or not isinstance(coverage, dict)
        or coverage.get("candidate_rows") != expected_candidates
        or coverage.get("groups") != expected_groups
        or coverage.get("invalid_candidate_rows") != 0
        or coverage.get("invalid_groups") != 0
        or coverage.get("all_invalid_groups") != 0
        or coverage.get("invalid_cache_key_histogram") != {}
        or coverage.get("error_code_histogram") != {}
        or coverage.get("invalid_task_histogram") != {}
        or coverage.get("invalid_target_length_histogram") != {}
        or coverage.get("invalid_update_histogram")
        != {str(update): 0 for update in range(next_update)}
        or not isinstance(temporal_histogram, dict)
        or set(temporal_histogram) != {str(step) for step in range(8)}
        or any(type(value) is not int or value < 0 for value in temporal_histogram.values())
        or sum(temporal_histogram.values()) != expected_groups
    ):
        raise RuntimeError("contrastive resume checkpoint cursor/coverage/histogram is invalid")
    expected_contrastive_groups = (next_update - 96) * 4
    tier_counts = state.get("contrastive_tier_candidate_counts")
    effective = state.get("contrastive_effective_pair_groups")
    ineffective = state.get("contrastive_ineffective_groups")
    pair_count = state.get("contrastive_pair_count")
    if (
        type(effective) is not int
        or type(ineffective) is not int
        or type(pair_count) is not int
        or effective < 0
        or ineffective < 0
        or pair_count < 0
        or effective + ineffective != expected_contrastive_groups
        or state.get("contrastive_mixed_correctness_groups") != effective
        or state.get("contrastive_policy_gradient_nonzero_groups") != effective
        or state.get("contrastive_policy_gradient_finite") is not True
        or not isinstance(tier_counts, dict)
        or set(tier_counts) != {"0", "2", "3"}
        or any(type(value) is not int or value < 0 for value in tier_counts.values())
        or sum(tier_counts.values()) != expected_contrastive_groups * 12
        or pair_count < effective
        or pair_count
        > expected_contrastive_groups * 12 * 11 // 2
        or not isinstance(
            state.get("contrastive_behavior_delta_max_abs_error"), (int, float)
        )
        or not math.isfinite(
            float(state["contrastive_behavior_delta_max_abs_error"])
        )
        or float(state["contrastive_behavior_delta_max_abs_error"]) > 1e-5
    ):
        raise RuntimeError("contrastive resume checkpoint accounting is invalid")
    model_state = payload.get("trainable_model")
    if not isinstance(model_state, dict):
        raise RuntimeError("contrastive resume checkpoint has no trainable model state")
    manifest = optimizer_parameter_manifest_from_state_dict(model_state)
    if any(not torch.isfinite(value).all() for value in model_state.values()):
        raise RuntimeError("contrastive resume checkpoint parameters are non-finite")
    if payload.get("optimizer_parameter_manifest") != manifest:
        raise RuntimeError("contrastive resume optimizer parameter manifest mismatch")
    optimizer = payload.get("optimizer")
    groups = optimizer.get("param_groups") if isinstance(optimizer, dict) else None
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("contrastive resume optimizer parameter coverage mismatch")
    parameter_ids = [
        parameter_id for group in groups for parameter_id in group.get("params", [])
    ]
    optimizer_states = optimizer.get("state")
    if (
        len(parameter_ids) != len(manifest)
        or len(set(parameter_ids)) != len(parameter_ids)
        or not isinstance(optimizer_states, dict)
        or set(optimizer_states) != set(parameter_ids)
    ):
        raise RuntimeError("contrastive resume optimizer state does not cover every parameter")
    expected_step = 2 * next_update
    for parameter_id, parameter_manifest in zip(parameter_ids, manifest, strict=True):
        parameter_state = optimizer_states[parameter_id]
        if not isinstance(parameter_state, dict):
            raise RuntimeError("contrastive resume optimizer state is malformed")
        step = parameter_state.get("step")
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise RuntimeError("contrastive resume optimizer step is malformed")
            step = step.item()
        if step != expected_step:
            raise RuntimeError("contrastive resume optimizer step is not 2N")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state.get(moment_name)
            if (
                not isinstance(moment, torch.Tensor)
                or list(moment.shape) != parameter_manifest["shape"]
                or not torch.isfinite(moment).all()
            ):
                raise RuntimeError(
                    f"contrastive resume optimizer {moment_name} is missing or malformed"
                )
    if {float(group.get("lr", float("nan"))) for group in groups} != {
        float(contract.get("learning_rate", float("nan")))
    }:
        raise RuntimeError("contrastive resume optimizer learning rate changed")
    if {float(group.get("weight_decay", float("nan"))) for group in groups} != {
        float(contract.get("weight_decay", float("nan")))
    }:
        raise RuntimeError("contrastive resume optimizer weight decay changed")


def rank_local_call(context: DistributedContext, description: str, operation):
    """Run rank-local work, then make any rank failure fail every peer coherently."""
    value = None
    failure = None
    try:
        value = operation()
    except Exception as error:
        failure = f"rank {context.rank} {type(error).__name__}: {error}"
    if context.world_size > 1:
        failures: list[str | None] = [None] * context.world_size
        dist.all_gather_object(failures, failure)
    else:
        failures = [failure]
    reported = [item for item in failures if item is not None]
    if reported:
        raise RuntimeError(f"distributed {description} failed: {reported}")
    return value


def load_rank_local_resume_checkpoint(
    context: DistributedContext,
    path: Path | str,
    expected_contract: dict,
    expected_contract_sha256: str,
) -> tuple[dict, dict]:
    """Validate on rank zero, then require every rank to load the shared checkpoint."""
    path = Path(path)

    def primary_metadata() -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(
            payload, expected_contract, expected_contract_sha256, context.world_size
        )
        return {
            "checkpoint_sha256": checkpoint_file_sha256(path),
            "contract_sha256": payload["contract_sha256"],
            "world_size": context.world_size,
            "state": payload["state"],
        }

    metadata = primary_call(context, "resume validation", primary_metadata)

    def load_and_validate() -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(
            payload, expected_contract, expected_contract_sha256, context.world_size
        )
        if (
            checkpoint_file_sha256(path) != metadata["checkpoint_sha256"]
            or payload["state"] != metadata["state"]
            or metadata["contract_sha256"] != expected_contract_sha256
            or metadata["world_size"] != context.world_size
        ):
            raise RuntimeError("trajectory GRPO resume checkpoint metadata mismatch")
        return payload

    return rank_local_call(context, "resume checkpoint load", load_and_validate), metadata


def gather_rank_rng_states(context: DistributedContext, local_state: dict) -> list[dict]:
    if context.world_size == 1:
        return [local_state]
    states: list[dict | None] = [None] * context.world_size
    dist.all_gather_object(states, local_state)
    return validate_rank_rng_states(states, context.world_size)


def gather_objects(context: DistributedContext, value: object) -> list[object]:
    if context.world_size == 1:
        return [value]
    values: list[object | None] = [None] * context.world_size
    dist.all_gather_object(values, value)
    if any(value is None for value in values):
        raise RuntimeError("distributed object gather returned an incomplete rank set")
    return values


def primary_call(context: DistributedContext, description: str, operation):
    """Run a rank-zero-only operation and raise its error on every rank."""
    outcome: tuple[bool, object]
    if context.is_primary:
        try:
            outcome = (True, operation())
        except Exception as error:
            outcome = (False, f"{type(error).__name__}: {error}")
    else:
        outcome = (False, None)
    if context.world_size > 1:
        values = [outcome if context.is_primary else None]
        dist.broadcast_object_list(values, src=0)
        outcome = values[0]
    if not outcome[0]:
        raise RuntimeError(f"rank0 {description} failed: {outcome[1]}")
    return outcome[1]


def global_sum(context: DistributedContext, value: torch.Tensor) -> torch.Tensor:
    total = value.detach().clone()
    if context.world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total


def broadcast_primary(context: DistributedContext, value: object) -> object:
    if context.world_size == 1:
        return value
    values = [value if context.is_primary else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def global_max_memory(context: DistributedContext, local_memory: int) -> int:
    if context.world_size == 1:
        return local_memory
    memory = torch.tensor([local_memory], dtype=torch.long, device=context.device)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    return int(memory.item())
