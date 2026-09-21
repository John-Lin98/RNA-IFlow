"""Resume-safe TRAIN-only comparison of phase-2 policy objectives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

FAIR_DIR = Path(__file__).resolve().parents[1] / "rna-flow-fair-components"
sys.path.insert(0, str(FAIR_DIR))
from evaluate import evaluate_candidate, sha256  # noqa: E402
from model import FairRNAFlow  # noqa: E402
from train import FlowDataset, collate, compute_loss  # noqa: E402

from reward_cache import ViennaRewardCache  # noqa: E402
from rl_objectives import (  # noqa: E402
    OBJECTIVE_NAMES,
    anchored_policy_loss,
    grpo_loss,
    preference_dpo_loss,
    reinforce_loss,
    reward_weighted_loss,
)
from rl_primitives import (  # noqa: E402
    categorical_kl,
    official_terminal_reward,
    pair_credit_terminal_reward,
    sequence_mean_log_probability,
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def contract_sha256(contract: dict) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_model(checkpoint: dict, rnaernie: Path, device: torch.device) -> FairRNAFlow:
    contract = checkpoint["contract"]
    model = FairRNAFlow(
        rnaernie,
        structure_source=contract["structure_source"],
        injection=contract["injection"],
        structure_dim=contract.get("structure_dim"),
        lora_rank=contract["lora_rank"],
        lora_alpha=contract["lora_alpha"],
        lora_dropout=contract["lora_dropout"],
        backbone_mode=contract.get("backbone_mode"),
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.eval()
    return model


def policy_batch(rows: list[dict], device: torch.device, seed: int) -> tuple[dict, torch.Tensor]:
    batch = collate([
        {"sequence": row["sequence"], "target_structure": row["target_structure"]}
        for row in rows
    ])
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    clean = batch["clean"]
    clean_onehot = F.one_hot(clean, num_classes=4).float()
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch._sample_dirichlet(torch.ones_like(clean_onehot), generator=generator)
    batch["state"] = 0.65 * clean_onehot + 0.35 * noise
    batch["alpha"] = torch.full((clean.shape[0],), 3.0, device=device)
    return batch, clean


def logits(model: FairRNAFlow, batch: dict) -> torch.Tensor:
    return model(
        batch["state"],
        batch["alpha"],
        batch["mask"],
        structure_hidden=batch["structure_hidden"],
        structure_tokens=batch["structure_tokens"],
    )


def grouped_policy_loss(
    objective: str,
    current: torch.Tensor,
    reference: torch.Tensor,
    rewards: torch.Tensor,
    groups: list[list[int]],
) -> tuple[torch.Tensor, dict]:
    losses = []
    diagnostics = []
    for indices in groups:
        index = torch.tensor(indices, dtype=torch.long, device=current.device)
        current_group = current[index]
        reference_group = reference[index]
        reward_group = rewards[index]
        if objective == "grpo":
            loss, values = grpo_loss(
                current_group, reference_group, reward_group, clip_ratio=0.2
            )
        elif objective == "reinforce":
            loss, values = reinforce_loss(current_group, reward_group)
        elif objective == "reward_weighted":
            loss, values = reward_weighted_loss(
                current_group, reward_group, temperature=0.1
            )
        elif objective == "preference":
            chosen = int(torch.argmax(reward_group).item())
            rejected = int(torch.argmin(reward_group).item())
            loss, values = preference_dpo_loss(
                current_group[chosen : chosen + 1],
                current_group[rejected : rejected + 1],
                reference_group[chosen : chosen + 1],
                reference_group[rejected : rejected + 1],
                beta=0.1,
            )
        else:
            raise ValueError(f"unknown objective: {objective}")
        losses.append(loss)
        diagnostics.append(values)
    return torch.stack(losses).mean(), {"groups": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--supervised-train", type=Path, required=True)
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-limit", type=int, default=4)
    parser.add_argument("--supervised-batch-size", type=int, default=32)
    parser.add_argument("--reward-mode", choices=["terminal", "pair_credit"], default="terminal")
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVE_NAMES, default=list(OBJECTIVE_NAMES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.task_limit <= 0 or args.supervised_batch_size <= 0:
        raise ValueError("task and supervised batch sizes must be positive")
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError("resume output is absent")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(9176)
    torch.cuda.manual_seed_all(9176)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint.get("contract")
    if (
        not contract
        or checkpoint.get("contract_sha256") != contract_sha256(contract)
        or contract.get("method") != "pre-backbone-lora-dotbracket-x0-flow"
    ):
        raise RuntimeError("supervised checkpoint contract is absent, corrupt, or unsupported")

    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["candidate_index"]))
    reward_function = (
        official_terminal_reward if args.reward_mode == "terminal" else pair_credit_terminal_reward
    )
    ranked_groups = sorted(
        grouped.values(),
        key=lambda values: max(reward_function(row) for row in values)
        - min(reward_function(row) for row in values),
        reverse=True,
    )[: args.task_limit]
    selected = [row for values in ranked_groups for row in values]
    if len(ranked_groups) != args.task_limit or any(len(values) < 2 for values in ranked_groups):
        raise RuntimeError("candidate coverage is insufficient for objective debug")

    with ViennaRewardCache(args.output / "reward_cache.sqlite", evaluate_candidate) as cache:
        verified_rewards = []
        for row in selected:
            evaluation = cache.evaluate(row["sequence"], row["target_structure"])
            reward = reward_function(evaluation)
            if abs(reward - reward_function(row)) > 1e-10:
                raise RuntimeError("cached Vienna reward disagrees with candidate result")
            verified_rewards.append(reward)
        cache_stats = cache.stats()

    supervised_data = FlowDataset(args.supervised_train, "dotbracket", None)
    supervised_rows = [supervised_data[index] for index in range(args.supervised_batch_size)]
    supervised_batch = collate(supervised_rows)
    group_indices = []
    offset = 0
    for values in ranked_groups:
        group_indices.append(list(range(offset, offset + len(values))))
        offset += len(values)
    rewards = torch.tensor(verified_rewards, dtype=torch.float32, device=device)
    policy_inputs, actions = policy_batch(selected, device, 9176)

    objective_receipts = {}
    for objective in args.objectives:
        receipt_path = args.output / f"objective-{objective}.json"
        if receipt_path.is_file():
            if not args.resume:
                raise FileExistsError(f"objective receipt exists: {receipt_path}")
            objective_receipts[objective] = json.loads(receipt_path.read_text())
            continue
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        model = load_model(checkpoint, args.rnaernie, device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        with torch.no_grad():
            reference_logits = logits(model, policy_inputs).detach()
            reference_log_probabilities = sequence_mean_log_probability(
                reference_logits, actions, policy_inputs["mask"]
            ).detach()
        current_logits = logits(model, policy_inputs)
        current_log_probabilities = sequence_mean_log_probability(
            current_logits, actions, policy_inputs["mask"]
        )
        policy_loss, policy_diagnostics = grouped_policy_loss(
            objective,
            current_log_probabilities,
            reference_log_probabilities,
            rewards,
            group_indices,
        )
        torch.manual_seed(424242)
        torch.cuda.manual_seed_all(424242)
        ce_anchor_loss, ce_diagnostics = compute_loss(model, supervised_batch, device)
        total_loss, anchor_diagnostics = anchored_policy_loss(
            policy_loss,
            current_logits,
            reference_logits,
            policy_inputs["mask"],
            ce_anchor_loss,
            kl_coefficient=0.01,
            ce_coefficient=0.1,
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.01, foreach=False)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError(f"{objective} produced invalid gradient norm: {gradient_norm}")
        optimizer.step()
        maximum_update = max(
            float((parameter.detach() - before[name]).abs().max().item())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        with torch.no_grad():
            updated_logits = logits(model, policy_inputs)
            post_kl = (
                categorical_kl(updated_logits, reference_logits) * policy_inputs["mask"]
            ).sum() / policy_inputs["mask"].sum()
        receipt = {
            "status": "debug_complete",
            "formal_rl_started": False,
            "objective": objective,
            "tasks": len(ranked_groups),
            "candidates": len(selected),
            "reward_min": float(rewards.min().item()),
            "reward_max": float(rewards.max().item()),
            "reward_range": float((rewards.max() - rewards.min()).item()),
            "reward_mode": args.reward_mode,
            "policy_loss": float(policy_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "gradient_norm_before_clip": gradient_norm,
            "maximum_parameter_update": maximum_update,
            "post_update_reference_kl": float(post_kl.item()),
            "policy_diagnostics": policy_diagnostics,
            "anchor_diagnostics": anchor_diagnostics,
            "ce_diagnostics": ce_diagnostics,
            "runtime_seconds": time.time() - started,
            "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "nan_inf_count": 0,
        }
        atomic_json(receipt_path, receipt)
        objective_receipts[objective] = receipt
        del optimizer, model
        torch.cuda.empty_cache()

    final = {
        "status": "complete",
        "formal_rl_started": False,
        "role": "TRAIN-only phase-2 objective engineering; not a method claim",
        "reward_mode": args.reward_mode,
        "requested_objectives": args.objectives,
        "objectives": objective_receipts,
        "reward_cache": cache_stats,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_contract_sha256": checkpoint["contract_sha256"],
        "candidates": str(args.candidates),
        "candidates_sha256": sha256(args.candidates),
        "supervised_train": str(args.supervised_train),
        "supervised_train_sha256": sha256(args.supervised_train),
    }
    atomic_json(args.output / "receipt.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
