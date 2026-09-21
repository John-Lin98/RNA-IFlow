"""Exact-resume online terminal-reward GRPO for a frozen supervised Flow checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from reward_cache import ViennaRewardCache
from rl_objectives import anchored_policy_loss, grpo_loss
from rl_primitives import pair_credit_terminal_reward, sequence_mean_log_probability
from run_rl_objective_debug import (
    atomic_json,
    contract_sha256,
    load_model,
    logits,
    policy_batch,
)

import sys

FAIR_DIR = Path(__file__).resolve().parents[1] / "rna-flow-fair-components"
sys.path.insert(0, str(FAIR_DIR))
from evaluate import evaluate_candidate, sample, sha256  # noqa: E402
from train import FlowDataset, collate, compute_loss  # noqa: E402
from finalize_supervised_selection import validate_authorized_checkpoint  # noqa: E402


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def task_batch_indices(
    task_count: int,
    epoch: int,
    cursor: int,
    batch_size: int,
    seed: int,
) -> tuple[list[int], int, int]:
    if task_count <= 0 or batch_size <= 0 or batch_size > task_count:
        raise ValueError("invalid task batching contract")
    selected = []
    while len(selected) < batch_size:
        generator = torch.Generator().manual_seed(seed + epoch * 1_000_003)
        order = torch.randperm(task_count, generator=generator).tolist()
        take = min(batch_size - len(selected), task_count - cursor)
        selected.extend(order[cursor : cursor + take])
        cursor += take
        if cursor == task_count:
            epoch += 1
            cursor = 0
    return selected, epoch, cursor


def reconcile_history(path: Path, payload: dict | None) -> None:
    if payload is None or payload.get("last_record") is None:
        return
    record = payload["last_record"]
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    if rows and int(rows[-1]["update"]) > int(record["update"]):
        raise RuntimeError("RL history is ahead of the resume checkpoint")
    if not rows or int(rows[-1]["update"]) < int(record["update"]):
        append_jsonl(path, record)
    elif rows[-1] != record:
        raise RuntimeError("RL history/checkpoint record mismatch")


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--supervised-train", type=Path, required=True)
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--rnaernie-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9176)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--tasks-per-update", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=50)
    parser.add_argument("--supervised-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--kl-coefficient", type=float, default=0.01)
    parser.add_argument("--ce-coefficient", type=float, default=0.1)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-updates", type=int)
    parser.add_argument("--formal-rl", action="store_true")
    parser.add_argument("--supervised-selection-receipt", type=Path)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.tasks_per_update,
        args.candidates,
        args.flow_steps,
        args.supervised_batch_size,
        args.learning_rate,
    )
    if any(value <= 0 for value in positive) or args.weight_decay < 0:
        raise ValueError("RL sizes and learning rate must be positive")
    if args.preflight_updates is not None and args.preflight_updates <= 0:
        raise ValueError("preflight updates must be positive")
    if args.formal_rl == (args.preflight_updates is not None):
        raise ValueError("choose either formal RL or an explicitly bounded preflight")

    supervised = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
    supervised_contract = supervised.get("contract")
    if (
        not supervised_contract
        or supervised.get("contract_sha256") != contract_sha256(supervised_contract)
        or supervised_contract.get("method") != "pre-backbone-lora-dotbracket-x0-flow"
        or supervised_contract.get("rnaernie_revision") != args.rnaernie_revision
    ):
        raise RuntimeError("unsupported or corrupt supervised initialization checkpoint")
    if args.formal_rl:
        if args.supervised_selection_receipt is None:
            raise ValueError("formal RL requires the final supervised selection receipt")
        validate_authorized_checkpoint(
            args.supervised_selection_receipt, args.initialize_from
        )

    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    supervised_data = FlowDataset(args.supervised_train, "dotbracket", None)
    if args.tasks_per_update > len(tasks) or args.supervised_batch_size > len(supervised_data):
        raise ValueError("RL or supervised batch exceeds its dataset")
    task_ids = [str(task["id"]) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("RL task identifiers are not unique")

    repository_root = Path(__file__).resolve().parents[2]
    repository_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    contract = {
        "schema_version": 1,
        "method": "online-x0-proxy-grpo-pair-credit",
        "formal_rl": args.formal_rl,
        "supervised_checkpoint_sha256": sha256(args.initialize_from),
        "supervised_contract_sha256": supervised["contract_sha256"],
        "supervised_selection_receipt_sha256": (
            sha256(args.supervised_selection_receipt) if args.supervised_selection_receipt else None
        ),
        "tasks_sha256": sha256(args.tasks),
        "supervised_train_sha256": sha256(args.supervised_train),
        "rnaernie_revision": args.rnaernie_revision,
        "seed": args.seed,
        "updates": args.updates,
        "tasks_per_update": args.tasks_per_update,
        "candidates": args.candidates,
        "flow_steps": args.flow_steps,
        "supervised_batch_size": args.supervised_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "clip_ratio": 0.2,
        "gradient_clip_norm": 1.0,
        "kl_coefficient": args.kl_coefficient,
        "ce_coefficient": args.ce_coefficient,
        "reward": "0.8*(0.5*target_probability+0.25*MFE+0.25*uMFE)+0.2*Pair-F1",
        "precision": args.precision,
        "code_revision": repository_revision,
        "script_sha256": sha256(Path(__file__)),
    }
    encoded_contract = contract_sha256(contract)
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError("resume output directory is absent")
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_payload.get("contract_sha256") != encoded_contract:
            raise RuntimeError("RL resume contract mismatch")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        resume_payload = None
        atomic_json(args.output / "contract.json", contract | {"contract_sha256": encoded_contract})
    rollout_dir = args.output / "rollouts"
    checkpoint_dir = args.output / "checkpoints"
    rollout_dir.mkdir(exist_ok=bool(args.resume))
    checkpoint_dir.mkdir(exist_ok=bool(args.resume))

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    current = load_model(supervised, args.rnaernie, device)
    reference = load_model(supervised, args.rnaernie, device)
    reference.requires_grad_(False)
    reference.eval()
    if resume_payload:
        current.load_trainable_state_dict(resume_payload["trainable_model"])
    trainable = [parameter for parameter in current.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay, foreach=False
    )
    if resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])
        state = dict(resume_payload["state"])
    else:
        state = {"next_update": 0, "task_epoch": 0, "task_cursor": 0}
    history_path = args.output / "train_history.jsonl"
    reconcile_history(history_path, resume_payload)
    target_updates = min(
        args.updates,
        state["next_update"] + args.preflight_updates
        if args.preflight_updates is not None
        else args.updates,
    )
    started = time.time()
    maximum_memory = 0

    with ViennaRewardCache(args.output / "reward_cache.sqlite", evaluate_candidate) as cache:
        while state["next_update"] < target_updates:
            update = int(state["next_update"])
            indices, next_epoch, next_cursor = task_batch_indices(
                len(tasks),
                int(state["task_epoch"]),
                int(state["task_cursor"]),
                args.tasks_per_update,
                args.seed,
            )
            selected_tasks = [tasks[index] for index in indices]
            rollout_path = rollout_dir / f"rollout-{update:06d}.jsonl"
            if rollout_path.is_file():
                rollout_rows = [
                    json.loads(line) for line in rollout_path.read_text().splitlines() if line.strip()
                ]
            else:
                rollout_rows = []
                current.eval()
                for task_offset, task in enumerate(selected_tasks):
                    structure = task["target_structure"]
                    sequences = sample(
                        current,
                        structure,
                        None,
                        args.candidates,
                        args.flow_steps,
                        args.seed + update * 1_000_003 + task_offset * 10_007,
                        "native",
                        device,
                    )
                    for candidate_index, sequence in enumerate(sequences):
                        evaluation = cache.evaluate(sequence, structure)
                        rollout_rows.append({
                            "update": update,
                            "task_id": str(task["id"]),
                            "candidate_index": candidate_index,
                            "target_structure": structure,
                            "sequence": sequence,
                            "reward": pair_credit_terminal_reward(evaluation),
                            **evaluation,
                        })
                atomic_jsonl(rollout_path, rollout_rows)
            expected_rows = args.tasks_per_update * args.candidates
            if len(rollout_rows) != expected_rows:
                raise RuntimeError("RL rollout coverage mismatch")
            grouped = []
            offset = 0
            for task in selected_tasks:
                values = [row for row in rollout_rows if row["task_id"] == str(task["id"])]
                values.sort(key=lambda row: int(row["candidate_index"]))
                if len(values) != args.candidates:
                    raise RuntimeError("RL per-task candidate coverage mismatch")
                grouped.append(values)
            ordered_rows = [row for values in grouped for row in values]
            group_indices = [
                list(range(index * args.candidates, (index + 1) * args.candidates))
                for index in range(args.tasks_per_update)
            ]
            rewards = torch.tensor(
                [row["reward"] for row in ordered_rows], dtype=torch.float32, device=device
            )
            batch, actions = policy_batch(ordered_rows, device, args.seed + update * 97)
            current.eval()
            with torch.no_grad(), autocast_context(device, args.precision):
                behaviour_logits = logits(current, batch).detach()
                reference_logits = logits(reference, batch).detach()
                behaviour_log_probabilities = sequence_mean_log_probability(
                    behaviour_logits, actions, batch["mask"]
                ).detach()
            with autocast_context(device, args.precision):
                current_logits = logits(current, batch)
                current_log_probabilities = sequence_mean_log_probability(
                    current_logits, actions, batch["mask"]
                )
                group_losses = []
                group_diagnostics = []
                for indices_in_group in group_indices:
                    index = torch.tensor(indices_in_group, device=device)
                    loss, diagnostics = grpo_loss(
                        current_log_probabilities[index],
                        behaviour_log_probabilities[index],
                        rewards[index],
                        clip_ratio=0.2,
                    )
                    group_losses.append(loss)
                    group_diagnostics.append(diagnostics)
                policy_loss = torch.stack(group_losses).mean()
                supervised_rows = [
                    supervised_data[(update * args.supervised_batch_size + index) % len(supervised_data)]
                    for index in range(args.supervised_batch_size)
                ]
                torch.manual_seed(args.seed + update * 193)
                torch.cuda.manual_seed_all(args.seed + update * 193)
                ce_loss, ce_diagnostics = compute_loss(current, collate(supervised_rows), device)
                total_loss, anchor_diagnostics = anchored_policy_loss(
                    policy_loss,
                    current_logits,
                    reference_logits,
                    batch["mask"],
                    ce_loss,
                    kl_coefficient=args.kl_coefficient,
                    ce_coefficient=args.ce_coefficient,
                )
            if not torch.isfinite(total_loss):
                raise RuntimeError("non-finite RL loss")
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                raise RuntimeError("invalid RL gradient norm")
            optimizer.step()
            state.update({
                "next_update": update + 1,
                "task_epoch": next_epoch,
                "task_cursor": next_cursor,
            })
            maximum_memory = max(maximum_memory, torch.cuda.max_memory_allocated(device))
            record = {
                "update": update,
                "policy_loss": float(policy_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "gradient_norm_before_clip": gradient_norm,
                "reward_mean": float(rewards.mean().item()),
                "reward_min": float(rewards.min().item()),
                "reward_max": float(rewards.max().item()),
                "effective_groups": sum(bool(item["effective"]) for item in group_diagnostics),
                "reference_kl_before_update": anchor_diagnostics["reference_kl"],
                "ce_anchor_loss": anchor_diagnostics["ce_anchor_loss"],
                "ce_accuracy": ce_diagnostics["accuracy"],
                "max_cuda_memory_bytes": maximum_memory,
                "rollout_sha256": sha256(rollout_path),
            }
            checkpoint_payload = {
                "contract": contract,
                "contract_sha256": encoded_contract,
                "state": state,
                "trainable_model": current.trainable_state_dict(),
                "optimizer": optimizer.state_dict(),
                "last_record": record,
            }
            checkpoint_path = checkpoint_dir / f"checkpoint-update-{update + 1:06d}.pt"
            if checkpoint_path.exists():
                raise FileExistsError(f"immutable RL checkpoint exists: {checkpoint_path}")
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint_payload, temporary)
            os.replace(temporary, checkpoint_path)
            append_jsonl(history_path, record)
            atomic_json(args.output / "last.json", {
                "path": str(checkpoint_path),
                "sha256": sha256(checkpoint_path),
                "next_update": state["next_update"],
            })
            print(json.dumps(record, sort_keys=True), flush=True)
        cache_stats = cache.stats()

    receipt = {
        "status": "preflight_complete" if not args.formal_rl else "complete",
        "formal_rl_started": args.formal_rl,
        "updates_complete": state["next_update"],
        "target_updates": target_updates,
        "contract_sha256": encoded_contract,
        "last_checkpoint": json.loads((args.output / "last.json").read_text()),
        "reward_cache": cache_stats,
        "runtime_seconds": time.time() - started,
        "max_cuda_memory_bytes": maximum_memory,
        "nan_inf_count": 0,
    }
    atomic_json(args.output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
