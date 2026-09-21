"""Progressive x0 Flow training with thermodynamic selection and exact resume."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

FAIR_DIR = Path(__file__).resolve().parents[1] / "rna-flow-fair-components"
sys.path.insert(0, str(FAIR_DIR))

from evaluate import aggregate, evaluate_many, sample  # noqa: E402
from model import FairRNAFlow  # noqa: E402
from train import (  # noqa: E402
    FlowDataset,
    atomic_checkpoint,
    autocast_context,
    capture_rng_state,
    collate,
    compute_loss,
    contract_sha256,
    file_sha256,
    restore_rng_state,
    seed_everything,
)

LEGACY_DIR = Path(__file__).resolve().parents[1] / "dual-prior-rna-flow"
sys.path.insert(0, str(LEGACY_DIR))
from constraints import valid_pair_fraction  # noqa: E402
from arrow_data import ArrowFlowDataset  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def distributed_context(enabled: bool) -> tuple[int, int, int]:
    if not enabled:
        return 0, 1, 0
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    if any(name not in os.environ for name in required):
        raise RuntimeError("distributed training must be launched with torchrun")
    dist.init_process_group(backend="nccl")
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])


def rank_rng_states(world_size: int) -> list[dict]:
    local_state = capture_rng_state()
    if world_size == 1:
        return [local_state]
    states: list[dict | None] = [None] * world_size
    dist.all_gather_object(states, local_state)
    return [state for state in states if state is not None]


def broadcast_primary(value: object, rank: int, world_size: int) -> object:
    if world_size == 1:
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def validate_shadow_fork(source: dict, contract: dict) -> None:
    source_contract = source.get("contract")
    if not isinstance(source_contract, dict):
        raise RuntimeError("shadow-fork checkpoint has no training contract")
    immutable = (
        "method", "parameterization", "structure_source", "injection", "seed",
        "learning_rate", "weight_decay", "gradient_clip_norm", "precision",
        "lora_rank", "backbone_mode", "lora_alpha", "lora_dropout",
        "rnaernie_revision", "rnaernie_weight_sha256", "train_format",
        "train_sha256", "train_rows", "thermodynamic_validation", "nested_data",
    )
    mismatches = [key for key in immutable if source_contract.get(key) != contract.get(key)]
    if source_contract.get("effective_batch_size") != contract.get("effective_batch_size"):
        mismatches.append("effective_batch_size")
    source_world_size = int(source_contract.get("distributed", {}).get("world_size", 1))
    target_world_size = int(contract.get("distributed", {}).get("world_size", 1))
    if (
        source_contract.get("gradient_accumulation")
        != contract.get("gradient_accumulation")
    ):
        mismatches.append("gradient_accumulation")
    if (
        source_contract.get("micro_batch_size") * source_world_size
        != contract.get("micro_batch_size") * target_world_size
    ):
        mismatches.append("synchronized_global_micro_batch_size")
    if (
        source_world_size != target_world_size
        and int(contract.get("train_rows", 0)) % int(contract["effective_batch_size"]) != 0
    ):
        mismatches.append("distributed_epoch_tail")
    if mismatches:
        raise RuntimeError(f"shadow-fork checkpoint contract mismatch: {sorted(set(mismatches))}")


def resumed_batch_sampler(
    sampler, batch_size: int, drop_last: bool, start_batch: int
) -> tuple[BatchSampler, object]:
    full = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
    if not 0 <= start_batch <= len(full):
        raise RuntimeError("resume batch cursor is outside the frozen epoch")
    return full, itertools.islice(full, start_batch, None)


def load_thermo_tasks(path: Path) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not tasks:
        raise RuntimeError("thermodynamic validation set is empty")
    structures = [row["target_structure"] for row in tasks]
    if len(set(structures)) != len(structures):
        raise RuntimeError("thermodynamic validation structures are not unique")
    return tasks


def validation_subset_is_authorized(
    requested_tasks: list[dict], exclusions: dict, requested_path: Path
) -> bool:
    requested_sha = file_sha256(requested_path)
    if exclusions.get("validation_sha256") == requested_sha:
        return True
    authorized = exclusions.get("allowed_validation_subsets", [])
    ordered_ids = [str(row["id"]) for row in requested_tasks]
    membership_sha = hashlib.sha256(
        json.dumps(ordered_ids, separators=(",", ":")).encode()
    ).hexdigest()
    return any(
        row.get("sha256") == requested_sha
        and row.get("rows") == len(requested_tasks)
        and row.get("ordered_id_membership_sha256") == membership_sha
        for row in authorized
        if isinstance(row, dict)
    )


def thermo_validate(
    model: FairRNAFlow,
    tasks: list[dict],
    candidates: int,
    flow_steps: int,
    seed: int,
    workers: int,
    device: torch.device,
) -> tuple[dict, tuple[float, ...]]:
    model.eval()
    rows: list[dict] = []
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    if deterministic_enabled:
        torch.use_deterministic_algorithms(False)
    try:
        for task_index, task in enumerate(tasks):
            structure = task["target_structure"]
            task_seed = seed + task_index * 1_000_003
            sequences = sample(
                model,
                structure,
                cached_hidden=None,
                candidates=candidates,
                steps=flow_steps,
                seed=task_seed,
                condition="native",
                device=device,
            )
            metrics = evaluate_many(sequences, structure, workers)
            for candidate_index, (sequence, values) in enumerate(zip(sequences, metrics)):
                pair_validity = valid_pair_fraction(sequence, structure)
                valid = (
                    len(sequence) == len(structure)
                    and not (set(sequence) - set("ACGU"))
                    and pair_validity == 1
                )
                rows.append({
                    "task_id": str(task.get("id", task_index)),
                    "seed": seed,
                    "candidate_index": candidate_index,
                    "sequence": sequence,
                    "valid": float(valid),
                    "valid_pair_fraction": pair_validity,
                    **values,
                })
    finally:
        if deterministic_enabled:
            torch.use_deterministic_algorithms(True)
    _, summary = aggregate(rows, candidates)
    summary["tasks"] = len(tasks)
    summary["candidates_per_task"] = candidates
    summary["flow_steps"] = flow_steps
    summary["seed"] = seed
    score = (
        summary[f"pass_at_{candidates}"],
        summary["best_pair_f1"],
        summary["best_target_probability"],
        -summary["best_NED"],
        summary[f"mfe_at_{candidates}"],
        summary["valid_rate"],
    )
    model.train()
    return summary, score


def checkpoint_compatibility(model: FairRNAFlow, payload: dict) -> None:
    expected = set(model.trainable_state_dict())
    actual = set(payload.get("trainable_model", {}))
    if expected != actual:
        raise RuntimeError(
            f"initialization checkpoint mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def expand_lora_initialization(
    model: FairRNAFlow,
    payload: dict,
    *,
    target_rank: int,
    target_alpha: float,
) -> tuple[int, dict[str, torch.Tensor]]:
    source_contract = payload.get("contract", {})
    source_rank = int(source_contract.get("lora_rank", 0))
    source_alpha = float(source_contract.get("lora_alpha", 0.0))
    if source_rank <= 0 or target_rank <= source_rank or source_alpha <= 0 or target_alpha <= 0:
        raise ValueError("LoRA expansion requires a larger positive target rank and valid alphas")
    source = payload.get("trainable_model", {})
    target = model.trainable_state_dict()
    if set(source) != set(target):
        raise RuntimeError("LoRA expansion source and target parameter names differ")
    expanded = {}
    for name, target_value in target.items():
        source_value = source[name]
        if name.endswith("lora_a.weight"):
            if source_value.shape[0] != source_rank or target_value.shape[0] != target_rank:
                raise RuntimeError(f"unexpected LoRA A shape during expansion: {name}")
            value = torch.zeros_like(target_value)
            value[:source_rank].copy_(source_value.to(value))
        elif name.endswith("lora_b.weight"):
            if source_value.shape[1] != source_rank or target_value.shape[1] != target_rank:
                raise RuntimeError(f"unexpected LoRA B shape during expansion: {name}")
            value = torch.zeros_like(target_value)
            scale_correction = (source_alpha / source_rank) / (target_alpha / target_rank)
            value[:, :source_rank].copy_(source_value.to(value) * scale_correction)
        else:
            if source_value.shape != target_value.shape:
                raise RuntimeError(f"non-LoRA parameter changed shape during expansion: {name}")
            value = source_value.to(dtype=target_value.dtype).clone()
        expanded[name] = value
    return source_rank, expanded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-format", choices=["jsonl", "arrow"], default="jsonl")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--thermo-validation", type=Path, required=True)
    parser.add_argument("--heldout-benchmark", type=Path)
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--rnaernie-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--backbone-mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--thermo-candidates", type=int, default=8)
    parser.add_argument("--thermo-flow-steps", type=int, default=50)
    parser.add_argument("--thermo-seed", type=int, default=9176)
    parser.add_argument("--thermo-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--shadow-fork-from", type=Path)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--initialize-rank-expand-from", type=Path)
    parser.add_argument("--preflight-steps", type=int)
    parser.add_argument("--preflight-epochs", type=int)
    args = parser.parse_args()
    positive = (
        args.batch_size,
        args.gradient_accumulation,
        args.save_every,
        args.minimum_epochs,
        args.patience,
        args.thermo_candidates,
        args.thermo_flow_steps,
        args.thermo_workers,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("batch, checkpoint, epoch, patience, and validation values must be positive")
    initialization_modes = [
        args.resume, args.shadow_fork_from, args.initialize_from,
        args.initialize_rank_expand_from,
    ]
    if sum(value is not None for value in initialization_modes) > 1:
        raise ValueError("resume, initialize-from, and rank expansion are mutually exclusive")
    if args.preflight_steps is not None and args.preflight_steps <= 0:
        raise ValueError("preflight steps must be positive")
    if args.preflight_epochs is not None and args.preflight_epochs <= 0:
        raise ValueError("preflight epochs must be positive")
    if args.preflight_steps is not None and args.preflight_epochs is not None:
        raise ValueError("preflight steps and epochs are mutually exclusive")
    if args.train_format == "arrow" and (args.train_limit is None or args.train_manifest is None):
        raise ValueError("Arrow training requires a stage limit and nested-data manifest")
    if args.train_format == "jsonl" and (args.train_limit is not None or args.train_manifest is not None):
        raise ValueError("JSONL training does not accept Arrow stage arguments")
    if args.train_format == "jsonl" and args.heldout_benchmark is None:
        raise ValueError("JSONL training requires a held-out benchmark exclusion set")
    if args.backbone_mode == "lora" and args.lora_rank <= 0:
        raise ValueError("LoRA training requires a positive rank")
    if args.backbone_mode == "full" and args.lora_rank != 0:
        raise ValueError("full RNAErnie tuning requires lora-rank 0")
    return args


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = distributed_context(args.distributed)
    if world_size > 1 and args.gradient_accumulation != 1:
        raise ValueError("distributed training currently requires gradient accumulation 1")
    is_primary = rank == 0
    if is_primary:
        args.output.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_everything(args.seed + rank * 100_003)
    device = torch.device(f"cuda:{local_rank}" if args.distributed else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_data = (
        FlowDataset(args.train, "dotbracket", None)
        if args.train_format == "jsonl"
        else ArrowFlowDataset(args.train, int(args.train_limit))
    )
    thermo_tasks = load_thermo_tasks(args.thermo_validation)
    thermo_structures = {row["target_structure"] for row in thermo_tasks}
    nested_manifest = None
    if args.train_format == "jsonl":
        train_structures = {row["target_structure"] for row in train_data.rows}
        overlap = train_structures & thermo_structures
        if overlap:
            raise RuntimeError(f"train/thermodynamic-validation structure overlap: {len(overlap)}")
        heldout_structures = {
            row["target_structure"]
            for row in load_thermo_tasks(args.heldout_benchmark)
        }
        heldout_overlap = train_structures & heldout_structures
        if heldout_overlap:
            raise RuntimeError(f"train/held-out benchmark structure overlap: {len(heldout_overlap)}")
    else:
        nested_manifest = json.loads(args.train_manifest.read_text())
        prefix = nested_manifest.get("prefixes", {}).get(str(args.train_limit))
        exclusions = nested_manifest.get("exclusions", {})
        manifest_file = nested_manifest.get("file", {})
        verification = None
        if is_primary:
            try:
                verification = {
                    "train_sha256": file_sha256(args.train),
                    "thermo_validation_sha256": file_sha256(args.thermo_validation),
                    "train_manifest_sha256": file_sha256(args.train_manifest),
                    "rnaernie_weight_sha256": file_sha256(
                        args.rnaernie / "model.safetensors"
                    ),
                    "validation_subset_authorized": validation_subset_is_authorized(
                        thermo_tasks, exclusions, args.thermo_validation
                    ),
                    "error": None,
                }
            except Exception as error:  # propagated identically to every rank
                verification = {"error": f"{type(error).__name__}: {error}"}
        verification = broadcast_primary(verification, rank, world_size)
        if verification.get("error") is not None:
            raise RuntimeError(f"distributed file identity verification failed: {verification['error']}")
        if (
            nested_manifest.get("status") != "complete"
            or nested_manifest.get("bad_count") != 0
            or prefix is None
            or prefix.get("rows") != args.train_limit
            or exclusions.get("exact_structure_overlap_validation") != 0
            or exclusions.get("exact_structure_overlap_eterna100v2") != 0
            or manifest_file.get("rows") != train_data.source_rows
            or manifest_file.get("sha256") != verification["train_sha256"]
            or not verification["validation_subset_authorized"]
        ):
            raise RuntimeError("nested Arrow manifest does not authorize this stage prefix")

    sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True,
        seed=args.seed, drop_last=True,
    ) if world_size > 1 else None
    loader_options = {
        "collate_fn": collate,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    model = FairRNAFlow(
        args.rnaernie,
        structure_source="dotbracket",
        injection="pre",
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        backbone_mode=args.backbone_mode,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    continuation_path = args.resume or args.shadow_fork_from
    continuation_sha256 = broadcast_primary(
        file_sha256(continuation_path) if is_primary and continuation_path else None,
        rank,
        world_size,
    )
    resume_payload = (
        torch.load(continuation_path, map_location=device, weights_only=False)
        if continuation_path else None
    )
    repository_root = Path(__file__).resolve().parents[2]
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty and os.environ.get("RNA_FLOW_ALLOW_DIRTY_PREFLIGHT") != "1":
        raise RuntimeError("formal progressive training requires a clean source tree")
    repository_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if (
        args.resume is not None
        and resume_payload["contract"]["code"]["repository_revision"]
        != repository_revision
    ):
        raise RuntimeError("resume checkpoint source revision mismatch")
    if args.resume is not None:
        initialization_contract = resume_payload["contract"].get("initialization")
    elif args.shadow_fork_from is not None:
        initialization_contract = {
            "mode": "provisional-distributed-shadow-fork",
            "checkpoint_sha256": continuation_sha256,
            "checkpoint_contract_sha256": resume_payload.get("contract_sha256"),
        }
    elif args.initialize_from is not None:
        initialization_contract = {
            "mode": "same-rank-continuation",
            "checkpoint_sha256": file_sha256(args.initialize_from),
        }
    elif args.initialize_rank_expand_from is not None:
        initialization_contract = {
            "mode": "function-preserving-rank-expansion",
            "checkpoint_sha256": file_sha256(args.initialize_rank_expand_from),
        }
    else:
        initialization_contract = None

    contract = {
        "schema_version": 4,
        "method": f"pre-backbone-{args.backbone_mode}-dotbracket-x0-flow",
        "parameterization": "x0",
        "structure_source": "dotbracket",
        "injection": "pre",
        "seed": args.seed,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation * world_size,
        "distributed": {
            "enabled": args.distributed,
            "world_size": world_size,
            "backend": "nccl" if args.distributed else None,
            "sampler": "DistributedSampler(drop_last=True)" if args.distributed else None,
            "loss_normalization": "global-valid-token-count",
            "find_unused_parameters": True if args.distributed else None,
            "dataloader_epoch_generator": True,
        },
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": 1.0,
        "precision": args.precision,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "allow_tf32": False,
        "optimizer_foreach": False,
        "initialization": initialization_contract,
        "code": {
            "repository_revision": repository_revision,
            "train_script_sha256": file_sha256(Path(__file__)),
            "model_script_sha256": file_sha256(FAIR_DIR / "model.py"),
            "evaluator_script_sha256": file_sha256(FAIR_DIR / "evaluate.py"),
            "training_primitives_sha256": file_sha256(FAIR_DIR / "train.py"),
            "arrow_data_sha256": file_sha256(Path(__file__).with_name("arrow_data.py")),
            "constraints_sha256": file_sha256(LEGACY_DIR / "constraints.py"),
            "flow_core_sha256": file_sha256(LEGACY_DIR / "flow_core.py"),
        },
        "lora_rank": args.lora_rank,
        "backbone_mode": args.backbone_mode,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "minimum_epochs": args.minimum_epochs,
        "early_stopping_patience": args.patience,
        "selection_order": [
            f"pass_at_{args.thermo_candidates}",
            "best_pair_f1",
            "best_target_probability",
            "negative_best_NED",
            f"mfe_at_{args.thermo_candidates}",
            "valid_rate",
        ],
        "thermodynamic_validation": {
            "sha256": file_sha256(args.thermo_validation),
            "tasks": len(thermo_tasks),
            "candidates": args.thermo_candidates,
            "flow_steps": args.thermo_flow_steps,
            "seed": args.thermo_seed,
            "viennarna": "2.7.2",
        },
        "rnaernie_revision": args.rnaernie_revision,
        "rnaernie_weight_sha256": (
            verification["rnaernie_weight_sha256"]
            if nested_manifest is not None
            else file_sha256(args.rnaernie / "model.safetensors")
        ),
        "train_format": args.train_format,
        "train_sha256": (
            verification["train_sha256"]
            if nested_manifest is not None else file_sha256(args.train)
        ),
        "train_rows": len(train_data),
        "train_validation_structure_overlap": 0,
        "heldout_benchmark": (
            {
                "sha256": file_sha256(args.heldout_benchmark),
                "train_structure_overlap": 0,
            }
            if args.heldout_benchmark is not None else {
                "sha256": nested_manifest["exclusions"]["eterna100v2_sha256"],
                "train_structure_overlap": 0,
            }
        ),
        "nested_data": (
            {
                "manifest_sha256": verification["train_manifest_sha256"],
                "prefix_membership_sha256": nested_manifest["prefixes"][str(args.train_limit)][
                    "membership_sha256"
                ],
            }
            if nested_manifest is not None
            else None
        ),
    }
    digest = contract_sha256(contract)

    state = {
        "step": 0,
        "next_epoch": 0,
        "next_batch_index": 0,
        "completed_epochs": 0,
        "best_score": None,
        "best_metrics": None,
        "best_epoch": None,
        "best_checkpoint": None,
        "epochs_without_improvement": 0,
        "current_epoch_loss_sum": 0.0,
        "current_epoch_loss_count": 0,
    }
    resumed_from = None
    shadow_forked_from = None
    initialized_from = None
    if args.resume:
        payload = resume_payload
        if payload.get("contract") != contract or payload.get("contract_sha256") != digest:
            raise RuntimeError("resume checkpoint contract mismatch")
        model.load_trainable_state_dict(payload["trainable_model"])
        optimizer.load_state_dict(payload["optimizer"])
        state.update(payload["training_state"])
        saved_rng = payload.get("rng_state_by_rank")
        if not isinstance(saved_rng, list) or len(saved_rng) != world_size:
            raise RuntimeError("resume checkpoint lacks matching per-rank RNG state")
        restore_rng_state(saved_rng[rank])
        resumed_from = {"path": str(args.resume), "sha256": continuation_sha256}
    elif args.shadow_fork_from:
        payload = resume_payload
        validate_shadow_fork(payload, contract)
        checkpoint_compatibility(model, payload)
        model.load_trainable_state_dict(payload["trainable_model"])
        optimizer.load_state_dict(payload["optimizer"])
        state.update(payload["training_state"])
        seed_everything(args.seed + rank * 100_003 + int(state["step"]))
        shadow_forked_from = {
            "path": str(args.shadow_fork_from),
            "sha256": continuation_sha256,
            "source_contract_sha256": payload.get("contract_sha256"),
        }
    elif args.initialize_from:
        payload = torch.load(args.initialize_from, map_location=device, weights_only=False)
        checkpoint_compatibility(model, payload)
        model.load_trainable_state_dict(payload["trainable_model"])
        initialized_from = {"path": str(args.initialize_from), "sha256": file_sha256(args.initialize_from)}
    elif args.initialize_rank_expand_from:
        payload = torch.load(args.initialize_rank_expand_from, map_location="cpu", weights_only=False)
        source_rank, expanded = expand_lora_initialization(
            model,
            payload,
            target_rank=args.lora_rank,
            target_alpha=args.lora_alpha,
        )
        model.load_trainable_state_dict(expanded)
        initialized_from = {
            "path": str(args.initialize_rank_expand_from),
            "sha256": file_sha256(args.initialize_rank_expand_from),
            "mode": "function-preserving-rank-expansion",
            "source_rank": source_rank,
            "target_rank": args.lora_rank,
        }

    training_model = (
        DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if world_size > 1 else model
    )

    started = time.time()
    optimization_started = None
    optimization_ended = None
    max_memory = 0
    updates_this_run = 0
    stopped_for_preflight = False
    stopped_for_patience = False
    optimizer.zero_grad(set_to_none=True)

    def checkpoint_payload(rng_states: list[dict]) -> dict:
        return {
            "trainable_model": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_state": dict(state),
            "rng_state_by_rank": rng_states,
            "contract": contract,
            "contract_sha256": digest,
        }

    def save_checkpoint(name: str, kind: str) -> Path:
        path = args.output / name
        rng_states = rank_rng_states(world_size)
        if is_primary and path.exists():
            raise FileExistsError(f"immutable checkpoint already exists: {path}")
        if is_primary:
            atomic_checkpoint(path, checkpoint_payload(rng_states))
            checkpoint_hash = file_sha256(path)
            append_jsonl(args.output / "checkpoint_index.jsonl", {
                "kind": kind,
                "path": str(path),
                "sha256": checkpoint_hash,
                "step": state["step"],
                "completed_epochs": state["completed_epochs"],
                "created_unix": time.time(),
            })
            atomic_json(args.output / "last.json", {
                "path": str(path),
                "sha256": checkpoint_hash,
                "step": state["step"],
                "completed_epochs": state["completed_epochs"],
            })
        if world_size > 1:
            dist.barrier()
        return path

    epoch = int(state["next_epoch"])
    while True:
        generator = torch.Generator().manual_seed(args.seed + epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
            epoch_sampler = sampler
        else:
            epoch_sampler = RandomSampler(train_data, generator=generator)
        skip_batches = int(state["next_batch_index"]) if epoch == state["next_epoch"] else 0
        full_batch_sampler, remaining_batch_sampler = resumed_batch_sampler(
            epoch_sampler, args.batch_size, world_size > 1, skip_batches
        )
        train_loader = DataLoader(
            train_data,
            batch_sampler=remaining_batch_sampler,
            generator=generator,
            **loader_options,
        )
        accumulated = 0
        accumulated_loss = 0.0
        diagnostics: dict = {}
        remaining_batches = len(full_batch_sampler) - skip_batches
        for relative_batch_index, batch in enumerate(train_loader):
            if optimization_started is None:
                optimization_started = time.time()
            batch_index = skip_batches + relative_batch_index
            state["next_epoch"] = epoch
            state["next_batch_index"] = batch_index + 1
            with autocast_context(device, args.precision):
                loss, diagnostics = compute_loss(training_model, batch, device)
                local_tokens = batch["mask"].sum().to(device)
                global_tokens = local_tokens.clone()
                if world_size > 1:
                    dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
                scaled_loss = (
                    loss * (world_size * local_tokens / global_tokens)
                    / args.gradient_accumulation
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss before optimizer step {state['step'] + 1}")
            scaled_loss.backward()
            accumulated += 1
            weighted_loss = loss.detach() * local_tokens
            if world_size > 1:
                dist.all_reduce(weighted_loss, op=dist.ReduceOp.SUM)
            accumulated_loss += float((weighted_loss / global_tokens).cpu())
            final_micro_batch = relative_batch_index + 1 == remaining_batches
            if accumulated < args.gradient_accumulation and not final_micro_batch:
                continue
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            if not math.isfinite(gradient_norm) or gradient_norm == 0:
                raise RuntimeError(
                    f"invalid gradient norm at optimizer step {state['step'] + 1}: {gradient_norm}"
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            updates_this_run += 1
            mean_loss = accumulated_loss / accumulated
            state["current_epoch_loss_sum"] += mean_loss
            state["current_epoch_loss_count"] += 1
            record = {
                "step": state["step"],
                "epoch": epoch,
                "next_batch_index": state["next_batch_index"],
                "loss": mean_loss,
                "gradient_norm": gradient_norm,
                **diagnostics,
            }
            if state["step"] == 1 or state["step"] % 10 == 0:
                if is_primary:
                    print(json.dumps(record, sort_keys=True), flush=True)
            if is_primary and state["step"] % 100 == 0:
                append_jsonl(args.output / "train_history.jsonl", record)
            accumulated = 0
            accumulated_loss = 0.0
            if device.type == "cuda":
                max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))
            if state["step"] % args.save_every == 0:
                save_checkpoint(f"checkpoint-step-{state['step']:09d}.pt", "periodic")
            if args.preflight_steps is not None and updates_this_run >= args.preflight_steps:
                stopped_for_preflight = True
                break
        if stopped_for_preflight:
            break

        state["completed_epochs"] = epoch + 1
        state["next_epoch"] = epoch + 1
        state["next_batch_index"] = 0
        validation_started = time.time()
        if is_primary:
            metrics, score = thermo_validate(
                model, thermo_tasks, args.thermo_candidates,
                args.thermo_flow_steps, args.thermo_seed,
                args.thermo_workers, device,
            )
        else:
            metrics, score = None, None
        if world_size > 1:
            shared = [metrics, score]
            dist.broadcast_object_list(shared, src=0)
            metrics, score = shared
        improved = state["best_score"] is None or tuple(score) > tuple(state["best_score"])
        if improved:
            state["best_score"] = list(score)
            state["best_metrics"] = metrics
            state["best_epoch"] = epoch + 1
            state["epochs_without_improvement"] = 0
        else:
            state["epochs_without_improvement"] += 1
        if state["current_epoch_loss_count"] <= 0:
            raise RuntimeError("completed epoch has no optimizer-step loss records")
        epoch_record = {
            "epoch": epoch + 1,
            "step": state["step"],
            "mean_train_loss": (
                state["current_epoch_loss_sum"] / state["current_epoch_loss_count"]
            ),
            "thermodynamic_validation": metrics,
            "selection_score": list(score),
            "improved": improved,
            "epochs_without_improvement": state["epochs_without_improvement"],
            "validation_runtime_seconds": time.time() - validation_started,
        }
        if is_primary:
            append_jsonl(args.output / "epoch_history.jsonl", epoch_record)
        state["current_epoch_loss_sum"] = 0.0
        state["current_epoch_loss_count"] = 0
        epoch_path = save_checkpoint(f"checkpoint-epoch-{epoch + 1:04d}.pt", "epoch")
        if improved:
            intended_best_path = args.output / f"checkpoint-best-epoch-{epoch + 1:04d}.pt"
            state["best_checkpoint"] = {
                "path": str(intended_best_path),
                "sha256": None,
                "source_epoch_checkpoint": str(epoch_path),
            }
            best_path = save_checkpoint(f"checkpoint-best-epoch-{epoch + 1:04d}.pt", "best")
            best_hash = file_sha256(best_path) if is_primary else None
            if world_size > 1:
                values = [best_hash]
                dist.broadcast_object_list(values, src=0)
                best_hash = values[0]
            state["best_checkpoint"]["sha256"] = best_hash
            if is_primary:
                atomic_json(args.output / "best.json", state["best_checkpoint"] | {
                    "epoch": epoch + 1,
                    "metrics": metrics,
                    "selection_score": list(score),
                })
        if is_primary:
            print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if (
            args.preflight_epochs is not None
            and state["completed_epochs"] >= args.preflight_epochs
        ):
            stopped_for_preflight = True
            break
        if (
            state["completed_epochs"] >= args.minimum_epochs
            and state["epochs_without_improvement"] >= args.patience
        ):
            stopped_for_patience = True
            break
        epoch += 1

    optimization_ended = time.time()
    final_kind = "preflight" if stopped_for_preflight else "stage-boundary"
    final_path = save_checkpoint(
        f"checkpoint-{final_kind}-step-{state['step']:09d}.pt",
        final_kind,
    )
    status = "preflight_complete" if stopped_for_preflight else "complete"
    optimization_seconds = (
        optimization_ended - optimization_started
        if optimization_started is not None else 0.0
    )
    receipt = {
        "status": status,
        "method": contract["method"],
        "seed": args.seed,
        "contract": contract,
        "contract_sha256": digest,
        "training_state": state,
        "resumed_from": resumed_from,
        "shadow_forked_from": shadow_forked_from,
        "initialized_from": initialized_from,
        "updates_this_run": updates_this_run,
        "stopped_for_patience": stopped_for_patience,
        "stopped_for_preflight": stopped_for_preflight,
        "runtime_seconds": time.time() - started,
        "optimization_runtime_seconds": optimization_seconds,
        "samples_per_second": (
            updates_this_run * contract["effective_batch_size"]
            / max(optimization_seconds, 1e-9)
        ),
        "max_cuda_memory_bytes": max_memory,
        "nan_inf_count": 0,
        "trainable_parameters": model.trainable_parameters(),
        "backbone_trainable_parameters": model.backbone_trainable_parameters(),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "stage_boundary_checkpoint": {
            "path": str(final_path),
            "sha256": file_sha256(final_path),
        },
    }
    if world_size > 1:
        memory = torch.tensor([max_memory], dtype=torch.long, device=device)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        receipt["max_cuda_memory_bytes"] = int(memory.item())
    if is_primary:
        atomic_json(args.output / "train_receipt.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
