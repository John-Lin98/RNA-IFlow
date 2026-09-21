"""Resumable fair x0 training for structure-injection and backbone variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LEGACY_DIR = Path(__file__).resolve().parents[1] / "dual-prior-rna-flow"
sys.path.append(str(LEGACY_DIR))
from flow_core import sample_dirichlet_path  # noqa: E402
from model import FairRNAFlow, NUCLEOTIDES, STRUCTURE_SYMBOLS  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_sha256(contract: dict) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def structure_key(structure: str) -> str:
    return hashlib.sha256(structure.encode()).hexdigest()


class FlowDataset(Dataset):
    def __init__(self, path: Path, structure_source: str, structure_cache: Path | None) -> None:
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.structure_source = structure_source
        self.structure_cache = structure_cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        result = {"target_structure": row["target_structure"], "sequence": row["sequence"]}
        if self.structure_source.startswith("omnigenome"):
            cached = torch.load(
                self.structure_cache / f"{structure_key(row['target_structure'])}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if cached["target_structure"] != row["target_structure"]:
                raise RuntimeError("structure cache key/content mismatch")
            result["structure_hidden"] = cached["hidden"].float()
        return result


def collate(rows: list[dict]) -> dict:
    max_length = max(len(row["sequence"]) for row in rows)
    batch_size = len(rows)
    clean = torch.zeros(batch_size, max_length, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.float32)
    structure_tokens = torch.zeros(batch_size, max_length, dtype=torch.long)
    structure_hidden = None
    if "structure_hidden" in rows[0]:
        structure_hidden = torch.zeros(batch_size, max_length, rows[0]["structure_hidden"].shape[-1])
    for index, row in enumerate(rows):
        length = len(row["sequence"])
        clean[index, :length] = torch.tensor([NUCLEOTIDES.index(base) for base in row["sequence"]])
        mask[index, :length] = 1
        structure_tokens[index, :length] = torch.tensor([
            STRUCTURE_SYMBOLS.index(symbol) for symbol in row["target_structure"]
        ])
        if structure_hidden is not None:
            structure_hidden[index, :length] = row["structure_hidden"]
    return {
        "clean": clean,
        "mask": mask,
        "structure_tokens": structure_tokens,
        "structure_hidden": structure_hidden,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_loss(model: FairRNAFlow, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    batch = move_batch(batch, device)
    clean = batch["clean"]
    mask = batch["mask"]
    clean_onehot = F.one_hot(clean, num_classes=4).float()
    alpha_scalar = 1 + torch.rand((), device=device) * 7
    state, alpha = sample_dirichlet_path(clean_onehot, alpha=alpha_scalar.expand(clean.shape[0]))
    prediction = model(
        state,
        alpha,
        mask,
        structure_hidden=batch["structure_hidden"],
        structure_tokens=batch["structure_tokens"],
    )
    token_loss = F.cross_entropy(prediction.transpose(1, 2), clean, reduction="none")
    loss = (token_loss * mask).sum() / mask.sum()
    accuracy = ((prediction.argmax(-1) == clean) * mask.bool()).sum() / mask.sum()
    return loss, {"accuracy": float(accuracy.detach().cpu()), "alpha": float(alpha_scalar.detach().cpu())}


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_loss(
    model: FairRNAFlow,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    batches: int | None,
) -> float:
    model.eval()
    values = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(424242)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(424242)
        for index, batch in enumerate(loader):
            if batches is not None and index >= batches:
                break
            with autocast_context(device, precision):
                loss, _ = compute_loss(model, batch, device)
            values.append(float(loss.cpu()))
    model.train()
    if not values:
        raise RuntimeError("validation loader produced no batches")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--structure-source", choices=["none", "dotbracket", "omnigenome52", "omnigenome186"], required=True)
    parser.add_argument("--injection", choices=["none", "post", "pre"], required=True)
    parser.add_argument("--structure-cache", type=Path)
    parser.add_argument("--structure-cache-manifest", type=Path)
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--rnaernie-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.gradient_accumulation <= 0 or args.save_every <= 0:
        raise ValueError("batch, accumulation, and checkpoint intervals must be positive")
    cached_source = args.structure_source.startswith("omnigenome")
    if cached_source != bool(args.structure_cache and args.structure_cache_manifest):
        raise ValueError("OmniGenome sources require exactly one cache and manifest pair")
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_data = FlowDataset(args.train, args.structure_source, args.structure_cache)
    validation_data = FlowDataset(args.validation, args.structure_source, args.structure_cache)
    loader_options = {
        "batch_size": args.batch_size,
        "collate_fn": collate,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    structure_dim = None
    if cached_source:
        structure_dim = train_data[0]["structure_hidden"].shape[-1]
    model = FairRNAFlow(
        args.rnaernie,
        structure_source=args.structure_source,
        injection=args.injection,
        structure_dim=structure_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    rnaernie_weights = args.rnaernie / "model.safetensors"
    contract = {
        "schema_version": 2,
        "parameterization": "x0",
        "structure_source": args.structure_source,
        "injection": args.injection,
        "structure_dim": structure_dim,
        "seed": args.seed,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": 1.0,
        "precision": args.precision,
        "validation_batches": args.validation_batches,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "rnaernie_revision": args.rnaernie_revision,
        "rnaernie_weight_sha256": file_sha256(rnaernie_weights),
        "structure_cache_manifest_sha256": (
            file_sha256(args.structure_cache_manifest) if cached_source else None
        ),
        "data_sha256": {
            "train": file_sha256(args.train),
            "validation": file_sha256(args.validation),
        },
    }
    digest = contract_sha256(contract)
    start_step = start_epoch = start_batch_index = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload.get("contract") != contract or payload.get("contract_sha256") != digest:
            raise RuntimeError("resume checkpoint contract mismatch")
        model.load_trainable_state_dict(payload["trainable_model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        start_epoch = int(payload["epoch"])
        start_batch_index = int(payload["next_batch_index"])
        restore_rng_state(payload["rng_state"])

    resumed_epoch = start_epoch
    resumed_batch_index = start_batch_index
    initial_validation = evaluate_loss(
        model, validation_loader, device, args.precision, args.validation_batches
    )
    started = time.time()
    step = start_step
    last_epoch = start_epoch
    next_batch_index = start_batch_index
    history = []
    max_memory = 0
    optimizer.zero_grad(set_to_none=True)

    def save_checkpoint(path: Path) -> None:
        atomic_checkpoint(path, {
            "trainable_model": model.trainable_state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": last_epoch,
            "next_batch_index": next_batch_index,
            "rng_state": capture_rng_state(),
            "contract": contract,
            "contract_sha256": digest,
        })

    stop = args.max_steps is not None and step >= args.max_steps
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
        skip_batches = start_batch_index if epoch == start_epoch else 0
        accumulated = 0
        diagnostics = {}
        accumulated_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            if batch_index < skip_batches:
                continue
            last_epoch = epoch
            next_batch_index = batch_index + 1
            with autocast_context(device, args.precision):
                loss, diagnostics = compute_loss(model, batch, device)
                scaled_loss = loss / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss before optimizer step {step + 1}")
            scaled_loss.backward()
            accumulated += 1
            accumulated_loss += float(loss.detach().cpu())
            final_micro_batch = batch_index + 1 == len(train_loader)
            if accumulated < args.gradient_accumulation and not final_micro_batch:
                continue
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            if not math.isfinite(gradient_norm) or gradient_norm == 0:
                raise RuntimeError(f"invalid gradient norm at optimizer step {step + 1}: {gradient_norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = {
                "step": step,
                "epoch": epoch,
                "next_batch_index": next_batch_index,
                "micro_batches": accumulated,
                "loss": accumulated_loss / accumulated,
                "gradient_norm": gradient_norm,
                **diagnostics,
            }
            history.append(record)
            accumulated = 0
            accumulated_loss = 0.0
            if device.type == "cuda":
                max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))
            if step % 10 == 0 or step == 1:
                print(json.dumps(record, sort_keys=True), flush=True)
            if step % args.save_every == 0:
                save_checkpoint(args.output / f"checkpoint-{step:06d}.pt")
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        start_batch_index = 0

    final_validation = evaluate_loss(
        model, validation_loader, device, args.precision, args.validation_batches
    )
    final_checkpoint = args.output / f"checkpoint-{step:06d}.pt"
    save_checkpoint(final_checkpoint)
    receipt = {
        "status": "complete",
        "method": "fair-rna-flow-x0",
        "structure_source": args.structure_source,
        "injection": args.injection,
        "seed": args.seed,
        "steps": step,
        "initial_validation_loss": initial_validation,
        "final_validation_loss": final_validation,
        "loss_decreased": final_validation < initial_validation if history else None,
        "last_train_loss": history[-1]["loss"] if history else None,
        "last_gradient_norm": history[-1]["gradient_norm"] if history else None,
        "resumed_from_step": start_step,
        "updates_this_run": step - start_step,
        "resume_epoch": resumed_epoch,
        "resume_batch_index": resumed_batch_index,
        "contract": contract,
        "contract_sha256": digest,
        "nan_inf_count": 0,
        "trainable_parameters": model.trainable_parameters(),
        "backbone_trainable_parameters": model.backbone_trainable_parameters(),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "structure_gate": (
            float(torch.sigmoid(model.gate_logit).detach().cpu()) if model.gate_logit is not None else None
        ),
        "runtime_seconds": time.time() - started,
        "max_cuda_memory_bytes": max_memory,
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": None,
    }
    receipt["checkpoint_sha256"] = file_sha256(final_checkpoint)
    (args.output / "history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history)
    )
    (args.output / "train_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
