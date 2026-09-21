"""Shared training helpers; historical filename retained for source manifests."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

FAIR_DIR = Path(__file__).resolve().parents[1] / "rna-flow-fair-components"
sys.path.insert(0, str(FAIR_DIR))
from model import FairRNAFlow  # noqa: E402
from train import collate  # noqa: E402


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
