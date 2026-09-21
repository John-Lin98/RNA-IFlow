"""Fair x0 RNA flow variants with explicit structure injection and LoRA."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


NUCLEOTIDES = "ACGU"
STRUCTURE_SYMBOLS = ".()"
LORA_TARGET_PROFILES = {"query_value", "all_attention_ffn"}
ALL_ATTENTION_FFN_SUFFIXES = {
    "attention.self.query", "attention.self.key", "attention.self.value",
    "attention.output.dense", "intermediate.dense", "output.dense",
}


def freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def configure_rl_trainable_scope(model: nn.Module, scope: str) -> None:
    """Set the explicit RNAErnie adaptation boundary for post-training."""
    if scope == "inherited":
        return
    if scope not in {
        "adapter_and_head", "last_2_backbone_and_head", "full_backbone_and_head"
    }:
        raise ValueError(f"unsupported RL trainable scope: {scope}")
    for name, parameter in model.rnaernie.named_parameters():
        is_lora = name.endswith("lora_a.weight") or name.endswith("lora_b.weight")
        parameter.requires_grad_(
            (
                not name.startswith("pooler.")
                and not name.endswith("embeddings.word_embeddings.weight")
            ) if scope == "full_backbone_and_head" else is_lora
        )
    if scope == "last_2_backbone_and_head":
        layers = getattr(getattr(model.rnaernie, "encoder", None), "layer", None)
        if layers is None or len(layers) < 2:
            raise RuntimeError("RNAErnie does not expose at least two encoder layers")
        for layer in layers[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    model.rnaernie.eval()


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper that preserves the frozen base linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        freeze(self.base)
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_b(self.lora_a(self.dropout(inputs))) * self.scale
        return self.base(inputs) + update


def inject_lora(
    module: nn.Module, rank: int, alpha: float, dropout: float,
    target_profile: str = "query_value",
) -> list[str]:
    if target_profile not in LORA_TARGET_PROFILES:
        raise ValueError(f"unsupported LoRA target profile: {target_profile}")
    replaced = []
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            selected = (
                child_name in {"query", "value"}
                if target_profile == "query_value"
                else any(full_name.endswith(suffix) for suffix in ALL_ATTENTION_FFN_SUFFIXES)
            )
            if selected and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced.append(full_name)
    if not replaced:
        raise RuntimeError(f"no RNAErnie modules matched LoRA profile {target_profile}")
    return replaced


def adapt_loaded_supervised_model_with_lora(
    model: "FairRNAFlow", rank: int, alpha: float, dropout: float,
    target_profile: str = "all_attention_ffn",
) -> list[str]:
    """Freeze a loaded full-supervised backbone, then add fresh RL-only LoRA."""
    if model.lora_rank or any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("the loaded supervised model already contains LoRA modules")
    if model.backbone_mode != "full":
        raise RuntimeError("LoRA adaptation requires a loaded full-backbone supervised model")
    freeze(model.rnaernie)
    replaced = inject_lora(
        model.rnaernie, rank=rank, alpha=alpha, dropout=dropout,
        target_profile=target_profile,
    )
    model.lora_rank = rank
    model.lora_target_profile = target_profile
    model.lora_modules = replaced
    model.backbone_mode = "lora"
    return replaced


class FairRNAFlow(nn.Module):
    def __init__(
        self,
        rnaernie_path: str | Path,
        structure_source: str,
        injection: str,
        structure_dim: int | None = None,
        hidden_dim: int = 256,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_profile: str = "query_value",
        backbone_mode: str | None = None,
        backbone_config: dict | None = None,
    ) -> None:
        super().__init__()
        if structure_source not in {"none", "dotbracket", "omnigenome52", "omnigenome186"}:
            raise ValueError(f"unsupported structure source: {structure_source}")
        if injection not in {"none", "post", "pre"}:
            raise ValueError(f"unsupported injection: {injection}")
        if (structure_source == "none") != (injection == "none"):
            raise ValueError("none structure source and none injection must be selected together")
        if structure_source.startswith("omnigenome") and not structure_dim:
            raise ValueError("cached OmniGenome sources require structure_dim")
        if lora_rank and injection != "pre":
            raise ValueError("the fair LoRA variant is defined only for pre-backbone injection")
        if backbone_mode is None:
            backbone_mode = "lora" if lora_rank else "frozen"
        if backbone_mode not in {"frozen", "lora", "full"}:
            raise ValueError(f"unsupported RNAErnie backbone mode: {backbone_mode}")
        if (backbone_mode == "lora") != bool(lora_rank):
            raise ValueError("LoRA backbone mode requires a positive rank and other modes require rank zero")
        if backbone_mode == "full" and injection != "pre":
            raise ValueError("full RNAErnie tuning is defined only for pre-backbone injection")

        self.structure_source = structure_source
        self.injection = injection
        self.lora_rank = lora_rank
        self.lora_target_profile = lora_target_profile
        self.backbone_mode = backbone_mode
        self.rnaernie = (
            AutoModel.from_pretrained(str(rnaernie_path), local_files_only=True)
            if backbone_config is None else AutoModel.from_config(
                AutoConfig.for_model(**backbone_config)
            )
        )
        freeze(self.rnaernie)
        self.lora_modules = inject_lora(
            self.rnaernie, lora_rank, lora_alpha, lora_dropout,
            target_profile=lora_target_profile,
        ) if lora_rank else []
        if backbone_mode == "full":
            for name, parameter in self.rnaernie.named_parameters():
                parameter.requires_grad_(not name.startswith("pooler."))
        config = self.rnaernie.config

        # Full exports restore these buffers with strict=True; no original vocab or weights needed.
        vocab = (Path(rnaernie_path, "vocab.txt").read_text().splitlines()
                 if backbone_config is None else ["A", "C", "G", "U", "[CLS]", "[SEP]"])
        token_to_id = {token: index for index, token in enumerate(vocab)}
        uracil_token = "U" if "U" in token_to_id else "T"
        nucleotide_ids = [token_to_id[token] for token in ("A", "C", "G", uracil_token)]
        special_ids = [token_to_id["[CLS]"], token_to_id["[SEP]"]]
        embeddings = self.rnaernie.get_input_embeddings().weight.detach()
        self.register_buffer("nucleotide_embeddings", embeddings[nucleotide_ids].clone())
        self.register_buffer("special_embeddings", embeddings[special_ids].clone())

        if structure_source == "dotbracket":
            structure_dim = 64
            self.structure_embedding = nn.Embedding(len(STRUCTURE_SYMBOLS), structure_dim)
        else:
            self.structure_embedding = None
        self.sequence_adapter = nn.Sequential(
            nn.Linear(config.hidden_size, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.soft_input_adapter = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        if injection == "pre":
            self.structure_projector = nn.Sequential(
                nn.Linear(int(structure_dim), config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
            )
        elif injection == "post":
            self.structure_projector = nn.Sequential(
                nn.Linear(int(structure_dim), hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
        else:
            self.structure_projector = None
        self.time_adapter = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate_logit = nn.Parameter(torch.tensor(-4.0)) if injection != "none" else None
        self.output_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 4))

    def train(self, mode: bool = True) -> "FairRNAFlow":
        super().train(mode)
        if self.backbone_mode == "frozen":
            self.rnaernie.eval()
        return self

    def structure_features(
        self,
        structure_hidden: torch.Tensor | None,
        structure_tokens: torch.Tensor | None,
        structure_override: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.structure_source == "none":
            return None
        if structure_override is not None:
            return structure_override
        if self.structure_source == "dotbracket":
            if structure_tokens is None:
                raise ValueError("dot-bracket tokens are required")
            return self.structure_embedding(structure_tokens)
        if structure_hidden is None:
            raise ValueError("cached structure hidden states are required")
        return structure_hidden

    def forward(
        self,
        state: torch.Tensor,
        alpha: torch.Tensor,
        attention_mask: torch.Tensor,
        structure_hidden: torch.Tensor | None = None,
        structure_tokens: torch.Tensor | None = None,
        structure_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, alphabet = state.shape
        if alphabet != 4:
            raise ValueError("state must have four nucleotide channels")
        features = self.structure_features(structure_hidden, structure_tokens, structure_override)
        soft_embeddings = state @ self.nucleotide_embeddings.to(state.dtype)
        if self.injection == "pre":
            soft_embeddings = soft_embeddings + torch.sigmoid(self.gate_logit) * self.structure_projector(features)
        cls = self.special_embeddings[0].to(state.dtype).expand(batch, 1, -1)
        sep = self.special_embeddings[1].to(state.dtype).expand(batch, 1, -1)
        inputs_embeds = torch.cat([cls, soft_embeddings, sep], dim=1)
        backbone_mask = torch.cat([
            torch.ones(batch, 1, device=state.device, dtype=attention_mask.dtype),
            attention_mask,
            torch.ones(batch, 1, device=state.device, dtype=attention_mask.dtype),
        ], dim=1)
        grad_enabled = bool(self.lora_rank or self.injection == "pre")
        with torch.set_grad_enabled(torch.is_grad_enabled() and grad_enabled):
            sequence_hidden = self.rnaernie(
                inputs_embeds=inputs_embeds,
                attention_mask=backbone_mask,
                return_dict=True,
            ).last_hidden_state[:, 1:length + 1]
        fused = self.sequence_adapter(sequence_hidden) + self.soft_input_adapter(state)
        time = torch.stack([
            (alpha - 1) / 7,
            torch.sin(math.pi * (alpha - 1) / 7),
            torch.cos(math.pi * (alpha - 1) / 7),
        ], dim=-1)
        fused = fused + self.time_adapter(time)[:, None, :]
        if self.injection == "post":
            fused = fused + torch.sigmoid(self.gate_logit) * self.structure_projector(features)
        return self.output_head(fused) * attention_mask.unsqueeze(-1)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in names}

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if set(state) != expected:
            raise RuntimeError(
                f"trainable checkpoint mismatch: missing={sorted(expected - set(state))}, "
                f"unexpected={sorted(set(state) - expected)}"
            )
        current = self.state_dict()
        for name, value in state.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))

    def backbone_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.rnaernie.parameters() if parameter.requires_grad)

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
