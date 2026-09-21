"""Pure GRPO primitives for terminal-reward adaptation of x0 RNA Flow."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def official_terminal_reward(evaluation: Mapping[str, object]) -> float:
    probability = float(evaluation["target_probability"])
    mfe_hit = evaluation["mfe_hit"]
    umfe_hit = evaluation["uMFE_hit"]
    if (
        not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
        or not isinstance(mfe_hit, bool)
        or not isinstance(umfe_hit, bool)
    ):
        raise ValueError("terminal reward fields are invalid")
    return 0.5 * probability + 0.25 * float(mfe_hit) + 0.25 * float(umfe_hit)


def pair_credit_terminal_reward(evaluation: Mapping[str, object]) -> float:
    """Preserve terminal success while adding dense structural credit."""
    terminal = official_terminal_reward(evaluation)
    pair_f1 = float(evaluation["pair_f1"])
    if not math.isfinite(pair_f1) or not 0.0 <= pair_f1 <= 1.0:
        raise ValueError("Pair-F1 reward field is invalid")
    return 0.8 * terminal + 0.2 * pair_f1


def pair_rival_terminal_reward(evaluation: Mapping[str, object]) -> float:
    """Add bounded target-versus-MFE energy-gap credit to Pair-Credit."""
    reward = pair_credit_terminal_reward(evaluation)
    rival_credit = float(evaluation["rival_margin_credit"])
    if not math.isfinite(rival_credit) or not 0.0 <= rival_credit <= 1.0:
        raise ValueError("rival-margin reward field is invalid")
    return reward + 0.1 * rival_credit


def normalized_group_advantages(
    rewards: torch.Tensor,
    *,
    epsilon: float = 1e-4,
    zero_range: float = 1e-6,
) -> tuple[torch.Tensor, bool]:
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO requires at least two rewards per target")
    if epsilon <= 0 or zero_range < 0 or not torch.isfinite(rewards).all():
        raise ValueError("invalid GRPO reward normalization")
    if float((rewards.max() - rewards.min()).item()) < zero_range:
        return torch.zeros_like(rewards), False
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)
    return advantages, True


def stepwise_pair_advantages(
    final_rewards: torch.Tensor,
    provisional_pair_f1: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Normalize final reward plus per-transition Pair-F1 improvement by step."""
    if (
        final_rewards.ndim != 1
        or final_rewards.numel() < 2
        or provisional_pair_f1.ndim != 2
        or provisional_pair_f1.shape[0] != final_rewards.shape[0]
        or provisional_pair_f1.shape[1] < 2
        or not torch.isfinite(provisional_pair_f1).all()
        or (provisional_pair_f1 < 0).any()
        or (provisional_pair_f1 > 1).any()
    ):
        raise ValueError("stepwise Pair-F1 advantage tensors are invalid")
    improvements = provisional_pair_f1[:, 1:] - provisional_pair_f1[:, :-1]
    raw = final_rewards[:, None] + improvements
    columns = []
    effective_steps = 0
    for step in range(raw.shape[1]):
        values, effective = normalized_group_advantages(raw[:, step])
        columns.append(values)
        effective_steps += int(effective)
    return torch.stack(columns, dim=1), effective_steps


def clipped_grpo_loss(
    current_log_probs: torch.Tensor,
    behaviour_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_ratio: float,
) -> tuple[torch.Tensor, dict]:
    if (
        current_log_probs.ndim != 1
        or behaviour_log_probs.shape != current_log_probs.shape
        or advantages.shape != current_log_probs.shape
        or not 0 < clip_ratio < 1
    ):
        raise ValueError("GRPO clipped-loss tensor contract is invalid")
    if not torch.isfinite(current_log_probs).all() or not torch.isfinite(behaviour_log_probs).all():
        raise ValueError("GRPO log probabilities must be finite")
    ratios = torch.exp(current_log_probs - behaviour_log_probs)
    clipped = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio)
    surrogate = torch.minimum(ratios * advantages, clipped * advantages)
    loss = -surrogate.mean()
    return loss, {
        "mean_ratio": float(ratios.detach().mean().item()),
        "clip_fraction": float(((ratios < 1 - clip_ratio) | (ratios > 1 + clip_ratio)).float().mean().item()),
    }


def categorical_kl(current_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if current_logits.shape != reference_logits.shape or current_logits.ndim != 3:
        raise ValueError("Flow KL logits must be aligned [batch,length,alphabet] tensors")
    current_log_probs = torch.log_softmax(current_logits.float(), dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits.float(), dim=-1)
    current_probs = current_log_probs.exp()
    values = (current_probs * (current_log_probs - reference_log_probs)).sum(dim=-1)
    if not torch.isfinite(values).all():
        raise ValueError("Flow KL is non-finite")
    return values


def sequence_mean_log_probability(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if (
        logits.ndim != 3
        or actions.shape != logits.shape[:2]
        or mask.shape != actions.shape
        or logits.shape[-1] != 4
    ):
        raise ValueError("Flow sequence log-probability tensors are not aligned")
    token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    lengths = mask.sum(dim=-1)
    if (lengths <= 0).any():
        raise ValueError("Flow sequence mask contains an empty sequence")
    return (token_log_probs * mask).sum(dim=-1) / lengths
