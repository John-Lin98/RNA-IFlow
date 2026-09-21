"""Pluggable phase-2 policy objectives and supervision anchors."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rl_primitives import categorical_kl, clipped_grpo_loss, normalized_group_advantages


def reinforce_loss(log_probabilities: torch.Tensor, rewards: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if log_probabilities.shape != rewards.shape or log_probabilities.ndim != 1:
        raise ValueError("REINFORCE log probabilities and rewards must be aligned vectors")
    advantages = (rewards - rewards.mean()).detach()
    loss = -(advantages * log_probabilities).mean()
    return loss, {"effective": bool(float(advantages.abs().max().item()) > 0)}


def reward_weighted_loss(
    log_probabilities: torch.Tensor,
    rewards: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict]:
    if (
        log_probabilities.shape != rewards.shape
        or log_probabilities.ndim != 1
        or temperature <= 0
        or not math.isfinite(temperature)
    ):
        raise ValueError("reward-weighted objective arguments are invalid")
    weights = torch.softmax(rewards.detach() / temperature, dim=0)
    return -(weights * log_probabilities).sum(), {
        "maximum_weight": float(weights.max().item()),
        "effective_sample_size": float(1 / weights.square().sum().item()),
    }


def preference_dpo_loss(
    chosen_log_probabilities: torch.Tensor,
    rejected_log_probabilities: torch.Tensor,
    reference_chosen_log_probabilities: torch.Tensor,
    reference_rejected_log_probabilities: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    shape = chosen_log_probabilities.shape
    if (
        chosen_log_probabilities.ndim != 1
        or any(value.shape != shape for value in (
            rejected_log_probabilities,
            reference_chosen_log_probabilities,
            reference_rejected_log_probabilities,
        ))
        or beta <= 0
        or not math.isfinite(beta)
    ):
        raise ValueError("preference objective arguments are invalid")
    policy_margin = chosen_log_probabilities - rejected_log_probabilities
    reference_margin = reference_chosen_log_probabilities - reference_rejected_log_probabilities
    logits = beta * (policy_margin - reference_margin)
    return -F.logsigmoid(logits).mean(), {
        "preference_accuracy": float((logits.detach() > 0).float().mean().item()),
        "mean_logit": float(logits.detach().mean().item()),
    }


def grpo_loss(
    log_probabilities: torch.Tensor,
    behaviour_log_probabilities: torch.Tensor,
    rewards: torch.Tensor,
    *,
    clip_ratio: float,
) -> tuple[torch.Tensor, dict]:
    advantages, effective = normalized_group_advantages(rewards)
    loss, diagnostics = clipped_grpo_loss(
        log_probabilities,
        behaviour_log_probabilities,
        advantages,
        clip_ratio=clip_ratio,
    )
    diagnostics["effective"] = effective
    return loss, diagnostics


def anchored_policy_loss(
    policy_loss: torch.Tensor,
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    mask: torch.Tensor,
    ce_anchor_loss: torch.Tensor,
    *,
    kl_coefficient: float,
    ce_coefficient: float,
) -> tuple[torch.Tensor, dict]:
    if (
        mask.shape != current_logits.shape[:2]
        or ce_anchor_loss.ndim != 0
        or kl_coefficient < 0
        or ce_coefficient < 0
        or not math.isfinite(kl_coefficient)
        or not math.isfinite(ce_coefficient)
    ):
        raise ValueError("policy anchor arguments are invalid")
    kl_values = categorical_kl(current_logits, reference_logits)
    kl = (kl_values * mask).sum() / mask.sum()
    total = policy_loss + kl_coefficient * kl + ce_coefficient * ce_anchor_loss
    return total, {
        "policy_loss": float(policy_loss.detach().item()),
        "reference_kl": float(kl.detach().item()),
        "ce_anchor_loss": float(ce_anchor_loss.detach().item()),
        "kl_coefficient": kl_coefficient,
        "ce_coefficient": ce_coefficient,
    }


OBJECTIVE_NAMES = ("preference", "reward_weighted", "grpo", "reinforce")
