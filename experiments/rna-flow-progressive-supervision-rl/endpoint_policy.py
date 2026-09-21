"""Categorical clean-endpoint policy for stochastic RNA simplex trajectories."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

FAIR_DIR = Path(__file__).resolve().parents[1] / "rna-flow-fair-components"
LEGACY_DIR = Path(__file__).resolve().parents[1] / "dual-prior-rna-flow"
sys.path.insert(0, str(LEGACY_DIR))
sys.path.insert(0, str(FAIR_DIR))
from constraints import constrained_decode, target_pairs  # noqa: E402
from evaluate import TorchDirichletConditionalFlow  # noqa: E402
from model import FairRNAFlow, NUCLEOTIDES, STRUCTURE_SYMBOLS  # noqa: E402

LEGAL_PAIRS = ("AU", "UA", "GC", "CG", "GU", "UG")
LEGAL_PAIR_IDS = torch.tensor(
    [[NUCLEOTIDES.index(left), NUCLEOTIDES.index(right)] for left, right in LEGAL_PAIRS],
    dtype=torch.long,
)


def temporal_credit_steps(
    mode: str,
    *,
    seed: int,
    update: int,
    global_task_offset: int,
    trajectory_steps: int,
    causal_ticket_counts: Sequence[int] | None = None,
    causal_policy_sha256: str | None = None,
) -> tuple[int, ...]:
    """Return the globally keyed flow steps that receive policy credit."""
    return temporal_credit_selection(
        mode,
        seed=seed,
        update=update,
        global_task_offset=global_task_offset,
        trajectory_steps=trajectory_steps,
        causal_ticket_counts=causal_ticket_counts,
        causal_policy_sha256=causal_policy_sha256,
    )[0]


def _validate_causal_ticket_policy(
    trajectory_steps: int,
    causal_ticket_counts: Sequence[int] | None,
    causal_policy_sha256: str | None,
) -> tuple[int, ...]:
    """Validate and normalize the frozen causal ticket policy.

    Both ``causal_is_single`` and ``causal_window_2`` sample the same start
    ticket.  Keeping the validation and keyed draw in one helper is important:
    changing the key would make the window-1 compatibility path differ from
    the existing causal sampler.
    """
    tickets = tuple(causal_ticket_counts or ())
    if (
        len(tickets) != trajectory_steps
        or any(type(value) is not int or value <= 0 for value in tickets)
        or not isinstance(causal_policy_sha256, str)
        or len(causal_policy_sha256) != 64
        or any(character not in "0123456789abcdef" for character in causal_policy_sha256)
    ):
        raise ValueError("causal temporal credit requires positive tickets and a policy SHA-256")
    return tickets


def causal_window_step_probabilities(
    causal_ticket_counts: Sequence[int], *, window_size: int = 1,
) -> tuple[float, ...]:
    """Return ``p_t = P(t is in W(s))`` for the frozen causal ticket draw.

    The current contract has eight flow steps.  ``window_size=1`` is the
    compatibility path for ``causal_is_single`` (``p_t=q_t``), while size two
    uses ``W(s)={s,s+1}`` and clamps the final start to ``{6,7}``.
    """
    tickets = tuple(causal_ticket_counts)
    if len(tickets) != 8 or any(type(value) is not int or value <= 0 for value in tickets):
        raise ValueError("causal temporal credit requires eight positive ticket counts")
    if window_size not in (1, 2):
        raise ValueError("causal temporal credit supports window_size 1 or 2")
    total = float(sum(tickets))
    probabilities = [0.0] * 8
    for start, count in enumerate(tickets):
        if window_size == 1:
            window = (start,)
        else:
            window = (start, start + 1) if start < 7 else (6, 7)
        for step in window:
            probabilities[step] += float(count) / total
    if any(not math.isfinite(value) or value <= 0 for value in probabilities):
        raise ValueError("causal temporal-credit step probabilities must be finite and positive")
    return tuple(probabilities)


def causal_window_selection(
    *,
    seed: int,
    update: int,
    global_task_offset: int,
    trajectory_steps: int,
    causal_ticket_counts: Sequence[int],
    causal_policy_sha256: str,
    window_size: int = 1,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Draw one causal start and return its selected steps plus HT weights.

    The start draw deliberately uses the exact legacy
    ``endpoint-causal-is-temporal-credit-v1`` key.  For size two, each selected
    step receives ``1/(8*p_t)`` and the caller must sum the two weighted
    step-local objectives; no scalar window weight is returned.
    """
    if min(seed, update, global_task_offset) < 0 or trajectory_steps != 8:
        raise ValueError("causal temporal-credit identifiers and steps must be valid")
    if window_size not in (1, 2):
        raise ValueError("causal temporal credit supports window_size 1 or 2")
    tickets = _validate_causal_ticket_policy(
        trajectory_steps, causal_ticket_counts, causal_policy_sha256
    )
    total = sum(tickets)
    encoded = (
        "endpoint-causal-is-temporal-credit-v1:"
        f"{causal_policy_sha256}:{seed}:{update}:{global_task_offset}"
    ).encode()
    ticket = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % total
    cumulative = 0
    start = None
    for index, count in enumerate(tickets):
        cumulative += count
        if ticket < cumulative:
            start = index
            break
    if start is None:
        raise AssertionError("causal temporal-credit ticket selection fell outside its population")
    if window_size == 1:
        selected = (start,)
    else:
        selected = (start, start + 1) if start < 7 else (6, 7)
    probabilities = causal_window_step_probabilities(tickets, window_size=window_size)
    if window_size == 1:
        # Keep the legacy scalar arithmetic exactly for the compatibility
        # helper; the vector path below is used only for window size two.
        weights = (total / (trajectory_steps * tickets[start]),)
    else:
        weights = tuple(1.0 / (8.0 * probabilities[step]) for step in selected)
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("causal temporal-credit HT weights must be finite and positive")
    return selected, weights


# Explicit alias used by tests and downstream callers that want to document the
# Horvitz--Thompson estimator rather than the sampler implementation.
causal_temporal_credit = causal_window_selection


def temporal_credit_selection(
    mode: str,
    *,
    seed: int,
    update: int,
    global_task_offset: int,
    trajectory_steps: int,
    causal_ticket_counts: Sequence[int] | None = None,
    causal_policy_sha256: str | None = None,
) -> tuple[tuple[int, ...], float | tuple[float, ...]]:
    """Return selected steps and their uniform-objective importance weight.

    Legacy modes retain their scalar second return value.  ``causal_window_2``
    returns a per-step tuple because the HT estimator is a sum of step-local
    terms; callers must not collapse it to one scalar.
    """
    if min(seed, update, global_task_offset) < 0 or trajectory_steps <= 0:
        raise ValueError("temporal-credit identifiers and trajectory steps must be positive")
    if mode == "all_steps":
        return tuple(range(trajectory_steps)), 1.0
    if trajectory_steps != 8:
        raise ValueError("single-step temporal credit requires an eight-step trajectory")
    if mode in {"causal_is_single", "causal_window_2"}:
        selected, weights = causal_window_selection(
            seed=seed,
            update=update,
            global_task_offset=global_task_offset,
            trajectory_steps=trajectory_steps,
            causal_ticket_counts=causal_ticket_counts or (),
            causal_policy_sha256=causal_policy_sha256 or "",
            window_size=1 if mode == "causal_is_single" else 2,
        )
        # Preserve the old scalar return contract for causal_is_single while
        # exposing the necessary per-step HT vector for causal_window_2.
        return selected, weights[0] if mode == "causal_is_single" else weights
    populations = {
        "uniform_single": 8,
        "informative_early_single": 4,
    }
    population = populations.get(mode)
    if population is None:
        raise ValueError(f"unknown temporal credit mode: {mode}")
    encoded = (
        f"endpoint-temporal-credit-v1:{seed}:{update}:{global_task_offset}"
    ).encode()
    selected = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % population
    return (selected,), 1.0


def temporal_credit_step_weights(
    mode: str,
    *,
    selected_steps: Sequence[int],
    importance_weight: float | Sequence[float],
) -> tuple[float, ...] | None:
    """Normalize temporal-credit metadata into explicit per-step HT weights.

    ``None`` denotes ``all_steps`` (the pre-existing mean-over-steps path).
    For single-step modes the scalar is wrapped in a one-element tuple.  A
    window mode must provide exactly one finite positive weight per selected
    step, preventing accidental reintroduction of a scalar window weight.
    """
    steps = tuple(selected_steps)
    if mode == "all_steps":
        if not isinstance(importance_weight, (int, float)) or float(importance_weight) != 1.0:
            raise ValueError("all_steps temporal credit has unit importance weight")
        return None
    if mode == "causal_window_2":
        if not isinstance(importance_weight, Sequence) or isinstance(
            importance_weight, (str, bytes)
        ):
            raise ValueError("causal_window_2 requires per-step HT weights")
        values = tuple(float(value) for value in importance_weight)
        if len(values) != len(steps) or len(values) != 2:
            raise ValueError("causal_window_2 requires exactly two selected-step weights")
    else:
        if not isinstance(importance_weight, (int, float)):
            raise ValueError("single-step temporal credit requires a scalar importance weight")
        values = (float(importance_weight),)
        if len(steps) != 1:
            raise ValueError("single-step temporal credit requires exactly one selected step")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("temporal-credit per-step weights must be finite and positive")
    return values


def aggregate_step_local_values(
    values: torch.Tensor,
    step_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate per-candidate, per-step scalars under temporal credit.

    ``values`` has shape ``(candidates, steps)``.  Without explicit weights it
    retains the legacy mean-over-candidates-and-steps.  With weights it computes
    ``mean_candidates(sum_steps(w_t * value_t))`` for the HT estimator.
    """
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("step-local values must be a non-empty candidate/step matrix")
    if not torch.isfinite(values).all():
        raise ValueError("step-local values must be finite")
    if step_weights is None:
        return values.mean()
    weights = torch.as_tensor(step_weights, dtype=values.dtype, device=values.device)
    if (
        weights.ndim != 1
        or weights.shape[0] != values.shape[1]
        or not torch.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("step-local weights must be finite positive vector")
    return (values * weights[None, :]).sum(dim=1).mean()


# Descriptive alias for callers that want to emphasize the weighted path.
weighted_step_local_mean = aggregate_step_local_values


def endpoint_units(structure: str) -> tuple[tuple[int, ...], ...]:
    """Return one action unit per unpaired position or target base pair."""
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    paired = set()
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError("target structure has an unmatched closing bracket")
            left = stack.pop()
            pairs.append((left, index))
            paired.update((left, index))
        elif symbol != ".":
            raise ValueError(f"unsupported target-structure symbol: {symbol}")
    if stack:
        raise ValueError("target structure has an unmatched opening bracket")
    units = [(index,) for index in range(len(structure)) if index not in paired]
    units.extend(pairs)
    return tuple(sorted(units, key=lambda unit: unit[0]))


def structure_error_credit_weights(
    predicted_structure: str,
    target_structure: str,
    advantage: float,
    background_weight: float = 0.25,
    positive_uniform: bool = False,
    strength: float = 1.0,
) -> torch.Tensor:
    """Focus positive credit on correct units and negative credit on errors."""
    if (
        len(predicted_structure) != len(target_structure)
        or set(predicted_structure) - set("().")
        or not math.isfinite(advantage)
        or not 0 < background_weight <= 1
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise ValueError("structure-error credit inputs are invalid")
    units = endpoint_units(target_structure)
    if positive_uniform and advantage >= 0:
        return torch.ones(len(units), dtype=torch.float32)
    predicted_pairs = set(target_pairs(predicted_structure))
    predicted_paired = {position for pair in predicted_pairs for position in pair}
    correct = []
    for unit in units:
        if len(unit) == 2:
            correct.append(tuple(unit) in predicted_pairs)
        else:
            correct.append(unit[0] not in predicted_paired)
    focus_on_correct = advantage >= 0
    weights = torch.tensor(
        [1.0 if value == focus_on_correct else background_weight for value in correct],
        dtype=torch.float32,
    )
    focused = weights / weights.mean()
    if strength == 1.0:
        return focused
    if strength == 0.0:
        return torch.ones_like(focused)
    return 1.0 + strength * (focused - 1.0)


def _temperatures(
    temperature: float | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(temperature, dtype=torch.float32, device=device)
    if values.ndim == 0:
        values = values.expand(batch_size)
    if values.shape != (batch_size,) or not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("endpoint-policy temperature must be a positive batch vector")
    return values


def endpoint_action_unit_log_probabilities(
    logits: torch.Tensor,
    actions: torch.Tensor,
    structure: str,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Log-probability of every independent single/pair action factor."""
    if logits.ndim != 3 or logits.shape[-1] != 4 or actions.shape != logits.shape[:2]:
        raise ValueError("endpoint-policy logits/actions are not aligned")
    if logits.shape[1] != len(structure):
        raise ValueError("endpoint-policy structure length mismatch")
    batch = logits.shape[0]
    temperatures = _temperatures(temperature, batch, logits.device)
    token_log_probs = torch.log_softmax(logits.float() / temperatures[:, None, None], dim=-1)
    pair_ids = LEGAL_PAIR_IDS.to(logits.device)
    values = []
    for unit in endpoint_units(structure):
        if len(unit) == 1:
            position = unit[0]
            values.append(token_log_probs[:, position].gather(
                -1, actions[:, position, None]
            ).squeeze(-1))
            continue
        left, right = unit
        pair_scores = token_log_probs[:, left, pair_ids[:, 0]] + token_log_probs[
            :, right, pair_ids[:, 1]
        ]
        pair_log_probs = torch.log_softmax(pair_scores, dim=-1)
        matches = (actions[:, left, None] == pair_ids[None, :, 0]) & (
            actions[:, right, None] == pair_ids[None, :, 1]
        )
        if not torch.equal(matches.sum(dim=-1), torch.ones(batch, dtype=torch.long, device=logits.device)):
            raise ValueError("endpoint-policy action contains an illegal target pair")
        selected = matches.to(torch.long).argmax(dim=-1)
        values.append(pair_log_probs.gather(-1, selected[:, None]).squeeze(-1))
    if not values:
        raise ValueError("endpoint policy has no action units")
    result = torch.stack(values, dim=-1)
    if not torch.isfinite(result).all():
        raise RuntimeError("endpoint-policy log-probability is non-finite")
    return result


def endpoint_action_log_probabilities(
    logits: torch.Tensor,
    actions: torch.Tensor,
    structure: str,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Mean action-factor log-probability for scale-independent diagnostics."""
    return endpoint_action_unit_log_probabilities(
        logits, actions, structure, temperature
    ).mean(dim=-1)


def sample_endpoint_actions(
    logits: torch.Tensor,
    structure: str,
    temperature: float | torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample legal clean endpoint proposals and return mean log-prob/entropy."""
    if logits.ndim != 3 or logits.shape[1:] != (len(structure), 4):
        raise ValueError("endpoint-policy logits/structure are not aligned")
    batch = logits.shape[0]
    temperatures = _temperatures(temperature, batch, logits.device)
    token_log_probs = torch.log_softmax(logits.float() / temperatures[:, None, None], dim=-1)
    pair_ids = LEGAL_PAIR_IDS.to(logits.device)
    actions = torch.empty(batch, len(structure), dtype=torch.long, device=logits.device)
    entropies = []
    for unit in endpoint_units(structure):
        if len(unit) == 1:
            position = unit[0]
            probabilities = token_log_probs[:, position].exp()
            actions[:, position] = torch.multinomial(
                probabilities, 1, replacement=True, generator=generator
            ).squeeze(-1)
            entropies.append(-(probabilities * token_log_probs[:, position]).sum(dim=-1))
            continue
        left, right = unit
        pair_scores = token_log_probs[:, left, pair_ids[:, 0]] + token_log_probs[
            :, right, pair_ids[:, 1]
        ]
        pair_log_probs = torch.log_softmax(pair_scores, dim=-1)
        probabilities = pair_log_probs.exp()
        selected = torch.multinomial(
            probabilities, 1, replacement=True, generator=generator
        ).squeeze(-1)
        actions[:, left] = pair_ids[selected, 0]
        actions[:, right] = pair_ids[selected, 1]
        entropies.append(-(probabilities * pair_log_probs).sum(dim=-1))
    log_probabilities = endpoint_action_log_probabilities(
        logits, actions, structure, temperatures
    )
    entropy = torch.stack(entropies, dim=-1).mean(dim=-1)
    return actions, log_probabilities, entropy


def discrete_mixture_replacement_probability(step: int, steps: int) -> float:
    """Exact finite-step replacement probability for the linear DFM mixture path.

    For kappa_t=t with an evenly discretized horizon, conditioning on the current
    state gives rho=(kappa_{t+h}-kappa_t)/(1-kappa_t)=1/(steps-step).  The last
    transition therefore replaces from the learned clean posterior with rho=1.
    """
    if type(step) is not int or type(steps) is not int or steps <= 0 or not 0 <= step < steps:
        raise ValueError("discrete DFM step geometry is invalid")
    return 1.0 / float(steps - step)


def discrete_domino_dirichlet_mean_bridge(
    tokens: torch.Tensor, alpha: torch.Tensor | float,
) -> torch.Tensor:
    """Map a discrete RNA state onto the supervised Dirichlet-path mean.

    The transferred 87.55M denoiser was trained on
    Dirichlet(1 + onehot(x0) * (alpha - 1)).  Feeding raw one-hot states is
    therefore strongly out of distribution, especially for large alpha.  This
    deterministic bridge keeps the policy state discrete while presenting the
    denoiser with the exact conditional mean of its supervised corruption path:
    clean channel alpha/(alpha+3), other channels 1/(alpha+3).
    """
    if tokens.ndim != 2 or tokens.numel() == 0 or int(tokens.min()) < 0 or int(tokens.max()) >= 4:
        raise ValueError("discrete DoMinO bridge tokens are invalid")
    alpha_values = torch.as_tensor(alpha, dtype=torch.float32, device=tokens.device)
    if alpha_values.ndim == 0:
        alpha_values = alpha_values.expand(tokens.shape[0])
    if (
        alpha_values.shape != (tokens.shape[0],)
        or not torch.isfinite(alpha_values).all()
        or (alpha_values < 1).any()
    ):
        raise ValueError("discrete DoMinO bridge alpha is invalid")
    onehot = F.one_hot(tokens, num_classes=4).to(torch.float32)
    concentrations = 1.0 + onehot * (alpha_values[:, None, None] - 1.0)
    state = concentrations / (alpha_values[:, None, None] + 3.0)
    if not torch.isfinite(state).all() or float((state.sum(dim=-1) - 1).abs().max()) > 1e-6:
        raise RuntimeError("discrete DoMinO bridge left the probability simplex")
    return state


def _legal_unit_indices(tokens: torch.Tensor, unit: tuple[int, ...], pair_ids: torch.Tensor) -> torch.Tensor:
    if len(unit) == 1:
        values = tokens[:, unit[0]]
        if int(values.min()) < 0 or int(values.max()) >= 4:
            raise ValueError("discrete DFM token is outside nucleotide vocabulary")
        return values
    left, right = unit
    matches = (tokens[:, left, None] == pair_ids[None, :, 0]) & (
        tokens[:, right, None] == pair_ids[None, :, 1]
    )
    if not torch.equal(
        matches.sum(dim=-1), torch.ones(tokens.shape[0], dtype=torch.long, device=tokens.device)
    ):
        raise ValueError("discrete DFM state contains an illegal target pair")
    return matches.to(torch.long).argmax(dim=-1)


def discrete_domino_transition_unit_log_probabilities(
    logits: torch.Tensor,
    current_tokens: torch.Tensor,
    next_tokens: torch.Tensor,
    structure: str,
    temperature: float | torch.Tensor,
    replacement_probability: float,
) -> torch.Tensor:
    """Exact log-probabilities of the structured finite-step discrete transition.

    Each structure unit follows the x1-independent mixture-path kernel
    P(x_next|x_t)=(1-rho)delta(x_next=x_t)+rho*p_theta(x1|x_t).
    Unpaired positions have four states; target base-pair units have the six legal
    canonical/wobble pair states.  This is a genuine discrete transition kernel,
    not the continuous simplex endpoint surrogate used by B1.
    """
    if (
        logits.ndim != 3
        or logits.shape[1:] != (len(structure), 4)
        or current_tokens.shape != logits.shape[:2]
        or next_tokens.shape != current_tokens.shape
        or not math.isfinite(replacement_probability)
        or not 0 < replacement_probability <= 1
    ):
        raise ValueError("discrete DoMinO transition tensor contract is invalid")
    batch = logits.shape[0]
    temperatures = _temperatures(temperature, batch, logits.device)
    token_log_probs = torch.log_softmax(
        logits.float() / temperatures[:, None, None], dim=-1
    )
    pair_ids = LEGAL_PAIR_IDS.to(logits.device)
    values = []
    rho = float(replacement_probability)
    for unit in endpoint_units(structure):
        current_index = _legal_unit_indices(current_tokens, unit, pair_ids)
        next_index = _legal_unit_indices(next_tokens, unit, pair_ids)
        if len(unit) == 1:
            posterior = token_log_probs[:, unit[0]].exp()
        else:
            left, right = unit
            pair_scores = token_log_probs[:, left, pair_ids[:, 0]] + token_log_probs[
                :, right, pair_ids[:, 1]
            ]
            posterior = torch.softmax(pair_scores, dim=-1)
        stay = F.one_hot(current_index, num_classes=posterior.shape[-1]).to(posterior.dtype)
        transition = rho * posterior + (1.0 - rho) * stay
        selected = transition.gather(-1, next_index[:, None]).squeeze(-1)
        values.append(torch.log(selected.clamp_min(torch.finfo(selected.dtype).tiny)))
    result = torch.stack(values, dim=-1)
    if not torch.isfinite(result).all():
        raise RuntimeError("discrete DoMinO transition log-probability is non-finite")
    return result


def structured_discrete_reference_transition_tv(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    current_tokens: torch.Tensor,
    structure: str,
    temperature: float | torch.Tensor,
    replacement_probability: float,
) -> dict[str, torch.Tensor | bool]:
    """TV-inspired penalty between structured clean posteriors and transitions.

    This is deliberately a reference-transition regularizer, not a claim that
    the underlying continuous flow has a native discrete DFM likelihood.  For
    each saved discrete state, clean distributions are four-way for unpaired
    units and six-way over ``LEGAL_PAIRS`` for paired units.  The transition
    kernel mixes the same point mass into both policies, hence its TV is
    ``rho * TV(clean_current, clean_reference)``.
    """
    if (
        current_logits.ndim != 3
        or reference_logits.shape != current_logits.shape
        or current_logits.shape[1:] != (len(structure), 4)
        or current_tokens.shape != current_logits.shape[:2]
        or not math.isfinite(replacement_probability)
        or not 0 <= replacement_probability <= 1
    ):
        raise ValueError("structured discrete reference-transition TV tensors are invalid")
    batch = current_logits.shape[0]
    temperatures = _temperatures(temperature, batch, current_logits.device)
    current_log_probs = torch.log_softmax(
        current_logits.float() / temperatures[:, None, None], dim=-1
    )
    # The frozen reference must remain detached even when a caller accidentally
    # passes a grad-enabled tensor rather than logits from ``torch.no_grad``.
    reference_log_probs = torch.log_softmax(
        reference_logits.detach().float() / temperatures[:, None, None], dim=-1
    )
    pair_ids = LEGAL_PAIR_IDS.to(current_logits.device)
    posterior_values: list[torch.Tensor] = []
    for unit in endpoint_units(structure):
        # The saved rollout state is part of the transition contract even
        # though the common stay mass algebraically cancels from TV.
        _legal_unit_indices(current_tokens, unit, pair_ids)
        if len(unit) == 1:
            position = unit[0]
            current_posterior = current_log_probs[:, position].exp()
            reference_posterior = reference_log_probs[:, position].exp()
        else:
            left, right = unit
            current_scores = (
                current_log_probs[:, left, pair_ids[:, 0]]
                + current_log_probs[:, right, pair_ids[:, 1]]
            )
            reference_scores = (
                reference_log_probs[:, left, pair_ids[:, 0]]
                + reference_log_probs[:, right, pair_ids[:, 1]]
            )
            current_posterior = torch.softmax(current_scores, dim=-1)
            reference_posterior = torch.softmax(reference_scores, dim=-1)
        posterior_values.append(
            0.5 * (current_posterior - reference_posterior).abs().sum(dim=-1)
        )
    posterior_tv = torch.stack(posterior_values, dim=-1)
    transition_tv = posterior_tv * float(replacement_probability)
    finite = bool(torch.isfinite(posterior_tv.detach()).all() and torch.isfinite(transition_tv.detach()).all())
    if not finite:
        raise RuntimeError("structured discrete reference-transition TV is non-finite")
    return {
        "posterior_tv": posterior_tv,
        "transition_tv": transition_tv,
        "posterior_tv_mean": posterior_tv.mean(),
        "transition_tv_mean": transition_tv.mean(),
        "finite": finite,
    }


# Shorter public spelling used by the trainer and downstream unit tests.
discrete_reference_transition_tv = structured_discrete_reference_transition_tv


def sample_discrete_domino_transition(
    logits: torch.Tensor,
    current_tokens: torch.Tensor,
    structure: str,
    temperature: float | torch.Tensor,
    replacement_probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one structured DFM transition and return unit log-probs/entropy."""
    if logits.ndim != 3 or logits.shape[1:] != (len(structure), 4):
        raise ValueError("discrete DoMinO logits/structure are not aligned")
    batch = logits.shape[0]
    temperatures = _temperatures(temperature, batch, logits.device)
    token_log_probs = torch.log_softmax(
        logits.float() / temperatures[:, None, None], dim=-1
    )
    pair_ids = LEGAL_PAIR_IDS.to(logits.device)
    next_tokens = current_tokens.clone()
    entropies = []
    rho = float(replacement_probability)
    if not math.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError("discrete DoMinO replacement probability is invalid")
    for unit in endpoint_units(structure):
        current_index = _legal_unit_indices(current_tokens, unit, pair_ids)
        if len(unit) == 1:
            posterior = token_log_probs[:, unit[0]].exp()
        else:
            left, right = unit
            pair_scores = token_log_probs[:, left, pair_ids[:, 0]] + token_log_probs[
                :, right, pair_ids[:, 1]
            ]
            posterior = torch.softmax(pair_scores, dim=-1)
        stay = F.one_hot(current_index, num_classes=posterior.shape[-1]).to(posterior.dtype)
        transition = rho * posterior + (1.0 - rho) * stay
        selected = torch.multinomial(
            transition, 1, replacement=True, generator=generator
        ).squeeze(-1)
        if len(unit) == 1:
            next_tokens[:, unit[0]] = selected
        else:
            left, right = unit
            next_tokens[:, left] = pair_ids[selected, 0]
            next_tokens[:, right] = pair_ids[selected, 1]
        entropies.append(
            -(transition * torch.log(transition.clamp_min(torch.finfo(transition.dtype).tiny))).sum(dim=-1)
        )
    unit_log_probabilities = discrete_domino_transition_unit_log_probabilities(
        logits, current_tokens, next_tokens, structure, temperatures, rho
    )
    entropy = torch.stack(entropies, dim=-1).mean(dim=-1)
    return next_tokens, unit_log_probabilities, entropy


@torch.no_grad()
def rollout_discrete_domino_trajectory(
    model: FairRNAFlow,
    structure: str,
    candidates: int,
    steps: int,
    seed: int,
    device: torch.device,
    temperature: float | torch.Tensor = 1.0,
) -> dict:
    """Roll out the structured discrete mixture-path policy used by DoMinO."""
    if candidates <= 0 or steps <= 0:
        raise ValueError("discrete DoMinO rollout sizes must be positive")
    generator = torch.Generator(device=device).manual_seed(seed)
    pair_ids = LEGAL_PAIR_IDS.to(device)
    tokens = torch.empty(candidates, len(structure), dtype=torch.long, device=device)
    for unit in endpoint_units(structure):
        if len(unit) == 1:
            tokens[:, unit[0]] = torch.randint(
                0, 4, (candidates,), generator=generator, device=device
            )
        else:
            selected = torch.randint(
                0, len(LEGAL_PAIRS), (candidates,), generator=generator, device=device
            )
            tokens[:, unit[0]] = pair_ids[selected, 0]
            tokens[:, unit[1]] = pair_ids[selected, 1]
    mask = torch.ones(candidates, len(structure), device=device)
    structure_tokens = torch.tensor(
        [STRUCTURE_SYMBOLS.index(symbol) for symbol in structure],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(candidates, -1)
    temperatures = _temperatures(temperature, candidates, device)
    alpha_schedule = torch.linspace(1.001, 8.0, steps + 1, device=device)
    states = [tokens.detach().cpu()]
    actions_by_step = []
    old_unit_log_probs = []
    entropies = []
    replacement_probabilities = []
    model.eval()
    for step in range(steps):
        alpha = alpha_schedule[step].expand(candidates)
        simplex_state = discrete_domino_dirichlet_mean_bridge(tokens, alpha)
        logits = model(
            simplex_state, alpha, mask, structure_tokens=structure_tokens
        )
        rho = discrete_mixture_replacement_probability(step, steps)
        next_tokens, unit_log_probabilities, entropy = sample_discrete_domino_transition(
            logits, tokens, structure, temperatures, rho, generator
        )
        actions_by_step.append(next_tokens.detach().cpu())
        old_unit_log_probs.append(unit_log_probabilities.detach().cpu())
        entropies.append(entropy.detach().cpu())
        replacement_probabilities.append(rho)
        tokens = next_tokens
        states.append(tokens.detach().cpu())
    final_sequences = [
        "".join(NUCLEOTIDES[int(token)] for token in tokens[index].tolist())
        for index in range(candidates)
    ]
    return {
        "schema_version": 3,
        "policy": "structured-discrete-mixture-dfm-domino-v1",
        "structure": structure,
        "seed": seed,
        "steps": steps,
        "states": torch.stack(states, dim=1),
        "actions": torch.stack(actions_by_step, dim=1),
        "old_unit_log_probabilities": torch.stack(old_unit_log_probs, dim=1),
        "entropies": torch.stack(entropies, dim=1),
        "alphas": alpha_schedule.detach().cpu(),
        "replacement_probabilities": torch.tensor(replacement_probabilities),
        "temperatures": temperatures.detach().cpu(),
        "final_sequences": final_sequences,
    }


def recompute_discrete_domino_unit_log_probabilities(
    model: FairRNAFlow,
    trajectory: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    structure = str(trajectory["structure"])
    states = trajectory["states"].to(device)
    actions = trajectory["actions"].to(device)
    alphas = trajectory["alphas"].to(device)
    rhos = trajectory["replacement_probabilities"].to(device)
    temperatures = trajectory["temperatures"].to(device)
    batch, steps, length = actions.shape
    if (
        states.shape != (batch, steps + 1, length)
        or alphas.shape != (steps + 1,)
        or rhos.shape != (steps,)
    ):
        raise ValueError("saved discrete DoMinO trajectory has an invalid tensor contract")
    mask = torch.ones(batch, length, device=device)
    structure_tokens = torch.tensor(
        [STRUCTURE_SYMBOLS.index(symbol) for symbol in structure],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(batch, -1)
    log_probabilities = []
    logits_by_step = []
    for step in range(steps):
        alpha = alphas[step].expand(batch)
        simplex_state = discrete_domino_dirichlet_mean_bridge(states[:, step], alpha)
        logits = model(
            simplex_state,
            alpha,
            mask,
            structure_tokens=structure_tokens,
        )
        log_probabilities.append(discrete_domino_transition_unit_log_probabilities(
            logits, states[:, step], actions[:, step], structure,
            temperatures, float(rhos[step].item())
        ))
        logits_by_step.append(logits)
    return torch.stack(log_probabilities, dim=1), torch.stack(logits_by_step, dim=1)


@torch.no_grad()
def validate_discrete_domino_trajectory(
    trajectory: dict,
    device: torch.device,
    *,
    expected_structure: str,
    expected_seed: int,
    expected_candidates: int,
    expected_steps: int,
    expected_temperatures: torch.Tensor,
) -> dict:
    required = {
        "schema_version", "policy", "structure", "seed", "steps", "states", "actions",
        "old_unit_log_probabilities", "entropies", "alphas", "replacement_probabilities",
        "temperatures", "final_sequences",
    }
    if set(trajectory) != required:
        raise ValueError("discrete DoMinO trajectory fields mismatch")
    if (
        trajectory["schema_version"] != 3
        or trajectory["policy"] != "structured-discrete-mixture-dfm-domino-v1"
        or trajectory["structure"] != expected_structure
        or trajectory["seed"] != expected_seed
        or trajectory["steps"] != expected_steps
    ):
        raise ValueError("discrete DoMinO trajectory identity mismatch")
    states = trajectory["states"]
    actions = trajectory["actions"]
    old_units = trajectory["old_unit_log_probabilities"]
    unit_count = len(endpoint_units(expected_structure))
    if (
        states.shape != (expected_candidates, expected_steps + 1, len(expected_structure))
        or actions.shape != (expected_candidates, expected_steps, len(expected_structure))
        or old_units.shape != (expected_candidates, expected_steps, unit_count)
        or trajectory["entropies"].shape != (expected_candidates, expected_steps)
        or trajectory["alphas"].shape != (expected_steps + 1,)
        or trajectory["replacement_probabilities"].shape != (expected_steps,)
        or trajectory["temperatures"].shape != (expected_candidates,)
        or len(trajectory["final_sequences"]) != expected_candidates
        or not torch.equal(trajectory["temperatures"], expected_temperatures.cpu())
    ):
        raise ValueError("discrete DoMinO trajectory coverage mismatch")
    expected_rhos = torch.tensor([
        discrete_mixture_replacement_probability(step, expected_steps)
        for step in range(expected_steps)
    ])
    if not torch.allclose(trajectory["replacement_probabilities"], expected_rhos, atol=0, rtol=0):
        raise ValueError("discrete DoMinO transition schedule mismatch")
    if not torch.equal(states[:, 1:], actions):
        raise ValueError("discrete DoMinO replay mismatch")
    pair_ids = LEGAL_PAIR_IDS
    for step in range(expected_steps + 1):
        for unit in endpoint_units(expected_structure):
            _legal_unit_indices(states[:, step], unit, pair_ids)
    decoded = [
        "".join(NUCLEOTIDES[int(token)] for token in states[index, -1].tolist())
        for index in range(expected_candidates)
    ]
    if decoded != trajectory["final_sequences"] or not torch.isfinite(old_units).all():
        raise ValueError("discrete DoMinO final decode/log-prob mismatch")
    return {
        "replay_exact": True,
        "action_factors": unit_count,
        "structured_pair_actions_legal": True,
        "transition_kernel": "(1-rho)delta+rho*p_clean",
    }


def endpoint_policy_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    structure: str,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """KL(current || reference) over the coupled endpoint action units."""
    if current_logits.shape != reference_logits.shape or current_logits.ndim != 3:
        raise ValueError("endpoint-policy KL logits are not aligned")
    batch = current_logits.shape[0]
    temperatures = _temperatures(temperature, batch, current_logits.device)
    current = torch.log_softmax(current_logits.float() / temperatures[:, None, None], dim=-1)
    reference = torch.log_softmax(reference_logits.float() / temperatures[:, None, None], dim=-1)
    pair_ids = LEGAL_PAIR_IDS.to(current_logits.device)
    values = []
    for unit in endpoint_units(structure):
        if len(unit) == 1:
            position = unit[0]
            values.append((current[:, position].exp() * (
                current[:, position] - reference[:, position]
            )).sum(dim=-1))
            continue
        left, right = unit
        current_pair = torch.log_softmax(
            current[:, left, pair_ids[:, 0]] + current[:, right, pair_ids[:, 1]], dim=-1
        )
        reference_pair = torch.log_softmax(
            reference[:, left, pair_ids[:, 0]] + reference[:, right, pair_ids[:, 1]], dim=-1
        )
        values.append((current_pair.exp() * (current_pair - reference_pair)).sum(dim=-1))
    result = torch.stack(values, dim=-1).mean(dim=-1)
    if not torch.isfinite(result).all():
        raise RuntimeError("endpoint-policy KL is non-finite")
    return result


def endpoint_policy_entropy(
    logits: torch.Tensor,
    structure: str,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Mean entropy over independent single/pair endpoint action units."""
    if logits.ndim != 3 or logits.shape[1:] != (len(structure), 4):
        raise ValueError("endpoint-policy entropy logits/structure are not aligned")
    batch = logits.shape[0]
    temperatures = _temperatures(temperature, batch, logits.device)
    token_log_probs = torch.log_softmax(logits.float() / temperatures[:, None, None], dim=-1)
    pair_ids = LEGAL_PAIR_IDS.to(logits.device)
    values = []
    for unit in endpoint_units(structure):
        if len(unit) == 1:
            position = unit[0]
            probabilities = token_log_probs[:, position].exp()
            values.append(-(probabilities * token_log_probs[:, position]).sum(dim=-1))
            continue
        left, right = unit
        pair_log_probs = torch.log_softmax(
            token_log_probs[:, left, pair_ids[:, 0]]
            + token_log_probs[:, right, pair_ids[:, 1]],
            dim=-1,
        )
        probabilities = pair_log_probs.exp()
        values.append(-(probabilities * pair_log_probs).sum(dim=-1))
    result = torch.stack(values, dim=-1).mean(dim=-1)
    if not torch.isfinite(result).all():
        raise RuntimeError("endpoint-policy entropy is non-finite")
    return result


def endpoint_velocity_step(
    flow: TorchDirichletConditionalFlow,
    state: torch.Tensor,
    actions: torch.Tensor,
    alpha: torch.Tensor,
    next_alpha: torch.Tensor,
) -> torch.Tensor:
    endpoint = F.one_hot(actions, num_classes=4).to(state.dtype)
    velocity = flow.posterior_to_velocity(endpoint, state, alpha)
    return deterministic_simplex_project(state + velocity * (next_alpha - alpha))


def deterministic_simplex_project(value: torch.Tensor) -> torch.Tensor:
    """Four-channel simplex projection without CUDA's nondeterministic cumsum kernel."""
    if value.shape[-1] != 4:
        raise ValueError("RNA simplex projection requires four channels")
    flat = value.reshape(-1, 4)
    ordered, _ = torch.sort(flat, dim=-1, descending=True)
    prefix = torch.stack(
        (
            ordered[:, 0],
            ordered[:, 0] + ordered[:, 1],
            ordered[:, 0] + ordered[:, 1] + ordered[:, 2],
            ordered[:, 0] + ordered[:, 1] + ordered[:, 2] + ordered[:, 3],
        ),
        dim=-1,
    )
    divisors = torch.tensor((1, 2, 3, 4), dtype=value.dtype, device=value.device)
    thresholds = (prefix - 1) / divisors
    support = (ordered > thresholds).sum(dim=-1, keepdim=True)
    threshold = thresholds.gather(-1, support - 1)
    return torch.clamp(flat - threshold, min=0).view_as(value)


@torch.no_grad()
def rollout_endpoint_trajectory(
    model: FairRNAFlow,
    structure: str,
    candidates: int,
    steps: int,
    seed: int,
    device: torch.device,
    temperature: float | torch.Tensor = 1.0,
) -> dict:
    if candidates <= 0 or steps <= 0:
        raise ValueError("endpoint rollout candidates and steps must be positive")
    generator = torch.Generator(device=device).manual_seed(seed)
    concentrations = torch.ones(candidates, len(structure), 4, device=device)
    state = torch._sample_dirichlet(concentrations, generator=generator)
    mask = torch.ones(candidates, len(structure), device=device)
    structure_tokens = torch.tensor(
        [STRUCTURE_SYMBOLS.index(symbol) for symbol in structure],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(candidates, -1)
    temperatures = _temperatures(temperature, candidates, device)
    schedule = torch.linspace(1.001, 8.0, steps + 1, device=device)
    flow = TorchDirichletConditionalFlow(device)
    states = [state.detach().cpu()]
    actions_by_step = []
    old_log_probs = []
    old_unit_log_probs = []
    entropies = []
    model.eval()
    for step in range(steps):
        alpha = schedule[step].expand(candidates)
        logits = model(
            state,
            alpha,
            mask,
            structure_tokens=structure_tokens,
        )
        actions, log_probabilities, entropy = sample_endpoint_actions(
            logits, structure, temperatures, generator
        )
        unit_log_probabilities = endpoint_action_unit_log_probabilities(
            logits, actions, structure, temperatures
        )
        state = endpoint_velocity_step(
            flow, state, actions, schedule[step], schedule[step + 1]
        )
        actions_by_step.append(actions.detach().cpu())
        old_log_probs.append(log_probabilities.detach().cpu())
        old_unit_log_probs.append(unit_log_probabilities.detach().cpu())
        entropies.append(entropy.detach().cpu())
        states.append(state.detach().cpu())
    final_sequences = [constrained_decode(state[index], structure) for index in range(candidates)]
    return {
        "schema_version": 2,
        "policy": "simplex-clean-endpoint-categorical-paired-categorical",
        "structure": structure,
        "seed": seed,
        "steps": steps,
        "states": torch.stack(states, dim=1),
        "actions": torch.stack(actions_by_step, dim=1),
        "old_log_probabilities": torch.stack(old_log_probs, dim=1),
        "old_unit_log_probabilities": torch.stack(old_unit_log_probs, dim=1),
        "entropies": torch.stack(entropies, dim=1),
        "alphas": schedule.detach().cpu(),
        "temperatures": temperatures.detach().cpu(),
        "final_sequences": final_sequences,
    }


def recompute_trajectory_log_probabilities(
    model: FairRNAFlow,
    trajectory: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    structure = str(trajectory["structure"])
    states = trajectory["states"].to(device)
    actions = trajectory["actions"].to(device)
    alphas = trajectory["alphas"].to(device)
    temperatures = trajectory["temperatures"].to(device)
    batch, steps, length = actions.shape
    if states.shape != (batch, steps + 1, length, 4) or alphas.shape != (steps + 1,):
        raise ValueError("saved endpoint trajectory has an invalid tensor contract")
    mask = torch.ones(batch, length, device=device)
    structure_tokens = torch.tensor(
        [STRUCTURE_SYMBOLS.index(symbol) for symbol in structure],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(batch, -1)
    log_probabilities = []
    kls = []
    for step in range(steps):
        logits = model(
            states[:, step],
            alphas[step].expand(batch),
            mask,
            structure_tokens=structure_tokens,
        )
        log_probabilities.append(endpoint_action_log_probabilities(
            logits, actions[:, step], structure, temperatures
        ))
        kls.append(logits)
    return torch.stack(log_probabilities, dim=1), torch.stack(kls, dim=1)


def recompute_trajectory_unit_log_probabilities(
    model: FairRNAFlow,
    trajectory: dict,
    device: torch.device,
    *,
    selected_steps: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    structure = str(trajectory["structure"])
    states = trajectory["states"].to(device)
    actions = trajectory["actions"].to(device)
    alphas = trajectory["alphas"].to(device)
    temperatures = trajectory["temperatures"].to(device)
    batch, steps, length = actions.shape
    if states.shape != (batch, steps + 1, length, 4) or alphas.shape != (steps + 1,):
        raise ValueError("saved endpoint trajectory has an invalid tensor contract")
    if selected_steps is None:
        selected_steps = tuple(range(steps))
    else:
        selected_steps = tuple(selected_steps)
        if not selected_steps or len(set(selected_steps)) != len(selected_steps):
            raise ValueError("selected trajectory steps must be unique and non-empty")
        if any(not isinstance(step, int) or not 0 <= step < steps for step in selected_steps):
            raise ValueError("selected trajectory step is outside the saved rollout")
    mask = torch.ones(batch, length, device=device)
    structure_tokens = torch.tensor(
        [STRUCTURE_SYMBOLS.index(symbol) for symbol in structure],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(batch, -1)
    log_probabilities = []
    logits_by_step = []
    for step in selected_steps:
        logits = model(
            states[:, step],
            alphas[step].expand(batch),
            mask,
            structure_tokens=structure_tokens,
        )
        log_probabilities.append(endpoint_action_unit_log_probabilities(
            logits, actions[:, step], structure, temperatures
        ))
        logits_by_step.append(logits)
    return torch.stack(log_probabilities, dim=1), torch.stack(logits_by_step, dim=1)


@torch.no_grad()
def replay_endpoint_trajectory(trajectory: dict, device: torch.device) -> torch.Tensor:
    states = trajectory["states"].to(device)
    actions = trajectory["actions"].to(device)
    alphas = trajectory["alphas"].to(device)
    flow = TorchDirichletConditionalFlow(device)
    replayed = [states[:, 0]]
    state = states[:, 0]
    for step in range(actions.shape[1]):
        state = endpoint_velocity_step(flow, state, actions[:, step], alphas[step], alphas[step + 1])
        replayed.append(state)
    return torch.stack(replayed, dim=1)


@torch.no_grad()
def validate_endpoint_trajectory(
    trajectory: dict,
    device: torch.device,
    *,
    expected_structure: str,
    expected_seed: int,
    expected_candidates: int,
    expected_steps: int,
    expected_temperatures: torch.Tensor,
) -> dict:
    required = {
        "schema_version", "policy", "structure", "seed", "steps", "states", "actions",
        "old_log_probabilities", "old_unit_log_probabilities", "entropies", "alphas",
        "temperatures", "final_sequences",
    }
    if set(trajectory) != required:
        raise ValueError(
            f"endpoint trajectory fields mismatch: missing={sorted(required - set(trajectory))}, "
            f"extra={sorted(set(trajectory) - required)}"
        )
    if (
        trajectory["schema_version"] != 2
        or trajectory["policy"] != "simplex-clean-endpoint-categorical-paired-categorical"
        or trajectory["structure"] != expected_structure
        or trajectory["seed"] != expected_seed
        or trajectory["steps"] != expected_steps
    ):
        raise ValueError("endpoint trajectory identity mismatch")
    states = trajectory["states"]
    actions = trajectory["actions"]
    old = trajectory["old_log_probabilities"]
    old_units = trajectory["old_unit_log_probabilities"]
    unit_count = len(endpoint_units(expected_structure))
    if (
        states.shape != (expected_candidates, expected_steps + 1, len(expected_structure), 4)
        or actions.shape != (expected_candidates, expected_steps, len(expected_structure))
        or old.shape != (expected_candidates, expected_steps)
        or old_units.shape != (expected_candidates, expected_steps, unit_count)
        or trajectory["entropies"].shape != (expected_candidates, expected_steps)
        or trajectory["alphas"].shape != (expected_steps + 1,)
        or trajectory["temperatures"].shape != (expected_candidates,)
        or len(trajectory["final_sequences"]) != expected_candidates
    ):
        raise ValueError("endpoint trajectory tensor coverage mismatch")
    if (
        not torch.equal(trajectory["temperatures"], expected_temperatures.cpu())
        or not torch.equal(
            trajectory["alphas"], torch.linspace(1.001, 8.0, expected_steps + 1)
        )
    ):
        raise ValueError("endpoint trajectory schedule/temperature mismatch")
    invariant_failures = []
    simplex_sum_error = float((states.sum(dim=-1) - 1).abs().max())
    mean_log_prob_error = float((old - old_units.mean(dim=-1)).abs().max())
    if not torch.isfinite(states).all():
        invariant_failures.append("non-finite state")
    if float(states.min()) < -1e-7 or simplex_sum_error > 2e-6:
        invariant_failures.append(f"simplex error={simplex_sum_error}")
    if int(actions.min()) < 0 or int(actions.max()) > 3:
        invariant_failures.append("action outside nucleotide vocabulary")
    if not torch.isfinite(old).all() or not torch.isfinite(old_units).all():
        invariant_failures.append("non-finite old log-probability")
    # The mean is a redundant diagnostic; GPU and CPU float32 reductions may differ by a few ulps.
    if mean_log_prob_error > 1e-6:
        invariant_failures.append(f"mean log-probability error={mean_log_prob_error}")
    if invariant_failures:
        raise ValueError(
            "endpoint trajectory invariant failed: " + "; ".join(invariant_failures)
        )
    dummy_logits = torch.zeros(
        expected_candidates, len(expected_structure), 4, device=device
    )
    for step in range(expected_steps):
        endpoint_action_unit_log_probabilities(
            dummy_logits, actions[:, step].to(device), expected_structure,
            trajectory["temperatures"].to(device),
        )
    replayed = replay_endpoint_trajectory(trajectory, device).cpu()
    replay_error = float((replayed - states).abs().max())
    decoded = [
        constrained_decode(states[index, -1].to(device), expected_structure)
        for index in range(expected_candidates)
    ]
    if replay_error > 1e-6 or decoded != trajectory["final_sequences"]:
        raise ValueError("endpoint trajectory replay/final decode mismatch")
    return {
        "replay_max_abs_error": replay_error,
        "action_factors": unit_count,
        "paired_actions_legal": True,
        "final_decode_exact": True,
    }


def trajectory_grpo_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> tuple[torch.Tensor, dict]:
    if (
        new_log_probabilities.ndim != 2
        or old_log_probabilities.shape != new_log_probabilities.shape
        or advantages.shape != new_log_probabilities.shape[:1]
        or not 0 < clip_ratio < 1
    ):
        raise ValueError("trajectory GRPO tensor contract is invalid")
    ratios = torch.exp(new_log_probabilities - old_log_probabilities)
    expanded_advantages = advantages[:, None]
    clipped = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio)
    surrogate = torch.minimum(ratios * expanded_advantages, clipped * expanded_advantages)
    loss = -surrogate.mean()
    return loss, {
        "mean_ratio": float(ratios.detach().mean().item()),
        "minimum_ratio": float(ratios.detach().min().item()),
        "maximum_ratio": float(ratios.detach().max().item()),
        "clip_fraction": float(
            ((ratios < 1 - clip_ratio) | (ratios > 1 + clip_ratio)).float().mean().item()
        ),
    }


def factorized_trajectory_grpo_loss(
    new_unit_log_probabilities: torch.Tensor,
    old_unit_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
    *,
    step_weights: Sequence[float] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """PPO/GRPO surrogate for each joint Flow-step action.

    Independent action-unit log probabilities are summed before constructing the
    likelihood ratio. Unit-level advantages preserve that true joint ratio for the
    PPO clipping decision while shaping the score-function gradient across action
    units. Their unit mean remains the scalar step advantage, so the forward
    surrogate and its scale match joint-action PPO.
    """
    if (
        new_unit_log_probabilities.ndim != 3
        or old_unit_log_probabilities.shape != new_unit_log_probabilities.shape
        or advantages.shape not in (
            new_unit_log_probabilities.shape[:1],
            (new_unit_log_probabilities.shape[0], new_unit_log_probabilities.shape[2]),
            new_unit_log_probabilities.shape,
        )
        or not 0 < clip_ratio < 1
    ):
        raise ValueError("factorized trajectory GRPO tensor contract is invalid")
    joint_log_ratios = (
        new_unit_log_probabilities - old_unit_log_probabilities
    ).sum(dim=-1)
    bounded_log_ratios = torch.clamp(joint_log_ratios, min=-20.0, max=20.0)
    stable_log_ratios = joint_log_ratios + (bounded_log_ratios - joint_log_ratios).detach()
    ratios = torch.exp(stable_log_ratios)
    unit_advantages = None
    if advantages.ndim == 1:
        expanded_advantages = advantages[:, None]
    elif advantages.ndim == 2:
        unit_advantages = advantages[:, None, :].expand(
            -1, ratios.shape[1], -1
        )
        expanded_advantages = unit_advantages.mean(dim=-1)
    else:
        unit_advantages = advantages
        expanded_advantages = unit_advantages.mean(dim=-1)
    clipped = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio)
    if unit_advantages is None:
        surrogate = torch.minimum(
            ratios * expanded_advantages, clipped * expanded_advantages
        )
        credit_gradient = "joint_action"
    else:
        unit_log_ratios = new_unit_log_probabilities - old_unit_log_probabilities
        localized_score = (unit_log_ratios * unit_advantages).sum(dim=-1)
        joint_forward = (ratios * expanded_advantages).detach()
        localized_surrogate = joint_forward + ratios.detach() * (
            localized_score - localized_score.detach()
        )
        clip_active = (
            ((expanded_advantages >= 0) & (ratios > 1 + clip_ratio))
            | ((expanded_advantages < 0) & (ratios < 1 - clip_ratio))
        )
        clipped_surrogate = (clipped * expanded_advantages).detach()
        surrogate = torch.where(clip_active, clipped_surrogate, localized_surrogate)
        credit_gradient = "unit_shaped_joint_ratio"
    if step_weights is None:
        # Legacy all-steps and one-step paths: average over candidates and
        # selected steps exactly as before.
        loss = -surrogate.mean()
        normalized_step_weights = None
    else:
        normalized_step_weights = torch.as_tensor(
            step_weights, dtype=surrogate.dtype, device=surrogate.device
        )
        if (
            normalized_step_weights.ndim != 1
            or normalized_step_weights.shape[0] != surrogate.shape[1]
            or not torch.isfinite(normalized_step_weights).all()
            or (normalized_step_weights <= 0).any()
        ):
            raise ValueError("trajectory step weights must be finite positive vector")
        # Horvitz--Thompson aggregation is a sum of step-local estimators,
        # with a candidate mean retained within each step.  In particular,
        # this is not ``(weights.mean() * surrogate.mean())``.
        loss = -(surrogate * normalized_step_weights[None, :]).sum(dim=1).mean()
    diagnostics = {
        "mean_ratio": float(ratios.detach().mean().item()),
        "minimum_ratio": float(ratios.detach().min().item()),
        "maximum_ratio": float(ratios.detach().max().item()),
        "clip_fraction": float(
            ((ratios < 1 - clip_ratio) | (ratios > 1 + clip_ratio)).float().mean().item()
        ),
        "action_factors": int(new_unit_log_probabilities.shape[-1]),
        "trajectory_steps": int(new_unit_log_probabilities.shape[1]),
        "ratio_unit": "joint_flow_step_action",
        "credit_gradient": credit_gradient,
        "log_ratio_clamp": [-20.0, 20.0],
    }
    if normalized_step_weights is not None:
        diagnostics.update({
            "step_weighted": True,
            "step_weights": [
                float(value) for value in normalized_step_weights.detach().cpu()
            ],
            "step_weight_sum": float(normalized_step_weights.detach().sum().item()),
        })
    return loss, diagnostics


def domino_endpoint_ppo_loss(
    new_unit_log_probabilities: torch.Tensor,
    old_unit_log_probabilities: torch.Tensor,
    trajectory_advantages: torch.Tensor,
    clip_ratio: float,
    *,
    normalize_time: bool = False,
) -> tuple[torch.Tensor, dict]:
    """DoMinO-PPO objective for the stochastic endpoint-policy sampler.

    DoMinO-PPO treats each one-step sampler transition as an MDP action and
    applies a clipped old/new transition likelihood ratio.  The published
    terminal-reward setting explicitly permits one trajectory-level advantage
    to be shared across time.  Our 87.55M model is *not* a native discrete-flow
    CTMC: it is a Dirichlet-simplex flow.  Therefore this helper is deliberately
    named ``endpoint`` and uses the exact categorical endpoint-action
    probability of the existing stochastic endpoint sampler, not a claimed
    native DFM transition probability.

    The paper writes a sum over time.  ``normalize_time=True`` divides by the
    number of trajectory steps and is useful only for scale-matched diagnostics;
    it is not a different policy-gradient estimator.
    """
    if (
        new_unit_log_probabilities.ndim != 3
        or old_unit_log_probabilities.shape != new_unit_log_probabilities.shape
        or trajectory_advantages.shape != new_unit_log_probabilities.shape[:1]
        or new_unit_log_probabilities.shape[1] <= 0
        or new_unit_log_probabilities.shape[2] <= 0
        or not torch.isfinite(new_unit_log_probabilities).all()
        or not torch.isfinite(old_unit_log_probabilities).all()
        or not torch.isfinite(trajectory_advantages).all()
        or not 0 < clip_ratio < 1
    ):
        raise ValueError("DoMinO endpoint PPO tensor contract is invalid")

    joint_log_ratios = (
        new_unit_log_probabilities - old_unit_log_probabilities
    ).sum(dim=-1)
    bounded_log_ratios = torch.clamp(joint_log_ratios, min=-20.0, max=20.0)
    stable_log_ratios = joint_log_ratios + (
        bounded_log_ratios - joint_log_ratios
    ).detach()
    ratios = torch.exp(stable_log_ratios)
    advantages = trajectory_advantages[:, None]
    clipped = torch.clamp(ratios, 1 - clip_ratio, 1 + clip_ratio)
    surrogate = torch.minimum(ratios * advantages, clipped * advantages)
    trajectory_surrogate = surrogate.sum(dim=1)
    if normalize_time:
        trajectory_surrogate = trajectory_surrogate / surrogate.shape[1]
    loss = -trajectory_surrogate.mean()
    return loss, {
        "objective": "domino-endpoint-ppo",
        "mean_ratio": float(ratios.detach().mean().item()),
        "minimum_ratio": float(ratios.detach().min().item()),
        "maximum_ratio": float(ratios.detach().max().item()),
        "clip_fraction": float(
            ((ratios < 1 - clip_ratio) | (ratios > 1 + clip_ratio))
            .float().mean().item()
        ),
        "trajectory_steps": int(new_unit_log_probabilities.shape[1]),
        "action_factors": int(new_unit_log_probabilities.shape[2]),
        "ratio_unit": "one-endpoint-sampler-step-joint-action",
        "advantage_scope": "trajectory-level-terminal-advantage-shared-across-time",
        "time_aggregation": "mean" if normalize_time else "sum",
        "native_discrete_flow_transition_exact": False,
        "native_model_family": "dirichlet-simplex-flow-matching",
        "sequence_likelihood_claimed": False,
    }


def sequence_group_fpo_loss(
    new_unit_log_probabilities: torch.Tensor,
    old_unit_log_probabilities: torch.Tensor,
    sequence_advantages: torch.Tensor,
    clip_ratio: float,
) -> tuple[torch.Tensor, dict]:
    """Terminal-reward Sequence-Group FPO over every Flow-Matching step.

    This deliberately does *not* construct or claim a likelihood for the full
    sampled sequence.  A complete endpoint trajectory has one categorical
    endpoint action at each Flow-Matching step; the terminal reward produces a
    single group-relative advantage for that candidate and the same value is
    applied to each of those legal step-local action ratios.
    """
    if (
        new_unit_log_probabilities.ndim != 3
        or old_unit_log_probabilities.shape != new_unit_log_probabilities.shape
        or sequence_advantages.shape != new_unit_log_probabilities.shape[:1]
        or new_unit_log_probabilities.shape[1] <= 0
        or not torch.isfinite(sequence_advantages).all()
    ):
        raise ValueError("sequence-group FPO tensor contract is invalid")
    loss, diagnostics = factorized_trajectory_grpo_loss(
        new_unit_log_probabilities,
        old_unit_log_probabilities,
        sequence_advantages,
        clip_ratio,
    )
    return loss, diagnostics | {
        "objective": "terminal-group-relative-advantage-weighted-flow-matching",
        "advantage_scope": "one-terminal-sequence-advantage-broadcast-to-all-steps",
        "temporal_credit": "none",
        "sequence_likelihood_claimed": False,
    }
