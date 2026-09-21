"""RNACG-compatible Dirichlet conditional flow primitives.

The project calls the clean endpoint x0. RNACG's source names the same clean
endpoint x1; this module keeps the project notation at its public boundary.
"""

from __future__ import annotations

import numpy as np
import scipy.special
import torch


def simplex_project(x: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex (RNACG algorithm)."""
    flat = x.reshape(-1, x.shape[-1])
    ordered, _ = torch.sort(flat, dim=-1, descending=True)
    cumsum = torch.cumsum(ordered, dim=-1) - 1
    divisor = torch.arange(1, flat.shape[-1] + 1, dtype=x.dtype, device=x.device)
    threshold_candidates = cumsum / divisor
    support = (ordered > threshold_candidates).sum(dim=1, keepdim=True)
    rows = torch.arange(flat.shape[0], device=x.device).unsqueeze(1)
    threshold = threshold_candidates[rows, support - 1]
    return torch.clamp(flat - threshold, min=0).view_as(x)


def sample_dirichlet_path(
    clean_onehot: torch.Tensor,
    alpha: torch.Tensor | None = None,
    alpha_scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample RNACG's Dirichlet path q(x_t | x0)."""
    batch, length, alphabet = clean_onehot.shape
    if alpha is None:
        alpha = 1 + torch.empty(batch, device=clean_onehot.device).exponential_(1 / alpha_scale)
        alpha = alpha.clamp(max=8.0)
    concentrations = torch.ones(batch, length, alphabet, device=clean_onehot.device)
    concentrations = concentrations + clean_onehot * (alpha[:, None, None] - 1)
    return torch.distributions.Dirichlet(concentrations).sample(), alpha


class DirichletConditionalFlow:
    """Numerical conditional-flow coefficient used by RNACG."""

    def __init__(self, alphabet_size: int = 4, alpha_min: float = 1, alpha_max: float = 8,
                 alpha_spacing: float = 0.01) -> None:
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.bs = np.linspace(0, 1, 1000)
        beta_cdfs = [scipy.special.betainc(a, alphabet_size - 1, self.bs) for a in self.alphas]
        self.beta_cdfs_derivative = np.diff(np.asarray(beta_cdfs), axis=0) / alpha_spacing
        self.alphabet_size = alphabet_size

    def c_factor(self, probabilities: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
        alpha_array = np.asarray(alpha)
        if alpha_array.ndim:
            alpha_scalar = float(alpha_array.reshape(-1)[0])
        else:
            alpha_scalar = float(alpha_array)
        alpha_scalar = min(max(alpha_scalar, self.alphas[0]), self.alphas[-2])
        beta = scipy.special.beta(alpha_scalar, self.alphabet_size - 1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denom = (1 - probabilities) ** (self.alphabet_size - 1)
            ratio = np.where(probabilities < 1, beta / denom, 0)
            ratio = np.where(probabilities ** (alpha_scalar - 1) > 0,
                             ratio / probabilities ** (alpha_scalar - 1), 0)
        idx = int(np.argmin(np.abs(alpha_scalar - self.alphas[:-1])))
        derivative = self.beta_cdfs_derivative[idx]
        interp = -np.interp(probabilities, self.bs, derivative)
        return np.nan_to_num(interp * ratio, nan=0.0, posinf=0.0, neginf=0.0)

    def posterior_to_velocity(
        self, posterior: torch.Tensor, state: torch.Tensor, alpha: torch.Tensor | float
    ) -> torch.Tensor:
        """Convert clean-token posterior to RNACG analytic velocity."""
        alpha_value = float(alpha.reshape(-1)[0].item()) if isinstance(alpha, torch.Tensor) else float(alpha)
        coefficient = torch.from_numpy(self.c_factor(state.detach().cpu().numpy(), alpha_value)).to(state)
        eye = torch.eye(state.shape[-1], dtype=state.dtype, device=state.device)
        conditional = (eye - state.unsqueeze(-1)) * coefficient.unsqueeze(-2)
        velocity = (posterior.unsqueeze(-2) * conditional).sum(-1)
        return velocity - velocity.mean(dim=-1, keepdim=True)


def tangent_project(velocity: torch.Tensor) -> torch.Tensor:
    return velocity - velocity.mean(dim=-1, keepdim=True)
