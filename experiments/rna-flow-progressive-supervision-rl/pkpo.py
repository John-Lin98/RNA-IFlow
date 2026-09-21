"""Pure continuous max@k PKPO group-relative advantages.

The implementation intentionally follows the subset definition of
``sloo_minus_one`` from Walder & Karkhanis (arXiv:2505.15201).  D3 only ever
uses eight candidates, so direct enumeration is both clearer and less error
prone than a rank-based shortcut.
"""

from __future__ import annotations

from itertools import combinations
import math

import torch


def continuous_maxk_advantages(rewards: torch.Tensor, k: int) -> torch.Tensor:
    """Return per-candidate continuous max@k ``sloo_minus_one`` advantages.

    For ``k > 1`` this is exactly
    ``1/C(n,k) sum_{I: |I|=k, i in I}(max(g_I) - max(g_{I\\{i}}))``.  The D3
    ``k=1`` stage is deliberately the experiment-contract LOO raw-reward
    centering, rather than attempting to evaluate a nonexistent max over an
    empty set.
    """
    if (
        not isinstance(rewards, torch.Tensor)
        or rewards.ndim != 1
        or rewards.numel() < 2
        or type(k) is not int
        or not 1 <= k <= rewards.numel()
        or not torch.isfinite(rewards).all()
    ):
        raise ValueError("PKPO rewards must be a finite one-dimensional group and valid k")
    n = int(rewards.numel())
    if k == 1:
        # This intentionally has denominator n - 1, not k - 1.
        return rewards - (rewards.sum() - rewards) / float(n - 1)

    values: list[torch.Tensor] = []
    indices = range(n)
    for candidate in indices:
        contributions: list[torch.Tensor] = []
        for others in combinations([index for index in indices if index != candidate], k - 1):
            subset = (candidate, *others)
            subset_values = rewards[list(subset)]
            other_values = rewards[list(others)]
            contributions.append(subset_values.max() - other_values.max())
        if not contributions:
            raise AssertionError("valid PKPO subset geometry produced no subsets")
        # Eq. ``sloo_minus_one`` normalizes by *all* size-k subsets, not by
        # the C(n-1, k-1) subsets containing this candidate.  The two differ
        # by k/n except for k=n.
        values.append(torch.stack(contributions).sum() / float(math.comb(n, k)))
    result = torch.stack(values)
    if not torch.isfinite(result).all():
        raise RuntimeError("PKPO advantages are non-finite")
    return result


def pkpo_schedule_k(update: int) -> int:
    """Frozen D3-256 update schedule, indexed from zero."""
    if type(update) is not int or not 0 <= update < 256:
        raise ValueError("D3 PKPO schedule update must be in [0, 255]")
    return 8 if update <= 84 else 4 if update <= 169 else 1
