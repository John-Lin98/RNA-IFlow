"""Small analytic checks of the paper's finite policy; no checkpoints or GPU."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/rna-flow-progressive-supervision-rl'))
from endpoint_policy import (
    LEGAL_PAIR_IDS, discrete_domino_dirichlet_mean_bridge,
    discrete_domino_transition_unit_log_probabilities,
    discrete_mixture_replacement_probability, domino_endpoint_ppo_loss,
)
from rl_primitives import official_terminal_reward, normalized_group_advantages


class FinitePolicyTest(unittest.TestCase):
    def test_bridge(self):
        tokens = torch.tensor([[0, 1, 2, 3]])
        self.assertTrue(torch.equal(discrete_domino_dirichlet_mean_bridge(tokens, 1), torch.full((1, 4, 4), .25)))
        expected = (1 + 7 * torch.nn.functional.one_hot(tokens, 4)) / 11
        torch.testing.assert_close(discrete_domino_dirichlet_mean_bridge(tokens, 8), expected)

    def test_transition_normalization_and_stay(self):
        for structure, outcomes in [('.', torch.arange(4)[:, None]), ('()', LEGAL_PAIR_IDS)]:
            n = len(outcomes)
            current = outcomes[:1].expand(n, -1)
            logits = torch.zeros(n, len(structure), 4)
            for step in (0, 7):
                rho = discrete_mixture_replacement_probability(step, 8)
                probs = discrete_domino_transition_unit_log_probabilities(
                    logits, current, outcomes, structure, .8, rho).exp().flatten()
                expected = torch.full((n,), rho / n)
                expected[0] += 1 - rho
                torch.testing.assert_close(probs, expected)
                torch.testing.assert_close(probs.sum(), torch.tensor(1.))
        self.assertEqual(discrete_mixture_replacement_probability(7, 8), 1.)

    def test_reward_and_joint_time_sum(self):
        self.assertEqual(official_terminal_reward(dict(target_probability=.4, mfe_hit=True, uMFE_hit=False)), .45)
        advantages, effective = normalized_group_advantages(torch.ones(8))
        self.assertFalse(effective)
        self.assertTrue(torch.equal(advantages, torch.zeros(8)))
        old = torch.zeros(2, 3, 2)
        new = torch.full_like(old, .1, requires_grad=True)
        advantage = torch.tensor([1., -1.])
        loss, diagnostics = domino_endpoint_ppo_loss(new, old, advantage, .2)
        ratio = torch.exp(torch.tensor(.2))  # sum two unit log-ratios
        expected = -3 * (torch.minimum(ratio, torch.tensor(1.2)) - ratio) / 2
        torch.testing.assert_close(loss, expected)
        self.assertEqual(diagnostics['time_aggregation'], 'sum')
        loss.backward()
        self.assertTrue(torch.isfinite(new.grad).all())


if __name__ == '__main__':
    unittest.main()
