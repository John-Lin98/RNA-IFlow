# Paper method and implementation map

This document maps RNA-IFlow / RNA-IFlow-RL terminology to the executable code and the main U2442 contract. Historical function names are retained for provenance; they are not additional paper methods.

| Component | Implementation | Main-model behavior |
|---|---|---|
| Structure-conditioned Dirichlet flow | `experiments/rna-flow-fair-components/model.py`, `experiments/dual-prior-rna-flow/flow_core.py`, `experiments/rna-flow-progressive-supervision-rl/train_progressive.py` | Structure-conditioned clean-endpoint predictor trained on Dirichlet-corrupted sequence states |
| Flow-to-Policy Mapping (FPM) | `endpoint_policy.py`: `discrete_domino_dirichlet_mean_bridge`, `discrete_mixture_replacement_probability`, `discrete_domino_transition_unit_log_probabilities` | Conditional-mean bridge and an explicitly defined finite keep-or-resample kernel |
| Structure-Preserving Policy Dynamics (SPD) | `endpoint_policy.py`: `endpoint_units`, `sample_discrete_domino_transition`, `rollout_discrete_domino_trajectory` | Four states per unpaired site; six legal states per target pair, at initialization and every transition |
| Thermodynamic Trajectory Refinement (TTR) | `rl_primitives.py`: `official_terminal_reward`, `normalized_group_advantages`; `endpoint_policy.py`: `domino_endpoint_ppo_loss`; `train_endpoint_trajectory_grpo.py` | Terminal reward, group-relative advantage, joint-transition ratios and time-summed clipped surrogate |

Unqualified module names in this table are under `experiments/rna-flow-progressive-supervision-rl/`.

## Finite policy

For current nucleotide state x and concentration alpha, the bridge supplies `(1 + (alpha - 1) onehot(x)) / (alpha + 3)` to the predictor. It is the conditional mean of the supervised corruption path, not a sampled corruption and not an assertion that the continuous-flow endpoint likelihood is tractable.

For zero-based step t in horizon H, `rho = 1 / (H - t)`. Each structural unit samples from `(1-rho) delta_current + rho p_clean`. A pair posterior renormalizes the product of its two nucleotide marginals over AU, UA, GC, CG, GU and UG. Staying in the current state includes both the keep probability and resampling that same state. The final step has rho=1. Target-pair legality does not guarantee that folding produces the target structure.

The exact probabilities are those of this explicit finite policy, not a native continuous-time flow likelihood or a marginal sequence likelihood. The implementation sums unit log-probability differences to form one joint ratio per transition; it does not average unit ratios. The PPO surrogate sums over time and averages over candidate trajectories. The numerical log-ratio bound is [-20,20], with a straight-through gradient for the bound.

## Main U2442 training contract

The path-free source contract is `results/provenance/contract.json`. It fixes:

- H=8; G=8 trajectories per task; four global tasks per logical update; DDP world size 2.
- Reward `0.5 * target_probability + 0.25 * mfe_hit + 0.25 * uMFE_hit` on complete terminal sequences only.
- Group advantages use population standard deviation plus 1e-4; a reward range below 1e-6 gives zero advantage.
- PPO clip 0.2; learning rate 5e-6; weight decay 0.01; gradient norm clip 1.
- Frozen U96 reference, KL coefficient 0.01 over structured clean-update distributions; supervised CE coefficient 0.1 with global batch 32; entropy coefficient 0.
- Last two backbone blocks and output head trainable. All transitions receive the same terminal trajectory advantage; no single-step causal estimator or reference-transition TV penalty is used in this main run.
- Two rollout refreshes and two optimizer steps per logical update. A logical update produces 64 candidate rollouts in total and advances the task cursor by four; checkpoints lie after both refreshes.
- The recorded segment continues from U2093 to U2442, restoring optimizer, per-rank RNG, cursor and coverage. U2442 denotes cumulative logical updates, not this segment's update count or optimizer steps.

The initial scaled policy used U96 weights with fresh optimizer state; later continuation segments restore optimizer state. Do not conflate these two boundaries. The exported HF package is inference-only and cannot replace a continuation checkpoint.

## Verification and reproduction limits

`tests/test_finite_policy.py` checks the conditional-mean bridge, finite-kernel normalization and stay mass, terminal replacement, terminal reward and clipped joint-ratio/time-sum arithmetic without weights or GPUs. These checks complement, rather than replace, checkpoint export equivalence and benchmark evaluation.

The main evaluator uses the three frozen seed/temperature pairs and K=8; G above is a training group size. See [experiments](experiments.md) for result mappings. A complete portable training-data/parent-checkpoint recipe and all paper ablation adapters remain release gates; this implementation map is not a claim that those assets are already complete.
