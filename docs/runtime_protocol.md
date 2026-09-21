# Table 2 runtime protocol

This describes the recorded experiment, not a new timing result. Aggregate scripts
are portable; the multi-model execution adapter and external model dependencies
are not yet fully packaged. Single-model inference is not a replacement for this comparison.

## Resident neural comparison

Input: Eterna100-v2, original 100-target order, SHA256
`7514e053c8044d2dc96909e40383474375add1e8b926aa19417980a1e37c9412`.
Each target has seeds 1009, 2027 and 3037, K8, two repetitions. All five models
reside simultaneously on one shared A100. PyTorch uses two CPU threads; subsequent
ViennaRNA scoring uses four processes. This is not isolated-device timing.

| Method | Frozen sampler | Checkpoint SHA256 |
|---|---|---|
| RNA-IFlow | Native continuous flow, 50 steps | `8e221cc4c4382a5421890724c0548cf64492fe1be4556d498878e86e83056bc5` |
| RNA-IFlow-RL | U2442, H8, seed/T 1009/.8, 2027/1, 3037/1.2 | `198fd79e7680f4b01e758f063aadab00c0d4e6709ac4d2a220249a267a70ebe8` |
| RNA-Design-LM SL | Official model, T=2, condition seed without target offset | `a466271bfdbf108bbb55aa19a30cdc485b4ce2bbb2d648cae1d85e29a5aabe8c` |
| RNA-Design-LM SL+RL | Official model, T=2, condition seed without target offset | `970a3132fcb64c95141faa3c3bc041eccf38ecfa8dc2890936cdbd467d041741` |
| GoForth | pretrained_small, T=.1, batch8, condition seed | `a28e650ba0a8fd61a92ade424d939f3d95631df63f6f9f2ae78a5f81932b472f` |

RNA-IFlow and RNA-IFlow-RL use `condition_seed + original_task_index * 1000003`.
The three temperature values describe RNA-IFlow-RL, **not all baseline samplers**.

1. Load all models and precompute the flow lookup. Warm each sampler on the longest
   target with seed1009; require exact equality to its frozen candidate sequences.
2. Repetition zero traverses original target order. Rotate methods from table order
   by `(target_loop_index * 3 + condition_index) % 5`.
3. Repetition one reverses target order and reverses each rotated method order.
4. Synchronize CUDA before timing and after eight sequences return. Use
   `perf_counter_ns`; exclude loading, lookup setup and warmup.
5. Reject any difference from the eight frozen candidate sequences. Separately
   time uniform post-scoring; do not include it in generation latency.
6. Take the median of two repetitions per target/condition, then the median of
   those 300 groups—not a pooled median of 600 timings or a 12-target pilot.

The recorded run has 3,000 rows, zero candidate identity mismatches and zero
generation-failure groups. `results/tables/resident_runtime/` holds compact results
and hashes. Original execution source SHA256 was rechecked:
`aa4c2e13e02ec2b9b8f0b745281acc535e001f8eba8a37eb2f08911a579c161e`.
See `scripts/aggregate_resident_runtime.py --help` for source-aggregation inputs.

## Native-search supplement

Native time includes required process/model setup and internal folding; subsequent
common scoring is separate. It is not compute-matched resident neural latency.

- RNAinverse-pf: eight sequential fresh Python processes per group, `RNA.inverse_pf_fold`,
  45-second timeout per candidate. Draw starts from `AUGC` using
  `random.Random(condition_seed * 1000003 + target_index)`; run four groups concurrently.
  ViennaRNA internal RNG stays at its library default. Frozen starts do not establish
  exact output identity.
- DRAG: one group at a time, K8, one episode, 30 steps, eval seed equal to condition
  seed, init seed zero, no extra trajectory/final refinement, task pool enabled,
  slack threshold450, fold version2, freeze enabled, 360-second group timeout.
- Empty/timeout slots remain in denominators. RNAinverse returned 1,575/2,400
  candidates; this timing run is separate from the original Table1 accuracy ledger.

See `scripts/aggregate_native_runtime.py` and
`results/tables/native_runtime/provenance.json`. Historical execution source SHA256:
`175dd6f32085e3fa8d24f9cb7484e36f16d63ce1204ddd39ddc9b3135b2463f2`.
External wrappers, upstream revisions, environments and redistribution permissions
must still be resolved before claiming turnkey timing reproduction.
