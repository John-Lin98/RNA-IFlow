# Experiment-to-paper map

This map records verified assets without substituting engineering smoke tests for paper experiments. Final manuscript table/figure numbering, baseline tables, ablation data, and figure source-data mapping are still pending reconciliation; this is not a complete reproduction release.

## Frozen main-model protocol

Checkpoint: C3+D5 U2442, original SHA256 `198fd79e7680f4b01e758f063aadab00c0d4e6709ac4d2a220249a267a70ebe8`.

HF package: [jojojoojooo/RNA-IFlow-RL](https://huggingface.co/jojojoojooo/RNA-IFlow-RL/tree/774a352f55016ff32012d2936d629b12bd6045dc), private. Independent fresh-download SHA, strict load and clean-environment inference smoke passed; see `results/provenance/hf_readback_qa.json`.

The [method implementation map](methods.md) identifies FPM, SPD and TTR functions and distinguishes logical updates, optimizer steps, training group size and inference candidate budget.

See [dataset limitations](datasets.md) before interpreting these results: Eterna100-v2 participated in historical selection, and RNAsolo-764 has nine exact target structures overlapping SFT. Main values and benchmark membership remain unchanged; they are not claimed to establish untouched-test generalization.

All main results use seed/temperature pairs `(1009, 0.8)`, `(2027, 1.0)`, `(3037, 1.2)`, eight candidates per task and condition, eight transition steps, and ViennaRNA 2.7.2. Pass@k uses `uMFE_hit`; MFE@k uses `mfe_hit`. Averages across these conditions do not establish training-seed uncertainty. No unapproved SD table format is frozen.

| Paper asset | Code | Config / evidence | Result |
|---|---|---|---|
| Main-model benchmark rows: Eterna100-v2, Eterna100, Rfam-Taneda-27, RNAsolo-764 | `experiments/rna-flow-fair-components/evaluate.py`; `scripts/aggregate_main_results.py` | `results/provenance/main_evaluation_receipts.json`; HF `evaluation_protocol.json` | `results/tables/main_model_metrics.csv`; `results/tables/per_condition_metrics.csv` |
| Table 1 benchmark comparisons and RNAsolo supplement (snapshot mapping; latest approval pending) | `scripts/package_benchmark_table.py`; individual baseline generation adapters still to be packaged | Candidate SHA and reaggregation checks in `results/tables/benchmark_quality/benchmark_quality_provenance.json` | `results/tables/benchmark_quality/benchmark_quality.csv` (26 rows, no SD formatting) |
| Table 2 resident neural runtime rows | `scripts/measure_resident_runtime.py`; `runtime_samplers.py`; `aggregate_resident_runtime.py` | Frozen asset/candidate SHA, five-model resident protocol, ledger and summary SHA in `results/tables/resident_runtime/provenance.json`; portable runner CUDA preflight pending | `results/tables/resident_runtime/resident_runtime.csv` (five methods; excludes native-search rows) |
| Appendix RNAsolo exact-overlap sensitivity | `scripts/package_overlap_sensitivity.py --source /path/to/historical-sensitivity --tables /path/to/task-condition-tables --output /path/to/fresh-summary` | Historical overlap/result receipt hashes, excluded IDs/structure hashes, five task-condition source hashes; no new SFT scan | `results/tables/rnasolo_overlap/sensitivity.csv`; `provenance.json` (15 rows; full764 remains primary) |
| Table 2 native-search runtime rows | `scripts/aggregate_native_runtime.py`; portable `scripts/measure_inverse_runtime.py`; DRAG wrapper pending | Summary/plan SHA and failure semantics in `results/tables/native_runtime/provenance.json`; [execution protocol](runtime_protocol.md) | `results/tables/native_runtime/native_runtime.csv` |
| H/G sensitivity figure inputs (R7 source package; final manuscript mapping pending) | `scripts/package_hg_source_data.py`; `scripts/resolve_hg_contracts.py` | 24 summary hashes; all 24 contracts and terminal receipts verified, including eight references resolved separately | `results/figures/source_data/hg/HG_per_seed.csv`; `HG_mean_sd.csv`; `contract_resolution.json` |
| Training composition figure (R7 source package) | `scripts/package_composition_data.py`; original membership-construction recipe pending | Four membership-source hashes, evaluation summary hashes and semantic audit | `results/figures/source_data/composition/composition_counts.csv`; `composition_quality.csv`; `provenance.json` |
| H/G sensitivity cost axes | `scripts/package_hg_cost.py` | H values checked against recorded 12-target matched timing summary; G medians reaggregated from 64 updates and receipt hashes checked | `results/figures/source_data/hg_cost/HG_cost.csv`; `provenance.json` |
| Late-checkpoint plateau curve (four saved evaluations) | `scripts/package_checkpoint_curve.py` | 400 candidate-file hashes, complete 100-task × 3-condition × K8 coverage at each checkpoint; summary protocol and metric agreement | `results/figures/source_data/checkpoint_curve/late_checkpoints.csv`; `provenance.json` |
| Supervised structure-conditioned flow | `experiments/supervised_snapshot/` preserves the original seven-file dependency closure | `results/provenance/supervised_training_contract.json`; `supervised_source_manifest.json`; `docs/training.md` | Epoch-six parent; full asset acquisition remains pending |
| Finite-policy RL post-training | `experiments/rna-flow-progressive-supervision-rl/endpoint_policy.py`, `rl_primitives.py`, `train_endpoint_trajectory_grpo.py` | U2442 formal contract in `results/provenance/contract.json` | Main-model rows above; ablation mapping pending |
| Independent inference export validation (engineering, not a paper result) | `scripts/export_model.py`; `scripts/check_portable_equivalence.py`; `scripts/load_export.py` | HF `export_manifest.json` and model SHA | 223 tensor states equal; fixed mini-batch max absolute output difference 0 |

## Reaggregate main-model results

### Render H/G sensitivity panels

```bash
pip install -r requirements-figures.txt
python scripts/render_hg_panels.py --output /path/to/fresh-figure-output
```

The renderer produces Figure 2(d,e) PDF/PNG panels from the packaged H/G quality
and cost CSVs, with input/output hashes in `render_manifest.json`. Geometry,
colors, axis ranges and error-bar layering follow the R7 plotting source;
DejaVu Sans replaces Arial for portability, so pixel identity is not claimed.
Error bars remain sample SD over three continuation training seeds. Both panels
have been rendered in the independent CPU environment and visually checked for
legible labels and visible error bars. This is not the complete Figure 2, a new
experiment or approval of the final manuscript layout.

### Main-model aggregate tables

Given normalized candidate files named `RNA-IFlow-RL__<benchmark>.jsonl`, run:

```bash
python scripts/aggregate_main_results.py --input /path/to/normalized-candidates --output /path/to/fresh-summary
```

Benchmark keys are `eterna100v2`, `eterna100`, `rfam27`, and `rnasolo764`. The script rejects incomplete seed/task/candidate coverage and failed evaluations, and checks the frozen main-model values. It does not regenerate candidates or calculate an approved final SD table. Required raw candidates are not redistributed here; hashes are recorded in `results/tables/aggregation_provenance.json`.

The single-target inference example is not a benchmark reproduction command. Use the portable main-model adapter with the original benchmark JSONL bytes (including task order):

```bash
python scripts/evaluate_benchmark.py --model /path/to/downloaded-model --benchmark eterna100v2 --tasks /path/to/eterna100v2.jsonl --output /path/to/fresh-evaluation --device cpu
```

Supported benchmark keys are `eterna100v2`, `eterna100`, `rfam27`, and `rnasolo764`. Their required SHA values are recorded in `results/provenance/main_evaluation_receipts.json`; modified/reordered input files are rejected. The adapter fixes K=8, H=8 and the three temperature conditions, with `condition_seed + original_task_index * 1000003`. Output uses the corrected paper NED, not the legacy extra normalization. Failed runs do not receive a completion summary; existing output directories are never overwritten.

Add `--smoke-first-task` for an explicitly smoke-only run preserving original index zero. CPU smoke has passed for one task and all 24 candidates. Full-dataset reproduction and device-level numerical equivalence have not been rerun for this release; no new benchmark claim is made. Dataset acquisition/redistribution instructions and baseline adapters remain pending.

## Exclusions

H/G cost axes are separate measurements from the three-seed quality error bars. H inference costs use 12 outcome-independent length-stratified targets, not the main Table 2 full100 timing run. G costs use two eight-update profiles per setting on shared hardware, including rollout, folding, backward, optimizer and ordinary rollout writes but excluding model loading and checkpoint export. Neither axis establishes training-seed cost uncertainty. The packaging script verifies summary agreement, 64-update median aggregation and source receipt hashes; it does not rerun timing or reconstruct the raw H timing ledger.

The [runtime protocol](runtime_protocol.md) records exact sampler settings, checkpoint hashes, interleaving, synchronization and native-search timeout semantics. Baseline temperatures are not replaced with the RNA-IFlow-RL temperature schedule. Timing execution dependencies remain an explicit release gap.

H/G error bars use sample SD across three continuation training seeds, not the three evaluation-temperature conditions. Reaggregation of 24 rows into eight settings matches the R7 figure inputs. These are figure source data, not an approved Table 1 SD layout. Eight references absent in the source table were resolved through sibling formal-training directories and checked against evaluation scientific-contract hashes, H/G/seed values and terminal receipts. `contract_resolution.json` preserves that additional evidence without rewriting the source table or its initial packaging provenance. Portable experiment recipes and checkpoint-level validation remain separate work.

The benchmark-quality export rechecks candidate SHA, seed/task/K coverage and all four success metrics for every row. Other quality and parameter columns are preserved from the source table and have not been independently recomputed by that packaging script. Eight returned candidates do not imply equal compute budgets between search and neural generators; `compute_budget_matched` is retained explicitly.

Native-search timing includes necessary setup/internal search and retains failed/timeout slots in its denominators. The RNAinverse-pf timing run returned 1,575 of 2,400 attempted candidates (825 failures/timeouts); this is separate from the original Table 1 accuracy ledger. Do not infer identical sequences, identical success rates, or compute-matched speedup from the timing table. Both native timing medians were recomputed over 300 task/condition groups each.

Composition source labels are preserved: `Original mix`, `Core-only`, and `Hard-enriched` (paper display labels: Original mix, Easy, Easy + Hard). The learnability-selected core is not an original-membership subset: only 1,237 of its 2,790 IDs overlap the original mix. Do not relabel that category as "original members." Each condition contains 2,790 targets, and counts/fractions and corrected summary metrics were verified. Full training-recipe and final figure-rendering closure remain pending.

U2790 is plateau evidence only, not the main checkpoint. Fixed-T1 results, historical failed runs, raw result trees, structure caches, private operational logs, and duplicate checkpoints are excluded from this release. Benchmark redistribution and third-party model licensing must be resolved before public publication.

The late-checkpoint curve contains U1744, U2093, U2442 and U2790, reaggregated from 9,600 original candidate records without rerunning inference. All four use the frozen multi-temperature protocol, H8, K8 and ViennaRNA 2.7.2. The curve does not select a new checkpoint: U2442 remains the main model even when another point has a larger observed score. Candidate sequences are not duplicated in the release; per-file hashes support traceability. The broader historical early curve is not included here: its update-zero point is an already-trained U96 policy, not supervised-only weights. Final figure rendering and the separate on-policy training-reward curve remain pending.
