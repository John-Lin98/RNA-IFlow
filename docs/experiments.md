# Experiment-to-paper map

This map records verified assets without substituting engineering smoke tests for paper experiments. Final manuscript table/figure numbering, baseline tables, ablation data, and figure source-data mapping are still pending reconciliation; this is not a complete reproduction release.

## Frozen main-model protocol

Checkpoint: C3+D5 U2442, original SHA256 `198fd79e7680f4b01e758f063aadab00c0d4e6709ac4d2a220249a267a70ebe8`.

HF package: [jojojoojooo/RNA-IFlow-RL](https://huggingface.co/jojojoojooo/RNA-IFlow-RL/tree/774a352f55016ff32012d2936d629b12bd6045dc), private. Uploaded revision is pinned here; fresh-download validation remains pending.

All main results use seed/temperature pairs `(1009, 0.8)`, `(2027, 1.0)`, `(3037, 1.2)`, eight candidates per task and condition, eight transition steps, and ViennaRNA 2.7.2. Pass@k uses `uMFE_hit`; MFE@k uses `mfe_hit`. Averages across these conditions do not establish training-seed uncertainty. No unapproved SD table format is frozen.

| Paper asset | Code | Config / evidence | Result |
|---|---|---|---|
| Main-model benchmark rows: Eterna100-v2, Eterna100, Rfam-Taneda-27, RNAsolo-764 | `experiments/rna-flow-fair-components/evaluate.py`; `scripts/aggregate_main_results.py` | `results/provenance/main_evaluation_receipts.json`; HF `evaluation_protocol.json` | `results/tables/main_model_metrics.csv`; `results/tables/per_condition_metrics.csv` |
| Table 1 benchmark comparisons and RNAsolo supplement (snapshot mapping; latest approval pending) | `scripts/package_benchmark_table.py`; individual baseline generation adapters still to be packaged | Candidate SHA and reaggregation checks in `results/tables/benchmark_quality/benchmark_quality_provenance.json` | `results/tables/benchmark_quality/benchmark_quality.csv` (26 rows, no SD formatting) |
| Table 2 resident neural runtime rows | `scripts/aggregate_resident_runtime.py`; original timing execution adapter still to be packaged | Ledger and summary SHA plus measurement plan in `results/tables/resident_runtime/provenance.json` | `results/tables/resident_runtime/resident_runtime.csv` (five methods; excludes native-search rows) |
| H/G sensitivity figure inputs (R7 source package; final manuscript mapping pending) | `scripts/package_hg_source_data.py` | 24 summary hashes and available contract hashes; eight source rows omit contract references | `results/figures/source_data/hg/HG_per_seed.csv`; `HG_mean_sd.csv`; `provenance.json` |
| Supervised structure-conditioned flow | `experiments/rna-flow-fair-components/model.py`; `experiments/dual-prior-rna-flow/flow_core.py`; `experiments/rna-flow-progressive-supervision-rl/train_progressive.py` | Full portable training recipe pending | Supervised comparison table pending |
| Finite-policy RL post-training | `experiments/rna-flow-progressive-supervision-rl/endpoint_policy.py`, `rl_primitives.py`, `train_endpoint_trajectory_grpo.py` | U2442 formal contract in `results/provenance/contract.json` | Main-model rows above; ablation mapping pending |
| Independent inference export validation (engineering, not a paper result) | `scripts/export_model.py`; `scripts/check_portable_equivalence.py`; `scripts/load_export.py` | HF `export_manifest.json` and model SHA | 223 tensor states equal; fixed mini-batch max absolute output difference 0 |

## Reaggregate main-model results

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

H/G error bars use sample SD across three continuation training seeds, not the three evaluation-temperature conditions. Reaggregation of 24 rows into eight settings matches the R7 figure inputs. These are figure source data, not an approved Table 1 SD layout. Missing source contract references and full experimental config closure remain documented release work.

The benchmark-quality export rechecks candidate SHA, seed/task/K coverage and all four success metrics for every row. Other quality and parameter columns are preserved from the source table and have not been independently recomputed by that packaging script. Eight returned candidates do not imply equal compute budgets between search and neural generators; `compute_budget_matched` is retained explicitly.

U2790 is plateau evidence only, not the main checkpoint. Fixed-T1 results, historical failed runs, raw result trees, structure caches, private operational logs, and duplicate checkpoints are excluded from this release. Benchmark redistribution and third-party model licensing must be resolved before public publication.
