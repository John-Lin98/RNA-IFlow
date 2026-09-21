# Dataset identity and interpretation limits

The existing streaming data audit is packaged in `results/provenance/dataset_composition.json`. Release preparation rechecks its referenced small-file hashes and aggregate counts; it does not rerun the 10M-row overlap scan or rehash the large Arrow file. These evidence scopes remain distinct.

## Training provenance

SFT data originate from `Milanmg/LLM-RNA-Design-2026`, revision `609f573b1a373a4f4e6748057c90dcca07e02294`. The recorded builder retains a frozen 100K prefix, fills the nested prefixes from published shards, filters invalid/overlength samples and excluded structures, and removes duplicate structure/sequence pairs. The resulting training set has 10,000,000 rows and observed lengths 6–500 nt. These are published solver-generated pairs, not a claim of 10M natural RNA molecules; upstream solver budgets and composition remain unverified.

The 2,790 RL targets use the published YRL EternaWeb set without method-specific resampling. The existing audit records exact ID/order/structure mapping and unique structures. Auxiliary RL CE uses the frozen 100K supervised JSONL, not a new 10M batch source. Training parameters and asset hashes are in [training](training.md) and the recorded contracts.

## Benchmark limitations

| Set | Targets | Exact target structures overlapping SFT | Matching SFT rows | Overlap with RL2790 |
|---|---:|---:|---:|---:|
| Eterna100-v2 | 100 | 0 | 0 | 0 |
| Eterna100 | 100 | 0 | 0 | 0 |
| Rfam-Taneda-27 | 27 | 0 | 0 | 0 |
| RNAsolo-764 | 764 | 9 | 90 | 0 |

**Eterna100-v2 participated in historical checkpoint/method/hyperparameter selection.** It must not be described as a purely untouched held-out test. Zero structure overlap does not remove model-selection bias.

This role is corroborated by the historical route-selection contract (SHA256 `1b44f9ef61f6bd9798d4f38a54a91befe4d7e803cde25ae080f1c1f5074f3608`, Eterna100-v2 marked eligible) and the experiment authorization record (SHA256 `1ffabfee4fadaba5f1420c16967d6292844d584a3cc82edaae19a3d6844cc1e1`, explicitly allowing Eterna-based parameter selection). Both hashes were rechecked during packaging; private operational records are not redistributed.

**RNAsolo-764 is the frozen main reporting set and has the overlap shown above.** Do not silently substitute a clean subset for its main score or claim the 764 set has no training overlap. Exact full dot-bracket matching is not a sequence-homology or family-disjointness audit. The separate historical sensitivity analysis is in `results/tables/rnasolo_overlap/sensitivity.csv`: five methods, each with full764, exact-SFT-disjoint755 and overlap9 subsets. Membership uses recorded training-structure overlap only, not model outcomes. For RNA-IFlow-RL, full764 Pass@1/8 is 0.7120419/0.7312391; the 755 subset is 0.7086093/0.7280353. Release preparation rechecks five task-condition source hashes, complete task/seed coverage and eight aggregate metrics; it does not repeat the 10M-row overlap scan or candidate scoring. The provenance retains excluded task IDs and structure hashes without redistributing raw structures.

The benchmark adapter requires the original benchmark bytes and order, bound by `results/provenance/main_evaluation_receipts.json`. No dataset or raw candidate tree is redistributed in this repository. Full acquisition/build instructions and redistribution permission remain outstanding release gates; dataset names and hashes alone do not make the release turnkey.
