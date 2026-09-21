# Release checks and outstanding gates

GitHub and Hugging Face must remain private until explicit publication approval.

## Fast regression tests

Run `python -m unittest discover -s tests -v` in an isolated environment. The suite covers frozen benchmark/condition identity, scoring, finite-policy arithmetic, export configuration identity, runtime-source admission and RNAinverse failure semantics. Tests do not replace real model loading or full benchmark reproduction; current counts and results are reported by the command rather than frozen in this document.

## Inventory

`test_finite_policy.py` checks bridge values, finite transition normalization/stay mass, terminal replacement, reward arithmetic and joint-ratio/time-sum PPO arithmetic. The configuration-identity test rejects activation and byte tampering. Actual fresh-download package loading also passed. The independent review closed the configuration-binding issue; full-benchmark equivalence and asset-acquisition gaps remain open in `results/reports/release_review_zh.md`.

Run `python scripts/update_inventory.py` from any directory. It indexes only tracked and non-ignored files within this release repository, excludes its own self-referential inventory entry, rejects model binaries and files larger than 5 MB, and records destination SHA and size separately from upstream source SHA and size. Original-source hashes are not rewritten after portability adaptations. Release-authored files use logical `release:` paths, not private server paths. A complete current-file inventory is not evidence that all required paper assets have been included.

## License

The rights holder selected Apache-2.0 for this project. The complete license is stored in the root `LICENSE`, and `CITATION.cff` records the approved Chinese author order. Installed direct-dependency version and license-file evidence is recorded in `results/provenance/direct_dependency_licenses.json`; see [third-party scope](third_party.md). Dependencies and datasets are not bundled or relicensed by this repository.

## Manuscript and timing

The inspected Table 1 snapshot agrees with the source benchmark-success table at displayed precision. The official protocol is fixed to seed/temperature pairs `(1009, 0.8)`, `(2027, 1.0)`, and `(3037, 1.2)`, K=8, with ViennaRNA 2.7.2.

The Table 2 resident-model timing source has been reaggregated from 3,000 timing records: median across two repeats within each task/condition, then median across 300 groups. RNA-IFlow-RL is 0.73166435575 s/K8 and RNA-IFlow is 0.88450397525 s/K8, matching the displayed 0.73/0.88. See `results/tables/resident_runtime/`. Native-search runtime is packaged separately with failure counts retained; resident neural timing excludes setup/warmup and later scoring. The portable five-model execution entrypoint is `scripts/measure_resident_runtime.py`; its real-model GPU equivalence preflight remains distinct from CPU orchestration tests.

## Current release gates

Completed engineering gates: HF fresh-download SHA and strict-load/K8 smoke; clean-environment installation and 17 CPU regression tests; compact table/figure source packaging with provenance; paper-only inventory; final tracked-file path/secret/size scan; clean pushed release branch; and a Chinese Draft PR. Full benchmark or CUDA timing reruns remain optional reproduction strengthening and are not represented as completed experiments.

The project license, author order and official evaluation protocol are now approved. English author names, affiliations and ORCIDs remain a metadata refinement rather than a code-release blocker. GitHub and Hugging Face remain private until explicit publication approval.
