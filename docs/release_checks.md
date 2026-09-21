# Release checks and outstanding gates

GitHub and Hugging Face must remain private until explicit publication approval.

## Fast regression tests

Run `python -m unittest discover -s tests -v` in an isolated environment. The suite covers frozen benchmark/condition identity, scoring, finite-policy arithmetic, export configuration identity, runtime-source admission and RNAinverse failure semantics. Tests do not replace real model loading or full benchmark reproduction; current counts and results are reported by the command rather than frozen in this document.

## Inventory

`test_finite_policy.py` checks bridge values, finite transition normalization/stay mass, terminal replacement, reward arithmetic and joint-ratio/time-sum PPO arithmetic. The configuration-identity test rejects activation and byte tampering. Actual fresh-download package loading also passed. The independent review closed the configuration-binding issue; full-benchmark equivalence and asset-acquisition gaps remain open in `results/reports/release_review_zh.md`.

Run `python scripts/update_inventory.py` from any directory. It indexes only tracked and non-ignored files within this release repository, excludes its own self-referential inventory entry, rejects model binaries and files larger than 5 MB, and records destination SHA and size separately from upstream source SHA and size. Original-source hashes are not rewritten after portability adaptations. Release-authored files use logical `release:` paths, not private server paths. A complete current-file inventory is not evidence that all required paper assets have been included.

## License

No root LICENSE exists in the inspected authoritative source revision. A license for the new project cannot be inferred from dependency licenses or repository ownership. The RNAErnie model card declares Apache-2.0 metadata, but this alone does not complete the code, dataset, or derived-weight redistribution review. Author/rights-holder approval and third-party notices remain required. No LICENSE is invented, and no redistribution permission is asserted.

Installed direct-dependency version and license-file evidence is recorded in `results/provenance/direct_dependency_licenses.json`; see [third-party scope](third_party.md). This is a partial evidence ledger, not a completed compatibility or redistribution audit.

## Manuscript and timing

The inspected Table 1 snapshot agrees with the source benchmark-success table at displayed precision. The latest manuscript approval is not established.

The Table 2 resident-model timing source has been reaggregated from 3,000 timing records: median across two repeats within each task/condition, then median across 300 groups. RNA-IFlow-RL is 0.73166435575 s/K8 and RNA-IFlow is 0.88450397525 s/K8, matching the displayed 0.73/0.88. See `results/tables/resident_runtime/`. Native-search runtime is packaged separately with failure counts retained; resident neural timing excludes setup/warmup and later scoring. The portable five-model execution entrypoint is `scripts/measure_resident_runtime.py`; its real-model GPU equivalence preflight remains distinct from CPU orchestration tests.

## Remaining release gates

- HF fresh-download SHA and strict-load/K8 smoke passed; evidence is in `results/provenance/hf_readback_qa.json`. Full benchmark reproduction remains separate.
- Remaining paper figure/source and approved baseline/ablation mapping closure.
- Portable full-benchmark reproduction (clean CPU installation and minimal inference have passed).
- Latest approved manuscript reconciliation and approved citation metadata.
- Rights-holder license decision and third-party redistribution audit.
- Final secret/size scan, Chinese PR and review, clean committed release branch.
