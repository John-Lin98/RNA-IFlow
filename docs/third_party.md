# Third-party rights and license scope

The rights holder licenses this project's original code and documentation under Apache-2.0. Third-party dependencies, models and datasets retain their own terms and are not relicensed by the root `LICENSE`. GitHub and HF remain private until explicit publication approval.

`results/provenance/direct_dependency_licenses.json` records the installed versions, license classifiers and SHA256 of license files shipped with the seven pinned direct dependencies. Regenerate in the installed environment with:

```bash
python scripts/inspect_dependency_licenses.py --output /path/to/new-license-evidence.json
```

The script checks version pins and requires a fresh output. It records evidence only; it does not classify compatibility or grant rights. Dependency wheel contents are not bundled in this repository.

The inspected ViennaRNA 2.7.2 wheel includes `COPYING` with specific credit and redistribution terms, and refers to a separate copyright for `naview.c`. It is not represented here as simply Apache-2.0 or BSD. Its text must be included in the full dependency/redistribution review. The safetensors wheel includes an Apache-2.0 license text despite an empty license metadata header; missing metadata is not evidence that a package has no license.

Recorded boundaries:

- Full transitive and bundled-native dependency review, including required notices.
- RNAErnie source/model redistribution relies on its Apache-2.0 model-card metadata; its attribution and notices must remain with public model distribution.
- Dataset and benchmark acquisition, version identity, attribution and redistribution permissions. No dataset redistribution is assumed.
- English author names, affiliations and ORCIDs will be aligned with final paper metadata.

The included GitHub source package passes its license-scope check because dependencies and datasets are referenced rather than bundled. Do not change repository visibility until explicit publication approval.
