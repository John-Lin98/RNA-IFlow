# Third-party rights: evidence and unresolved scope

This release has **not** passed redistribution clearance. GitHub and HF remain private. No project LICENSE is inferred from dependency metadata.

`results/provenance/direct_dependency_licenses.json` records the installed versions, license classifiers and SHA256 of license files shipped with the seven pinned direct dependencies. Regenerate in the installed environment with:

```bash
python scripts/inspect_dependency_licenses.py --output /path/to/new-license-evidence.json
```

The script checks version pins and requires a fresh output. It records evidence only; it does not classify compatibility or grant rights. Dependency wheel contents are not bundled in this repository.

The inspected ViennaRNA 2.7.2 wheel includes `COPYING` with specific credit and redistribution terms, and refers to a separate copyright for `naview.c`. It is not represented here as simply Apache-2.0 or BSD. Its text must be included in the full dependency/redistribution review. The safetensors wheel includes an Apache-2.0 license text despite an empty license metadata header; missing metadata is not evidence that a package has no license.

Remaining decisions/evidence:

- Rights-holder authorization and a project license for the original method code.
- Full transitive and bundled-native dependency review, including required notices.
- RNAErnie source/model redistribution terms and obligations for the exported derived weights; model-card metadata alone does not close this gate.
- Dataset and benchmark acquisition, version identity, attribution and redistribution permissions. No dataset redistribution is assumed.
- Approved author/citation metadata. Private upload is not described as public distribution approval.

Until these are resolved, do not mark the release license audit PASS, publish the repositories, or imply unrestricted downstream use.
