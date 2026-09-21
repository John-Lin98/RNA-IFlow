# RNA Design via Conditioned Flow Matching and Finite-Policy Reinforcement Learning

RNA-IFlow is a structure-conditioned RNA inverse-folding framework. RNA-IFlow-RL adds finite-policy reinforcement learning post-training.

Paper and method figure: forthcoming.

This private repository is being prepared for reproducible paper release. The release branch is work in progress, not yet a validated public release.

Main model: [RNA-IFlow-RL on Hugging Face](https://huggingface.co/jojojoojooo/RNA-IFlow-RL) (private). The U2442 inference export is uploaded; independent download verification is pending. See [model reference](model/README.md).

## Installation and inference

Use Python 3.10 in an isolated environment:

```bash
python -m pip install -r requirements.txt
python scripts/load_export.py /path/to/downloaded-model
python scripts/infer.py --model /path/to/downloaded-model --structure '(((...)))' --candidates 8 --seed 1009 --temperature 0.8
```

The model directory must contain the HF package, including its config and export manifest. CPU inference has passed a local smoke test; a clean installation remains to be validated. See [inference details](docs/inference.md).

## Main results

| Benchmark | Pass@1 | Pass@8 |
|---|---:|---:|
| Eterna100-v2 | 0.5400 | 0.6500 |
| Eterna100 | 0.5066667 | 0.6166667 |
| Rfam-Taneda-27 | 0.8518519 | 0.8765432 |
| RNAsolo-764 | 0.7120419 | 0.7312391 |

U2442; arithmetic mean over seed/temperature conditions `(1009, 0.8)`, `(2027, 1.0)`, `(3037, 1.2)`, K=8, ViennaRNA 2.7.2. Pass uses the unique-MFE success indicator. These are evaluation conditions, not independent training seeds. Source CSVs are in `results/tables/`; original evaluation receipt provenance is in `results/provenance/`. These values match the frozen release specification; latest approved manuscript reconciliation remains pending.

## Reproduction and layout

See [experiment map](docs/experiments.md) for entrypoints and current reproduction boundaries. No datasets or model binaries are stored in GitHub.

- `experiments/`: source implementation and its dependency closure.
- `scripts/`: export, portable inference, equivalence checks, and aggregation.
- `results/tables/`: compact main-model and per-condition results.
- `results/provenance/`: contracts and evaluation receipts.
- `docs/`, `model/`: reproduction and model references.

## Citation and license

Paper link, final author list, and citation metadata will be added after approval; no provisional authorship or venue is asserted.

Licensing and third-party redistribution review are in progress. No public release or redistribution permission is asserted by this scaffold.
