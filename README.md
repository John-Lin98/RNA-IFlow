# RNA Design via Conditioned Flow Matching and Finite-Policy Reinforcement Learning

RNA-IFlow is a structure-conditioned RNA inverse-folding framework. RNA-IFlow-RL adds finite-policy reinforcement learning post-training.

Paper and method figure: forthcoming.

This private repository is being prepared for reproducible paper release. The release branch is work in progress, not yet a validated public release.

Main model: [RNA-IFlow-RL on Hugging Face](https://huggingface.co/jojojoojooo/RNA-IFlow-RL) (private). The U2442 inference export is uploaded; an independent fresh download, SHA verification, strict load and K8 CPU smoke passed. See [model reference](model/README.md).

## Installation and inference

Use Python 3.10 in an isolated environment:

```bash
python -m pip install -r requirements.txt
python scripts/load_export.py /path/to/downloaded-model
python scripts/infer.py --model /path/to/downloaded-model --structure '(((...)))' --candidates 8 --seed 1009 --temperature 0.8
```

The model directory must contain the HF package, including its config and export manifest. Independent download SHA checks and CPU inference in a clean installation passed. See [inference details](docs/inference.md).

## Main results

| Benchmark | Pass@1 | Pass@8 |
|---|---:|---:|
| Eterna100-v2 | 0.5400 | 0.6500 |
| Eterna100 | 0.5066667 | 0.6166667 |
| Rfam-Taneda-27 | 0.8518519 | 0.8765432 |
| RNAsolo-764 | 0.7120419 | 0.7312391 |

U2442; arithmetic mean over seed/temperature conditions `(1009, 0.8)`, `(2027, 1.0)`, `(3037, 1.2)`, K=8, ViennaRNA 2.7.2. Pass uses the unique-MFE success indicator. These are evaluation conditions, not independent training seeds. Source CSVs are in `results/tables/`; original evaluation receipt provenance is in `results/provenance/`. These are the official evaluation results for this release.

## Reproduction and layout

See [experiment map](docs/experiments.md), [training recipe](docs/training.md), and [dataset limitations](docs/datasets.md) for reproduction boundaries. Eterna100-v2 participated in historical model selection; RNAsolo-764 includes nine target structures overlapping SFT. No datasets or model binaries are stored in GitHub.

- `experiments/`: source implementation and its dependency closure.
- `scripts/`: export, portable inference/evaluation, training entrypoints, runtime reproduction, figure rendering and aggregation.
- `results/tables/`: compact main-model and per-condition results.
- `results/provenance/`: contracts and evaluation receipts.
- `docs/`, `model/`: reproduction and model references.

## Citation and license

The current author order is 林泽丰, 方贤勇, 符天凡, 徐小华. See [`CITATION.cff`](CITATION.cff); English names, affiliations and ORCIDs will be aligned with the final paper metadata.

The project is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies and datasets remain subject to their own terms; no datasets are redistributed here.
