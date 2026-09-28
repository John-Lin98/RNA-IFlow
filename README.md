# RNA Design via Conditioned Flow Matching and Finite-Policy Reinforcement Learning

RNA-IFlow generates RNA sequences conditioned on a target secondary structure using Dirichlet flow matching. RNA-IFlow-RL maps the learned flow to a pairing-preserving finite policy and refines it with thermodynamic feedback.

**Paper:** arXiv link will be added after the public preprint receives an identifier.

## Code and model availability

This repository contains research code, evaluation adapters, and compact result tables. It does not contain training datasets or model weights. The RNA-IFlow-RL weights remain in a separate private model repository; public weight access has not been announced. Inference requires an authorized local copy of the exported model.

## Installation and inference

Use Python 3.10 in an isolated environment:

```bash
python -m pip install -r requirements.txt
python scripts/load_export.py /path/to/model
python scripts/infer.py --model /path/to/model --structure '(((...)))' --candidates 8
```

The model directory must contain the export manifest, configuration, and weights. This single-target command is a usage example and does not reproduce the paper benchmarks. See [inference details](docs/inference.md).

## Results

| Benchmark | Pass@1 | Pass@8 |
| --- | ---: | ---: |
| Eterna100-v2 | 0.5400 | 0.6500 |
| Eterna100 | 0.5067 | 0.6167 |
| Rfam-27 | 0.8519 | 0.8765 |

These RNA-IFlow-RL values match the displayed Pass@1 and Pass@8 values in Table 1 of the submitted manuscript. They are means over the paper's evaluation conditions, with eight returned candidates per target. Compact source tables are in [`results/tables/`](results/tables/). The supplemental RNAsolo-764 analysis and its overlap limitations are documented in [dataset limitations](docs/datasets.md). These packaged results are frozen artifacts, not a fresh benchmark rerun.

## Reproduction and layout

See [experiment map](docs/experiments.md), [training recipe](docs/training.md), and [dataset limitations](docs/datasets.md) for reproduction boundaries. No datasets, raw candidate records, or model binaries are stored in this repository. Full training and benchmark reproduction requires the original assets; passing the included tests does not establish that those experiments were rerun.

- `experiments/`: source implementation and its dependency closure.
- `scripts/`: export, portable inference/evaluation, training entrypoints, runtime reproduction, figure rendering and aggregation.
- `results/tables/`: compact main-model and per-condition results.
- `results/provenance/`: contracts and evaluation receipts.
- `docs/`, `model/`: reproduction and model references.

## Citation and license

The author order is Zefeng Lin, Xianyong Fang, Tianfan Fu, and Xiaohua Xu. See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

The project is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies and datasets remain subject to their own terms; no datasets are redistributed here.
