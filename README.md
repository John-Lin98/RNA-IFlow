# RNA Design via Conditioned Flow Matching and Finite-Policy Reinforcement Learning

RNA-IFlow generates RNA sequences conditioned on a target secondary structure using Dirichlet flow matching. RNA-IFlow-RL maps the learned flow to a pairing-preserving finite policy and refines it with thermodynamic feedback.

**Paper:** arXiv identifier pending after submission.

## Paper, code, and model weights

This repository contains research code, evaluation adapters, and compact result tables. The paper's preprint was submitted to arXiv; its public link will be added when arXiv assigns an identifier. Model weights are archived separately at [Hugging Face](https://huggingface.co/jojojoojooo/RNA-IFlow). The model repository is public under Apache-2.0; both complete inference exports passed upload, independent download, SHA256, and CPU single-target smoke checks. No training datasets or model binaries are stored in this GitHub repository.

### Model Weights

The model repository contains two complete inference exports: `RNA-IFlow/` for the supervised flow model and `RNA-IFlow-RL/` for the C3+D5 U2442 final model. The `arxiv-v1` revision will identify the exports corresponding to the paper's arXiv v1 after the final verification and public identifier are available. The original server checkpoints are retained separately; the portable exports cannot resume training exactly. See the [model card](https://huggingface.co/jojojoojooo/RNA-IFlow) and [inference details](docs/inference.md).

## Installation and inference

Use Python 3.10 in an isolated environment:

```bash
python -m pip install -r requirements.txt
python scripts/load_export.py /path/to/RNA-IFlow-RL
python scripts/infer_flow.py --model /path/to/RNA-IFlow --structure '(((...)))...' --candidates 8
python scripts/infer.py --model /path/to/RNA-IFlow-RL --structure '(((...)))...' --candidates 8
```

Each model directory must contain its export manifest, configuration, and weights. These single-target commands are usage examples and do not reproduce the paper benchmarks. See [inference details](docs/inference.md).

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
- `scripts/`: export, portable inference/evaluation for both models, training entrypoints, runtime reproduction, figure rendering and aggregation.
- `results/tables/`: compact main-model and per-condition results.
- `results/provenance/`: contracts and evaluation receipts.
- `docs/`, `model/`: reproduction and model references.

## Citation and license

The author order is Zefeng Lin, Xianyong Fang, Tianfan Fu, and Xiaohua Xu. See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

The project is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies and datasets remain subject to their own terms; no datasets are redistributed here.
