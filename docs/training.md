# Supervised training and RL continuation

## Two source snapshots, two roles

The supervised parent was trained at `c332d2c9bace0dd586cfbfd119a24c762f8232ce`, whereas the main RL contract identifies `91a71ed91780e278369aead5aeae3fa957f72fb7`. The later trainer is not byte-identical to the original supervised trainer. `experiments/supervised_snapshot/` therefore preserves the seven-file original supervised dependency closure, with exact Git-blob hashes in `results/provenance/supervised_source_manifest.json`. Use separate Python processes for this snapshot and the main RL code: both retain historical module names such as `model` and `evaluate`.

The original supervised receipt's path-free contract is in `results/provenance/supervised_training_contract.json`. It records 10,000,000 Arrow rows, single-device batch 128, full-backbone training, bf16, LR 1e-5, seed 1009, minimum two epochs and patience two. Training completed eight epochs and selected epoch six. These are the paper parent's settings, not the later RiNALMo experiments or current CLI defaults.

## Original supervised command

Supply the original, hash-matching assets through local paths. This command starts training and is documentation only; no training was launched during release preparation.

```bash
python experiments/supervised_snapshot/rna-flow-progressive-supervision-rl/train_progressive.py \
  --train /path/to/training.arrow --train-format arrow --train-limit 10000000 \
  --train-manifest /path/to/nested_manifest.json \
  --thermo-validation /path/to/validation48.jsonl \
  --rnaernie /path/to/RNAErnie \
  --rnaernie-revision df26681ed44bb671b61536c5e2a196d6058a44f7 \
  --output /path/to/fresh-supervised-output --device cuda:0 \
  --seed 1009 --batch-size 128 --gradient-accumulation 1 \
  --learning-rate 1e-5 --weight-decay 0.01 --precision bf16 \
  --backbone-mode full --lora-rank 0 --lora-alpha 16 --lora-dropout .05 \
  --minimum-epochs 2 --patience 2 \
  --thermo-candidates 8 --thermo-flow-steps 50 --thermo-seed 9176
```

Arrow fields are `sequence` and `target_structure`. The nested manifest authorizes the exact prefix, membership and exclusion hashes; an arbitrary Arrow dataset is not an equivalent substitute. Asset SHA values are in the supervised contract. Dataset acquisition and redistribution instructions remain incomplete, so the command is not yet a turnkey reproduction. The original source used single-device training; adding DDP changes this historical recipe.

The supervised snapshot intentionally retains the historical scorer, including its extra length normalization of ensemble defect, because it participated in checkpoint selection. Do not use it to produce paper quality tables. Paper-facing evaluation uses `scripts/paper_metrics.py` and the corrected NED convention. This distinction preserves training provenance without silently rewriting historical selection.

## Nested Arrow construction

The original `build_nested_arrow.py` is now preserved alongside the supervised trainer, with SHA256 `d638b83175dd92993fd89f856376de138d10e0aa9c21e3b5f8119a7e582c17ac`, matching both the original Git blob and the audited source copy. It verifies the shard set, hashes and row counts before building the nested prefixes. Run only after acquiring the authorized original inputs:

```bash
python experiments/supervised_snapshot/rna-flow-progressive-supervision-rl/build_nested_arrow.py \
  --source-shards /path/to/parquet-shards --base-train /path/to/train100k.jsonl \
  --validation /path/to/validation5000.jsonl --eterna100v2 /path/to/eterna100v2.jsonl \
  --source-manifest /path/to/source-manifest.json --output /path/to/fresh-nested-output
```

This is a full 10M builder, not a cheap smoke command. Only import/CLI was tested during packaging; the dataset was not rebuilt. `valid_pair` in this builder checks sequence/structure syntax, length and alphabet, not thermodynamic fold correctness.

### Prerequisite split and source manifest

The same snapshot now includes `rna-flow-fair-components/data_contract.py` and `rna-flow-progressive-supervision-rl/build_contract.py`. They respectively build the 100K/5K structure-disjoint split and verify the 11 source shards/select thermo48. Their SHA values are in the source manifest; the original base-split contract and output identities are preserved without private paths in `results/provenance/base_split_contract.json`.

```bash
python experiments/supervised_snapshot/rna-flow-fair-components/data_contract.py \
  --sl-parquet /path/to/part-00000.parquet \
  --eterna-v1-dir /path/to/Eterna100V1_inputs --eterna-v2-dir /path/to/Eterna100V2_inputs \
  --extra-benchmark-jsonl /path/to/public_test_tasks.v1.jsonl \
  --out /path/to/fresh-base-data --train-count 100000 --validation-count 5000 --seed 1009
python experiments/supervised_snapshot/rna-flow-progressive-supervision-rl/build_contract.py \
  --source-shards /path/to/parquet-shards --base-data /path/to/frozen-base-data \
  --official-code /path/to/RNA-Design-LM --rnaernie /path/to/RNAErnie \
  --output /path/to/fresh-source-contract --seed 9176 --tasks-per-bin 12
```

The source verifier pins the RNA-Design-LM code revision and every source-shard hash/row count. Its historic `monitor-only` benchmark label describes that supervised-stage contract only, not the later Eterna selection history disclosed in [datasets](datasets.md).

**Byte-level reproduction caveat:** the original split builder iterates a Python set before shuffling validation rows. A synthetic check with identical seed 1009 but `PYTHONHASHSEED=1` versus `2` produced the same training rows and validation membership but different validation row order. The original process hash seed is not established. Do not claim that `--seed 1009` alone reconstructs the historical validation JSONL SHA. Preserve the original frozen split or establish exact ordering before claiming byte-level reproduction; no historical split is silently rewritten. The downstream thermo48 selector ranks by a stable content hash, but this does not retroactively make the original validation file ordering deterministic.

These source restorations do not resolve source acquisition/redistribution permissions or the exact extra-exclusion asset. Full data pipeline execution has not been rerun for the release.

## RL continuation boundaries

The main RL implementation remains outside `supervised_snapshot/`. See [methods](methods.md) and `results/provenance/contract.json` for the finite-policy settings. Scaled C3 starts from U96 policy weights with fresh optimizer state; later C3+D5 segments perform exact continuation. Their validators require matching source checkpoint, receipt, contract, optimizer manifest, per-rank RNG and task coverage. A safetensors inference export is not an exact-resume checkpoint and cannot satisfy these requirements.

The portable HF inference route needs none of those training parents. Full from-scratch RL reproduction still requires the U96 warm-start recipe, original task construction and parent artifacts; do not bypass the validators or present the inference export as a training-resume substitute.

## Release checks

The preserved supervised trainer's `--help` imports passed in the isolated CPU environment. All seven copied files matched their source Git blobs byte-for-byte, and trainer/model/evaluator hashes matched the supervised receipt. This verifies source closure and CLI import, not long-training numerical equivalence or a new trained checkpoint.
