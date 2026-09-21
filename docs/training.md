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

## RL boundaries

The main RL implementation remains outside `supervised_snapshot/`. See [methods](methods.md) and `results/provenance/contract.json` for the finite-policy settings. Scaled C3 starts from U96 policy weights with fresh optimizer state; later C3+D5 segments perform exact continuation. Their validators require matching source checkpoint, receipt, contract, optimizer manifest, per-rank RNG and task coverage. A safetensors inference export is not an exact-resume checkpoint and cannot satisfy these requirements.

The portable HF inference route needs none of those training parents. Full from-scratch RL reproduction still requires the U96 warm-start recipe, original task construction and parent artifacts; do not bypass the validators or present the inference export as a training-resume substitute.

## Release checks

The preserved supervised trainer's `--help` imports passed in the isolated CPU environment. All seven copied files matched their source Git blobs byte-for-byte, and trainer/model/evaluator hashes matched the supervised receipt. This verifies source closure and CLI import, not long-training numerical equivalence or a new trained checkpoint.
