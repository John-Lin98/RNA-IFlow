# Authoritative implementation and export

The authoritative method revision is `91a71ed91780e278369aead5aeae3fa957f72fb7`, as recorded by the U2442 formal contract. All 13 files in its implementation manifest matched both the Git blobs at that revision and the inspected clean runtime checkout.

The `commit7856d04` label in the historical result-directory name is a naming/launch lineage marker, not the authoritative U2442 implementation. The trainer at `7856d04301853b54165ca117a96c588c579d1a19` does not match the formal implementation manifest. Files in this release originate from the contract-backed snapshot, with portability adaptations recorded separately by source and destination hashes in `paper_release_inventory.tsv`.

Evidence limitation: four current operator scripts do not match their older command-receipt hashes. The old receipts were not modified or retroactively rewritten. Current launcher files therefore do not prove an immutable original command-script chain. The implementation determination instead rests on the formal contract, receipt/checkpoint linkage, Git content hashes and clean runtime implementation.

## Full inference export

The U2442 source checkpoint contains partial trainable state, not a standalone full model. The export reconstructs the actual model in order:

1. Frozen RNAErnie backbone.
2. Supervised epoch-6 model state.
3. Causal U96 trainable state.
4. U2442 trainable state.

Input SHA gates and strict loads are implemented in `scripts/export_model.py`. The full output contains 223 tensor states; tensor equivalence and a fixed two-example CPU forward comparison passed with maximum absolute error zero. Portable config-only loading was also compared against the authoritative original class with zero output difference.

The upstream backbone checkpoint omits unused pooler parameters. Their deterministic constructor values are preserved in the export; they are not claimed to originate from the source checkpoint and are not used by the model's last-hidden-state inference path.

The inference export omits optimizer and RNG state and is not an exact-resume training checkpoint. Its HF revision and hashes are listed in `model/README.md`. Download/load validation is a separate gate from local export equivalence.

## Supervised source is a separate snapshot

The supervised epoch-six parent predates the RL source revision. Its trainer, model, and evaluator Git blobs match the recorded source hashes. The seven-file closure is preserved byte-for-byte under `experiments/supervised_snapshot/`, independently of the later RL/inference code. The original supervised scorer is not the corrected paper-quality scorer. See `docs/training.md` for the command, asset identities, and this distinction. These files are actual parent-training dependencies.
