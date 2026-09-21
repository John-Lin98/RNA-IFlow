"""Export a verified full inference state from the original partial checkpoints.

Run against the authoritative source checkout; no training or GPU required.
Only load trusted PyTorch checkpoints. Output must be a fresh directory outside Git.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'backbone', 'supervised', 'u96', 'checkpoint', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing an existing export directory')
    torch.set_num_threads(4)
    torch.manual_seed(1009)
    assert digest(args.checkpoint) == '198fd79e7680f4b01e758f063aadab00c0d4e6709ac4d2a220249a267a70ebe8'
    candidate = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    contract = candidate['contract']
    assert digest(args.supervised) == contract['supervised_checkpoint_sha256']
    assert digest(args.u96) == contract['policy_initialization']['source_checkpoint_sha256']
    assert digest(args.backbone / 'model.safetensors') == contract['rnaernie_weight_sha256']
    source_file = args.source / 'experiments/rna-flow-fair-components/model.py'
    assert digest(source_file) == contract['implementation_manifest']['experiments/rna-flow-fair-components/model.py']
    spec = importlib.util.spec_from_file_location('rna_iflow_original_model', source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    supervised = torch.load(args.supervised, map_location='cpu', weights_only=False)
    u96 = torch.load(args.u96, map_location='cpu', weights_only=False)
    sc = supervised['contract']
    model_args = {k: sc.get(k) for k in ('structure_source', 'injection', 'structure_dim', 'lora_rank', 'lora_alpha', 'lora_dropout', 'backbone_mode')}
    model = module.FairRNAFlow(args.backbone, **model_args)
    model.load_trainable_state_dict(supervised['trainable_model'])
    module.configure_rl_trainable_scope(model, 'last_2_backbone_and_head')
    model.load_trainable_state_dict(u96['trainable_model'])
    model.load_trainable_state_dict(candidate['trainable_model'])
    model.eval()
    state = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    args.output.mkdir(parents=True)
    save_file(state, str(args.output / 'model.safetensors'))
    restored = load_file(str(args.output / 'model.safetensors'))
    assert set(state) == set(restored)
    for name in state:
        assert state[name].shape == restored[name].shape
        assert state[name].dtype == restored[name].dtype
        assert torch.equal(state[name], restored[name]), name
    # Independently instantiate then strict-load every parameter and buffer.
    fresh = module.FairRNAFlow(args.backbone, **model_args)
    fresh.load_state_dict(restored, strict=True)
    fresh.eval()
    generator = torch.Generator().manual_seed(2027)
    simplex = torch.rand((2, 12, 4), generator=generator)
    simplex /= simplex.sum(-1, keepdim=True)
    structures = ['(((...)))...', '....(....)..']
    inputs = dict(state=simplex, alpha=torch.tensor([1.5, 6.0]),
                  attention_mask=torch.ones(2, 12),
                  structure_tokens=torch.tensor([[module.STRUCTURE_SYMBOLS.index(s) for s in row] for row in structures]))
    with torch.inference_mode():
        expected, actual = model(**inputs), fresh(**inputs)
    assert torch.isfinite(expected).all() and torch.equal(expected, actual)
    manifest = dict(source_checkpoint_sha256=digest(args.checkpoint),
                    supervised_sha256=digest(args.supervised), u96_sha256=digest(args.u96),
                    backbone_sha256=digest(args.backbone / 'model.safetensors'),
                    source_revision=contract['repository_revision'],
                    model_source_sha256=digest(source_file), export_script_sha256=digest(__file__),
                    tensor_count=len(state), tensor_equivalence=True,
                    fixed_minibatch_output_equivalence=True, max_abs_diff=float((expected-actual).abs().max()),
                    test_device='cpu', test_shape=[2, 12, 4], test_structures=structures,
                    torch_version=torch.__version__, inference_only=True,
                    original_resume_checkpoint_included=False,
                    model_sha256=digest(args.output / 'model.safetensors'))
    backbone_config = model.rnaernie.config.to_dict()
    backbone_config['_name_or_path'] = 'RNAErnie'
    for name, value in [('export_manifest.json', manifest), ('config.json', {'model_args': model_args, 'backbone_config': backbone_config})]:
        (args.output / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
