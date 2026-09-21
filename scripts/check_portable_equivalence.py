"""Compare portable loading with the authoritative model class on fixed inputs."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from load_export import load_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--original-model-file', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('authoritative_model', args.original_model_file)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    config = json.loads((args.model/'config.json').read_text())
    reference = original.FairRNAFlow(args.backbone, **config['model_args']).eval()
    reference.load_state_dict(load_file(str(args.model/'model.safetensors')), strict=True)
    portable = load_export(args.model)
    state = torch.rand((2,12,4), generator=torch.Generator().manual_seed(2027))
    state /= state.sum(-1,keepdim=True)
    inputs = dict(state=state, alpha=torch.tensor([1.5,6.0]), attention_mask=torch.ones(2,12),
                  structure_tokens=torch.tensor([[original.STRUCTURE_SYMBOLS.index(c) for c in s]
                      for s in ['(((...)))...','....(....)..']]))
    with torch.inference_mode():
        a,b = reference(**inputs),portable(**inputs)
    assert torch.isfinite(a).all() and torch.equal(a,b)
    print(json.dumps({'portable_vs_authoritative_output_equivalence':True,
                      'max_abs_diff':float((a-b).abs().max()),'device':'cpu','shape':list(a.shape)}))


if __name__ == '__main__':
    main()
