"""Load the complete inference export without original backbone files."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

FAIR = Path(__file__).resolve().parents[1] / 'experiments/rna-flow-fair-components'
sys.path.insert(0, str(FAIR))
from model import FairRNAFlow

MODEL_SHA256 = '8a8dcf74014be2e3ab9571a19facebaad639ca4fd6574bc97d7bdf5d9ffe9f9d'
CONFIG_SHA256 = 'fd84cd17150e1ad1d9e756a05a78dc13c24463ac27bcf4c771e5c7e8f07375fb'


def parse_config(raw):
    """Bind architecture to the independently verified U2442 HF package."""
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError('U2442 config SHA256 mismatch')
    return json.loads(raw)


def load_export(directory, device='cpu'):
    directory = Path(directory)
    manifest = json.loads((directory / 'export_manifest.json').read_text())
    if manifest['model_sha256'] != MODEL_SHA256:
        raise ValueError('This loader is frozen to the U2442 inference export')
    config = parse_config((directory / 'config.json').read_bytes())
    weights = directory / 'model.safetensors'
    with weights.open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != manifest['model_sha256']:
        raise ValueError('Export weight SHA256 mismatch')
    model = FairRNAFlow('', **config['model_args'], backbone_config=config['backbone_config'])
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.to(device).eval()


if __name__ == '__main__':
    torch.set_num_threads(4)
    model = load_export(sys.argv[1])
    with torch.inference_mode():
        output = model(torch.full((1, 9, 4), .25), torch.tensor([2.0]),
                       torch.ones(1, 9), structure_tokens=torch.tensor([[1,1,1,0,0,0,2,2,2]]))
    assert output.shape == (1, 9, 4) and torch.isfinite(output).all()
    print('PORTABLE_EXPORT_LOAD_PASS')
