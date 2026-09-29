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

EXPORTS = {
    'RNA-IFlow': (
        '4f33b51d28aa0fa7fab7a27cd6b0422c2eed9654f204bcedb80a771851c5a151',
        '948220c0bf8750c09108300d5745b3b16b2eeef8e5789ad48b111e13bf94c316',
    ),
    'RNA-IFlow-RL': (
        '8a8dcf74014be2e3ab9571a19facebaad639ca4fd6574bc97d7bdf5d9ffe9f9d',
        'fd84cd17150e1ad1d9e756a05a78dc13c24463ac27bcf4c771e5c7e8f07375fb',
    ),
}
MODEL_SHA256, CONFIG_SHA256 = EXPORTS['RNA-IFlow-RL']


def parse_config(raw, expected_sha256=CONFIG_SHA256):
    """Bind the architecture to a verified inference export."""
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Export config SHA256 mismatch')
    return json.loads(raw)


def load_export(directory, device='cpu'):
    directory = Path(directory)
    manifest = json.loads((directory / 'export_manifest.json').read_text())
    model_sha256 = manifest['model_sha256']
    matching = [item for item in EXPORTS.values() if item[0] == model_sha256]
    if len(matching) != 1:
        raise ValueError('Unrecognized inference export')
    config = parse_config((directory / 'config.json').read_bytes(), matching[0][1])
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
