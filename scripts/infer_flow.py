"""Generate a single-target RNA-IFlow example with the supervised flow model."""
import argparse
import json
from pathlib import Path
import sys

import torch

from load_export import load_export

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/rna-flow-fair-components'))
from evaluate import sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--structure', required=True)
    parser.add_argument('--seed', type=int, default=1009)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not args.structure or set(args.structure) - set('.()'):
        parser.error('Expected a nonempty dot-bracket target')
    if args.candidates < 1 or args.steps < 1:
        parser.error('Candidates and steps must be positive')
    torch.set_num_threads(4)
    model = load_export(args.model, args.device)
    sequences = sample(model, args.structure, None, args.candidates, args.steps,
                       args.seed, 'native', torch.device(args.device))
    print(json.dumps({'scope': 'single-target example, not official benchmark',
                      'structure': args.structure, 'seed': args.seed,
                      'steps': args.steps, 'sequences': sequences}, indent=2))


if __name__ == '__main__':
    main()
