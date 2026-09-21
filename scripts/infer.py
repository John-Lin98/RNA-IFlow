"""Generate RNA for one target; this example is not a benchmark aggregate."""
import argparse
import json
from pathlib import Path
import sys

import torch
from load_export import load_export

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/rna-flow-progressive-supervision-rl'))
from endpoint_policy import rollout_discrete_domino_trajectory, endpoint_units
from evaluate import evaluate_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--structure', required=True)
    parser.add_argument('--seed', type=int, default=1009)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not args.structure or set(args.structure) - set('.()'):
        parser.error('Expected a nonempty dot-bracket target')
    endpoint_units(args.structure)
    torch.set_num_threads(4)
    model = load_export(args.model, args.device)
    with torch.inference_mode():
        trajectory = rollout_discrete_domino_trajectory(
            model, args.structure, args.candidates, 8, args.seed,
            torch.device(args.device), args.temperature)
    rows = [{'sequence': sequence, **evaluate_candidate(sequence, args.structure)}
            for sequence in trajectory['final_sequences']]
    print(json.dumps({'scope': 'single-target example, not official benchmark',
                      'structure': args.structure, 'seed': args.seed,
                      'temperature': args.temperature, 'candidates': rows}, indent=2))


if __name__ == '__main__':
    main()
