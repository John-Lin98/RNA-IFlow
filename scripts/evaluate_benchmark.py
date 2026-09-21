"""Evaluate the U2442 export with the frozen paper sampler and corrected NED."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import RNA
import torch
from load_export import load_export, MODEL_SHA256, CONFIG_SHA256
from paper_metrics import evaluate_candidate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/rna-flow-progressive-supervision-rl'))
from endpoint_policy import rollout_discrete_domino_trajectory, validate_discrete_domino_trajectory
from evaluate import aggregate, atomic_json, global_task_seed, valid_pair_fraction, validate_coverage

CONDITIONS = {1009: .8, 2027: 1., 3037: 1.2}
COUNTS = {'eterna100v2': 100, 'eterna100': 100, 'rfam27': 27, 'rnasolo764': 764}
MODEL_SHA = MODEL_SHA256


def read_benchmark(path, name):
    raw = path.read_bytes()
    evidence = json.loads((ROOT / 'results/provenance/main_evaluation_receipts.json').read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != evidence[name]['receipt']['benchmark']['sha256']:
        raise ValueError('Benchmark bytes/order do not match the frozen source')
    tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(tasks) != COUNTS[name] or len({str(t['id']) for t in tasks}) != len(tasks):
        raise ValueError('Benchmark count or IDs differ')
    return tasks, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--benchmark', choices=COUNTS, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--smoke-first-task', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    if RNA.__version__ != '2.7.2':
        raise RuntimeError('ViennaRNA 2.7.2 is required')
    tasks, benchmark_sha = read_benchmark(args.tasks, args.benchmark)
    manifest = json.loads((args.model / 'export_manifest.json').read_text())
    if manifest['model_sha256'] != MODEL_SHA:
        raise ValueError('This adapter is frozen to the main U2442 export')
    if args.smoke_first_task:
        tasks = tasks[:1]
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = load_export(args.model, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(benchmark=args.benchmark, benchmark_sha256=benchmark_sha,
                    model_sha256=MODEL_SHA, config_sha256=CONFIG_SHA256,
                    conditions=CONDITIONS, candidates=8, steps=8,
                    seed_mapping='condition_seed + original_task_index * 1000003',
                    scope='smoke' if args.smoke_first_task else 'benchmark_reproduction',
                    viennarna=RNA.__version__, tasks=len(tasks),
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(args.output / 'protocol.json', protocol)
    rows = []
    with (args.output / 'candidates.jsonl').open('x') as handle, torch.inference_mode():
        for index, task in enumerate(tasks):
            structure = task['target_structure']
            for seed, temperature in CONDITIONS.items():
                temperatures = torch.full((8,), temperature, dtype=torch.float32, device=device)
                actual_seed = global_task_seed(seed, index)
                trajectory = rollout_discrete_domino_trajectory(model, structure, 8, 8,
                                                               actual_seed, device, temperatures)
                checked = validate_discrete_domino_trajectory(
                    trajectory, device, expected_structure=structure, expected_seed=actual_seed,
                    expected_candidates=8, expected_steps=8, expected_temperatures=temperatures)
                if not checked.get('replay_exact') or not checked.get('structured_pair_actions_legal'):
                    raise RuntimeError('Trajectory replay/legal validation failed')
                for candidate_index, sequence in enumerate(trajectory['final_sequences']):
                    score = evaluate_candidate(sequence, structure)
                    pair_validity = valid_pair_fraction(sequence, structure)
                    row = dict(task_id=str(task['id']), seed=seed, candidate_index=candidate_index,
                               sequence=sequence, target_structure=structure,
                               valid=float(pair_validity == 1), valid_pair_fraction=pair_validity, **score)
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + '\n')
                handle.flush()
            print(json.dumps(dict(tasks_complete=index + 1, tasks_total=len(tasks))), flush=True)
    validate_coverage(rows, [t['id'] for t in tasks], list(CONDITIONS), 8)
    per_condition, summary = aggregate(rows, 8)
    atomic_json(args.output / 'task_seed.json', per_condition)
    atomic_json(args.output / 'summary.json', summary | protocol | {'status': 'complete'})
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
