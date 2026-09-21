"""Reaggregate the paper's resident full100 K8 runtime without running inference."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = {'RNA-IFlow', 'RNA-IFlow-RL', 'RNA-Design-LM SL', 'RNA-Design-LM SL+RL', 'GoForth'}


def validate_source(summary, rows):
    """Reject partial/foreign plans and nonphysical timings before aggregation."""
    plan = summary['plan']
    if (plan['K'], plan['tasks'], plan['repeats'], plan['conditions']) != (8, 100, 2, [1009, 2027, 3037]):
        raise ValueError('Not the frozen full100 K8 two-repeat protocol')
    for methods in (plan['methods'], [r['method'] for r in summary['methods']]):
        if len(methods) != 5 or set(methods) != METHODS:
            raise ValueError('Expected exactly the five resident methods')
    if not plan['models_resident_simultaneously'] or len(rows) != 3000:
        raise ValueError('Not a complete resident-model ledger')
    for row in rows:
        for key in ('generation_seconds', 'post_scoring_seconds'):
            value = row[key]
            if not math.isfinite(value) or value < 0:
                raise ValueError('Nonfinite or negative timing: ' + key)
    for row in summary['methods']:
        value = row['run_time_seconds_per_K8_group']
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid reported runtime')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    raw = (args.input / 'timing_ledger.jsonl').read_bytes()
    source = (args.input / 'summary.json').read_bytes()
    summary = json.loads(source)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != summary['ledger_sha256'] or summary['status'] != 'complete':
        raise ValueError('Incomplete or mismatched source evidence')
    rows = list(map(json.loads, raw.splitlines()))
    validate_source(summary, rows)
    groups = collections.defaultdict(list)
    for row in rows:
        if row['returned_candidates'] != 8 or row['candidate_identity_mismatches'] != 0:
            raise ValueError('Candidate identity/coverage mismatch')
        groups[row['method'], row['task_id'], row['condition_seed']].append(row)
    results = []
    for expected in summary['methods']:
        method = expected['method']
        current = {k: v for k, v in groups.items() if k[0] == method}
        if len(current) != 300:
            raise ValueError('Expected 300 task-condition groups')
        task_sets = [{k[1] for k in current if k[2] == s} for s in (1009, 2027, 3037)]
        if len(task_sets[0]) != 100 or any(t != task_sets[0] for t in task_sets):
            raise ValueError('Task-condition coverage mismatch')
        for values in current.values():
            if len(values) != 2 or {v['replicate'] for v in values} != {0, 1}:
                raise ValueError('Expected exactly two repeats')
        median = statistics.median(statistics.median(r['generation_seconds'] for r in v)
                                   for v in current.values())
        if abs(median - expected['run_time_seconds_per_K8_group']) > 1e-12:
            raise ValueError('Reported runtime does not reproduce')
        results.append(dict(method=method, k=8, tasks=100, conditions=3, repeats=2,
                            resident_generation_seconds_per_group=median,
                            checkpoint_sha256=expected['identity']['checkpoint_sha256']))
    if set(k[0] for k in groups) != {r['method'] for r in results}:
        raise ValueError('Unexpected methods')
    args.output.mkdir(parents=True)
    with (args.output / 'resident_runtime.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    plan = {k: v for k, v in summary['plan'].items() if k not in ('gpu', 'gpu_uuid', 'started_unix')}
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        ledger_sha256=digest, summary_sha256=hashlib.sha256(source).hexdigest(), plan=plan,
        aggregation='Median of two repeats per task/condition, then median over 300 groups',
        scope='Resident neural generation only; setup/warmup and subsequent scoring excluded',
        native_search_rows='Separate protocol; not included here'), indent=2) + '\n')
    print('PASS: 5 methods, 3000 timing rows, exact reported medians')


if __name__ == '__main__':
    main()
