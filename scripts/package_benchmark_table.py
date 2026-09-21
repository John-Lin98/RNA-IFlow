"""Verify normalized candidate evidence and export a path-free benchmark table."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-table', type=Path, required=True)
    parser.add_argument('--candidate-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    source = args.source_table.read_bytes()
    rows = list(csv.DictReader(source.decode().splitlines()))
    columns = ['method', 'method_type', 'benchmark', 'tasks', 'K',
               'pass_at_1', 'pass_at_8', 'mfe_at_1', 'mfe_at_8', 'bad_count',
               'diversity', 'best_pair_f1', 'best_NED', 'best_target_probability',
               'total_parameters', 'trainable_parameters', 'compute_budget_matched',
               'candidate_budget_semantics', 'normalized_candidates_sha256',
               'source_summary_sha256']
    evidence = []
    keys = set()
    for row in rows:
        key = row['method'], row['benchmark']
        if key in keys:
            raise ValueError('Duplicate method/benchmark')
        keys.add(key)
        candidate_path = args.candidate_directory / Path(row['normalized_candidates_path']).name
        raw = candidate_path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != row['normalized_candidates_sha256']:
            raise ValueError('Candidate SHA mismatch: ' + candidate_path.name)
        groups = collections.defaultdict(list)
        for line in raw.splitlines():
            candidate = json.loads(line)
            if any(type(candidate[field]) is not bool for field in ('uMFE_hit', 'mfe_hit')):
                raise ValueError('Non-boolean success indicator')
            groups[candidate['seed'], candidate['task_id']].append(candidate)
        count, k = int(row['tasks']), int(row['K'])
        if k != 8 or len(groups) != 3 * count:
            raise ValueError('Invalid candidate budget or task coverage')
        task_sets = [{task for seed, task in groups if seed == s} for s in (1009, 2027, 3037)]
        if {s for s, _ in groups} != {1009, 2027, 3037} or any(t != task_sets[0] or len(t) != count for t in task_sets):
            raise ValueError('Invalid seed/task coverage')
        for group in groups.values():
            if len(group) != k or {c['candidate_index'] for c in group} != set(range(k)):
                raise ValueError('Incomplete candidate indices')
        for prefix, field in [('pass', 'uMFE_hit'), ('mfe', 'mfe_hit')]:
            first = sum(next(c[field] for c in g if c['candidate_index'] == 0) for g in groups.values()) / len(groups)
            best = sum(any(c[field] for c in g) for g in groups.values()) / len(groups)
            for suffix, value in [('1', first), ('8', best)]:
                if abs(value - float(row[prefix + '_at_' + suffix])) > 1e-12:
                    raise ValueError('Metric mismatch: ' + str(key))
        evidence.append(dict(method=key[0], benchmark=key[1], candidate_file=candidate_path.name,
                             candidate_sha256=sha, groups=len(groups), metrics_recomputed=True))
    args.output.mkdir(parents=True)
    with (args.output / 'benchmark_quality.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: row[c] for c in columns} for row in rows)
    (args.output / 'benchmark_quality_provenance.json').write_text(json.dumps(dict(
        source_file=args.source_table.name, source_sha256=hashlib.sha256(source).hexdigest(),
        candidate_checks=evidence,
        scope='P@1/P@8/MFE@1/MFE@8 recomputed; other columns preserved from source, not recomputed',
        manuscript_status='Matches inspected Table 1 snapshot after rounding; latest approval not established',
        sd_format='No SD presentation frozen',
        budget_note='K counts returned candidates; search and neural compute budgets are not matched'), indent=2) + '\n')
    print('PASS:', len(rows), 'method/benchmark rows; candidate hashes and four success metrics verified')


if __name__ == '__main__':
    main()
