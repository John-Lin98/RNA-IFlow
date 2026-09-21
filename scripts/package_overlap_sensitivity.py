"""Recheck historical RNAsolo subset aggregates; never choose subsets by outcomes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    receipt = json.loads((args.source / 'receipt.json').read_text())
    for name, key in [('results.json', 'results_sha256'),
                      ('overlap_targets.json', 'overlap_targets_sha256')]:
        if sha(args.source / name) != receipt[key]:
            raise ValueError('Historical artifact hash mismatch: ' + name)
    overlap = json.loads((args.source / 'overlap_targets.json').read_text())
    ids = {r['task_id'] for r in overlap['targets']}
    if len(ids) != 9 or sum(r['sft_matching_rows'] for r in overlap['targets']) != 90:
        raise ValueError('Unexpected historical overlap membership')
    rows = json.loads((args.source / 'results.json').read_text())['rows']
    metrics = ['pass_at_1', 'pass_at_8', 'mfe_at_1', 'mfe_at_8',
               'best_target_probability', 'best_NED', 'best_pair_f1', 'diversity']
    sources = {Path(r['path']).parent.name: r['sha256'] for r in receipt['original_sources']}
    methods = {r['method'] for r in rows}
    if len(methods) != 5 or len(rows) != 15:
        raise ValueError('Expected five methods and three subsets each')
    verified = []
    for method in sorted(methods):
        key = method.replace(' ', '_').replace('+', 'plus') + '__rnasolo764'
        path = args.tables / key / 'task_seed.jsonl'
        if sha(path) != sources[key]:
            raise ValueError('Task-condition source hash mismatch: ' + key)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        task_ids = {r['task_id'] for r in records}
        pairs = {(r['task_id'], r['seed']) for r in records}
        if (len(task_ids) != 764 or not ids <= task_ids or len(records) != 2292
                or pairs != {(task, seed) for task in task_ids for seed in (1009, 2027, 3037)}):
            raise ValueError('Incomplete or duplicated task-condition coverage')
        for subset, n in [('full764', 764), ('exact_SFT_disjoint755', 755), ('overlap9', 9)]:
            selected = [r for r in records if subset == 'full764'
                        or ((r['task_id'] in ids) == (subset == 'overlap9'))]
            expected = [r for r in rows if r['method'] == method and r['subset'] == subset]
            if len(expected) != 1 or len(selected) != n * 3:
                raise ValueError('Invalid subset coverage')
            row = expected[0]
            if (row['tasks'], row['task_condition_groups'], row['K']) != (n, n * 3, 8):
                raise ValueError('Subset metadata mismatch')
            for metric in metrics:
                value = sum(r[metric] for r in selected) / len(selected)
                if not math.isfinite(value) or not math.isfinite(row[metric]) or abs(value - row[metric]) > 1e-12:
                    raise ValueError('Aggregate mismatch: ' + metric)
        verified.append({'source_file': key + '/task_seed.jsonl', 'sha256': sources[key]})
    args.output.mkdir(parents=True)
    with (args.output / 'sensitivity.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        'primary_protocol': 'full764 unchanged; three frozen seed/temperature conditions, K8',
        'scope': 'Posthoc descriptive exact-structure sensitivity, not outcome-based filtering',
        'verification': 'Five task-condition file hashes, complete coverage, all eight aggregate metrics',
        'not_repeated': '10M-row SFT scan, large Arrow SHA, candidate generation and thermodynamic scoring',
        'historical_receipt_sha256': sha(args.source / 'receipt.json'),
        'historical_results_sha256': receipt['results_sha256'],
        'historical_overlap_sha256': receipt['overlap_targets_sha256'],
        'benchmark_sha256': receipt['benchmark_sha256'],
        'sft_arrow_sha256_inherited': receipt['SFT_source']['sha256'],
        'overlap_definition': overlap['definition'],
        'overlap_targets': [{'task_id': r['task_id'], 'sft_matching_rows': r['sft_matching_rows'],
                             'structure_sha256': hashlib.sha256(r['target_structure'].encode()).hexdigest()}
                            for r in overlap['targets']],
        'sources': verified,
    }
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('PASS: five methods, 15 subset rows, eight metrics; full764 unchanged')


if __name__ == '__main__':
    main()
