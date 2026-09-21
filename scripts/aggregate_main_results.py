"""Aggregate existing normalized candidates; no inference and no SD formatting."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

EXPECTED = {'eterna100v2': (100, .54, .65), 'eterna100': (100, .5066667, .6166667),
            'rfam27': (27, .8518519, .8765432), 'rnasolo764': (764, .7120419, .7312391)}
SEEDS = {1009, 2027, 3037}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    conditions, summary, sources = [], [], []
    for benchmark, (count, p1, p8) in EXPECTED.items():
        path = args.input / ('RNA-IFlow-RL__' + benchmark + '.jsonl')
        raw = path.read_bytes()
        groups = collections.defaultdict(list)
        for line in raw.splitlines():
            row = json.loads(line)
            assert type(row['uMFE_hit']) is bool and type(row['mfe_hit']) is bool
            assert row['evaluation_valid'] is True and row['generation_failed'] is False
            groups[row['seed'], row['task_id']].append(row)
        assert {seed for seed, _ in groups} == SEEDS
        assert len(groups) == count * 3
        task_sets = [{task for seed, task in groups if seed == s} for s in sorted(SEEDS)]
        assert len(task_sets[0]) == count and all(t == task_sets[0] for t in task_sets)
        for group in groups.values():
            assert len(group) == 8 and {x['candidate_index'] for x in group} == set(range(8))
        for seed in sorted(SEEDS):
            selected = [rows for (s, _), rows in groups.items() if s == seed]
            first = [next(x for x in rows if x['candidate_index'] == 0) for rows in selected]
            conditions.append(dict(benchmark=benchmark, seed=seed, tasks=count, k=8,
                pass_at_1=sum(x['uMFE_hit'] for x in first)/count,
                pass_at_8=sum(any(x['uMFE_hit'] for x in rows) for rows in selected)/count,
                mfe_at_1=sum(x['mfe_hit'] for x in first)/count,
                mfe_at_8=sum(any(x['mfe_hit'] for x in rows) for rows in selected)/count,
                bad_count=0))
        current = conditions[-3:]
        means = {key: sum(x[key] for x in current)/3 for key in ('pass_at_1','pass_at_8','mfe_at_1','mfe_at_8')}
        assert abs(means['pass_at_1']-p1)<1e-7 and abs(means['pass_at_8']-p8)<1e-7
        summary.append(dict(benchmark=benchmark, model='RNA-IFlow-RL', checkpoint='U2442', **means))
        sources.append(dict(source_file=path.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    args.output.mkdir(parents=True)
    for name, rows in [('per_condition_metrics.csv',conditions), ('main_model_metrics.csv',summary)]:
        with (args.output/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.output/'aggregation_provenance.json').write_text(json.dumps(dict(
        sources=sources, metric='P@k uses uMFE_hit; MFE@k uses mfe_hit',
        coverage_validation='PASS', frozen_main_values='PASS',
        protocol_binding='normalized rows omit temperature; original evaluation contract binding still required',
        manuscript_comparison='pending latest approved manuscript readback',
        training_seed_claim=False),indent=2)+'\n')
    print('AGGREGATION_PASS: 4 benchmarks, 12 seed conditions; not a final SD table')


if __name__ == '__main__':
    main()
