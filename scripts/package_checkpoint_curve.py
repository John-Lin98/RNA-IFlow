"""Reaggregate late-checkpoint figure points; never select a new main model."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Figure source_data directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    source_raw = (args.source / 'figure2_checkpoint_P1_P8.csv').read_bytes()
    audit_raw = (args.source / 'figure2_P1_P8_audit.json').read_bytes()
    audit = json.loads(audit_raw)
    hashes = {(r['update'], r['path']): r['sha256'] for r in audit['source_files']}
    checkpoints = [r for r in csv.DictReader(source_raw.decode().splitlines())
                   if int(r['update']) in (1744, 2093, 2442, 2790)]
    if sorted(int(r['update']) for r in checkpoints) != [1744, 2093, 2442, 2790]:
        raise ValueError('Late checkpoint coverage mismatch')
    points, evidence = [], []
    for cp in checkpoints:
        update = int(cp['update'])
        path = Path(cp['summary_path'])
        raw = path.read_bytes()
        if sha(raw) != cp['summary_sha256']:
            raise ValueError('Summary hash mismatch')
        summary = json.loads(raw)
        sampler = summary['evaluation_sampler']
        if (summary['status'] != 'complete' or summary['bad_count'] != 0
                or summary['viennarna_version'] != '2.7.2'
                or sampler['seed_temperature_map'] != {'1009': .8, '2027': 1., '3037': 1.2}
                or sampler['trajectory_steps'] != 8 or sampler['candidates'] != 8
                or summary['benchmark_sha256'] != '7514e053c8044d2dc96909e40383474375add1e8b926aa19417980a1e37c9412'):
            raise ValueError('Formal evaluation protocol mismatch')
        groups, file_evidence = defaultdict(dict), []
        files = sorted((path.parent / 'task_progress').glob('task-*.jsonl'))
        if len(files) != 100:
            raise ValueError('Expected 100 task files')
        for file in files:
            candidate_raw = file.read_bytes()
            digest = sha(candidate_raw)
            if digest != hashes[(update, str(file))]:
                raise ValueError('Candidate source hash mismatch')
            file_evidence.append({'file': file.name, 'sha256': digest})
            for line in candidate_raw.decode().splitlines():
                row = json.loads(line)
                key, index = (row['task_id'], int(row['seed'])), int(row['candidate_index'])
                if index in groups[key] or type(row['uMFE_hit']) is not bool:
                    raise ValueError('Duplicate or invalid candidate')
                groups[key][index] = row['uMFE_hit']
        tasks = {task for task, seed in groups}
        if len(tasks) != 100 or set(groups) != {(t, s) for t in tasks for s in (1009, 2027, 3037)}:
            raise ValueError('Task/seed coverage mismatch')
        if any(set(rows) != set(range(8)) for rows in groups.values()):
            raise ValueError('K8 coverage mismatch')
        p1 = statistics.mean(int(rows[0]) for rows in groups.values())
        p8 = statistics.mean(int(any(rows.values())) for rows in groups.values())
        for measured, field, original in [(p1, 'pass_at_1', 'P@1'), (p8, 'pass_at_8', 'P@8')]:
            if abs(measured - summary[field]) > 1e-12 or abs(measured - float(cp[original])) > 1e-12:
                raise ValueError('Candidate/summary/figure disagreement')
        points.append({'update': update, 'pass_at_1': p1, 'pass_at_8': p8,
                       'task_seed_groups': 300, 'candidates': 2400,
                       'role': 'main_model' if update == 2442 else 'plateau_evidence',
                       'checkpoint_sha256': summary['source_sha256']['scale_checkpoint']})
        evidence.append({'update': update, 'summary_sha256': sha(raw),
                         'sampler': sampler, 'candidate_files': file_evidence})
    args.output.mkdir(parents=True)
    with (args.output / 'late_checkpoints.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(points)
    provenance = {'source_csv_sha256': sha(source_raw), 'source_audit_sha256': sha(audit_raw),
                  'scope': 'Four late Eterna100-v2 checkpoints; no selection or new inference',
                  'criterion': 'uMFE_hit; P1=index0, P8=any index0..7', 'checkpoints': evidence}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('PASS: 4 checkpoints, 400 source hashes, 9600 candidates; main remains U2442')


if __name__ == '__main__':
    main()
