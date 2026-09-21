"""Package verified H/G figure inputs without private paths or raw candidates."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    raw = args.source.read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    metrics = ['pass_at_1', 'pass_at_8', 'best_target_probability', 'best_NED', 'diversity']
    compact, groups = [], collections.defaultdict(list)
    for row in rows:
        source = Path(row['source_path']).read_bytes()
        if hashlib.sha256(source).hexdigest() != row['source_sha256']:
            raise ValueError('Summary hash mismatch')
        summary = json.loads(source)
        values = {k: float(row[k]) for k in metrics}
        if any(abs(values[k] - float(summary[k])) > 1e-12 for k in metrics):
            raise ValueError('Metric mismatch with summary')
        contract_sha = ''
        if row['contract_path']:
            contract_sha = hashlib.sha256(Path(row['contract_path']).read_bytes()).hexdigest()
            if contract_sha != row['contract_sha256']:
                raise ValueError('Contract file hash mismatch')
        item = dict(H=int(row['H']), G=int(row['G']), training_seed=int(row['training_seed']),
                    **values, checkpoint_sha256=row['checkpoint_sha256'],
                    summary_sha256=row['source_sha256'], contract_file_sha256=contract_sha)
        compact.append(item)
        groups[item['H'], item['G']].append(item)
    means = []
    for (h, g), items in sorted(groups.items()):
        if len(items) != 3 or {r['training_seed'] for r in items} != {1009, 2027, 3037}:
            raise ValueError('Expected three distinct continuation training seeds')
        record = dict(H=h, G=g, n=3)
        for metric in metrics:
            values = [r[metric] for r in items]
            record[metric + '_mean'] = statistics.mean(values)
            record[metric + '_sd'] = statistics.stdev(values)
        means.append(record)
    args.output.mkdir(parents=True)
    for name, values in [('HG_per_seed.csv', compact), ('HG_mean_sd.csv', means)]:
        with (args.output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        source_file=args.source.name, source_sha256=hashlib.sha256(raw).hexdigest(),
        scope='H/G continuation-seed figure inputs; not main Table 1 SD formatting',
        sd='sample standard deviation, ddof=1',
        verification='All summary hashes and metric values checked; available contract file hashes checked',
        missing_contract_references=sum(not r['contract_file_sha256'] for r in compact),
        limitations='Some source rows omit contract references; protocol/config closure and latest manuscript mapping remain pending'), indent=2) + '\n')
    print('PASS:', len(compact), 'per-seed rows;', len(means), 'three-seed settings')


if __name__ == '__main__':
    main()
