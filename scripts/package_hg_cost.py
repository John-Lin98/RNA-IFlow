"""Verify Figure 2 H/G cost data separately from continuation-seed quality SD."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readcsv(path):
    return list(csv.DictReader(path.open()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figure-source', type=Path, required=True)
    p.add_argument('--profile-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    source = args.figure_source / 'HG_per_seed.csv'
    rows = readcsv(source)
    costs, evidence = [], {}
    for h in (1, 2, 4, 8, 16):
        matches = [r for r in rows if int(r['H']) == h and int(r['G']) == 8 and int(r['training_seed']) == 1009]
        if len(matches) != 1:
            raise ValueError('Missing or duplicate H cost row')
        row = matches[0]
        path = Path(row['generation_latency_source'])
        digest = sha(path)
        if digest != row['generation_latency_source_sha256']:
            raise ValueError('Inference timing source SHA mismatch')
        summary = json.loads(path.read_text())
        arm = summary['arms'][f'H{h}']
        value = float(row['inference_seconds_per_K8_group'])
        if not math.isfinite(value) or abs(value - arm['generation_latency_ms_per_group_median'] / 1000) > 1e-12:
            raise ValueError('Inference cost mismatch')
        costs.append(dict(axis='H', value=h, seconds=value, scope='12 length-stratified targets; K8 inference; shared-card profile'))
        evidence['inference_summary_sha256'] = digest
    updates_path = args.profile_source / 'g_training_profile_updates.csv'
    updates = readcsv(updates_path)
    profile = readcsv(args.figure_source / 'HG_cost_profile.csv')
    receipts = {}
    for row in updates:
        path = Path(row['source_path'])
        if path not in receipts:
            receipts[path] = sha(path)
        if receipts[path] != row['source_sha256']:
            raise ValueError('Training cost receipt SHA mismatch')
    for g in (2, 4, 8, 16):
        selected = [r for r in updates if int(r['G']) == g and int(r['H']) == 8]
        if len(selected) != 16 or {(int(r['replicate']), int(r['update'])) for r in selected} != {(rep, u) for rep in (1, 2) for u in range(1, 9)}:
            raise ValueError('Expected two profiles of eight updates')
        values = [float(r['seconds_per_update']) for r in selected]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid update cost')
        matches = [r for r in profile if int(r['G']) == g and int(r['H']) == 8]
        if len(matches) != 1:
            raise ValueError('Missing or duplicate G cost row')
        median = statistics.median(values)
        if abs(median - float(matches[0]['seconds_per_update_median'])) > 1e-12:
            raise ValueError('Training cost median mismatch')
        costs.append(dict(axis='G', value=g, seconds=median, scope=matches[0]['timing_scope']))
    evidence.update(figure_per_seed_sha256=sha(source), profile_rows_sha256=sha(updates_path),
                    figure_cost_sha256=sha(args.figure_source / 'HG_cost_profile.csv'),
                    profile_receipts=[dict(source_file=p.name, sha256=h) for p, h in receipts.items()],
                    verification='H costs checked against recorded summary; G medians reaggregated over 64 update rows; receipt hashes verified',
                    limitations='No timing rerun; no raw inference-ledger reaggregation; shared-hardware cost is not uncertainty across training seeds')
    args.output.mkdir(parents=True)
    with (args.output / 'HG_cost.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(costs[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(costs)
    (args.output / 'provenance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print('PASS: five H inference costs, four G medians, 64 update rows')


if __name__ == '__main__':
    main()
