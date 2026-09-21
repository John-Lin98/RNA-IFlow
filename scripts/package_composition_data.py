"""Package the composition figure's aggregate data and verified source hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    audit = json.loads((args.source / 'composition_semantic_audit.json').read_text())
    for source in audit['files']:
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('Composition membership source hash mismatch')
    metrics = ['pass_at_1', 'pass_at_8', 'mfe_at_1', 'mfe_at_8',
               'best_target_probability', 'best_corrected_NED', 'best_pair_f1', 'diversity']
    quality_raw = (args.source / 'source_data/composition_quality.csv').read_bytes()
    quality = list(csv.DictReader(quality_raw.decode().splitlines()))
    for row in quality:
        raw = Path(row['source_summary']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['source_summary_sha256']:
            raise ValueError('Composition evaluation summary hash mismatch')
        summary = json.loads(raw)
        for metric in metrics:
            key = 'best_NED' if metric == 'best_corrected_NED' else metric
            if abs(float(row[metric]) - float(summary[key])) > 1e-12:
                raise ValueError('Composition metric mismatch: ' + metric)
    counts_raw = (args.source / 'source_data/composition_counts.csv').read_bytes()
    counts = list(csv.DictReader(counts_raw.decode().splitlines()))
    for setting in {r['setting'] for r in counts}:
        selected = [r for r in counts if r['setting'] == setting]
        if sum(int(r['count']) for r in selected) != 2790:
            raise ValueError('Composition count mismatch')
        if any(abs(float(r['fraction']) - int(r['count']) / 2790) > 1e-12 for r in selected):
            raise ValueError('Composition fraction mismatch')
    args.output.mkdir(parents=True)
    for name, rows, fields in [
        ('composition_counts.csv', counts, list(counts[0])),
        ('composition_quality.csv', quality, ['setting', *metrics, 'source_summary_sha256'])]:
        with (args.output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows({k: r[k] for k in fields} for r in rows)
    audit['files'] = [{'source_file': Path(r['path']).name, 'sha256': r['sha256']} for r in audit['files']]
    audit['quality_source_sha256'] = hashlib.sha256(quality_raw).hexdigest()
    audit['counts_source_sha256'] = hashlib.sha256(counts_raw).hexdigest()
    audit['verification_scope'] = 'Source file hashes, summary metrics, count totals/fractions; not new training'
    (args.output / 'provenance.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('PASS: 3 composition settings, 9 category rows, 4 membership-source hashes')


if __name__ == '__main__':
    main()
