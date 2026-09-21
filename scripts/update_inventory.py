"""Index only release-repository files; never scan external experiment storage."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'paper_release_inventory.tsv'


def main():
    old = {r['destination_path']: r for r in csv.DictReader(INVENTORY.open(), delimiter='\t')}
    for source in json.loads((ROOT / 'results/provenance/supervised_source_manifest.json').read_text()):
        old[source['destination_path']] = dict(
            source, role='training', paper_artifact='Supervised RNA-IFlow parent',
            include_exclude='include', reason='Exact supervised-training source dependency; Git blob verified')
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    ).decode().split('\0')
    rows = []
    for name in sorted(set(names) - {'', INVENTORY.name}):
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            raise ValueError('Expected regular release file: ' + name)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) > 5_000_000 or path.suffix in {'.pt', '.pth', '.bin', '.safetensors'}:
            raise ValueError('Unexpected binary/large release asset: ' + name)
        role = 'docs'
        if name.startswith('results/'):
            role = 'results'
        elif name.startswith('model/'):
            role = 'model_reference'
        elif name.startswith('scripts/'):
            role = 'analysis' if any(x in name for x in ('aggregate', 'table')) else 'inference'
            if 'inventory' in name:
                role = 'docs'
        row = old.get(name, dict(destination_path=name, source_path='release:' + name,
            role=role, paper_artifact='Release reproduction; see docs/experiments.md',
            source_revision='release-working-tree', sha256=digest, size=str(len(raw)),
            include_exclude='include', reason='Release-authored artifact; final QA pending'))
        if row['source_path'].startswith('release:'):
            row.update(sha256=digest, size=str(len(raw)),
                       reason='Release-authored artifact; QA status in docs/release_checks.md')
        row.update(destination_sha256=digest, destination_size=str(len(raw)))
        rows.append(row)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    INVENTORY.write_text(output.getvalue())
    print('INVENTORY:', len(rows), 'files;', sum(int(r['destination_size']) for r in rows), 'bytes')


if __name__ == '__main__':
    main()
