"""Package compact R7 figure data, stripping server paths but preserving source hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


FILES = (
    'figure2_density_curve.csv', 'figure2_density_summary.json',
    'flow_state_aggregate.csv', 'policy_mechanics.csv', 'sequence_evolution.csv',
    'flow_initial_state_boxes.json', 'process_arc_selection.json',
    'case_frozen_and_window.json',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    manifest_path = args.source / 'delivery_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    expected = {row['path'].removeprefix('source_data/'): row for row in manifest['files']
                if row['path'].startswith('source_data/')}
    selected = set(FILES) | {'figure2_training.csv', 'figure2_checkpoint_P1_P8.csv', 'coverage_tradeoff.csv'}
    for name in selected:
        path = args.source / 'source_data' / name
        row = expected.get(name)
        if not row or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('R7 delivery identity mismatch: ' + name)
    figure2 = args.output / 'figure2'
    flow = args.output / 'flow'
    case = args.output / 'case'
    figure4 = args.output / 'figure4'
    for directory in (figure2, flow, case, figure4):
        directory.mkdir(parents=True)
    for name in ('figure2_density_curve.csv', 'figure2_density_summary.json'):
        (figure2 / name).write_bytes((args.source / 'source_data' / name).read_bytes())
    for name in ('flow_state_aggregate.csv', 'policy_mechanics.csv', 'sequence_evolution.csv',
                 'flow_initial_state_boxes.json', 'process_arc_selection.json'):
        (flow / name).write_bytes((args.source / 'source_data' / name).read_bytes())
    (case / 'case_frozen_and_window.json').write_bytes(
        (args.source / 'source_data/case_frozen_and_window.json').read_bytes())

    training_path = args.source / 'source_data/figure2_training.csv'
    training = list(csv.DictReader(training_path.open()))
    source_hashes = set()
    for row in training:
        path = Path(row['source_file'])
        digest = sha(path)
        source_hashes.add(digest)
        row['source_file_sha256'] = digest
    write_csv(figure2 / 'training_reward.csv', training,
              ['completed_update', 'reward', 'logged_update', 'refresh_rounds', 'source_file_sha256'])

    checkpoint_path = args.source / 'source_data/figure2_checkpoint_P1_P8.csv'
    checkpoints = list(csv.DictReader(checkpoint_path.open()))
    for row in checkpoints:
        if sha(Path(row['summary_path'])) != row['summary_sha256']:
            raise ValueError('Checkpoint summary hash mismatch')
    write_csv(figure2 / 'checkpoint_P1_P8.csv', checkpoints,
              ['update', 'P@1', 'P@8', 'task_seed_groups', 'candidates', 'summary_sha256'])

    coverage_path = args.source / 'source_data/coverage_tradeoff.csv'
    coverage = list(csv.DictReader(coverage_path.open()))
    for row in coverage:
        if sha(Path(row['source_summary'])) != row['source_summary_sha256']:
            raise ValueError('Coverage summary hash mismatch')
    coverage_fields = [key for key in coverage[0] if key not in ('source_summary', 'source_raw')]
    write_csv(figure4 / 'coverage_tradeoff.csv', coverage, coverage_fields)

    provenance = {
        'status': 'complete',
        'r7_delivery_manifest_sha256': sha(manifest_path),
        'r7_render_script_sha256': '28d33e3ef0d3fa14bf7a2b36bfd23b7b419dc790f36499a329ee1d6eb0b4155b',
        'source_files': {name: expected[name]['sha256'] for name in sorted(selected)},
        'training_history_sources': [
            {'source_file': f'training_history_{index:02d}.jsonl', 'sha256': digest}
            for index, digest in enumerate(sorted(source_hashes), start=1)
        ],
        'verification': 'R7 delivery hashes; referenced training/checkpoint/coverage source hashes',
        'scope': 'Compact source data for Figure2, Flow, Figure4 coverage and case study; no raw candidate distributions',
        'excluded': 'Candidate-level distribution/hit tables and rendered binaries are omitted as duplicate raw/derived assets',
    }
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('PASS: compact source data for four R7 figure families')


if __name__ == '__main__':
    main()
