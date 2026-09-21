"""Package existing dataset audit aggregates; do not rescan or distribute data."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output')
    raw = args.source.read_bytes()
    data = json.loads(raw)
    if data['status'] != 'verified_streaming':
        raise ValueError('Unverified source audit')
    for key in ('SFT_source_shard_counts', 'SFT_length_histogram', 'SFT_length_bins'):
        if sum(data[key].values()) != data['SFT_rows']:
            raise ValueError('SFT aggregate count mismatch')
    if sum(data['RL_length_bins'].values()) != data['RL_rows']:
        raise ValueError('RL aggregate count mismatch')
    references = []
    for source, expected in data['source_sha256'].items():
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError('Dataset audit reference hash mismatch')
        references.append({'source_file': Path(source).name, 'sha256': digest})
    data['source_sha256'] = references
    data['large_arrow_source'].pop('path')
    data['release_verification'] = {
        'audit_source_sha256': hashlib.sha256(raw).hexdigest(),
        'scope': 'Reference hashes and aggregate count consistency verified; original streaming overlap audit inherited, not rerun',
        'raw_data_redistributed': False,
        'large_arrow_rehashed': False,
    }
    with args.output.open('x') as handle:
        handle.write(json.dumps(data, indent=2) + '\n')
    print('PASS:', len(references), 'reference hashes; aggregate counts consistent; overlap scan inherited')


if __name__ == '__main__':
    main()
