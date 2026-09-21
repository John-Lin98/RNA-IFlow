"""Resolve omitted H/G contract references without changing historical tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh evidence file')
    records = []
    for row in csv.DictReader(args.source.open()):
        source = Path(row['source_path'])
        raw_summary = source.read_bytes()
        if sha(raw_summary) != row['source_sha256']:
            raise ValueError('Summary hash mismatch')
        summary = json.loads(raw_summary)
        original = bool(row['contract_path'])
        contract_path = Path(row['contract_path']) if original else (
            source.parent.with_name(source.parent.name.replace('_evaluation', '_formal')) / 'contract.json')
        raw_contract = contract_path.read_bytes()
        if original and sha(raw_contract) != row['contract_sha256']:
            raise ValueError('Original contract file hash mismatch')
        contract = json.loads(raw_contract)
        digest = contract.pop('contract_sha256')
        if sha(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()) != digest:
            raise ValueError('Scientific contract content hash mismatch')
        if summary.get('checkpoint_contract_sha256', digest) != digest:
            raise ValueError('Evaluation/contract link mismatch')
        h, g, seed = int(row['H']), int(row['G']), int(row['training_seed'])
        if (contract['trajectory_steps'], contract['candidates_per_task'], contract['seed']) != (h, g, seed):
            raise ValueError('H/G/seed mismatch')
        receipt_raw = (contract_path.parent / 'receipt.json').read_bytes()
        receipt = json.loads(receipt_raw)
        if receipt['contract_sha256'] != digest or receipt['status'] != 'complete':
            raise ValueError('Training receipt mismatch')
        records.append(dict(H=h, G=g, training_seed=seed,
            reference_resolution='original_table' if original else 'verified_sibling_formal_directory',
            summary_sha256=sha(raw_summary), contract_file_sha256=sha(raw_contract),
            scientific_contract_sha256=digest, training_receipt_sha256=sha(receipt_raw),
            updates_complete=receipt['updates_complete'], repository_revision=contract['repository_revision'],
            temperatures=contract['temperatures'], learning_rate=contract['learning_rate'],
            policy_warmstart=contract.get('policy_warmstart'), method_analysis=contract.get('method_analysis')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(dict(source_sha256=sha(args.source.read_bytes()), records=records,
            status='contract_and_terminal_receipt_verified',
            scope='Does not assert independent checkpoint binary or candidate re-evaluation'), handle, indent=2)
        handle.write('\n')
    print('PASS:', len(records), 'contracts and terminal receipts')


if __name__ == '__main__':
    main()
