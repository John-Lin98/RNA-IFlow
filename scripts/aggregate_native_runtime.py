"""Package Table 2 native-search timing separately from resident generation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    output, evidence = [], []
    for method in ['DRAG', 'RNAinverse-pf']:
        raw = (args.input / method / 'formal/summary.json').read_bytes()
        summary = json.loads(raw)
        groups = summary['groups']
        keys = {(r['task_id'], r['condition_seed']) for r in groups}
        task_sets = [{t for t, s in keys if s == seed} for seed in (1009, 2027, 3037)]
        if len(groups) != 300 or len(keys) != 300 or len(task_sets[0]) != 100 or any(t != task_sets[0] for t in task_sets):
            raise ValueError('Native timing task/seed coverage mismatch')
        if summary['status'] != 'complete' or summary['K'] != 8 or any(r['K'] != 8 for r in groups):
            raise ValueError('Native timing status/budget mismatch')
        median = statistics.median(r['run_time_seconds'] for r in groups)
        if abs(median - summary['run_time_seconds_per_K8_group_median']) > 1e-12:
            raise ValueError('Native timing median mismatch')
        returned = sum(r['returned_valid_candidates'] for r in groups)
        if returned != summary['returned_candidates'] or 2400 - returned != summary['failed_candidates']:
            raise ValueError('Native returned/failed count mismatch')
        output.append(dict(method=method, k=8, task_condition_groups=300,
            native_return_seconds_median=median, attempted_candidates=2400,
            returned_candidates=returned, failed_candidates=2400-returned,
            timeout_candidates=summary['timeout_candidates'], timing_scope=summary['scope']))
        evidence.append(dict(method=method, summary_sha256=hashlib.sha256(raw).hexdigest(),
                             plan_sha256=summary['plan_sha256'], failure_semantics=summary['failure_semantics']))
    args.output.mkdir(parents=True)
    with (args.output / 'native_runtime.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(output)
    (args.output / 'provenance.json').write_text(json.dumps(dict(
        sources=evidence, verification='300 groups per method, median and returned/failed counts recomputed',
        scope='Native return/search including setup; not resident neural generation',
        quality_table_link='Timing run is distinct from original Table 1 accuracy ledger; no sequence identity asserted'), indent=2) + '\n')
    print('PASS: native medians and failure denominators; 600 groups')


if __name__ == '__main__':
    main()
