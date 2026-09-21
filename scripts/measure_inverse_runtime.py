"""Portable RNAinverse-pf K8 native-return timing; not resident neural latency.

Adapted from measure_search_return_k8.py, original SHA256
175dd6f32085e3fa8d24f9cb7484e36f16d63ce1204ddd39ddc9b3135b2463f2.
No promise of historical sequence identity: ViennaRNA internal RNG is unchanged.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

BENCHMARK_SHA = '7514e053c8044d2dc96909e40383474375add1e8b926aa19417980a1e37c9412'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def candidate():
    started = time.perf_counter()
    import RNA
    request = json.load(sys.stdin)
    if RNA.__version__ != '2.7.2':
        raise RuntimeError('ViennaRNA 2.7.2 required')
    setup = time.perf_counter() - started
    started = time.perf_counter()
    result = RNA.inverse_pf_fold(request['start'], request['target'])
    seconds = time.perf_counter() - started
    sequence = str(result[0] if isinstance(result, (list, tuple)) else result).upper().replace('T', 'U')
    if len(sequence) != len(request['target']) or set(sequence) - set('ACGU'):
        raise RuntimeError('Invalid native inverse output')
    print(json.dumps(dict(sequence=sequence, native_search_seconds=seconds,
                          RNA_import_setup_seconds=setup, status='ok', viennarna_version=RNA.__version__)))


def generate(index, target, seed):
    rng = random.Random(seed * 1000003 + index)
    rows = []
    for k in range(8):
        start = ''.join(rng.choice('AUGC') for _ in target)
        started = time.perf_counter()
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--candidate-mode'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(json.dumps(dict(start=start, target=target)), timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            row = dict(sequence='', status='timeout', native_search_seconds=None)
        else:
            if proc.returncode:
                raise RuntimeError('RNAinverse child failed: ' + stderr[-600:])
            row = json.loads(stdout.splitlines()[-1])
        row.update(candidate_index=k, start_sequence_sha256=hashlib.sha256(start.encode()).hexdigest(),
                   return_wall_seconds=time.perf_counter() - started)
        rows.append(row)
    return rows


def run_group(spec, output):
    from paper_metrics import evaluate_candidate
    index, task, seed = spec
    directory = output / f'task-{index:03d}-seed-{seed}'
    directory.mkdir()
    rows = generate(index, task['target_structure'], seed)
    started = time.perf_counter()
    scored = [evaluate_candidate(r['sequence'], task['target_structure']) if r['sequence'] else
              dict(generation_failed=True, mfe_hit=False, uMFE_hit=False, target_probability=0., NED=1., pair_f1=0.)
              for r in rows]
    post = time.perf_counter() - started
    dump(directory / 'candidates.json', [dict(m, sequence=r['sequence'], candidate_index=r['candidate_index'])
                                          for r, m in zip(rows, scored)])
    generation = sum(r['return_wall_seconds'] for r in rows)
    receipt = dict(status='complete', method='RNAinverse-pf', task_id=str(task['id']),
                   global_target_index=index, condition_seed=seed, K=8, run_time_seconds=generation,
                   post_scoring_seconds=post, end_to_end_seconds=generation + post,
                   returned_valid_candidates=sum(bool(r['sequence']) for r in rows),
                   timeout_candidates=sum(r['status'] == 'timeout' for r in rows), candidate_timings=rows,
                   P1=float(scored[0]['uMFE_hit']), P8=float(any(m['uMFE_hit'] for m in scored)))
    dump(directory / 'receipt.json', receipt)
    return receipt


def main():
    if '--candidate-mode' in sys.argv:
        candidate()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-first-task', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    raw = args.tasks.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BENCHMARK_SHA:
        raise ValueError('Frozen Eterna100-v2 bytes/order required')
    tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    specs = [(i, task, seed) for i, task in enumerate(tasks) for seed in (1009, 2027, 3037)]
    if args.smoke_first_task:
        specs = specs[:1]
    # Match original native child thread limits, not an inference throughput tuning knob.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import RNA
    if RNA.__version__ != '2.7.2':
        raise RuntimeError('ViennaRNA 2.7.2 required')
    # Import once before worker threads modify module search paths.
    import paper_metrics
    args.output.mkdir(parents=True)
    plan = dict(method='RNAinverse-pf', benchmark_sha256=BENCHMARK_SHA, K=8, workers=4,
                groups=len(specs), hard_timeout_seconds_per_candidate=45,
                scope='smoke' if args.smoke_first_task else 'native_search_reproduction',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                internal_rng='ViennaRNA library default; no historical sequence identity claim',
                scoring='Corrected paper NED; differs from historical timing-only extra-normalized NED')
    dump(args.output / 'plan.json', plan)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda spec: run_group(spec, args.output), specs))
    returned = sum(r['returned_valid_candidates'] for r in results)
    dump(args.output / 'summary.json', dict(
        status='complete_smoke' if args.smoke_first_task else 'complete', method='RNAinverse-pf',
        task_condition_groups=len(results), K=8, attempted_candidates=8 * len(results),
        returned_candidates=returned, failed_candidates=8 * len(results) - returned,
        timeout_candidates=sum(r['timeout_candidates'] for r in results),
        run_time_seconds_per_K8_group_median=statistics.median(r['run_time_seconds'] for r in results),
        post_scoring_seconds_per_K8_group_median=statistics.median(r['post_scoring_seconds'] for r in results),
        end_to_end_seconds_per_K8_group_median=statistics.median(r['end_to_end_seconds'] for r in results),
        failure_semantics='Timeout slots stay in denominators; child/scorer errors abort without completion',
        groups=results))


if __name__ == '__main__':
    main()
