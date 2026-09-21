"""Portable five-model resident K=8 runtime entrypoint.

The runner records fresh evidence only.  It keeps all five models resident,
warms each model once, enforces the frozen candidate prefixes, and times only
CUDA-synchronized generation.  Model assets and GoForth source stay external.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from runtime_samplers import (
    BENCHMARK_SHA256,
    CHECKPOINT_SHA256,
    K,
    METHODS,
    SEEDS,
    SOURCE_CANDIDATE_SHA256,
    build_samplers,
    release_scorer,
    require_vienna_version,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequences: list[str]) -> str:
    return hashlib.sha256(("\n".join(sequences)).encode()).hexdigest()


def candidate_filename(method: str) -> str:
    return method.replace(" ", "_").replace("+", "plus") + "__eterna100v2.jsonl"


def load_tasks(path: Path) -> tuple[list[dict[str, Any]], str]:
    observed = sha256(path)
    if observed != BENCHMARK_SHA256:
        raise ValueError(f"Eterna100-v2 benchmark SHA256 mismatch: {observed}")
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(tasks) != 100 or len({str(task.get("id")) for task in tasks}) != 100:
        raise ValueError("Eterna100-v2 must contain 100 unique tasks")
    for task in tasks:
        structure = task.get("target_structure")
        if not isinstance(structure, str) or not structure or set(structure) - set(".()"):
            raise ValueError("Eterna100-v2 contains an invalid target structure")
    for index, task in enumerate(tasks):
        task["_global_index"] = index
    return tasks, observed


def load_expected_candidates(
    root: Path, tasks: list[dict[str, Any]]
) -> tuple[dict[str, dict[tuple[str, int], list[str]]], dict[str, str]]:
    task_ids = {str(task["id"]) for task in tasks}
    expected_keys = {(task_id, seed) for task_id in task_ids for seed in SEEDS}
    all_expected: dict[str, dict[tuple[str, int], list[str]]] = {}
    digests: dict[str, str] = {}
    for method in METHODS:
        path = Path(root) / candidate_filename(method)
        observed = sha256(path)
        if observed != SOURCE_CANDIDATE_SHA256[method]:
            raise ValueError(f"{method} candidate SHA256 mismatch: {observed}")
        groups: dict[tuple[str, int], list[tuple[int, str]]] = collections.defaultdict(list)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task_id"]), int(row["seed"]))
            groups[key].append((int(row["candidate_index"]), str(row["sequence"])))
        if set(groups) != expected_keys or any(len(values) != K for values in groups.values()):
            raise ValueError(f"{method} candidate coverage mismatch")
        result: dict[tuple[str, int], list[str]] = {}
        for key, values in groups.items():
            ordered = sorted(values)
            if [index for index, _sequence in ordered] != list(range(K)):
                raise ValueError(f"{method} candidate indices mismatch for {key}")
            result[key] = [sequence for _index, sequence in ordered]
        all_expected[method] = result
        digests[method] = observed
    return all_expected, digests


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def aggregate_rows(
    rows: list[dict[str, Any]], plan: dict[str, Any], identities: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    methods = []
    expected_rows = plan["tasks"] * len(SEEDS) * plan["repeats"]
    for method in plan["methods"]:
        values = [row for row in rows if row["method"] == method]
        groups: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
        for row in values:
            groups[(row["task_id"], row["condition_seed"])].append(row)
        per_group = [statistics.median(row["generation_seconds"] for row in group) for group in groups.values()]
        post_group = [statistics.median(row["post_scoring_seconds"] for row in group) for group in groups.values()]
        e2e_group = [
            statistics.median(row["generation_seconds"] + row["post_scoring_seconds"] for row in group)
            for group in groups.values()
        ]
        if not values:
            continue
        bins = {}
        for label, low, high in (("<=64", 0, 64), ("65-128", 65, 128), ("129-256", 129, 256), (">256", 257, 9999)):
            subset = [row for row in values if low <= row["length"] <= high]
            if subset:
                bins[label] = {
                    "timed_groups": len(subset),
                    "generation_seconds_median": statistics.median(
                        row["generation_seconds"] for row in subset
                    ),
                    "returned_candidates": sum(row["returned_candidates"] for row in subset),
                }
        methods.append(
            {
                "method": method,
                "status": "complete" if len(values) == expected_rows else "partial",
                "timed_groups": len(values),
                "target_condition_groups": len(groups),
                "repeats": plan["repeats"],
                "run_time_seconds_per_K8_group": statistics.median(per_group),
                "mean_seconds_per_K8_group": statistics.mean(per_group),
                "generation_ms_per_candidate": statistics.median(per_group) * 1000 / K,
                "returned_candidates_per_generation_second": sum(
                    row["returned_candidates"] for row in values
                )
                / sum(row["generation_seconds"] for row in values),
                "post_scoring_seconds_per_K8_group": statistics.median(post_group),
                "end_to_end_seconds_per_K8_group": statistics.median(e2e_group),
                "source_candidate_identity_mismatches": sum(
                    row["candidate_identity_mismatches"] for row in values
                ),
                "generation_failure_groups": sum(row["returned_candidates"] != K for row in values),
                "length_bins": bins,
                "identity": identities[method],
            }
        )
    return {
        "status": "complete" if len(methods) == len(METHODS) and all(item["status"] == "complete" for item in methods) else "running",
        "methods": methods,
        "plan": plan,
        "time_unix": time.time(),
    }


def run_timing(
    *,
    tasks: list[dict[str, Any]],
    samplers: dict[str, Callable[[dict[str, Any], int], list[str]]],
    expected: dict[str, dict[tuple[str, int], list[str]]],
    identities: dict[str, dict[str, Any]],
    output: Path,
    plan: dict[str, Any],
    score: Callable[[list[str], str], list[dict[str, Any]]],
    synchronize: Callable[[], None],
    time_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Run the timing protocol; ``synchronize`` and ``score`` are injectable for orchestration tests."""
    names = list(plan["methods"])
    warm = max(tasks, key=lambda task: len(task["target_structure"]))
    for name in names:
        synchronize()
        sequences = samplers[name](warm, 1009)
        synchronize()
        frozen = expected[name][(str(warm["id"]), 1009)]
        if sequences != frozen:
            raise RuntimeError(f"warm-up candidate identity mismatch: {name}/{warm['id']}/1009")

    rows: list[dict[str, Any]] = []
    ledger = Path(output) / "timing_ledger.jsonl"
    with ledger.open("x", encoding="utf-8") as handle:
        for replicate in range(plan["repeats"]):
            ordered_tasks = tasks if replicate % 2 == 0 else list(reversed(tasks))
            for loop_index, task in enumerate(ordered_tasks):
                for condition_seed in SEEDS:
                    offset = (loop_index * 3 + SEEDS.index(condition_seed)) % len(names)
                    method_order = names[offset:] + names[:offset]
                    if replicate % 2:
                        method_order = list(reversed(method_order))
                    for method in method_order:
                        synchronize()
                        started = time_ns()
                        sequences = samplers[method](task, condition_seed)
                        synchronize()
                        generation_seconds = (time_ns() - started) / 1e9
                        if not _finite(generation_seconds) or generation_seconds <= 0:
                            raise RuntimeError(
                                f"invalid generation duration {method}/{task['id']}/{condition_seed}: "
                                f"{generation_seconds}"
                            )
                        frozen = expected[method][(str(task["id"]), condition_seed)]
                        mismatch = sum(
                            actual != wanted for actual, wanted in zip(sequences, frozen)
                        ) + abs(len(sequences) - K)
                        if mismatch:
                            raise RuntimeError(
                                f"candidate identity mismatch {method}/{task['id']}/{condition_seed}: {mismatch}"
                            )
                        started = time_ns()
                        metrics = score(sequences, task["target_structure"])
                        post_scoring_seconds = (time_ns() - started) / 1e9
                        if not _finite(post_scoring_seconds) or post_scoring_seconds < 0:
                            raise RuntimeError(
                                f"invalid scoring duration {method}/{task['id']}/{condition_seed}: "
                                f"{post_scoring_seconds}"
                            )
                        if not isinstance(metrics, (list, tuple)) or len(metrics) != K:
                            raise RuntimeError(
                                f"scorer returned {len(metrics) if hasattr(metrics, '__len__') else 'non-sized'} "
                                f"metrics for {method}/{task['id']}/{condition_seed}; expected {K}"
                            )
                        if any(item.get("evaluation_valid") is False for item in metrics):
                            raise RuntimeError(f"scorer failure: {method}/{task['id']}/{condition_seed}")
                        row = {
                            "method": method,
                            "task_id": str(task["id"]),
                            "global_target_index": task["_global_index"],
                            "length": len(task["target_structure"]),
                            "condition_seed": condition_seed,
                            "replicate": replicate,
                            "generation_seconds": generation_seconds,
                            "post_scoring_seconds": post_scoring_seconds,
                            "returned_candidates": len(sequences),
                            "candidate_identity_mismatches": mismatch,
                            "candidate_sequence_sha256": sequence_sha256(sequences),
                        }
                        rows.append(row)
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        handle.flush()
    result = aggregate_rows(rows, plan, identities)
    result["ledger_sha256"] = sha256(ledger)
    return result


def _gpu_uuid(index: int) -> str:
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="RNA-IFlow release root")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--sft-checkpoint", type=Path, required=True)
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--rl-model", type=Path, required=True, help="complete U2442 export directory")
    parser.add_argument("--rna-dlm-root", type=Path, required=True, help="directory containing SL and SL+RL")
    parser.add_argument("--goforth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goforth-cache", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.repeats <= 0 or args.eval_workers <= 0:
        parser.error("gpu must be nonnegative; repeats and eval-workers must be positive")
    if args.repeats != 2 or args.eval_workers != 4:
        parser.error("formal resident runtime freezes repeats=2 and eval-workers=4")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError("fresh output required; refusing an existing output directory")
    tasks, benchmark_sha = load_tasks(args.benchmark)
    expected, candidate_sha = load_expected_candidates(args.candidate_root, tasks)
    selected = [tasks[0], max(tasks, key=lambda task: len(task["target_structure"]))] if args.preflight else tasks
    for task in selected:
        task["_global_index"] = next(index for index, item in enumerate(tasks) if item["id"] == task["id"])

    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal --gpu for physical-device accounting")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("resident runtime requires exactly one visible CUDA GPU")
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu_uuid = _gpu_uuid(args.gpu)
    output = args.output
    output.mkdir(parents=True)
    cache = args.goforth_cache or output / "goforth_runtime_cache"
    try:
        with torch.inference_mode():
            viennarna_version = require_vienna_version()
            samplers, identities, retained = build_samplers(
                repo_root=args.repo_root,
                sft_checkpoint=args.sft_checkpoint,
                rnaernie=args.rnaernie,
                rl_model=args.rl_model,
                rna_dlm_root=args.rna_dlm_root,
                goforth_root=args.goforth_root,
                goforth_cache=cache,
                device=device,
            )
            if tuple(samplers) != METHODS:
                raise RuntimeError(f"method order drifted: {tuple(samplers)}")
            plan = {
                "status": "admitted",
                "scope": "preflight" if args.preflight else "full100_target_interleaved_paired_runtime",
                "tasks": len(selected),
                "conditions": list(SEEDS),
                "K": K,
                "repeats": args.repeats,
                "gpu": args.gpu,
                "gpu_uuid": gpu_uuid,
                "methods": list(METHODS),
                "benchmark_sha256": benchmark_sha,
                "source_candidates_sha256": candidate_sha,
                "ordering": "cyclic rotated methods per target-condition; reverse task order and method order on repeat2",
                "models_resident_simultaneously": True,
                "torch_cpu_threads": 2,
                "scoring_processes": args.eval_workers,
                "viennarna_version": viennarna_version,
                "timing_definition": "seconds to return8sequences excluding loading/lookup setup/warmup; subsequent uniform Vienna scoring separate",
                "shared_hardware": True,
                "script_sha256": sha256(Path(__file__)),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "started_unix": time.time(),
            }
            (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
            scorer_impl = release_scorer(args.repo_root)
            scorer = lambda sequences, structure: scorer_impl(
                sequences, structure, args.eval_workers
            )
            result = run_timing(
                tasks=selected,
                samplers=samplers,
                expected=expected,
                identities=identities,
                output=output,
                plan=plan,
                score=scorer,
                synchronize=lambda: torch.cuda.synchronize(device),
            )
            (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"status": result["status"], "methods": len(result["methods"])}, sort_keys=True))
    except Exception as error:
        (output / "failure.json").write_text(
            json.dumps({"status": "blocked", "error": repr(error), "time_unix": time.time()}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
