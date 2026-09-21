"""Freeze official source evidence and the 100K thermodynamic-selection contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


SOURCE_REVISION = "609f573b1a373a4f4e6748057c90dcca07e02294"
CODE_REVISION = "8ad11f7b50fd65da893097934c5aab94508115a5"
EXPECTED_SHARDS = {
    "part-00000.parquet": (999656, "173a94a72035f588dcea2b235f7505c8d6b04c3b06fac43c82f16738bb3bb5e8"),
    "part-00001.parquet": (999992, "71461b5c1cfbc029cff75d0f92e5359e12984652c1f560e130d4dd9d3ad3cf45"),
    "part-00002.parquet": (999984, "069ed247b06140c2692e75168947acccc57f0398f483a760f3fc8e86682950a7"),
    "part-00003.parquet": (1000000, "3fc0cbfa41029b2afcde89380dd7ef2518c5adcb9b644eb9aa1012c969317b08"),
    "part-00004.parquet": (999982, "43938046e89ac2ce38de98b5447606f94202a0d34592be33954ec7250910806b"),
    "part-00005.parquet": (1000000, "79b4d985717eedf458c4fa76e535796b447e6c593307a9da987a8e3bc014224c"),
    "part-00006.parquet": (1000000, "d0f7dba248ff77a5a0177072696acb6d8d5d3ddfda36834f17dbf78ff12d81ce"),
    "part-00007.parquet": (991210, "b3c50b6c64d47180cf5b158d58ffd77e616ee8c78bc0fdd216148b80ab9eb781"),
    "part-00008.parquet": (1000000, "9ae2f1b480f019174011eebccdab6f020e9c5aa3b123e3ae64d64989fb125390"),
    "part-00009.parquet": (590250, "caca4b5d8d008058842f0484f7afd759679f17d7b3d9e5272feb5d06cb6bd087"),
    "part-00010.parquet": (896644, "5f870a71020353d566e050f0e2fc48397951f68f51095b2d8264149920a3df45"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_rank(seed: int, structure: str) -> str:
    return hashlib.sha256(f"{seed}\0{structure}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-shards", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9176)
    parser.add_argument("--tasks-per-bin", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    code_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.official_code,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if code_head != CODE_REVISION:
        raise RuntimeError(f"official code revision mismatch: {code_head}")
    license_path = args.official_code / "LICENSE"
    if "MIT License" not in license_path.read_text():
        raise RuntimeError("official code license is not the expected MIT text")

    source_files = []
    total_rows = 0
    for name, (expected_rows, expected_sha) in EXPECTED_SHARDS.items():
        path = args.source_shards / name
        actual_sha = sha256(path)
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        if actual_sha != expected_sha or rows != expected_rows:
            raise RuntimeError(
                f"source mismatch for {name}: rows={rows}/{expected_rows}, sha={actual_sha}/{expected_sha}"
            )
        schema = parquet.schema_arrow
        source_files.append({
            "name": name,
            "rows": rows,
            "sha256": actual_sha,
            "has_quality_columns": name == "part-00010.parquet",
            "schema": str(schema),
        })
        total_rows += rows

    base_manifest_path = args.base_data / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    train_path = args.base_data / "train.jsonl"
    validation_path = args.base_data / "validation.jsonl"
    benchmark_path = args.base_data / "eterna100v2.jsonl"
    expected_base = base_manifest["files"]
    for name, path in {
        "train.jsonl": train_path,
        "validation.jsonl": validation_path,
        "eterna100v2.jsonl": benchmark_path,
    }.items():
        if sha256(path) != expected_base[name]["sha256"]:
            raise RuntimeError(f"base data hash mismatch: {name}")

    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    benchmark_rows = read_jsonl(benchmark_path)
    train_structures = {row["target_structure"] for row in train_rows}
    benchmark_structures = {row["target_structure"] for row in benchmark_rows}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in validation_rows:
        grouped[row["length_bin"]].append(row)
    selected = []
    for length_bin in sorted(grouped):
        ranked = sorted(
            grouped[length_bin],
            key=lambda row: stable_rank(args.seed, row["target_structure"]),
        )
        if len(ranked) < args.tasks_per_bin:
            raise RuntimeError(f"not enough validation rows in {length_bin}")
        for row in ranked[: args.tasks_per_bin]:
            structure = row["target_structure"]
            selected.append({
                "id": f"thermo-{hashlib.sha256(structure.encode()).hexdigest()[:16]}",
                "target_structure": structure,
                "length": len(structure),
                "length_bin": length_bin,
            })
    selected.sort(key=lambda row: row["id"])
    selected_structures = {row["target_structure"] for row in selected}
    if selected_structures & train_structures or selected_structures & benchmark_structures:
        raise RuntimeError("thermodynamic validation overlap with train or benchmark")
    thermo_path = args.output / "thermo_validation_48.jsonl"
    thermo_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))

    rnaernie_weight = args.rnaernie / "model.safetensors"
    manifest = {
        "status": "complete",
        "contract_version": 1,
        "official_source": {
            "repository": "Milanmg/LLM-RNA-Design-2026",
            "revision": SOURCE_REVISION,
            "rows": total_rows,
            "files": source_files,
            "part_00010_policy": (
                "same structure-to-sequence schema as the other SL shards; prob/ned/id are provenance-only "
                "and are never used as supervised labels"
            ),
        },
        "official_code": {
            "repository": "KuNyaa/RNA-Design-LM",
            "revision": CODE_REVISION,
            "license": "MIT",
            "license_sha256": sha256(license_path),
        },
        "rnaernie": {
            "revision": "df26681ed44bb671b61536c5e2a196d6058a44f7",
            "weight_sha256": sha256(rnaernie_weight),
        },
        "nested_policy": (
            "the frozen 100K train.jsonl is the exact prefix of every later stage; later rows are appended "
            "from official shards in lexical shard/row order after validation, benchmark, invalid, and duplicate filtering"
        ),
        "stage_100k": {
            "path": str(train_path),
            "rows": len(train_rows),
            "sha256": sha256(train_path),
            "unique_pairs": len({(row["target_structure"], row["sequence"]) for row in train_rows}),
        },
        "thermodynamic_validation": {
            "path": str(thermo_path),
            "rows": len(selected),
            "sha256": sha256(thermo_path),
            "seed": args.seed,
            "tasks_per_length_bin": args.tasks_per_bin,
            "structure_overlap_train": len(selected_structures & train_structures),
            "structure_overlap_eterna100v2": len(selected_structures & benchmark_structures),
        },
        "eterna100v2": {
            "path": str(benchmark_path),
            "rows": len(benchmark_rows),
            "sha256": sha256(benchmark_path),
            "selection_role": "monitor-only after best checkpoint freeze",
        },
        "bad_count": 0,
    }
    manifest_path = args.output / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
