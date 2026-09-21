"""Build the exact 100k/5k fair-comparison data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


RNA = set("ACGU")
DOT_BRACKET = set("().")
SOURCE_REVISION = "609f573b1a373a4f4e6748057c90dcca07e02294"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pairs(structure: str) -> list[tuple[int, int]]:
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    if not structure or set(structure) - DOT_BRACKET:
        raise ValueError("invalid dot-bracket alphabet")
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced close")
            pairs.append((stack.pop(), index))
    if stack:
        raise ValueError("unbalanced open")
    return pairs


def valid_pair(structure: str, sequence: str) -> bool:
    return (
        bool(sequence)
        and len(structure) == len(sequence)
        and not (set(sequence) - RNA)
        and not (set(structure) - DOT_BRACKET)
    )


def read_benchmark_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            structure = row.get("target_structure") or row.get("structure")
            parse_pairs(structure)
            rows.append({"id": row.get("id", len(rows) + 1), "target_structure": structure})
    return rows


def read_desirna_directory(path: Path, source: str) -> list[dict]:
    rows = []
    for file_path in sorted(path.glob("*.txt")):
        lines = [line.strip() for line in file_path.read_text().splitlines()]
        fields = {
            lines[index][1:]: lines[index + 1]
            for index in range(0, len(lines) - 1, 2)
            if lines[index].startswith(">")
        }
        structure = fields["sec_struct"]
        parse_pairs(structure)
        rows.append({
            "id": fields.get("name", file_path.stem),
            "target_structure": structure,
            "source": source,
            "source_file": file_path.name,
            "source_sha256": sha256(file_path),
        })
    return rows


def iter_sl_pairs(path: Path) -> Iterable[tuple[str, str]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192, columns=["messages"]):
        for messages in batch.column(0).to_pylist():
            if not messages or len(messages) < 2:
                continue
            structure = messages[0]["content"].strip()
            sequence = messages[1]["content"].strip().upper().replace("T", "U")
            yield structure, sequence


def length_bin(length: int) -> str:
    if length <= 64:
        return "000-064"
    if length <= 128:
        return "065-128"
    if length <= 256:
        return "129-256"
    return "257-510"


def allocations(sizes: dict[str, int], count: int) -> dict[str, int]:
    total = sum(sizes.values())
    exact = {key: size * count / total for key, size in sizes.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(result.values())
    ranking = sorted(sizes, key=lambda key: (exact[key] - result[key], key), reverse=True)
    for key in ranking[:remaining]:
        result[key] += 1
    return result


def split_exact(
    sequences_by_structure: dict[str, list[str]], train_count: int, validation_count: int, seed: int
) -> tuple[list[dict], list[dict], dict]:
    if len(sequences_by_structure) <= validation_count:
        raise RuntimeError("not enough independent structures for validation")
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for structure in sequences_by_structure:
        groups[length_bin(len(structure))].append(structure)
    for values in groups.values():
        rng.shuffle(values)
    val_alloc = allocations({key: len(value) for key, value in groups.items()}, validation_count)
    validation_structures: set[str] = set()
    for key, values in groups.items():
        validation_structures.update(values[: val_alloc[key]])

    validation = [{
        "target_structure": structure,
        "sequence": sequences_by_structure[structure][0],
        "length": len(structure),
        "length_bin": length_bin(len(structure)),
    } for structure in validation_structures]
    train = [{
        "target_structure": structure,
        "sequence": sequences[0],
        "length": len(structure),
        "length_bin": length_bin(len(structure)),
    } for structure, sequences in sequences_by_structure.items() if structure not in validation_structures]
    if len(train) > train_count:
        rng.shuffle(train)
        train = train[:train_count]
    extras = [{
        "target_structure": structure,
        "sequence": sequence,
        "length": len(structure),
        "length_bin": length_bin(len(structure)),
    } for structure, sequences in sequences_by_structure.items()
      if structure not in validation_structures for sequence in sequences[1:]]
    rng.shuffle(extras)
    needed = train_count - len(train)
    if needed > len(extras):
        raise RuntimeError(f"not enough unique training pairs: need extras={needed}, have={len(extras)}")
    train.extend(extras[:needed])
    rng.shuffle(train)
    rng.shuffle(validation)
    metadata = {
        "unique_structures": len(sequences_by_structure),
        "train_unique_structures": len({row["target_structure"] for row in train}),
        "validation_unique_structures": len(validation_structures),
        "train_extra_sequence_pairs": needed,
    }
    return train, validation, metadata


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sl-parquet", type=Path, required=True)
    parser.add_argument("--eterna-v1-dir", type=Path, required=True)
    parser.add_argument("--eterna-v2-dir", type=Path, required=True)
    parser.add_argument("--extra-benchmark-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=100_000)
    parser.add_argument("--validation-count", type=int, default=5_000)
    parser.add_argument("--smoke-train-count", type=int, default=1_000)
    parser.add_argument("--smoke-validation-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1009)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    eterna_v1 = read_desirna_directory(args.eterna_v1_dir, "Eterna100-v1/DesiRNA")
    eterna_v2 = read_desirna_directory(args.eterna_v2_dir, "Eterna100-v2/DesiRNA")
    if len(eterna_v1) != 100 or len(eterna_v2) != 100:
        raise RuntimeError(f"benchmark cardinality mismatch: v1={len(eterna_v1)} v2={len(eterna_v2)}")
    benchmark_rows = eterna_v1 + eterna_v2
    for path in args.extra_benchmark_jsonl:
        benchmark_rows.extend(read_benchmark_jsonl(path))
    excluded = {row["target_structure"] for row in benchmark_rows}

    sequences_by_structure: dict[str, list[str]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    counters = Counter()
    for structure, sequence in iter_sl_pairs(args.sl_parquet):
        counters["source_rows"] += 1
        try:
            parse_pairs(structure)
            valid = valid_pair(structure, sequence)
        except ValueError:
            valid = False
        if not valid or len(structure) > 510:
            counters["invalid_or_too_long"] += 1
            continue
        if structure in excluded:
            counters["benchmark_overlap_removed"] += 1
            continue
        pair = (structure, sequence)
        if pair in seen_pairs:
            counters["duplicate_pair_removed"] += 1
            continue
        seen_pairs.add(pair)
        sequences_by_structure.setdefault(structure, []).append(sequence)

    train, validation, split_metadata = split_exact(
        sequences_by_structure, args.train_count, args.validation_count, args.seed
    )
    if len(train) != args.train_count or len(validation) != args.validation_count:
        raise RuntimeError(f"exact cardinality failed: train={len(train)} validation={len(validation)}")
    train_pairs = {(row["target_structure"], row["sequence"]) for row in train}
    validation_pairs = {(row["target_structure"], row["sequence"]) for row in validation}
    structure_overlap = {row["target_structure"] for row in train} & {
        row["target_structure"] for row in validation
    }
    if len(train_pairs) != len(train) or len(validation_pairs) != len(validation) or structure_overlap:
        raise RuntimeError("split uniqueness or structure isolation failed")

    outputs = {
        "train.jsonl": train,
        "validation.jsonl": validation,
        "smoke_train.jsonl": train[: args.smoke_train_count],
        "smoke_validation.jsonl": validation[: args.smoke_validation_count],
        "eterna100v2.jsonl": eterna_v2,
    }
    for name, rows in outputs.items():
        write_jsonl(args.out / name, rows)
    benchmark_overlap = sum(row["target_structure"] in excluded for row in train + validation)
    manifest = {
        "contract_version": 2,
        "split_policy": (
            "validation has unique structures; train/validation structures are disjoint; "
            "train contains unique (structure,sequence) pairs and may contain multiple sequences per structure"
        ),
        "seed": args.seed,
        "source": {
            "path": str(args.sl_parquet),
            "sha256": sha256(args.sl_parquet),
            "revision": SOURCE_REVISION,
            "repository": "Milanmg/LLM-RNA-Design-2026",
        },
        "benchmark_sources": [
            {"path": str(args.eterna_v1_dir)},
            {"path": str(args.eterna_v2_dir)},
        ] + [{"path": str(path), "sha256": sha256(path)} for path in args.extra_benchmark_jsonl],
        "counts": dict(counters) | split_metadata | {
            "unique_valid_pairs": len(seen_pairs),
            "train": len(train),
            "validation": len(validation),
            "smoke_train": len(outputs["smoke_train.jsonl"]),
            "smoke_validation": len(outputs["smoke_validation.jsonl"]),
            "eterna100v2": len(eterna_v2),
        },
        "length_bins_train": dict(Counter(row["length_bin"] for row in train)),
        "length_bins_validation": dict(Counter(row["length_bin"] for row in validation)),
        "exact_structure_overlap_train_validation": len(structure_overlap),
        "exact_structure_overlap_benchmarks": benchmark_overlap,
        "duplicate_pairs_after": len(train) + len(validation) - len(train_pairs | validation_pairs),
        "files": {},
    }
    for name, rows in outputs.items():
        manifest["files"][name] = {"sha256": sha256(args.out / name), "rows": len(rows)}
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if benchmark_overlap:
        raise RuntimeError(f"decontamination failed: {benchmark_overlap} overlaps remain")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
