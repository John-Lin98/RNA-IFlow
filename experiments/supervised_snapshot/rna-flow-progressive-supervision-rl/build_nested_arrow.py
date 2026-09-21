"""Build one canonical Arrow stream with frozen 100K/1M/3M/10M prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


RNA = set("ACGU")
DOT_BRACKET = set("().")
BOUNDARIES = (100_000, 1_000_000, 3_000_000, 10_000_000)


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


def valid_structure(structure: str) -> bool:
    if not structure or set(structure) - DOT_BRACKET:
        return False
    depth = 0
    for symbol in structure:
        depth += symbol == "("
        depth -= symbol == ")"
        if depth < 0:
            return False
    return depth == 0


def valid_pair(structure: str, sequence: str) -> bool:
    return (
        valid_structure(structure)
        and bool(sequence)
        and len(structure) == len(sequence)
        and len(structure) <= 510
        and not (set(sequence) - RNA)
    )


def pair_digest(structure: str, sequence: str) -> bytes:
    return hashlib.sha256(structure.encode() + b"\0" + sequence.encode()).digest()


def membership_update(digest: hashlib._Hash, structure: str, sequence: str) -> None:
    digest.update(structure.encode())
    digest.update(b"\0")
    digest.update(sequence.encode())
    digest.update(b"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-shards", type=Path, required=True)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--eterna100v2", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    source_manifest = json.loads(args.source_manifest.read_text())
    if source_manifest.get("status") != "complete" or source_manifest.get("bad_count") != 0:
        raise RuntimeError("official source manifest is incomplete")
    expected_files = {
        row["name"]: row for row in source_manifest["official_source"]["files"]
    }
    actual_files = sorted(args.source_shards.glob("part-*.parquet"))
    if [path.name for path in actual_files] != sorted(expected_files):
        raise RuntimeError("official source shard set differs from the frozen manifest")
    for path in actual_files:
        expected = expected_files[path.name]
        if sha256(path) != expected["sha256"] or pq.ParquetFile(path).metadata.num_rows != expected["rows"]:
            raise RuntimeError(f"official source shard changed: {path.name}")

    base_rows = read_jsonl(args.base_train)
    if len(base_rows) != BOUNDARIES[0]:
        raise RuntimeError(f"base prefix is not exactly 100K: {len(base_rows)}")
    validation_structures = {row["target_structure"] for row in read_jsonl(args.validation)}
    benchmark_structures = {row["target_structure"] for row in read_jsonl(args.eterna100v2)}
    excluded_structures = validation_structures | benchmark_structures

    schema = pa.schema([
        ("target_structure", pa.string()),
        ("sequence", pa.string()),
        ("length", pa.uint16()),
        ("source_shard", pa.string()),
        ("source_row", pa.int64()),
    ])
    arrow_path = args.output / "nested_10m.arrow"
    sink = pa.OSFile(str(arrow_path), "wb")
    writer = ipc.new_file(sink, schema)
    buffers = {name: [] for name in schema.names}
    seen: set[bytes] = set()
    membership = hashlib.sha256()
    stage_membership: dict[str, str] = {}
    counters = Counter()
    row_count = 0

    def flush() -> None:
        if not buffers["target_structure"]:
            return
        writer.write_batch(pa.record_batch(buffers, schema=schema))
        for values in buffers.values():
            values.clear()

    def append(structure: str, sequence: str, source_shard: str, source_row: int) -> None:
        nonlocal row_count
        digest = pair_digest(structure, sequence)
        if digest in seen:
            counters["duplicate_pair_removed"] += 1
            return
        seen.add(digest)
        membership_update(membership, structure, sequence)
        buffers["target_structure"].append(structure)
        buffers["sequence"].append(sequence)
        buffers["length"].append(len(structure))
        buffers["source_shard"].append(source_shard)
        buffers["source_row"].append(source_row)
        row_count += 1
        if row_count in BOUNDARIES:
            stage_membership[str(row_count)] = membership.copy().hexdigest()
        if len(buffers["target_structure"]) >= 8192:
            flush()

    for index, row in enumerate(base_rows):
        structure = row["target_structure"]
        sequence = row["sequence"].upper().replace("T", "U")
        if not valid_pair(structure, sequence) or structure in excluded_structures:
            raise RuntimeError(f"frozen 100K prefix violates the new contract at row {index}")
        append(structure, sequence, "frozen-100k", index)
    if row_count != BOUNDARIES[0]:
        raise RuntimeError("frozen 100K prefix contains duplicate pairs")

    for shard in actual_files:
        parquet = pq.ParquetFile(shard)
        source_row = 0
        for batch in parquet.iter_batches(batch_size=8192, columns=["messages"]):
            for messages in batch.column(0).to_pylist():
                counters["source_rows_scanned"] += 1
                current_row = source_row
                source_row += 1
                if not messages or len(messages) < 2:
                    counters["malformed_messages"] += 1
                    continue
                if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                    counters["role_mismatch"] += 1
                    continue
                structure = messages[0]["content"].strip()
                sequence = messages[1]["content"].strip().upper().replace("T", "U")
                if structure in excluded_structures:
                    counters["excluded_structure_removed"] += 1
                    continue
                if not valid_pair(structure, sequence):
                    counters["invalid_or_too_long_removed"] += 1
                    continue
                append(structure, sequence, shard.name, current_row)
                if row_count == BOUNDARIES[-1]:
                    break
            if row_count == BOUNDARIES[-1]:
                break
        if row_count == BOUNDARIES[-1]:
            break

    flush()
    writer.close()
    sink.close()
    if row_count != BOUNDARIES[-1] or set(stage_membership) != {str(value) for value in BOUNDARIES}:
        raise RuntimeError(
            f"canonical stream did not reach every boundary: rows={row_count}, stages={stage_membership}"
        )

    manifest = {
        "status": "complete",
        "contract_version": 1,
        "format": "Arrow IPC file",
        "file": {
            "path": str(arrow_path),
            "rows": row_count,
            "sha256": sha256(arrow_path),
            "bytes": arrow_path.stat().st_size,
        },
        "prefixes": {
            str(boundary): {
                "rows": boundary,
                "membership_sha256": stage_membership[str(boundary)],
            }
            for boundary in BOUNDARIES
        },
        "source_manifest": {
            "path": str(args.source_manifest),
            "sha256": sha256(args.source_manifest),
        },
        "frozen_100k": {
            "path": str(args.base_train),
            "sha256": sha256(args.base_train),
        },
        "exclusions": {
            "validation_path": str(args.validation),
            "validation_sha256": sha256(args.validation),
            "validation_structures": len(validation_structures),
            "eterna100v2_path": str(args.eterna100v2),
            "eterna100v2_sha256": sha256(args.eterna100v2),
            "eterna100v2_structures": len(benchmark_structures),
            "exact_structure_overlap_validation": 0,
            "exact_structure_overlap_eterna100v2": 0,
        },
        "counters": dict(counters),
        "bad_count": 0,
    }
    manifest_path = args.output / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
