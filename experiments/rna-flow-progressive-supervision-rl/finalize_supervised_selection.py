"""Create the immutable supervised-selection receipt after the 10M queue completes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


class SelectionPending(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_sha256(contract: dict) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def candidate_from_attempt(attempt: Path) -> dict:
    receipt_path = attempt / "train_receipt.json"
    best_path = attempt / "best.json"
    if not receipt_path.is_file() or not best_path.is_file():
        raise SelectionPending(f"10M attempt is not terminal: {attempt}")
    receipt = json.loads(receipt_path.read_text())
    best = json.loads(best_path.read_text())
    contract = receipt.get("contract")
    if (
        receipt.get("status") != "complete"
        or receipt.get("nan_inf_count") != 0
        or not contract
        or receipt.get("contract_sha256") != contract_sha256(contract)
    ):
        raise RuntimeError(f"invalid completed training receipt: {receipt_path}")
    checkpoint = Path(best["path"])
    if not checkpoint.is_file() or sha256(checkpoint) != best.get("sha256"):
        raise RuntimeError(f"best checkpoint hash mismatch: {checkpoint}")
    state = receipt.get("training_state", {})
    if (
        best.get("selection_score") != state.get("best_score")
        or best.get("metrics") != state.get("best_metrics")
        or best.get("epoch") != state.get("best_epoch")
    ):
        raise RuntimeError(f"best pointer disagrees with training state: {attempt}")
    metrics = best["metrics"]
    if (
        metrics.get("bad_count") != 0
        or metrics.get("valid_rate") != 1.0
        or metrics.get("valid_pair_rate") != 1.0
    ):
        raise RuntimeError(f"best checkpoint failed quality gate: {attempt}")
    return {
        "attempt": str(attempt),
        "attempt_name": attempt.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": best["sha256"],
        "epoch": best["epoch"],
        "metrics": metrics,
        "selection_score": best["selection_score"],
        "contract_sha256": receipt["contract_sha256"],
        "train_receipt_sha256": sha256(receipt_path),
        "best_pointer_sha256": sha256(best_path),
        "thermodynamic_validation": contract["thermodynamic_validation"],
        "selection_order": contract["selection_order"],
        "method": contract["method"],
        "backbone_mode": contract["backbone_mode"],
        "lora_rank": contract["lora_rank"],
        "training_data": {
            "train_sha256": contract["train_sha256"],
            "train_rows": contract["train_rows"],
            "nested_data": contract.get("nested_data"),
            "heldout_benchmark": contract.get("heldout_benchmark"),
        },
        "code": contract["code"],
        "rnaernie_revision": contract["rnaernie_revision"],
        "rnaernie_weight_sha256": contract["rnaernie_weight_sha256"],
    }


def finalize(queue_state: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"selection receipt already exists: {output}")
    queue = json.loads(queue_state.read_text())
    if queue.get("fatal_failure") is not None:
        raise RuntimeError(f"progressive queue failed: {queue['fatal_failure']}")
    if queue.get("status") != "complete":
        raise SelectionPending(f"progressive queue status is {queue.get('status')!r}")
    statuses = queue.get("stage_statuses", {}).get("stage_10m", {})
    if len(statuses) < 3 or any(value.get("status") != "complete" for value in statuses.values()):
        raise SelectionPending("10M candidate set is not complete")
    candidates = [
        candidate_from_attempt(Path(value["output"]))
        for _, value in sorted(statuses.items())
    ]
    validation = candidates[0]["thermodynamic_validation"]
    order = candidates[0]["selection_order"]
    for candidate in candidates[1:]:
        if candidate["thermodynamic_validation"] != validation:
            raise RuntimeError("10M candidates used different validation contracts")
        if candidate["selection_order"] != order:
            raise RuntimeError("10M candidates used different selection orders")
        if candidate["training_data"] != candidates[0]["training_data"]:
            raise RuntimeError("10M candidates used different frozen training data")
        if candidate["code"] != candidates[0]["code"]:
            raise RuntimeError("10M candidates used different source contracts")
    selected = max(candidates, key=lambda value: tuple(value["selection_score"]))
    receipt = {
        "status": "complete",
        "role": "frozen supervised selection; Eterna/public benchmark not accessed",
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_attempt": selected["attempt"],
        "selected_attempt_name": selected["attempt_name"],
        "selected_method": selected["method"],
        "selected_backbone_mode": selected["backbone_mode"],
        "selected_lora_rank": selected["lora_rank"],
        "epoch": selected["epoch"],
        "metrics": selected["metrics"],
        "selection_score": selected["selection_score"],
        "selection_order": order,
        "thermodynamic_validation": validation,
        "training_data": selected["training_data"],
        "code": selected["code"],
        "candidates": candidates,
        "queue_state": str(queue_state),
        "queue_state_sha256": sha256(queue_state),
        "created_unix": time.time(),
    }
    atomic_json(output, receipt)
    return receipt


def validate_selection(path: Path, queue_state: Path | None = None) -> dict:
    """Rebuild the immutable selection decision from its primary evidence."""
    receipt = json.loads(path.read_text())
    queue_path = queue_state or Path(receipt.get("queue_state", ""))
    if not queue_path.is_file() or receipt.get("queue_state_sha256") != sha256(queue_path):
        raise RuntimeError(f"selection queue evidence mismatch: {path}")
    queue = json.loads(queue_path.read_text())
    statuses = queue.get("stage_statuses", {}).get("stage_10m", {})
    if (
        queue.get("status") != "complete"
        or len(statuses) < 3
        or any(value.get("status") != "complete" for value in statuses.values())
    ):
        raise RuntimeError(f"selection queue is not complete: {queue_path}")
    candidates = [
        candidate_from_attempt(Path(value["output"]))
        for _, value in sorted(statuses.items())
    ]
    selected = max(candidates, key=lambda value: tuple(value["selection_score"]))
    validation = candidates[0]["thermodynamic_validation"]
    order = candidates[0]["selection_order"]
    if any(
        candidate["thermodynamic_validation"] != validation
        or candidate["selection_order"] != order
        or candidate["training_data"] != candidates[0]["training_data"]
        or candidate["code"] != candidates[0]["code"]
        for candidate in candidates[1:]
    ):
        raise RuntimeError("selection candidates use inconsistent frozen contracts")
    expected = {
        "status": "complete",
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_attempt": selected["attempt"],
        "selected_attempt_name": selected["attempt_name"],
        "selection_score": selected["selection_score"],
        "selection_order": selected["selection_order"],
        "thermodynamic_validation": selected["thermodynamic_validation"],
        "training_data": selected["training_data"],
        "code": selected["code"],
        "candidates": candidates,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"selection receipt disagrees with primary evidence: {path}")
    return receipt


def validate_authorized_checkpoint(selection_path: Path, checkpoint: Path) -> dict:
    receipt = validate_selection(selection_path)
    if receipt.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("supervised selection receipt does not authorize this checkpoint")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = finalize(args.queue_state, args.output)
    except SelectionPending as error:
        print(str(error), flush=True)
        raise SystemExit(75) from error
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
