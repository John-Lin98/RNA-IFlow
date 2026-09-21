"""Resume-safe constrained generation and ViennaRNA 2.7.2 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import RNA
import torch

LEGACY_DIR = Path(__file__).resolve().parents[1] / "dual-prior-rna-flow"
sys.path.append(str(LEGACY_DIR))
from constraints import constrained_decode, target_pairs, valid_pair_fraction  # noqa: E402
from flow_core import DirichletConditionalFlow, simplex_project  # noqa: E402
from model import FairRNAFlow, STRUCTURE_SYMBOLS  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_key(structure: str) -> str:
    return hashlib.sha256(structure.encode()).hexdigest()


def pair_f1(predicted: str, target: str) -> float:
    predicted_pairs = set(target_pairs(predicted))
    target_pair_set = set(target_pairs(target))
    if not predicted_pairs and not target_pair_set:
        return 1.0
    if not predicted_pairs or not target_pair_set:
        return 0.0
    overlap = len(predicted_pairs & target_pair_set)
    return 2 * overlap / (len(predicted_pairs) + len(target_pair_set))


def evaluate_candidate(sequence: str, target: str) -> dict:
    md = RNA.md()
    md.temperature = 37.0
    md.dangles = 2
    md.uniq_ML = 1
    compound = RNA.fold_compound(sequence, md)
    mfe_structure, mfe_energy = compound.mfe()
    mfe_structures = {solution.structure for solution in compound.subopt(0)}
    if not mfe_structures:
        raise RuntimeError("ViennaRNA returned no MFE structure")
    compound.exp_params_rescale(mfe_energy)
    compound.pf()
    probability = float(compound.pr_structure(target))
    ned = float(compound.ensemble_defect(target)) / len(target)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError(f"invalid target probability: {probability}")
    if not math.isfinite(ned) or not 0 <= ned <= 1:
        raise RuntimeError(f"invalid NED: {ned}")
    score_f1 = pair_f1(mfe_structure, target)
    return {
        "mfe_structure": mfe_structure,
        "mfe_energy": float(mfe_energy),
        "mfe_hit": target in mfe_structures,
        "uMFE_hit": len(mfe_structures) == 1 and target in mfe_structures,
        "mfe_degeneracy": len(mfe_structures),
        "pair_f1": score_f1,
        "pair_error": 1 - score_f1,
        "target_probability": probability,
        "NED": ned,
    }


def evaluate_many(sequences: list[str], target: str, workers: int) -> list[dict]:
    items = [(sequence, target) for sequence in sequences]
    if workers == 1:
        return [evaluate_candidate(*item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda item: evaluate_candidate(*item), items))


def structure_inputs(
    model: FairRNAFlow,
    structure: str,
    cached_hidden: torch.Tensor | None,
    candidates: int,
    condition: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    hidden = None
    tokens = torch.tensor(
        [STRUCTURE_SYMBOLS.index(symbol) for symbol in structure], dtype=torch.long, device=device
    ).unsqueeze(0).expand(candidates, -1)
    if cached_hidden is not None:
        hidden = cached_hidden.to(device).float().unsqueeze(0).expand(candidates, -1, -1)
    override = None
    if condition == "zero":
        dim = 64 if model.structure_source == "dotbracket" else hidden.shape[-1]
        override = torch.zeros(candidates, len(structure), dim, device=device)
    elif condition == "shuffle":
        generator = torch.Generator(device=device).manual_seed(9176 + len(structure))
        permutation = torch.randperm(len(structure), generator=generator, device=device)
        if hidden is not None:
            hidden = hidden[:, permutation]
        else:
            tokens = tokens[:, permutation]
    return hidden, tokens, override


class TorchDirichletConditionalFlow:
    """GPU-native equivalent of RNACG's uniform-grid NumPy interpolation."""

    def __init__(self, device: torch.device) -> None:
        reference = DirichletConditionalFlow(alphabet_size=4)
        self.derivatives = torch.from_numpy(reference.beta_cdfs_derivative).to(
            device=device, dtype=torch.float32
        )
        self.alpha_min = float(reference.alphas[0])
        self.alpha_spacing = float(reference.alphas[1] - reference.alphas[0])

    def posterior_to_velocity(
        self, posterior: torch.Tensor, state: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        alpha_value = float(alpha.reshape(-1)[0].item())
        row = int(round((alpha_value - self.alpha_min) / self.alpha_spacing))
        row = min(max(row, 0), self.derivatives.shape[0] - 1)
        probabilities = state.float().clamp(0, 1)
        positions = probabilities * (self.derivatives.shape[1] - 1)
        left = positions.floor().long().clamp(0, self.derivatives.shape[1] - 2)
        fraction = positions - left
        values = self.derivatives[row]
        interpolated = -(values[left] * (1 - fraction) + values[left + 1] * fraction)
        beta = 2.0 / (alpha_value * (alpha_value + 1) * (alpha_value + 2))
        denominator = (1 - probabilities).pow(3) * probabilities.pow(alpha_value - 1)
        ratio = torch.where(
            (probabilities < 1) & (denominator > 0), beta / denominator, torch.zeros_like(denominator)
        )
        coefficient = torch.nan_to_num(interpolated * ratio, nan=0.0, posinf=0.0, neginf=0.0).to(state)
        eye = torch.eye(state.shape[-1], dtype=state.dtype, device=state.device)
        conditional = (eye - state.unsqueeze(-1)) * coefficient.unsqueeze(-2)
        velocity = (posterior.unsqueeze(-2) * conditional).sum(-1)
        return velocity - velocity.mean(dim=-1, keepdim=True)


@torch.inference_mode()
def sample(
    model: FairRNAFlow,
    structure: str,
    cached_hidden: torch.Tensor | None,
    candidates: int,
    steps: int,
    seed: int,
    condition: str,
    device: torch.device,
) -> list[str]:
    generator = torch.Generator(device=device).manual_seed(seed)
    concentrations = torch.ones(candidates, len(structure), 4, device=device)
    state = torch._sample_dirichlet(concentrations, generator=generator)
    mask = torch.ones(candidates, len(structure), device=device)
    hidden, tokens, override = structure_inputs(
        model, structure, cached_hidden, candidates, condition, device
    )
    flow = TorchDirichletConditionalFlow(device)
    schedule = torch.linspace(1.001, 8.0, steps + 1, device=device)
    for index in range(steps):
        alpha = schedule[index].expand(candidates)
        logits = model(
            state,
            alpha,
            mask,
            structure_hidden=hidden,
            structure_tokens=tokens,
            structure_override=override,
        )
        posterior = torch.softmax(logits, dim=-1)
        velocity = flow.posterior_to_velocity(posterior, state, schedule[index])
        state = simplex_project(state + velocity * (schedule[index + 1] - schedule[index]))
    return [constrained_decode(state[index], structure) for index in range(candidates)]


def validate_coverage(rows: list[dict], tasks: list[str], seeds: list[int], candidates: int) -> None:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        groups[(str(row["task_id"]), int(row["seed"]))].append(int(row["candidate_index"]))
    expected = {(str(task), seed) for task in tasks for seed in seeds}
    if set(groups) != expected:
        raise RuntimeError(
            f"coverage group mismatch: missing={sorted(expected - set(groups))[:5]}, "
            f"extra={sorted(set(groups) - expected)[:5]}"
        )
    for key, indices in groups.items():
        if sorted(indices) != list(range(candidates)):
            raise RuntimeError(f"candidate coverage mismatch for {key}: {sorted(indices)}")


def aggregate(rows: list[dict], candidates: int) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_id"]), int(row["seed"]))].append(row)
    task_seed = []
    for (task_id, seed), values in sorted(grouped.items()):
        values.sort(key=lambda row: row["candidate_index"])
        task_seed.append({
            "task_id": task_id,
            "seed": seed,
            "pass_at_1": float(values[0]["uMFE_hit"]),
            f"pass_at_{candidates}": float(any(row["uMFE_hit"] for row in values)),
            "mfe_at_1": float(values[0]["mfe_hit"]),
            f"mfe_at_{candidates}": float(any(row["mfe_hit"] for row in values)),
            "best_pair_f1": max(row["pair_f1"] for row in values),
            "best_target_probability": max(row["target_probability"] for row in values),
            "best_NED": min(row["NED"] for row in values),
            "valid_rate": sum(row["valid"] for row in values) / candidates,
            "valid_pair_rate": sum(row["valid_pair_fraction"] for row in values) / candidates,
            "diversity": len({row["sequence"] for row in values}) / candidates,
        })
    keys = [
        "pass_at_1", f"pass_at_{candidates}", "mfe_at_1", f"mfe_at_{candidates}",
        "best_pair_f1", "best_target_probability", "best_NED", "valid_rate",
        "valid_pair_rate", "diversity",
    ]
    summary = {key: sum(row[key] for row in task_seed) / len(task_seed) for key in keys}
    summary.update({"task_seed_groups": len(task_seed), "candidate_rows": len(rows), "bad_count": 0})
    return task_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rnaernie", type=Path, required=True)
    parser.add_argument("--rnaernie-revision", required=True)
    parser.add_argument("--structure-cache", type=Path)
    parser.add_argument("--structure-cache-manifest", type=Path)
    parser.add_argument("--training-structure-cache-manifest", type=Path)
    parser.add_argument("--condition", choices=["native", "zero", "shuffle"], default="native")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1009, 2027, 3037])
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.eval_workers <= 0 or args.candidates <= 0 or args.steps <= 0:
        raise ValueError("workers, candidates, and flow steps must be positive")
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError("resume output directory is absent")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    progress_dir = args.output / "task_progress"
    progress_dir.mkdir(exist_ok=args.resume)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    contract = checkpoint.get("contract")
    if not contract or checkpoint.get("contract_sha256") != hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest():
        raise RuntimeError("checkpoint training contract is absent or corrupt")
    source = contract["structure_source"]
    cached_source = source.startswith("omnigenome")
    cache_inputs = (
        args.structure_cache,
        args.structure_cache_manifest,
        args.training_structure_cache_manifest,
    )
    if cached_source != all(cache_inputs) or (not cached_source and any(cache_inputs)):
        raise ValueError("cached evaluation requires cache plus training/evaluation manifests")
    if args.rnaernie_revision != contract["rnaernie_revision"]:
        raise RuntimeError("RNAErnie revision does not match training contract")
    if sha256(args.rnaernie / "model.safetensors") != contract["rnaernie_weight_sha256"]:
        raise RuntimeError("RNAErnie weight hash does not match training contract")
    if cached_source:
        training_manifest = json.loads(args.training_structure_cache_manifest.read_text())
        evaluation_manifest = json.loads(args.structure_cache_manifest.read_text())
        if sha256(args.training_structure_cache_manifest) != contract[
            "structure_cache_manifest_sha256"
        ]:
            raise RuntimeError("training cache manifest does not match checkpoint contract")
        identity_keys = ("model_revision", "model_weight_sha256", "hidden_size")
        if any(training_manifest[key] != evaluation_manifest[key] for key in identity_keys):
            raise RuntimeError("training/evaluation structure cache model identity mismatch")
        if evaluation_manifest.get("status") != "complete" or evaluation_manifest.get("bad_count") != 0:
            raise RuntimeError("evaluation structure cache is incomplete")
    if source == "none" and args.condition != "native":
        raise ValueError("unconditioned model has no zero/shuffle control")
    model = FairRNAFlow(
        args.rnaernie,
        structure_source=source,
        injection=contract["injection"],
        structure_dim=contract["structure_dim"],
        lora_rank=contract["lora_rank"],
        lora_alpha=contract["lora_alpha"],
        lora_dropout=contract["lora_dropout"],
    ).to(device)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.eval()
    tasks = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    if args.limit_tasks is not None:
        tasks = tasks[:args.limit_tasks]
    started = time.time()
    all_rows = []
    for task_index, task in enumerate(tasks):
        progress_path = progress_dir / f"task-{task_index:03d}.jsonl"
        if progress_path.is_file():
            task_rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
            validate_coverage(task_rows, [task["id"]], args.seeds, args.candidates)
            all_rows.extend(task_rows)
            continue
        structure = task["target_structure"]
        cached_hidden = None
        if cached_source:
            payload = torch.load(
                args.structure_cache / f"{structure_key(structure)}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if payload["target_structure"] != structure:
                raise RuntimeError("evaluation structure cache key/content mismatch")
            cached_hidden = payload["hidden"]
        task_rows = []
        task_started = time.time()
        for seed in args.seeds:
            sequences = sample(
                model,
                structure,
                cached_hidden,
                args.candidates,
                args.steps,
                seed + task_index * 1_000_003,
                args.condition,
                device,
            )
            metrics = evaluate_many(sequences, structure, args.eval_workers)
            for candidate_index, (sequence, values) in enumerate(zip(sequences, metrics)):
                pair_validity = valid_pair_fraction(sequence, structure)
                valid = len(sequence) == len(structure) and not (set(sequence) - set("ACGU")) and pair_validity == 1
                task_rows.append({
                    "task_id": task["id"],
                    "seed": seed,
                    "candidate_index": candidate_index,
                    "target_structure": structure,
                    "sequence": sequence,
                    "valid": float(valid),
                    "valid_pair_fraction": pair_validity,
                    **values,
                })
        validate_coverage(task_rows, [task["id"]], args.seeds, args.candidates)
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in task_rows))
        os.replace(temporary, progress_path)
        all_rows.extend(task_rows)
        print(json.dumps({
            "tasks_complete": task_index + 1,
            "tasks_total": len(tasks),
            "task_seconds": time.time() - task_started,
        }), flush=True)
    validate_coverage(all_rows, [task["id"] for task in tasks], args.seeds, args.candidates)
    task_seed, summary = aggregate(all_rows, args.candidates)
    summary.update({
        "method": "fair-rna-flow-x0",
        "structure_source": source,
        "injection": contract["injection"],
        "condition": args.condition,
        "flow_steps": args.steps,
        "seeds": args.seeds,
        "runtime_seconds": time.time() - started,
        "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "viennarna_version": RNA.__version__,
        "temperature_c": 37.0,
        "dangles": 2,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_contract_sha256": checkpoint["contract_sha256"],
        "structure_cache_manifest_sha256": sha256(args.structure_cache_manifest) if cached_source else None,
        "training_structure_cache_manifest_sha256": (
            sha256(args.training_structure_cache_manifest) if cached_source else None
        ),
        "coverage": {"tasks": len(tasks), "seeds": len(args.seeds), "candidates": args.candidates},
    })
    (args.output / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows)
    )
    (args.output / "task_seed.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in task_seed)
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
