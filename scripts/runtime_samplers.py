"""Lazy, path-parameterized samplers for the five resident runtime methods.

This module deliberately imports no model/runtime dependency at import time.  The
release owns RNA-IFlow code; RNA-DLM and GoForth are loaded from user-supplied
external paths and are never copied into the release tree.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


METHODS = (
    "RNA-IFlow",
    "RNA-IFlow-RL",
    "RNA-Design-LM SL",
    "RNA-Design-LM SL+RL",
    "GoForth",
)
SEEDS = (1009, 2027, 3037)
K = 8
FLOW_STEPS = 50
RL_STEPS = 8
RL_TEMPERATURES = {1009: 0.8, 2027: 1.0, 3037: 1.2}
DLM_TEMPERATURE = 2.0
GOFORTH_TEMPERATURE = 0.1
VIENNARNA_VERSION = "2.7.2"

BENCHMARK_SHA256 = "7514e053c8044d2dc96909e40383474375add1e8b926aa19417980a1e37c9412"
SOURCE_CANDIDATE_SHA256 = {
    "GoForth": "4fb872ba59fab57c8d923dafcd52682cdf0ad698aec640e0299b664e9bbb6caf",
    "RNA-Design-LM SL": "7bf0224ebc45438ef8cee2d3a2d63832b777ba81a9d598d356ba8eb7f46934ba",
    "RNA-Design-LM SL+RL": "d0aeb3bcbfd16bfc5801c0168ccfc7c9214060b71bd6b11e2824397e6abf5688",
    "RNA-IFlow": "c3e18f53527e237a1bf58e6eb4e2efd8c24f51c9dbebb1fc969c858bcb56e83c",
    "RNA-IFlow-RL": "da6bbcb19272ddae1cd61e56d70cd5610a5987235552972cc53c6c7a0c19b7cb",
}
CHECKPOINT_SHA256 = {
    "RNA-IFlow": "8e221cc4c4382a5421890724c0548cf64492fe1be4556d498878e86e83056bc5",
    "RNA-IFlow-RL": "198fd79e7680f4b01e758f063aadab00c0d4e6709ac4d2a220249a267a70ebe8",
    "RNA-Design-LM SL": "a466271bfdbf108bbb55aa19a30cdc485b4ce2bbb2d648cae1d85e29a5aabe8c",
    "RNA-Design-LM SL+RL": "970a3132fcb64c95141faa3c3bc041eccf38ecfa8dc2890936cdbd467d041741",
    "GoForth": "a28e650ba0a8fd61a92ade424d939f3d95631df63f6f9f2ae78a5f81932b472f",
}
RNAERNIE_WEIGHT_SHA256 = "0321a1402074927e1163027b425ae2d8bd5f26e577c87e51fbc632984eb233d3"
PORTABLE_EXPORT_SHA256 = "8a8dcf74014be2e3ab9571a19facebaad639ca4fd6574bc97d7bdf5d9ffe9f9d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected: str | None = None, label: str = "file") -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    if expected is not None:
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"{label} SHA256 mismatch: {path}: {observed}")
    return path


def require_vienna_version() -> str:
    """Require the exact ViennaRNA version frozen by the runtime protocol."""
    import RNA

    observed = str(RNA.__version__)
    if observed != VIENNARNA_VERSION:
        raise RuntimeError(
            f"resident runtime requires ViennaRNA {VIENNARNA_VERSION}; observed {observed}"
        )
    return observed


def _json_asset_hashes(model_dir: Path) -> tuple[dict[str, str], str]:
    """Hash local model/tokenizer JSON metadata without embedding its path."""
    metadata = {
        path.name: sha256(path)
        for path in sorted(Path(model_dir).glob("*.json"))
        if path.is_file()
    }
    if not metadata:
        raise FileNotFoundError(f"RNA-DLM config/tokenizer JSON files are absent: {model_dir}")
    manifest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return metadata, manifest


def _release_paths(repo_root: Path) -> tuple[Path, Path, Path]:
    root = Path(repo_root).resolve()
    scripts = root / "scripts"
    fair = root / "experiments/rna-flow-fair-components"
    progressive = root / "experiments/rna-flow-progressive-supervision-rl"
    legacy = root / "experiments/dual-prior-rna-flow"
    for path in (scripts, progressive, fair, legacy):
        if not path.is_dir():
            raise FileNotFoundError(f"release source directory is absent: {path}")
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return scripts, fair, progressive


def _import_external_server(root: Path):
    server_path = Path(root) / "apps/rna_workbench/server.py"
    require_file(server_path, label="GoForth server")
    module_name = "rna_iflow_runtime_goforth_" + hashlib.sha1(str(server_path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, server_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load GoForth server: {server_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _rna_dlm_template() -> str:
    return (
        "{% for message in messages %}"
        "{% if message.role == 'structure' %}<struct>{{ message.content }}</struct>"
        "{% elif message.role == 'sequence' %}{{ bos_token }}{{ message.content }}{{ eos_token }}"
        "{% endif %}{% endfor %}"
    )


def _build_tokenizer(path: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    tokenizer.chat_template = _rna_dlm_template()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _pair_map(structure: str) -> dict[int, int]:
    stack: list[int] = []
    result: dict[int, int] = {}
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced target structure")
            opening = stack.pop()
            result[index] = opening
            result[opening] = index
    if stack:
        raise ValueError("unbalanced target structure")
    return result


def _constrained_tokens(tokenizer, structure: str, prefix_length: int):
    import torch

    complements = {"A": "U", "C": "G", "G": "CU", "U": "AG"}
    base_ids = {base: tokenizer.convert_tokens_to_ids(base) for base in "ACGU"}
    if len(set(base_ids.values())) != 4:
        raise RuntimeError("RNA-DLM tokenizer does not expose four distinct nucleotide tokens")
    id_to_base = {token_id: base for base, token_id in base_ids.items()}
    partners = _pair_map(structure)

    def allowed(_batch_id: int, current_ids: torch.Tensor) -> list[int]:
        position = current_ids.shape[-1] - prefix_length
        if position >= len(structure):
            return [tokenizer.eos_token_id]
        partner = partners.get(position)
        if partner is None or partner > position:
            return list(base_ids.values())
        opening_id = int(current_ids[prefix_length + partner].item())
        return [base_ids[base] for base in complements[id_to_base[opening_id]]]

    return allowed


def _prompt_and_positions(tokenizer, structure: str) -> tuple[list[int], list[int]]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "structure", "content": structure}],
        tokenize=False,
        add_generation_prompt=True,
    ) + (tokenizer.bos_token or "")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    structure_ids = tokenizer.encode(structure, add_special_tokens=False)
    if len(structure_ids) != len(structure):
        raise RuntimeError("RNA-DLM tokenizer does not preserve one token per structure symbol")
    starts = [
        index
        for index in range(len(prompt_ids) - len(structure_ids) + 1)
        if prompt_ids[index : index + len(structure_ids)] == structure_ids
    ]
    if len(starts) != 1:
        raise RuntimeError(f"RNA-DLM structure token span is ambiguous: matches={starts}")
    start = starts[0]
    return prompt_ids, list(range(start, start + len(structure)))


def _generate_dlm(model, tokenizer, structure: str, seed: int, device: Any) -> list[str]:
    import numpy as np
    import random
    import torch

    prompt_ids, _positions = _prompt_and_positions(tokenizer, structure)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = input_ids.expand(K, -1).contiguous()
    attention_mask = torch.ones_like(input_ids)
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=len(structure) + 1,
            do_sample=True,
            temperature=DLM_TEMPERATURE,
            top_p=1.0,
            use_cache=True,
            prefix_allowed_tokens_fn=_constrained_tokens(tokenizer, structure, len(prompt_ids)),
        )
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        random.setstate(random_state)
        np.random.set_state(numpy_state)
    sequences = [
        "".join(tokenizer.decode(row[len(prompt_ids) :], skip_special_tokens=True).split())
        for row in output
    ]
    for sequence in sequences:
        if len(sequence) != len(structure) or set(sequence) - set("ACGU"):
            raise RuntimeError(f"RNA-DLM returned invalid sequence: {sequence!r}")
    return sequences


def build_samplers(
    *,
    repo_root: Path,
    sft_checkpoint: Path,
    rnaernie: Path,
    rl_model: Path,
    rna_dlm_root: Path,
    goforth_root: Path,
    goforth_cache: Path,
    device: Any,
) -> tuple[dict[str, Callable[[dict[str, Any], int], list[str]]], dict[str, dict[str, Any]], list[Any]]:
    """Load all five residents and return generators, identities and keep-alives."""
    import torch

    _release_paths(repo_root)
    fair = importlib.import_module("evaluate")
    retained: list[Any] = []
    samplers: dict[str, Callable[[dict[str, Any], int], list[str]]] = {}
    identities: dict[str, dict[str, Any]] = {}

    require_file(sft_checkpoint, CHECKPOINT_SHA256["RNA-IFlow"], "RNA-IFlow SFT checkpoint")
    require_file(Path(rnaernie) / "model.safetensors", RNAERNIE_WEIGHT_SHA256, "RNAErnie weights")
    started = time.monotonic()
    payload = torch.load(sft_checkpoint, map_location="cpu", weights_only=False)
    loader = importlib.import_module("run_rl_objective_debug").load_model
    flow = loader(payload, Path(rnaernie), device).eval()
    lookup = fair.TorchDirichletConditionalFlow(device)
    fair.TorchDirichletConditionalFlow = lambda _device: lookup
    retained.extend((flow, lookup))

    def generate_flow(task: dict[str, Any], seed: int, model=flow):
        return fair.sample(
            model,
            task["target_structure"],
            None,
            K,
            FLOW_STEPS,
            fair.global_task_seed(seed, task["_global_index"]),
            "native",
            device,
        )

    samplers["RNA-IFlow"] = generate_flow
    identities["RNA-IFlow"] = {
        "checkpoint_sha256": CHECKPOINT_SHA256["RNA-IFlow"],
        "rnaernie_weight_sha256": RNAERNIE_WEIGHT_SHA256,
        "steps": FLOW_STEPS,
        "setup_seconds": time.monotonic() - started,
        "lookup_setup_excluded": True,
    }

    started = time.monotonic()
    export_manifest = require_file(Path(rl_model) / "export_manifest.json", label="RNA-IFlow-RL export manifest")
    manifest = json.loads(export_manifest.read_text())
    if manifest.get("model_sha256") != PORTABLE_EXPORT_SHA256:
        raise ValueError("RNA-IFlow-RL portable export SHA256 is not the verified U2442 export")
    rl = importlib.import_module("load_export").load_export(Path(rl_model), device).eval()
    endpoint = importlib.import_module("endpoint_policy")
    retained.append(rl)

    def generate_rl(task: dict[str, Any], seed: int, model=rl):
        temperature = torch.full(
            (K,), RL_TEMPERATURES[seed], device=device, dtype=torch.float32
        )
        trajectory = endpoint.rollout_discrete_domino_trajectory(
            model,
            task["target_structure"],
            K,
            RL_STEPS,
            fair.global_task_seed(seed, task["_global_index"]),
            device,
            temperature,
        )
        return trajectory["final_sequences"]

    samplers["RNA-IFlow-RL"] = generate_rl
    identities["RNA-IFlow-RL"] = {
        "checkpoint_sha256": CHECKPOINT_SHA256["RNA-IFlow-RL"],
        "portable_export_sha256": sha256(Path(rl_model) / "model.safetensors"),
        "H": RL_STEPS,
        "temperature_map": RL_TEMPERATURES,
        "setup_seconds": time.monotonic() - started,
    }

    from transformers import AutoModelForCausalLM

    for label, subdir in (
        ("RNA-Design-LM SL", "SL"),
        ("RNA-Design-LM SL+RL", "SL+RL"),
    ):
        started = time.monotonic()
        model_dir = Path(rna_dlm_root) / subdir
        require_file(model_dir / "model.safetensors", CHECKPOINT_SHA256[label], f"{label} weights")
        metadata_hashes, metadata_manifest_sha256 = _json_asset_hashes(model_dir)
        tokenizer = _build_tokenizer(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, dtype=torch.bfloat16
        )
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.to(device).eval()
        retained.append((tokenizer, model))

        def generate_dlm(task: dict[str, Any], seed: int, model=model, tokenizer=tokenizer):
            return _generate_dlm(model, tokenizer, task["target_structure"], seed, device)

        samplers[label] = generate_dlm
        identities[label] = {
            "checkpoint_sha256": CHECKPOINT_SHA256[label],
            "temperature": DLM_TEMPERATURE,
            "seed_mapping": "condition seed without target offset",
            "metadata_json_sha256": metadata_hashes,
            "metadata_manifest_sha256": metadata_manifest_sha256,
            "setup_seconds": time.monotonic() - started,
        }

    goforth_root = Path(goforth_root).resolve()
    goforth_checkpoint = goforth_root / "checkpoints/full_structure_small.redownload.pt"
    require_file(goforth_checkpoint, CHECKPOINT_SHA256["GoForth"], "GoForth checkpoint")
    os.environ["RNA_WORKBENCH_FS_SMALL"] = str(goforth_checkpoint)
    started = time.monotonic()
    server = _import_external_server(goforth_root)
    server_path = goforth_root / "apps/rna_workbench/server.py"
    choices = server.checkpoint_choices()
    if "pretrained_small" not in choices:
        raise FileNotFoundError("GoForth pretrained_small checkpoint is unavailable")
    cache = server.ModelCache(choices, "cuda", 500)
    designer = server.Designer(cache, Path(goforth_cache))
    if cache.device().type != "cuda":
        raise RuntimeError("GoForth must use CUDA")
    retained.extend((cache, designer, server))

    def generate_goforth(task: dict[str, Any], seed: int):
        _dot, side, _exact = server.normalize_structure_condition(task["target_structure"])
        allowed, mask = server.normalize_base_mask("", len(side))
        allowed = server.tighten_pair_compatible_masks(
            allowed, server.partners_from_side(side)
        )
        sequences, _meta = designer.sample_sequences(
            condition_side=side,
            allowed=allowed,
            mask_text=mask,
            count=K,
            temperature=GOFORTH_TEMPERATURE,
            seed=seed,
            batch_size=K,
            checkpoint_key="pretrained_small",
            custom_checkpoint=None,
        )
        return sequences

    samplers["GoForth"] = generate_goforth
    identities["GoForth"] = {
        "checkpoint_sha256": CHECKPOINT_SHA256["GoForth"],
        "temperature": GOFORTH_TEMPERATURE,
        "batch_size": K,
        "setup_seconds": time.monotonic() - started,
        "source_files_sha256": {
            "apps/rna_workbench/server.py": sha256(server_path),
        },
    }
    return samplers, identities, retained


def release_scorer(repo_root: Path):
    _release_paths(repo_root)
    return importlib.import_module("evaluate").evaluate_many
