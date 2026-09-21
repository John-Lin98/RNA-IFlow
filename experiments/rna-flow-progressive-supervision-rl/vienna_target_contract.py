"""Target-only checks for structures represented by ViennaRNA's DP state space."""

from __future__ import annotations

import hashlib
import json

import RNA


TARGET_LENGTH_BINS = ("000-064", "065-128", "129-256", "257-510")
TARGET_REPRESENTABILITY_POLICY = {
    "schema_version": 1,
    "alphabet": "dot-bracket-no-pseudoknots",
    "maximum_target_length": 510,
    "length_bins": list(TARGET_LENGTH_BINS),
    "minimum_hairpin_unpaired": int(RNA.TURN),
    "maximum_interior_loop_unpaired": int(RNA.MAXLOOP),
}


def target_representability(structure: str) -> dict:
    reasons: list[str] = []
    if not isinstance(structure, str) or not structure or set(structure) - set("()."):
        return {"representable": False, "reasons": ["invalid_dot_bracket_alphabet"]}
    if len(structure) > TARGET_REPRESENTABILITY_POLICY["maximum_target_length"]:
        return {"representable": False, "reasons": ["target_length_above_contract"]}

    stack: list[dict] = []
    pairs: list[dict] = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            node = {"left": index, "right": None, "children": []}
            if stack:
                stack[-1]["children"].append(node)
            stack.append(node)
            pairs.append(node)
        elif symbol == ")":
            if not stack:
                return {"representable": False, "reasons": ["unbalanced_dot_bracket"]}
            stack.pop()["right"] = index
    if stack:
        return {"representable": False, "reasons": ["unbalanced_dot_bracket"]}

    maximum_interior_loop = 0
    minimum_hairpin = None
    for pair in pairs:
        left = int(pair["left"])
        right = int(pair["right"])
        children = pair["children"]
        if not children:
            hairpin = right - left - 1
            minimum_hairpin = hairpin if minimum_hairpin is None else min(
                minimum_hairpin, hairpin
            )
            if hairpin < RNA.TURN:
                reasons.append("hairpin_below_turn")
        elif len(children) == 1:
            child = children[0]
            interior = (
                int(child["left"]) - left - 1
                + right - int(child["right"]) - 1
            )
            maximum_interior_loop = max(maximum_interior_loop, interior)
            if interior > RNA.MAXLOOP:
                reasons.append("interior_loop_above_maxloop")

    return {
        "representable": not reasons,
        "reasons": sorted(set(reasons)),
        "pairs": len(pairs),
        "minimum_hairpin_unpaired": minimum_hairpin,
        "maximum_interior_loop_unpaired": maximum_interior_loop,
    }


def target_contract_digest(tasks: list[dict]) -> str:
    records = [
        {
            "id": str(task["id"]),
            "target_structure_sha256": hashlib.sha256(
                task["target_structure"].encode()
            ).hexdigest(),
            "representability": target_representability(task["target_structure"]),
        }
        for task in tasks
    ]
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def target_length_bin(length: int) -> str:
    if length <= 64:
        return "000-064"
    if length <= 128:
        return "065-128"
    if length <= 256:
        return "129-256"
    return "257-510"


def validate_target_contract(tasks: list[dict], evidence: dict) -> None:
    invalid = [
        str(task.get("id"))
        for task in tasks
        if (
            target_representability(task.get("target_structure"))["representable"] is False
            or task.get("length") != len(task.get("target_structure", ""))
            or task.get("length_bin")
            != target_length_bin(len(task.get("target_structure", "")))
        )
    ]
    if (
        evidence.get("policy") != TARGET_REPRESENTABILITY_POLICY
        or evidence.get("selected_rows") != len(tasks)
        or evidence.get("selected_bad_count") != 0
        or evidence.get("selected_contract_sha256") != target_contract_digest(tasks)
        or invalid
    ):
        raise RuntimeError("TRAIN target representability evidence fails the ViennaRNA contract")
