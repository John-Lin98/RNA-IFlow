"""Dot-bracket parsing and deterministic terminal pair constraints."""

from __future__ import annotations

import torch


NUCLEOTIDES = "ACGU"
ALLOWED_PAIRS = ("AU", "UA", "GC", "CG", "GU", "UG")


def target_pairs(structure: str) -> list[tuple[int, int]]:
    stack = []
    pairs = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced structure")
            pairs.append((stack.pop(), index))
        elif symbol != ".":
            raise ValueError("invalid structure symbol")
    if stack:
        raise ValueError("unbalanced structure")
    return pairs


def constrained_decode(probabilities: torch.Tensor, structure: str) -> str:
    if probabilities.shape != (len(structure), 4):
        raise ValueError("probability/structure length mismatch")
    indices = probabilities.argmax(dim=-1).tolist()
    for left, right in target_pairs(structure):
        best_pair = max(
            ALLOWED_PAIRS,
            key=lambda pair: float(probabilities[left, NUCLEOTIDES.index(pair[0])]
                                   * probabilities[right, NUCLEOTIDES.index(pair[1])]),
        )
        indices[left] = NUCLEOTIDES.index(best_pair[0])
        indices[right] = NUCLEOTIDES.index(best_pair[1])
    return "".join(NUCLEOTIDES[index] for index in indices)


def valid_pair_fraction(sequence: str, structure: str) -> float:
    pairs = target_pairs(structure)
    if not pairs:
        return 1.0
    return sum(sequence[i] + sequence[j] in ALLOWED_PAIRS for i, j in pairs) / len(pairs)
