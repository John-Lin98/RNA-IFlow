"""Memory-mapped Arrow dataset for progressive RNA Flow stages."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from torch.utils.data import Dataset


class ArrowFlowDataset(Dataset):
    def __init__(self, path: Path, limit: int) -> None:
        if limit <= 0:
            raise ValueError("Arrow stage limit must be positive")
        self.path = path
        self.limit = limit
        self._source = pa.memory_map(str(path), "r")
        self._reader = ipc.open_file(self._source)
        self._table = self._reader.read_all()
        required = {"target_structure", "sequence"}
        if not required.issubset(self._table.column_names):
            raise RuntimeError(f"Arrow dataset lacks columns: {sorted(required - set(self._table.column_names))}")
        if limit > self._table.num_rows:
            raise ValueError(f"Arrow stage limit {limit} exceeds {self._table.num_rows} rows")

    def __len__(self) -> int:
        return self.limit

    @staticmethod
    def _rows_from_table(table: pa.Table) -> list[dict]:
        structures = table["target_structure"].to_pylist()
        sequences = table["sequence"].to_pylist()
        return [
            {"target_structure": structure, "sequence": sequence}
            for structure, sequence in zip(structures, sequences)
        ]

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.limit:
            raise IndexError(index)
        return self._rows_from_table(self._table.slice(index, 1))[0]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        if any(index < 0 or index >= self.limit for index in indices):
            raise IndexError("Arrow batch index is outside the frozen stage prefix")
        # Taking from the full 10M-row ChunkedArray asks Arrow to concatenate
        # more than 2 GiB of 32-bit string offsets. Scalar chunk lookup avoids
        # that overflow while still touching only the requested random batch.
        structures = self._table["target_structure"]
        sequences = self._table["sequence"]
        return [
            {
                "target_structure": structures[index].as_py(),
                "sequence": sequences[index].as_py(),
            }
            for index in indices
        ]
