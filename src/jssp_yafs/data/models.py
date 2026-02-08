from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class JSSPInstance:
    name: str
    n_jobs: int
    n_machines: int
    machine_matrix: np.ndarray
    processing_matrix: np.ndarray

    @property
    def n_operations(self) -> int:
        return self.n_jobs * self.n_machines

    @property
    def total_work(self) -> float:
        return float(self.processing_matrix.sum())
