from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jssp_yafs.data.models import JSSPInstance


@dataclass(slots=True)
class DecodedOperation:
    step_index: int
    job: int
    operation: int
    machine: int
    processing_time: float



def decode_operation_order(sequence: np.ndarray, instance: JSSPInstance) -> list[DecodedOperation]:
    op_cursor = np.zeros(instance.n_jobs, dtype=np.int64)
    decoded: list[DecodedOperation] = []

    for idx, gene in enumerate(sequence.tolist()):
        j = int(gene)
        if j < 0 or j >= instance.n_jobs:
            continue
        op = int(op_cursor[j])
        if op >= instance.n_machines:
            continue

        m = int(instance.machine_matrix[j, op])
        p = float(instance.processing_matrix[j, op])
        decoded.append(
            DecodedOperation(step_index=idx, job=j, operation=op, machine=m, processing_time=p)
        )
        op_cursor[j] += 1

    if not np.all(op_cursor == instance.n_machines):
        raise ValueError("Decoded sequence is infeasible: not all job operations were scheduled")

    return decoded
