from __future__ import annotations

import numpy as np

from jssp_yafs.data.models import JSSPInstance


def priority_sequence(instance: JSSPInstance, rule: str, rng: np.random.Generator) -> np.ndarray:
    op_cursor = np.zeros(instance.n_jobs, dtype=np.int64)
    sequence: list[int] = []

    while len(sequence) < instance.n_operations:
        candidates = np.where(op_cursor < instance.n_machines)[0]
        if len(candidates) == 0:
            break

        if rule == "spt":
            scores = [instance.processing_matrix[j, op_cursor[j]] for j in candidates]
            best = int(candidates[int(np.argmin(scores))])
        elif rule == "lpt":
            scores = [instance.processing_matrix[j, op_cursor[j]] for j in candidates]
            best = int(candidates[int(np.argmax(scores))])
        elif rule == "mwr":
            scores = [instance.processing_matrix[j, op_cursor[j] :].sum() for j in candidates]
            best = int(candidates[int(np.argmax(scores))])
        else:
            best = int(rng.choice(candidates))

        sequence.append(best)
        op_cursor[best] += 1

    return np.array(sequence, dtype=np.int64)



def heuristic_machine_map(
    instance: JSSPInstance,
    compute_nodes: list[int],
    node_mips: dict[int, float],
) -> np.ndarray:
    workload = np.zeros(instance.n_machines, dtype=np.float64)
    for j in range(instance.n_jobs):
        for op in range(instance.n_machines):
            m = int(instance.machine_matrix[j, op])
            workload[m] += instance.processing_matrix[j, op]

    machine_order = np.argsort(-workload)
    nodes_by_speed = sorted(compute_nodes, key=lambda n: node_mips[n], reverse=True)

    assignment = np.zeros(instance.n_machines, dtype=np.int64)
    for i, m in enumerate(machine_order):
        assignment[m] = nodes_by_speed[i % len(nodes_by_speed)]
    return assignment
