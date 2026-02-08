from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jssp_yafs.data.models import JSSPInstance


@dataclass(slots=True)
class Chromosome:
    sequence: np.ndarray
    machine_map: np.ndarray

    def copy(self) -> "Chromosome":
        return Chromosome(self.sequence.copy(), self.machine_map.copy())



def random_sequence(instance: JSSPInstance, rng: np.random.Generator) -> np.ndarray:
    seq = np.repeat(np.arange(instance.n_jobs, dtype=np.int64), instance.n_machines)
    rng.shuffle(seq)
    return seq



def random_machine_map(
    instance: JSSPInstance, compute_nodes: list[int], rng: np.random.Generator
) -> np.ndarray:
    return rng.choice(np.array(compute_nodes, dtype=np.int64), size=instance.n_machines, replace=True)



def repair_sequence(
    sequence: np.ndarray,
    n_jobs: int,
    n_machines: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seq = sequence.astype(np.int64, copy=True)
    expected = n_machines

    valid_mask = (seq >= 0) & (seq < n_jobs)
    seq[~valid_mask] = -1

    counts = np.zeros(n_jobs, dtype=np.int64)
    for g in seq:
        if g >= 0:
            counts[g] += 1

    missing: list[int] = []
    for j in range(n_jobs):
        deficit = expected - counts[j]
        if deficit > 0:
            missing.extend([j] * int(deficit))
    rng.shuffle(missing)

    missing_idx = 0
    for idx, g in enumerate(seq):
        replace = False
        if g < 0:
            replace = True
        elif counts[g] > expected:
            counts[g] -= 1
            replace = True

        if replace:
            if missing_idx >= len(missing):
                # Safety fallback: choose a random job.
                seq[idx] = int(rng.integers(0, n_jobs))
            else:
                seq[idx] = missing[missing_idx]
                counts[seq[idx]] += 1
                missing_idx += 1

    return seq



def repair_machine_map(
    machine_map: np.ndarray,
    compute_nodes: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    fixed = machine_map.astype(np.int64, copy=True)
    allowed = set(compute_nodes)
    for i in range(len(fixed)):
        if int(fixed[i]) not in allowed:
            fixed[i] = int(rng.choice(compute_nodes))
    return fixed
