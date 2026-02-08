from __future__ import annotations

import numpy as np

from jssp_yafs.data.models import JSSPInstance
from jssp_yafs.scheduling.representation import repair_machine_map, repair_sequence


def crossover_uniform(
    a_seq: np.ndarray,
    b_seq: np.ndarray,
    a_map: np.ndarray,
    b_map: np.ndarray,
    instance: JSSPInstance,
    compute_nodes: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mask_seq = rng.random(len(a_seq)) < 0.5
    child_seq = np.where(mask_seq, a_seq, b_seq)

    mask_map = rng.random(len(a_map)) < 0.5
    child_map = np.where(mask_map, a_map, b_map)

    child_seq = repair_sequence(child_seq, instance.n_jobs, instance.n_machines, rng)
    child_map = repair_machine_map(child_map, compute_nodes, rng)
    return child_seq, child_map



def crossover_two_point(
    a_seq: np.ndarray,
    b_seq: np.ndarray,
    a_map: np.ndarray,
    b_map: np.ndarray,
    instance: JSSPInstance,
    compute_nodes: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(a_seq)
    i, j = sorted(rng.choice(np.arange(n), size=2, replace=False).tolist())
    child_seq = a_seq.copy()
    child_seq[i:j] = b_seq[i:j]

    m = len(a_map)
    p, q = sorted(rng.choice(np.arange(m), size=2, replace=False).tolist())
    child_map = a_map.copy()
    child_map[p:q] = b_map[p:q]

    child_seq = repair_sequence(child_seq, instance.n_jobs, instance.n_machines, rng)
    child_map = repair_machine_map(child_map, compute_nodes, rng)
    return child_seq, child_map



def crossover_job_preserving(
    a_seq: np.ndarray,
    b_seq: np.ndarray,
    a_map: np.ndarray,
    b_map: np.ndarray,
    instance: JSSPInstance,
    compute_nodes: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected_jobs = set(
        rng.choice(np.arange(instance.n_jobs), size=max(1, instance.n_jobs // 2), replace=False).tolist()
    )
    child_seq = np.full_like(a_seq, fill_value=-1)
    keep_mask = np.array([gene in selected_jobs for gene in a_seq], dtype=bool)
    child_seq[keep_mask] = a_seq[keep_mask]

    fill_vals = [gene for gene in b_seq.tolist() if gene not in selected_jobs]
    fill_idx = np.where(~keep_mask)[0]
    child_seq[fill_idx] = np.array(fill_vals[: len(fill_idx)], dtype=np.int64)

    child_map = a_map.copy()
    alt_mask = rng.random(len(a_map)) < 0.3
    child_map[alt_mask] = b_map[alt_mask]

    child_seq = repair_sequence(child_seq, instance.n_jobs, instance.n_machines, rng)
    child_map = repair_machine_map(child_map, compute_nodes, rng)
    return child_seq, child_map



def mutate_swap(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seq = sequence.copy()
    i, j = rng.choice(np.arange(len(seq)), size=2, replace=False)
    seq[i], seq[j] = seq[j], seq[i]
    return seq



def mutate_insert(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seq = sequence.copy()
    i, j = rng.choice(np.arange(len(seq)), size=2, replace=False)
    if i < j:
        seq = np.concatenate([seq[:i], seq[i + 1 : j + 1], seq[i : i + 1], seq[j + 1 :]])
    else:
        seq = np.concatenate([seq[:j], seq[i : i + 1], seq[j:i], seq[i + 1 :]])
    return seq



def mutate_scramble(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seq = sequence.copy()
    i, j = sorted(rng.choice(np.arange(len(seq)), size=2, replace=False).tolist())
    part = seq[i:j].copy()
    rng.shuffle(part)
    seq[i:j] = part
    return seq



def mutate_reassign_map(
    machine_map: np.ndarray,
    compute_nodes: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    out = machine_map.copy()
    idx = int(rng.integers(0, len(out)))
    out[idx] = int(rng.choice(compute_nodes))
    return out



def apply_crossover(
    op_name: str,
    a_seq: np.ndarray,
    b_seq: np.ndarray,
    a_map: np.ndarray,
    b_map: np.ndarray,
    instance: JSSPInstance,
    compute_nodes: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if op_name == "two_point":
        return crossover_two_point(a_seq, b_seq, a_map, b_map, instance, compute_nodes, rng)
    if op_name == "job_preserving":
        return crossover_job_preserving(a_seq, b_seq, a_map, b_map, instance, compute_nodes, rng)
    return crossover_uniform(a_seq, b_seq, a_map, b_map, instance, compute_nodes, rng)



def apply_mutation(
    op_name: str,
    sequence: np.ndarray,
    machine_map: np.ndarray,
    instance: JSSPInstance,
    compute_nodes: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    seq = sequence.copy()
    mmap = machine_map.copy()

    if op_name == "swap":
        seq = mutate_swap(seq, rng)
    elif op_name == "insert":
        seq = mutate_insert(seq, rng)
    elif op_name == "scramble":
        seq = mutate_scramble(seq, rng)
    else:
        mmap = mutate_reassign_map(mmap, compute_nodes, rng)

    seq = repair_sequence(seq, instance.n_jobs, instance.n_machines, rng)
    mmap = repair_machine_map(mmap, compute_nodes, rng)
    return seq, mmap
