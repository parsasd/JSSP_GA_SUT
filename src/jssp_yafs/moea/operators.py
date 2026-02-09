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



def crossover_jox(
    a_seq: np.ndarray,
    b_seq: np.ndarray,
    a_map: np.ndarray,
    b_map: np.ndarray,
    instance: JSSPInstance,
    compute_nodes: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Job Order Crossover (JOX) for operation-based JSSP encoding.

    Selects a random subset of jobs.  Operations belonging to the selected
    jobs keep their positions from parent A.  The remaining positions are
    filled, in order, by scanning parent B and taking only operations of
    the non-selected jobs.  This preserves the relative ordering inherited
    from each parent and guarantees each job appears exactly n_machines
    times, so no repair is needed for the sequence.
    """
    n_jobs = instance.n_jobs
    n_select = max(1, n_jobs // 2)
    selected_jobs = set(
        rng.choice(np.arange(n_jobs), size=n_select, replace=False).tolist()
    )

    child_seq = np.full_like(a_seq, fill_value=-1)

    # Keep selected-job genes at their positions from parent A.
    keep_mask = np.array([int(g) in selected_jobs for g in a_seq], dtype=bool)
    child_seq[keep_mask] = a_seq[keep_mask]

    # Fill remaining positions from parent B, preserving B's order for
    # the non-selected jobs.
    fill_vals = [g for g in b_seq.tolist() if int(g) not in selected_jobs]
    fill_idx = np.where(~keep_mask)[0]
    child_seq[fill_idx] = np.array(fill_vals, dtype=np.int64)

    # Machine map: uniform crossover.
    mask_map = rng.random(len(a_map)) < 0.5
    child_map = np.where(mask_map, a_map, b_map)
    child_map = repair_machine_map(child_map, compute_nodes, rng)

    return child_seq, child_map



def crossover_ppx(
    a_seq: np.ndarray,
    b_seq: np.ndarray,
    a_map: np.ndarray,
    b_map: np.ndarray,
    instance: JSSPInstance,
    compute_nodes: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Precedence Preserving Crossover (PPX) for operation-based JSSP encoding.

    For each position in the child, a random binary mask decides whether
    the next gene is taken from parent A or parent B.  Genes are consumed
    from each parent in order (via a cursor), and duplicate occurrences
    are skipped.  This guarantees each job appears exactly n_machines
    times and preserves operation precedence from both parents without
    repair.
    """
    n = len(a_seq)
    child_seq = np.empty(n, dtype=np.int64)

    # Per-job counters: how many times each job has been placed so far.
    n_jobs = instance.n_jobs
    n_machines = instance.n_machines
    placed = np.zeros(n_jobs, dtype=np.int64)

    # Cursors into each parent.
    ptr_a = 0
    ptr_b = 0

    mask = rng.random(n) < 0.5  # True -> take from A, False -> take from B

    for i in range(n):
        if mask[i]:
            # Take next valid gene from parent A.
            while ptr_a < n:
                g = int(a_seq[ptr_a])
                ptr_a += 1
                if placed[g] < n_machines:
                    child_seq[i] = g
                    placed[g] += 1
                    break
        else:
            # Take next valid gene from parent B.
            while ptr_b < n:
                g = int(b_seq[ptr_b])
                ptr_b += 1
                if placed[g] < n_machines:
                    child_seq[i] = g
                    placed[g] += 1
                    break

    # Machine map: uniform crossover.
    mask_map = rng.random(len(a_map)) < 0.5
    child_map = np.where(mask_map, a_map, b_map)
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
    if op_name == "jox":
        return crossover_jox(a_seq, b_seq, a_map, b_map, instance, compute_nodes, rng)
    if op_name == "ppx":
        return crossover_ppx(a_seq, b_seq, a_map, b_map, instance, compute_nodes, rng)
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
        mmap = repair_machine_map(mmap, compute_nodes, rng)

    # swap, insert, scramble preserve job counts by construction;
    # only reassign_map needs machine_map repair (applied above).
    return seq, mmap
