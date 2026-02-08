from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from jssp_yafs.config import GAConfig
from jssp_yafs.data.models import JSSPInstance
from jssp_yafs.moea.adaptive import UCB1Bandit
from jssp_yafs.moea.indicators import build_ref_point, hypervolume
from jssp_yafs.moea.model import Individual
from jssp_yafs.moea.operators import apply_crossover, apply_mutation
from jssp_yafs.scheduling.heuristics import heuristic_machine_map, priority_sequence
from jssp_yafs.scheduling.representation import Chromosome, random_machine_map, random_sequence
from jssp_yafs.simulation.yafs_simulator import YAFSScheduleSimulator


@dataclass(slots=True)
class NSGA2Result:
    population: list[Individual]
    pareto: list[Individual]
    history: list[dict[str, float]]
    runtime_sec: float
    evaluations: int



def dominates(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(
        x < y for x, y in zip(a, b, strict=True)
    )



def fast_non_dominated_sort(pop: list[Individual]) -> list[list[int]]:
    s: list[list[int]] = [[] for _ in pop]
    n = [0 for _ in pop]
    fronts: list[list[int]] = [[]]

    for p_idx, p in enumerate(pop):
        for q_idx, q in enumerate(pop):
            if p_idx == q_idx:
                continue
            if dominates(p.objectives, q.objectives):
                s[p_idx].append(q_idx)
            elif dominates(q.objectives, p.objectives):
                n[p_idx] += 1
        if n[p_idx] == 0:
            pop[p_idx].rank = 0
            fronts[0].append(p_idx)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[int] = []
        for p_idx in fronts[i]:
            for q_idx in s[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    pop[q_idx].rank = i + 1
                    next_front.append(q_idx)
        i += 1
        fronts.append(next_front)

    if fronts and not fronts[-1]:
        fronts.pop()
    return fronts



def crowding_distance(pop: list[Individual], front: list[int]) -> None:
    if not front:
        return

    for idx in front:
        pop[idx].crowding = 0.0

    n_obj = len(pop[front[0]].objectives)
    for obj in range(n_obj):
        sorted_front = sorted(front, key=lambda idx: pop[idx].objectives[obj])
        pop[sorted_front[0]].crowding = float("inf")
        pop[sorted_front[-1]].crowding = float("inf")

        min_val = pop[sorted_front[0]].objectives[obj]
        max_val = pop[sorted_front[-1]].objectives[obj]
        if max_val == min_val:
            continue

        for i in range(1, len(sorted_front) - 1):
            prev_v = pop[sorted_front[i - 1]].objectives[obj]
            next_v = pop[sorted_front[i + 1]].objectives[obj]
            pop[sorted_front[i]].crowding += (next_v - prev_v) / (max_val - min_val)



def tournament_select(pop: list[Individual], k: int, rng: np.random.Generator) -> Individual:
    cand_idx = rng.choice(np.arange(len(pop)), size=k, replace=False)
    best = pop[int(cand_idx[0])]
    for idx in cand_idx[1:]:
        c = pop[int(idx)]
        if c.rank < best.rank or (c.rank == best.rank and c.crowding > best.crowding):
            best = c
    return best



def _evaluate(
    simulator: YAFSScheduleSimulator,
    instance: JSSPInstance,
    chrom: Chromosome,
) -> Individual:
    sim = simulator.evaluate(instance, chrom.sequence, chrom.machine_map, with_traces=False)
    metrics = {
        "makespan": sim.makespan,
        "energy": sim.energy,
        "reliability": sim.reliability,
        "one_minus_reliability": 1.0 - sim.reliability,
        "log_reliability": sim.log_reliability,
    }
    return Individual(chromosome=chrom, objectives=sim.objective_vector, metrics=metrics)



def _scalar_score(ind: Individual) -> float:
    # Mildly reliability-aware scalar score for local search decisions.
    return ind.objectives[0] + 0.1 * ind.objectives[1] + 100.0 * ind.objectives[2]



def _local_search(
    base: Individual,
    simulator: YAFSScheduleSimulator,
    instance: JSSPInstance,
    compute_nodes: list[int],
    rng: np.random.Generator,
    tries: int = 6,
) -> tuple[Individual, int]:
    best = base
    evals = 0

    for _ in range(tries):
        seq = best.chromosome.sequence.copy()
        mmap = best.chromosome.machine_map.copy()

        if rng.random() < 0.7:
            i, j = sorted(rng.choice(np.arange(len(seq)), size=2, replace=False).tolist())
            seq[i], seq[j] = seq[j], seq[i]
        else:
            idx = int(rng.integers(0, len(mmap)))
            mmap[idx] = int(rng.choice(compute_nodes))

        cand = _evaluate(simulator, instance, Chromosome(seq, mmap))
        evals += 1

        if dominates(cand.objectives, best.objectives) or _scalar_score(cand) < _scalar_score(best):
            best = cand

    return best, evals



def _init_population(
    instance: JSSPInstance,
    simulator: YAFSScheduleSimulator,
    cfg: GAConfig,
    compute_nodes: list[int],
    node_mips: dict[int, float],
    rng: np.random.Generator,
    smart_init: bool,
) -> tuple[list[Individual], int]:
    population: list[Individual] = []
    evaluations = 0

    if smart_init:
        rules = ["spt", "mwr", "lpt"]
        for rule in rules:
            seq = priority_sequence(instance, rule, rng)
            mmap = heuristic_machine_map(instance, compute_nodes, node_mips)
            population.append(_evaluate(simulator, instance, Chromosome(seq, mmap)))
            evaluations += 1

    while len(population) < cfg.population_size:
        seq = random_sequence(instance, rng)
        mmap = random_machine_map(instance, compute_nodes, rng)
        population.append(_evaluate(simulator, instance, Chromosome(seq, mmap)))
        evaluations += 1

    return population[: cfg.population_size], evaluations



def run_nsga2(
    instance: JSSPInstance,
    simulator: YAFSScheduleSimulator,
    cfg: GAConfig,
    rng: np.random.Generator,
    compute_nodes: list[int],
    node_mips: dict[int, float],
    use_smart_init: bool,
    use_adaptive_ops: bool,
    use_local_search: bool,
    show_progress: bool,
) -> NSGA2Result:
    start = time.perf_counter()

    pop, evals = _init_population(
        instance=instance,
        simulator=simulator,
        cfg=cfg,
        compute_nodes=compute_nodes,
        node_mips=node_mips,
        rng=rng,
        smart_init=use_smart_init,
    )

    c_bandit = UCB1Bandit(cfg.crossover_ops, c=1.1)
    m_bandit = UCB1Bandit(cfg.mutation_ops, c=1.1)

    history: list[dict[str, float]] = []

    iterator = range(cfg.generations)
    if show_progress:
        iterator = tqdm(iterator, desc=f"NSGA-II {instance.name}", leave=False)

    for gen in iterator:
        fronts = fast_non_dominated_sort(pop)
        for front in fronts:
            crowding_distance(pop, front)

        offspring: list[Individual] = []

        while len(offspring) < cfg.population_size:
            p1 = tournament_select(pop, cfg.tournament_k, rng)
            p2 = tournament_select(pop, cfg.tournament_k, rng)

            c_op = c_bandit.select() if use_adaptive_ops else str(rng.choice(cfg.crossover_ops))
            m_op = m_bandit.select() if use_adaptive_ops else str(rng.choice(cfg.mutation_ops))

            if rng.random() < cfg.crossover_prob:
                c_seq, c_map = apply_crossover(
                    c_op,
                    p1.chromosome.sequence,
                    p2.chromosome.sequence,
                    p1.chromosome.machine_map,
                    p2.chromosome.machine_map,
                    instance,
                    compute_nodes,
                    rng,
                )
            else:
                c_seq = p1.chromosome.sequence.copy()
                c_map = p1.chromosome.machine_map.copy()

            if rng.random() < cfg.mutation_prob:
                c_seq, c_map = apply_mutation(
                    m_op,
                    c_seq,
                    c_map,
                    instance,
                    compute_nodes,
                    rng,
                )

            child = _evaluate(simulator, instance, Chromosome(c_seq, c_map))
            evals += 1

            if use_local_search and rng.random() < cfg.local_search_prob:
                child, extra_evals = _local_search(
                    child,
                    simulator,
                    instance,
                    compute_nodes,
                    rng,
                )
                evals += extra_evals

            if use_adaptive_ops:
                reward = 0.0
                if dominates(child.objectives, p1.objectives) or dominates(child.objectives, p2.objectives):
                    reward = 1.0
                elif dominates(p1.objectives, child.objectives) and dominates(p2.objectives, child.objectives):
                    reward = 0.0
                else:
                    reward = 0.35
                c_bandit.update(c_op, reward)
                m_bandit.update(m_op, reward)

            offspring.append(child)

        union = pop + offspring
        fronts = fast_non_dominated_sort(union)

        next_pop: list[Individual] = []
        for front in fronts:
            crowding_distance(union, front)
            front_inds = [union[idx] for idx in front]
            if len(next_pop) + len(front_inds) <= cfg.population_size:
                next_pop.extend(front_inds)
            else:
                front_inds.sort(key=lambda ind: ind.crowding, reverse=True)
                remain = cfg.population_size - len(next_pop)
                next_pop.extend(front_inds[:remain])
                break
        pop = next_pop

        objectives = np.array([ind.objectives for ind in pop], dtype=np.float64)
        ref = build_ref_point(objectives)
        hv = hypervolume(objectives, ref)
        best_mk = float(objectives[:, 0].min())
        best_en = float(objectives[:, 1].min())
        best_rl = float((1.0 - objectives[:, 2]).max())
        history.append(
            {
                "generation": gen,
                "hypervolume": hv,
                "best_makespan": best_mk,
                "best_energy": best_en,
                "best_reliability": best_rl,
            }
        )

    fronts = fast_non_dominated_sort(pop)
    pareto = [pop[idx] for idx in fronts[0]]

    return NSGA2Result(
        population=pop,
        pareto=pareto,
        history=history,
        runtime_sec=float(time.perf_counter() - start),
        evaluations=evals,
    )
