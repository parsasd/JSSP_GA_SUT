from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from jssp_yafs.config import ExperimentConfig, GAConfig
from jssp_yafs.data.benchmarks import BEST_KNOWN_MAKESPAN
from jssp_yafs.data.loader import load_instances_from_folder
from jssp_yafs.data.models import JSSPInstance
from jssp_yafs.experiments.aggregation import aggregate_metrics
from jssp_yafs.experiments.statistics import run_statistics
from jssp_yafs.moea.indicators import build_ref_point, hypervolume, igd, non_dominated_front
from jssp_yafs.moea.nsga2 import NSGA2Result, run_nsga2
from jssp_yafs.scheduling.heuristics import heuristic_machine_map, priority_sequence
from jssp_yafs.simulation.edge_topology import EdgeTopology, build_edge_fog_cloud_topology
from jssp_yafs.simulation.yafs_simulator import SimulationResult, YAFSScheduleSimulator

logger = logging.getLogger(__name__)

# ── Worker-process globals (set once by _init_worker, reused across tasks) ──
_worker_instances: dict[str, JSSPInstance] = {}
_worker_topology: EdgeTopology | None = None
_worker_simulator: YAFSScheduleSimulator | None = None
_worker_cfg: ExperimentConfig | None = None
_worker_compute_nodes: list[int] = []
_worker_node_mips: dict[int, float] = {}


def _init_worker(
    cfg: ExperimentConfig,
    split_name: str,
    instances_names: list[str],
) -> None:
    """Called once per worker process by ProcessPoolExecutor.

    Loads instances from disk, builds the topology, and creates a simulator
    whose _path_cache will persist across all tasks assigned to this worker.
    """
    global _worker_instances, _worker_topology, _worker_simulator
    global _worker_cfg, _worker_compute_nodes, _worker_node_mips

    _worker_cfg = cfg
    _worker_instances = load_instances_from_folder(
        Path("data/processed") / split_name, instances_names
    )
    _worker_topology = build_edge_fog_cloud_topology(cfg.topology)
    _worker_compute_nodes = _worker_topology.compute_nodes
    _worker_node_mips = {
        n: float(_worker_topology.topology.G.nodes[n]["IPT"])
        for n in _worker_compute_nodes
    }
    _worker_simulator = YAFSScheduleSimulator(
        _worker_topology, cfg.evaluation
    )



@dataclass(slots=True)
class RunnerOutput:
    run_root: Path
    per_run_csv: Path
    pareto_csv: Path
    convergence_csv: Path
    schedule_csv: Path
    indicator_reference_csv: Path
    aggregate_csv: Path
    stats_csv: Path
    budget_json: Path



def _best_compromise(objs: np.ndarray) -> int:
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    denom = np.where(maxs - mins > 0, maxs - mins, 1.0)
    norm = (objs - mins) / denom
    w = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    score = norm @ w
    return int(np.argmin(score))



def _single_heuristic(
    rule: str,
    instance,
    simulator: YAFSScheduleSimulator,
    compute_nodes: list[int],
    node_mips: dict[int, float],
    rng: np.random.Generator,
) -> tuple[list[SimulationResult], list[dict[str, float]], int, float, list[tuple[np.ndarray, np.ndarray]]]:
    seq = priority_sequence(instance, rule=rule, rng=rng)
    mmap = heuristic_machine_map(instance, compute_nodes, node_mips)
    sim = simulator.evaluate(instance, seq, mmap)
    history = [
        {
            "generation": 0,
            "hypervolume": 0.0,
            "best_makespan": sim.makespan,
            "best_energy": sim.energy,
            "best_reliability": sim.reliability,
        }
    ]
    return [sim], history, 1, 0.0, [(seq.copy(), mmap.copy())]



def _ga_to_sims(
    result: NSGA2Result,
    instance,
    simulator: YAFSScheduleSimulator,
) -> tuple[list[SimulationResult], list[dict[str, float]], int, float, list[tuple[np.ndarray, np.ndarray]]]:
    sims = [
        simulator.evaluate(
            instance,
            ind.chromosome.sequence,
            ind.chromosome.machine_map,
            with_traces=False,
        )
        for ind in result.pareto
    ]
    decisions = [
        (ind.chromosome.sequence.copy(), ind.chromosome.machine_map.copy())
        for ind in result.pareto
    ]
    return sims, result.history, result.evaluations, result.runtime_sec, decisions



def _run_algorithm(
    algorithm: str,
    instance,
    simulator: YAFSScheduleSimulator,
    ga_plain: GAConfig,
    ga_enhanced: GAConfig,
    compute_nodes: list[int],
    node_mips: dict[int, float],
    rng: np.random.Generator,
    show_progress: bool,
) -> tuple[list[SimulationResult], list[dict[str, float]], int, float, list[tuple[np.ndarray, np.ndarray]]]:
    if algorithm == "heuristic_spt":
        return _single_heuristic("spt", instance, simulator, compute_nodes, node_mips, rng)
    if algorithm == "heuristic_mwr":
        return _single_heuristic("mwr", instance, simulator, compute_nodes, node_mips, rng)

    if algorithm == "plain_ga":
        nsga = run_nsga2(
            instance=instance,
            simulator=simulator,
            cfg=ga_plain,
            rng=rng,
            compute_nodes=compute_nodes,
            node_mips=node_mips,
            use_smart_init=False,
            use_adaptive_ops=False,
            use_local_search=False,
            show_progress=show_progress,
        )
        return _ga_to_sims(nsga, instance, simulator)

    if algorithm == "enhanced_ga":
        nsga = run_nsga2(
            instance=instance,
            simulator=simulator,
            cfg=ga_enhanced,
            rng=rng,
            compute_nodes=compute_nodes,
            node_mips=node_mips,
            use_smart_init=True,
            use_adaptive_ops=True,
            use_local_search=True,
            show_progress=show_progress,
        )
        return _ga_to_sims(nsga, instance, simulator)

    # Ablations for enhanced GA components.
    feature_flags = {
        "ablation_no_aos": dict(smart=True, aos=False, local=True),
        "ablation_no_local_search": dict(smart=True, aos=True, local=False),
        "ablation_no_smart_init": dict(smart=False, aos=True, local=True),
    }
    if algorithm in feature_flags:
        f = feature_flags[algorithm]
        nsga = run_nsga2(
            instance=instance,
            simulator=simulator,
            cfg=ga_enhanced,
            rng=rng,
            compute_nodes=compute_nodes,
            node_mips=node_mips,
            use_smart_init=f["smart"],
            use_adaptive_ops=f["aos"],
            use_local_search=f["local"],
            show_progress=show_progress,
        )
        return _ga_to_sims(nsga, instance, simulator)

    raise ValueError(f"Unknown algorithm: {algorithm}")



def _run_one(task: tuple[int, str, str]) -> dict:
    """Worker for one (seed, instance, algorithm) run.

    Relies on module-level globals set once by _init_worker:
      _worker_instances, _worker_topology, _worker_simulator,
      _worker_cfg, _worker_compute_nodes, _worker_node_mips
    """
    seed, instance_name, algorithm = task

    cfg = _worker_cfg
    assert cfg is not None
    simulator = _worker_simulator
    assert simulator is not None
    instance = _worker_instances[instance_name]

    # Clear eval cache from the previous task to bound memory, but keep
    # _path_cache (topology-dependent shortest paths, only ~144 entries).
    simulator._eval_cache.clear()

    rng = np.random.default_rng(seed)
    sims, history, evals, runtime_sec, decisions = _run_algorithm(
        algorithm=algorithm,
        instance=instance,
        simulator=simulator,
        ga_plain=cfg.ga_plain,
        ga_enhanced=cfg.ga_enhanced,
        compute_nodes=_worker_compute_nodes,
        node_mips=_worker_node_mips,
        rng=rng,
        show_progress=False,
    )

    objs = np.array([s.objective_vector for s in sims], dtype=np.float64)
    best_idx = _best_compromise(objs)
    best = sims[best_idx]
    best_traces = best.traces
    if not best_traces:
        best_seq, best_map = decisions[best_idx]
        best_full = simulator.evaluate(instance, best_seq, best_map, with_traces=True)
        best_traces = best_full.traces

    bks = BEST_KNOWN_MAKESPAN.get(instance_name)

    per_run = {
        "instance": instance_name,
        "seed": seed,
        "algorithm": algorithm,
        "hypervolume": np.nan,
        "igd": np.nan,
        "best_makespan": best.makespan,
        "best_energy": best.energy,
        "best_reliability": best.reliability,
        "bks_makespan": bks if bks is not None else np.nan,
        "runtime_sec": runtime_sec,
        "evaluations": evals,
    }

    budget = {
        "instance": instance_name,
        "seed": seed,
        "algorithm": algorithm,
        "population_size": (
            cfg.ga_plain.population_size if algorithm == "plain_ga" else cfg.ga_enhanced.population_size
        ),
        "generations": (
            cfg.ga_plain.generations if algorithm == "plain_ga" else cfg.ga_enhanced.generations
        ),
        "evaluations": evals,
        "runtime_sec": runtime_sec,
    }

    pareto_list = [
        {
            "instance": instance_name,
            "seed": seed,
            "algorithm": algorithm,
            "point_id": i,
            "makespan": s.makespan,
            "energy": s.energy,
            "reliability": s.reliability,
            "one_minus_reliability": 1.0 - s.reliability,
        }
        for i, s in enumerate(sims)
    ]

    conv_list = [
        {"instance": instance_name, "seed": seed, "algorithm": algorithm, **row}
        for row in history
    ]

    schedule_list = [
        {
            "instance": instance_name,
            "seed": seed,
            "algorithm": algorithm,
            "job": tr.job,
            "operation": tr.operation,
            "machine": tr.machine,
            "node": tr.node,
            "predecessor_node": tr.predecessor_node,
            "start": tr.start,
            "end": tr.end,
            "processing_time": tr.processing_time,
            "comm_time": tr.comm_time,
        }
        for tr in best_traces
    ]

    return {
        "per_run": per_run,
        "budget": budget,
        "pareto": pareto_list,
        "convergence": conv_list,
        "schedule": schedule_list,
    }



def run_experiments(
    cfg: ExperimentConfig,
    instance_mode: str,
    include_ablations: bool,
    show_progress: bool,
    max_workers: int | None = None,
) -> RunnerOutput:
    output_root = cfg.run.output_dir
    run_root = output_root / "runs" / ("full" if instance_mode == "full" else "smoke")
    run_root.mkdir(parents=True, exist_ok=True)

    per_run_csv = run_root / "per_run_metrics.csv"
    pareto_csv = run_root / "pareto_points.csv"
    convergence_csv = run_root / "convergence.csv"
    schedule_csv = run_root / "schedule_traces.csv"
    indicator_reference_csv = run_root / "indicator_reference.csv"
    aggregate_csv = run_root / "aggregated_metrics.csv"
    stats_csv = run_root / "statistical_tests.csv"
    budget_json = run_root / "compute_budget.json"

    split_name = "full" if instance_mode == "full" else "quick"
    instances_names = (
        cfg.dataset.full_instances if instance_mode == "full" else cfg.dataset.quick_instances
    )
    seeds = cfg.run.random_seeds if instance_mode == "full" else cfg.run.quick_random_seeds

    algorithms = list(cfg.run.algorithms)
    if include_ablations:
        algorithms.extend(cfg.run.ablations)

    # Build task list: one lightweight tuple per (seed, instance, algorithm).
    tasks: list[tuple[int, str, str]] = [
        (seed, instance_name, algorithm)
        for seed in seeds
        for instance_name in instances_names
        for algorithm in algorithms
    ]

    n_workers = max_workers if max_workers is not None else os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(tasks)))

    logger.info(
        "Running %s experiments: instances=%s seeds=%s algorithms=%s workers=%d tasks=%d",
        instance_mode,
        instances_names,
        seeds,
        algorithms,
        n_workers,
        len(tasks),
    )

    per_run_rows: list[dict] = []
    pareto_rows: list[dict] = []
    conv_rows: list[dict] = []
    schedule_rows: list[dict] = []
    budget_rows: list[dict] = []
    indicator_rows: list[dict] = []

    if n_workers <= 1:
        # Sequential fallback: initialize once in this process.
        _init_worker(cfg, split_name, instances_names)
        iterator = tasks
        if show_progress:
            iterator = tqdm(iterator, desc="Experiments", unit="run")
        for task in iterator:
            result = _run_one(task)
            per_run_rows.append(result["per_run"])
            budget_rows.append(result["budget"])
            pareto_rows.extend(result["pareto"])
            conv_rows.extend(result["convergence"])
            schedule_rows.extend(result["schedule"])
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(cfg, split_name, instances_names),
        ) as pool:
            futures = {pool.submit(_run_one, task): task for task in tasks}
            completed_iter = as_completed(futures)
            if show_progress:
                completed_iter = tqdm(
                    completed_iter,
                    total=len(futures),
                    desc=f"Experiments ({n_workers} workers)",
                    unit="run",
                )
            for future in completed_iter:
                result = future.result()
                per_run_rows.append(result["per_run"])
                budget_rows.append(result["budget"])
                pareto_rows.extend(result["pareto"])
                conv_rows.extend(result["convergence"])
                schedule_rows.extend(result["schedule"])

    per_run_df = pd.DataFrame(per_run_rows)
    pareto_df = pd.DataFrame(pareto_rows)
    conv_df = pd.DataFrame(conv_rows)
    schedule_df = pd.DataFrame(schedule_rows)

    # Shared indicators: same reference point/front per instance for all algorithms and seeds.
    obj_cols = ["makespan", "energy", "one_minus_reliability"]
    for instance_name, inst_group in pareto_df.groupby("instance"):
        all_points = inst_group[obj_cols].to_numpy(dtype=np.float64)
        ref_point = build_ref_point(all_points, margin=0.10)
        reference_front = non_dominated_front(all_points)

        indicator_rows.append(
            {
                "instance": instance_name,
                "ref_makespan": float(ref_point[0]),
                "ref_energy": float(ref_point[1]),
                "ref_one_minus_reliability": float(ref_point[2]),
                "reference_front_size": int(len(reference_front)),
                "all_points_size": int(len(all_points)),
            }
        )

        for (seed, algorithm), run_group in inst_group.groupby(["seed", "algorithm"], sort=False):
            run_points = run_group[obj_cols].to_numpy(dtype=np.float64)
            run_hv = hypervolume(run_points, ref_point)
            run_igd = igd(run_points, reference_front)

            mask = (
                (per_run_df["instance"] == instance_name)
                & (per_run_df["seed"] == int(seed))
                & (per_run_df["algorithm"] == str(algorithm))
            )
            per_run_df.loc[mask, "hypervolume"] = run_hv
            per_run_df.loc[mask, "igd"] = run_igd

    per_run_df.to_csv(per_run_csv, index=False)
    pareto_df.to_csv(pareto_csv, index=False)
    conv_df.to_csv(convergence_csv, index=False)
    schedule_df.to_csv(schedule_csv, index=False)
    pd.DataFrame(indicator_rows).to_csv(indicator_reference_csv, index=False)

    aggregate_metrics(per_run_csv, aggregate_csv)
    run_statistics(per_run_csv, stats_csv)

    budget_payload = {
        "mode": instance_mode,
        "instances": instances_names,
        "seeds": seeds,
        "rows": budget_rows,
    }
    budget_json.write_text(json.dumps(budget_payload, indent=2), encoding="utf-8")

    return RunnerOutput(
        run_root=run_root,
        per_run_csv=per_run_csv,
        pareto_csv=pareto_csv,
        convergence_csv=convergence_csv,
        schedule_csv=schedule_csv,
        indicator_reference_csv=indicator_reference_csv,
        aggregate_csv=aggregate_csv,
        stats_csv=stats_csv,
        budget_json=budget_json,
    )
