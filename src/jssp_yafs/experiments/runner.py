from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jssp_yafs.config import ExperimentConfig, GAConfig
from jssp_yafs.data.loader import load_instances_from_folder
from jssp_yafs.experiments.aggregation import aggregate_metrics
from jssp_yafs.experiments.statistics import run_statistics
from jssp_yafs.moea.indicators import build_ref_point, hypervolume, igd, non_dominated_front
from jssp_yafs.moea.nsga2 import NSGA2Result, run_nsga2
from jssp_yafs.scheduling.heuristics import heuristic_machine_map, priority_sequence
from jssp_yafs.simulation.edge_topology import build_edge_fog_cloud_topology
from jssp_yafs.simulation.yafs_simulator import SimulationResult, YAFSScheduleSimulator

logger = logging.getLogger(__name__)


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



def run_experiments(
    cfg: ExperimentConfig,
    instance_mode: str,
    include_ablations: bool,
    show_progress: bool,
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

    instances = load_instances_from_folder(
        Path("data/processed") / split_name,
        instances_names,
    )

    edge_topology = build_edge_fog_cloud_topology(cfg.topology)
    node_mips = {n: float(edge_topology.topology.G.nodes[n]["IPT"]) for n in edge_topology.compute_nodes}
    simulator = YAFSScheduleSimulator(edge_topology, cfg.evaluation, cfg.run.cache_dir)

    algorithms = list(cfg.run.algorithms)
    if include_ablations:
        algorithms.extend(cfg.run.ablations)

    per_run_rows: list[dict[str, float | int | str]] = []
    pareto_rows: list[dict[str, float | int | str]] = []
    conv_rows: list[dict[str, float | int | str]] = []
    schedule_rows: list[dict[str, float | int | str]] = []
    indicator_rows: list[dict[str, float | int | str]] = []
    budget_rows: list[dict[str, float | int | str]] = []

    logger.info(
        "Running %s experiments: instances=%s seeds=%s algorithms=%s",
        instance_mode,
        instances_names,
        seeds,
        algorithms,
    )

    for seed in seeds:
        for instance_name in instances_names:
            instance = instances[instance_name]

            for algorithm in algorithms:
                rng = np.random.default_rng(seed)
                sims, history, evals, runtime_sec, decisions = _run_algorithm(
                    algorithm=algorithm,
                    instance=instance,
                    simulator=simulator,
                    ga_plain=cfg.ga_plain,
                    ga_enhanced=cfg.ga_enhanced,
                    compute_nodes=edge_topology.compute_nodes,
                    node_mips=node_mips,
                    rng=rng,
                    show_progress=show_progress,
                )

                objs = np.array([s.objective_vector for s in sims], dtype=np.float64)
                best_idx = _best_compromise(objs)
                best = sims[best_idx]
                best_traces = best.traces
                if not best_traces:
                    best_seq, best_map = decisions[best_idx]
                    best_full = simulator.evaluate(
                        instance,
                        best_seq,
                        best_map,
                        with_traces=True,
                    )
                    best_traces = best_full.traces

                per_run_rows.append(
                    {
                        "instance": instance_name,
                        "seed": seed,
                        "algorithm": algorithm,
                        # Filled after all runs using a shared per-instance reference point.
                        "hypervolume": np.nan,
                        # Inverted generational distance to shared empirical reference front.
                        "igd": np.nan,
                        "best_makespan": best.makespan,
                        "best_energy": best.energy,
                        "best_reliability": best.reliability,
                        "runtime_sec": runtime_sec,
                        "evaluations": evals,
                    }
                )

                budget_rows.append(
                    {
                        "instance": instance_name,
                        "seed": seed,
                        "algorithm": algorithm,
                        "population_size": (
                            cfg.ga_plain.population_size
                            if algorithm == "plain_ga"
                            else cfg.ga_enhanced.population_size
                        ),
                        "generations": (
                            cfg.ga_plain.generations if algorithm == "plain_ga" else cfg.ga_enhanced.generations
                        ),
                        "evaluations": evals,
                        "runtime_sec": runtime_sec,
                    }
                )

                for point_idx, sim in enumerate(sims):
                    pareto_rows.append(
                        {
                            "instance": instance_name,
                            "seed": seed,
                            "algorithm": algorithm,
                            "point_id": point_idx,
                            "makespan": sim.makespan,
                            "energy": sim.energy,
                            "reliability": sim.reliability,
                            "one_minus_reliability": 1.0 - sim.reliability,
                        }
                    )

                for row in history:
                    conv_rows.append(
                        {
                            "instance": instance_name,
                            "seed": seed,
                            "algorithm": algorithm,
                            **row,
                        }
                    )

                for tr in best_traces:
                    schedule_rows.append(
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
                    )

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

    simulator.close()

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
