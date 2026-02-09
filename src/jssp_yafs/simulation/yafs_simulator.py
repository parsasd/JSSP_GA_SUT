from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from jssp_yafs.config import EvalConfig
from jssp_yafs.data.models import JSSPInstance
from jssp_yafs.scheduling.decoder import decode_operation_order
from jssp_yafs.simulation.edge_topology import EdgeTopology


@dataclass(slots=True)
class OperationTrace:
    job: int
    operation: int
    machine: int
    node: int
    predecessor_node: int
    start: float
    end: float
    processing_time: float
    comm_time: float


@dataclass(slots=True)
class SimulationResult:
    makespan: float
    energy: float
    reliability: float
    objective_vector: tuple[float, float, float]
    traces: list[OperationTrace]
    log_reliability: float


class YAFSScheduleSimulator:
    """Deterministic JSSP schedule simulation over a YAFS topology model."""

    def __init__(
        self,
        edge_topology: EdgeTopology,
        eval_cfg: EvalConfig,
        cache_dir: str | Path = "",
        cache_traces: bool = False,
    ) -> None:
        self.edge_topology = edge_topology
        self.eval_cfg = eval_cfg
        self.cache_traces = cache_traces
        self._path_cache: dict[tuple[int, int, int], tuple[float, float, float, list[int]]] = {}
        self._eval_cache: dict[int, dict[str, Any]] = {}

    def close(self) -> None:
        self._eval_cache.clear()

    def _reset_cache(self) -> None:
        self._eval_cache.clear()

    def _source_node(self, job: int) -> int:
        # Deterministic source edge placement per job.
        return self.edge_topology.edge_nodes[job % len(self.edge_topology.edge_nodes)]

    def _path_latency_energy_reliability(
        self,
        src: int,
        dst: int,
        payload_bytes: int,
    ) -> tuple[float, float, float, list[int]]:
        key = (src, dst, payload_bytes)
        if key in self._path_cache:
            return self._path_cache[key]

        if src == dst:
            self._path_cache[key] = (0.0, 0.0, 1.0, [src])
            return self._path_cache[key]

        graph = self.edge_topology.topology.G
        path = nx.shortest_path(graph, src, dst, weight="PR")

        total_latency = 0.0
        total_energy = 0.0
        log_rel = 0.0

        for u, v in zip(path[:-1], path[1:], strict=True):
            edge_att = graph.edges[(u, v)]
            transmit = payload_bytes / (float(edge_att["BW"]) * 1_000_000.0)
            latency = transmit + float(edge_att["PR"])
            total_latency += latency
            total_energy += self.eval_cfg.communication_weight * float(edge_att["p_tx"]) * transmit
            log_rel += -float(edge_att["failure_lambda"]) * latency

        rel = math.exp(log_rel)
        self._path_cache[key] = (total_latency, total_energy, rel, path)
        return self._path_cache[key]

    def _cache_key(
        self,
        instance: JSSPInstance,
        sequence: np.ndarray,
        machine_map: np.ndarray,
    ) -> int:
        return hash((instance.name, sequence.data.tobytes(), machine_map.data.tobytes()))

    def evaluate(
        self,
        instance: JSSPInstance,
        sequence: np.ndarray,
        machine_map: np.ndarray,
        with_traces: bool = True,
    ) -> SimulationResult:
        cache_key = self._cache_key(instance, sequence, machine_map)
        cached: dict[str, Any] | None = self._eval_cache.get(cache_key)

        if cached is not None:
            if with_traces and "traces" not in cached:
                cached = None
            else:
                traces = [OperationTrace(**row) for row in cached.get("traces", [])] if with_traces else []
                return SimulationResult(
                    makespan=float(cached["makespan"]),
                    energy=float(cached["energy"]),
                    reliability=float(cached["reliability"]),
                    objective_vector=(
                        float(cached["makespan"]),
                        float(cached["energy"]),
                        float(cached["one_minus_reliability"]),
                    ),
                    traces=traces,
                    log_reliability=float(cached["log_reliability"]),
                )

        traces: list[OperationTrace] = []
        if not with_traces:
            traces = []

        decoded = decode_operation_order(sequence, instance)

        machine_ready = np.zeros(instance.n_machines, dtype=np.float64)
        job_ready = np.zeros(instance.n_jobs, dtype=np.float64)
        job_last_node = np.full(instance.n_jobs, fill_value=-1, dtype=np.int64)

        node_busy: dict[int, float] = {n: 0.0 for n in self.edge_topology.compute_nodes}
        total_energy = 0.0
        log_reliability = 0.0

        graph = self.edge_topology.topology.G

        for op in decoded:
            machine = op.machine
            node = int(machine_map[machine])
            prev_node = int(job_last_node[op.job])
            if prev_node < 0:
                prev_node = self._source_node(op.job)

            comm_t, comm_e, comm_r, _ = self._path_latency_energy_reliability(
                prev_node, node, self.eval_cfg.payload_bytes
            )

            release = job_ready[op.job] + comm_t
            start = max(release, machine_ready[machine])

            mips = float(graph.nodes[node]["IPT"])
            proc_time = op.processing_time / mips
            end = start + proc_time

            p_active = float(graph.nodes[node]["p_active"])
            node_lambda = float(graph.nodes[node]["failure_lambda"])

            total_energy += p_active * proc_time + comm_e
            node_busy[node] += proc_time

            op_log_rel = -node_lambda * proc_time
            if comm_r <= 0:
                op_log_rel += -1e9
            else:
                op_log_rel += math.log(comm_r)
            log_reliability += op_log_rel

            machine_ready[machine] = end
            job_ready[op.job] = end
            job_last_node[op.job] = node

            if with_traces:
                traces.append(
                    OperationTrace(
                        job=op.job,
                        operation=op.operation,
                        machine=machine,
                        node=node,
                        predecessor_node=prev_node,
                        start=float(start),
                        end=float(end),
                        processing_time=float(proc_time),
                        comm_time=float(comm_t),
                    )
                )

        makespan = float(job_ready.max())

        if self.eval_cfg.include_idle_energy:
            for node in self.edge_topology.compute_nodes:
                idle = max(0.0, makespan - node_busy[node])
                total_energy += float(graph.nodes[node]["p_idle"]) * idle

        reliability = float(math.exp(log_reliability)) if log_reliability > -745 else 0.0
        objective = (makespan, float(total_energy), 1.0 - reliability)

        payload: dict[str, Any] = {
            "makespan": makespan,
            "energy": float(total_energy),
            "reliability": reliability,
            "one_minus_reliability": 1.0 - reliability,
            "log_reliability": float(log_reliability),
        }
        if self.cache_traces and with_traces:
            from dataclasses import asdict
            payload["traces"] = [asdict(t) for t in traces]
        self._eval_cache[cache_key] = payload

        return SimulationResult(
            makespan=makespan,
            energy=float(total_energy),
            reliability=reliability,
            objective_vector=objective,
            traces=traces,
            log_reliability=float(log_reliability),
        )
