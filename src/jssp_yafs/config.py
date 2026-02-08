from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    source_file: Path
    quick_instances: list[str]
    full_instances: list[str]


@dataclass(slots=True)
class NodeTierConfig:
    count: int
    mips: float
    p_active: float
    p_idle: float
    failure_lambda: float


@dataclass(slots=True)
class LinkConfig:
    bw_mbps: float
    prop_delay: float
    p_tx: float
    failure_lambda: float


@dataclass(slots=True)
class TopologyConfig:
    edge: NodeTierConfig
    fog: NodeTierConfig
    cloud: NodeTierConfig
    links: dict[str, LinkConfig]


@dataclass(slots=True)
class EvalConfig:
    payload_bytes: int
    source_edge_policy: str
    communication_weight: float
    include_idle_energy: bool


@dataclass(slots=True)
class GAConfig:
    population_size: int
    generations: int
    crossover_prob: float
    mutation_prob: float
    local_search_prob: float
    tournament_k: int
    crossover_ops: list[str]
    mutation_ops: list[str]


@dataclass(slots=True)
class RunConfig:
    output_dir: Path
    cache_dir: Path
    random_seeds: list[int]
    quick_random_seeds: list[int]
    algorithms: list[str]
    ablations: list[str]


@dataclass(slots=True)
class ExperimentConfig:
    dataset: DatasetConfig
    topology: TopologyConfig
    evaluation: EvalConfig
    ga_plain: GAConfig
    ga_enhanced: GAConfig
    run: RunConfig



def _node_tier(data: dict[str, Any]) -> NodeTierConfig:
    return NodeTierConfig(
        count=int(data["count"]),
        mips=float(data["mips"]),
        p_active=float(data["p_active"]),
        p_idle=float(data["p_idle"]),
        failure_lambda=float(data["failure_lambda"]),
    )



def _link_cfg(data: dict[str, Any]) -> LinkConfig:
    return LinkConfig(
        bw_mbps=float(data["bw_mbps"]),
        prop_delay=float(data["prop_delay"]),
        p_tx=float(data["p_tx"]),
        failure_lambda=float(data["failure_lambda"]),
    )



def _ga_cfg(data: dict[str, Any]) -> GAConfig:
    return GAConfig(
        population_size=int(data["population_size"]),
        generations=int(data["generations"]),
        crossover_prob=float(data["crossover_prob"]),
        mutation_prob=float(data["mutation_prob"]),
        local_search_prob=float(data["local_search_prob"]),
        tournament_k=int(data["tournament_k"]),
        crossover_ops=list(data["crossover_ops"]),
        mutation_ops=list(data["mutation_ops"]),
    )



def load_config(path: str | Path) -> ExperimentConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text())
    dataset = DatasetConfig(
        source_file=Path(raw["dataset"]["source_file"]),
        quick_instances=list(raw["dataset"]["quick_instances"]),
        full_instances=list(raw["dataset"]["full_instances"]),
    )
    topology = TopologyConfig(
        edge=_node_tier(raw["topology"]["edge"]),
        fog=_node_tier(raw["topology"]["fog"]),
        cloud=_node_tier(raw["topology"]["cloud"]),
        links={k: _link_cfg(v) for k, v in raw["topology"]["links"].items()},
    )
    evaluation = EvalConfig(
        payload_bytes=int(raw["evaluation"]["payload_bytes"]),
        source_edge_policy=str(raw["evaluation"].get("source_edge_policy", "round_robin")),
        communication_weight=float(raw["evaluation"].get("communication_weight", 1.0)),
        include_idle_energy=bool(raw["evaluation"].get("include_idle_energy", True)),
    )
    run = RunConfig(
        output_dir=Path(raw["run"]["output_dir"]),
        cache_dir=Path(raw["run"]["cache_dir"]),
        random_seeds=[int(x) for x in raw["run"]["random_seeds"]],
        quick_random_seeds=[int(x) for x in raw["run"]["quick_random_seeds"]],
        algorithms=list(raw["run"]["algorithms"]),
        ablations=list(raw["run"]["ablations"]),
    )
    return ExperimentConfig(
        dataset=dataset,
        topology=topology,
        evaluation=evaluation,
        ga_plain=_ga_cfg(raw["ga_plain"]),
        ga_enhanced=_ga_cfg(raw["ga_enhanced"]),
        run=run,
    )
