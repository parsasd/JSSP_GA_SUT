from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from yafs.topology import Topology

from jssp_yafs.config import TopologyConfig


@dataclass(slots=True)
class EdgeTopology:
    topology: Topology
    edge_nodes: list[int]
    fog_nodes: list[int]
    cloud_nodes: list[int]
    compute_nodes: list[int]



def _link_key(a_tier: str, b_tier: str) -> str:
    pair = sorted([a_tier, b_tier])
    return f"{pair[0]}-{pair[1]}"



def build_edge_fog_cloud_topology(cfg: TopologyConfig) -> EdgeTopology:
    g = nx.Graph()

    edge_nodes: list[int] = []
    fog_nodes: list[int] = []
    cloud_nodes: list[int] = []

    node_id = 0

    def add_nodes(count: int, tier: str, mips: float, p_active: float, p_idle: float, lam: float) -> list[int]:
        nonlocal node_id
        nodes: list[int] = []
        for _ in range(count):
            g.add_node(
                node_id,
                IPT=float(mips),
                tier=tier,
                p_active=float(p_active),
                p_idle=float(p_idle),
                failure_lambda=float(lam),
            )
            nodes.append(node_id)
            node_id += 1
        return nodes

    edge_nodes = add_nodes(
        cfg.edge.count,
        "edge",
        cfg.edge.mips,
        cfg.edge.p_active,
        cfg.edge.p_idle,
        cfg.edge.failure_lambda,
    )
    fog_nodes = add_nodes(
        cfg.fog.count,
        "fog",
        cfg.fog.mips,
        cfg.fog.p_active,
        cfg.fog.p_idle,
        cfg.fog.failure_lambda,
    )
    cloud_nodes = add_nodes(
        cfg.cloud.count,
        "cloud",
        cfg.cloud.mips,
        cfg.cloud.p_active,
        cfg.cloud.p_idle,
        cfg.cloud.failure_lambda,
    )

    tier_nodes = {
        "edge": edge_nodes,
        "fog": fog_nodes,
        "cloud": cloud_nodes,
    }

    # Intra-tier links as rings to avoid disconnected components.
    for tier, nodes in tier_nodes.items():
        if len(nodes) <= 1:
            continue
        key = _link_key(tier, tier)
        link_cfg = cfg.links[key]
        for i in range(len(nodes)):
            a = nodes[i]
            b = nodes[(i + 1) % len(nodes)]
            g.add_edge(
                a,
                b,
                BW=float(link_cfg.bw_mbps),
                PR=float(link_cfg.prop_delay),
                p_tx=float(link_cfg.p_tx),
                failure_lambda=float(link_cfg.failure_lambda),
            )

    # Fully connect adjacent tiers (edge<->fog and fog<->cloud), plus sparse edge<->cloud backup.
    for a_tier, b_tier in [("edge", "fog"), ("fog", "cloud"), ("edge", "cloud")]:
        key = _link_key(a_tier, b_tier)
        link_cfg = cfg.links[key]
        for a in tier_nodes[a_tier]:
            for b in tier_nodes[b_tier]:
                if a_tier == "edge" and b_tier == "cloud" and (a + b) % 2 != 0:
                    # Keep edge-cloud sparse but existent.
                    continue
                g.add_edge(
                    a,
                    b,
                    BW=float(link_cfg.bw_mbps),
                    PR=float(link_cfg.prop_delay),
                    p_tx=float(link_cfg.p_tx),
                    failure_lambda=float(link_cfg.failure_lambda),
                )

    topo = Topology()
    topo.create_topology_from_graph(g)

    return EdgeTopology(
        topology=topo,
        edge_nodes=edge_nodes,
        fog_nodes=fog_nodes,
        cloud_nodes=cloud_nodes,
        compute_nodes=edge_nodes + fog_nodes + cloud_nodes,
    )
