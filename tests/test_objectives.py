import numpy as np

from jssp_yafs.config import EvalConfig, LinkConfig, NodeTierConfig, TopologyConfig
from jssp_yafs.data.models import JSSPInstance
from jssp_yafs.simulation.edge_topology import build_edge_fog_cloud_topology
from jssp_yafs.simulation.yafs_simulator import YAFSScheduleSimulator


def _toy_instance() -> JSSPInstance:
    machine = np.array([[0, 1], [1, 0]], dtype=np.int64)
    proc = np.array([[3.0, 4.0], [2.0, 5.0]], dtype=np.float64)
    return JSSPInstance("toy", 2, 2, machine, proc)



def _toy_topology() -> TopologyConfig:
    links = {
        "edge-edge": LinkConfig(100.0, 0.001, 1.0, 0.01),
        "fog-fog": LinkConfig(300.0, 0.001, 1.2, 0.006),
        "cloud-cloud": LinkConfig(600.0, 0.001, 1.5, 0.002),
        "edge-fog": LinkConfig(80.0, 0.003, 1.1, 0.01),
        "cloud-fog": LinkConfig(120.0, 0.005, 1.3, 0.008),
        "cloud-edge": LinkConfig(60.0, 0.01, 1.6, 0.015),
    }
    return TopologyConfig(
        edge=NodeTierConfig(2, 100.0, 10.0, 2.0, 0.02),
        fog=NodeTierConfig(1, 200.0, 20.0, 4.0, 0.01),
        cloud=NodeTierConfig(1, 400.0, 40.0, 8.0, 0.005),
        links=links,
    )



def test_simulation_objectives_are_valid(tmp_path) -> None:
    inst = _toy_instance()
    top = build_edge_fog_cloud_topology(_toy_topology())
    sim = YAFSScheduleSimulator(
        edge_topology=top,
        eval_cfg=EvalConfig(16000, "round_robin", 1.0, True),
        cache_dir=tmp_path / "cache",
    )

    seq = np.array([0, 1, 0, 1], dtype=np.int64)
    mmap = np.array([top.compute_nodes[0], top.compute_nodes[1]], dtype=np.int64)

    out1 = sim.evaluate(inst, seq, mmap)
    out2 = sim.evaluate(inst, seq, mmap)

    assert out1.makespan > 0
    assert out1.energy > 0
    assert 0 <= out1.reliability <= 1
    assert out1.objective_vector == out2.objective_vector
    sim.close()
