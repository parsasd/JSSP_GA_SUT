import numpy as np

from jssp_yafs.data.loader import parse_instance_corpus
from jssp_yafs.scheduling.decoder import decode_operation_order
from jssp_yafs.scheduling.representation import repair_sequence


def test_sequence_repair_and_decode_feasible() -> None:
    instance = parse_instance_corpus("data/raw/instance_data.txt")["ft06"]

    rng = np.random.default_rng(42)
    broken = np.array([999] * instance.n_operations, dtype=np.int64)
    repaired = repair_sequence(broken, instance.n_jobs, instance.n_machines, rng)

    decoded = decode_operation_order(repaired, instance)
    assert len(decoded) == instance.n_operations

    counts = np.bincount(repaired, minlength=instance.n_jobs)
    assert (counts == instance.n_machines).all()
