import numpy as np

from jssp_yafs.moea.indicators import hypervolume, igd, non_dominated_front


def test_non_dominated_front_filters_dominated_points() -> None:
    points = np.array(
        [
            [1.0, 3.0, 0.20],
            [2.0, 2.0, 0.10],
            [3.0, 1.0, 0.05],
            [2.5, 2.5, 0.30],
        ],
        dtype=np.float64,
    )
    front = non_dominated_front(points)

    assert len(front) == 3
    assert not np.any(np.all(np.isclose(front, [2.5, 2.5, 0.30]), axis=1))



def test_igd_is_zero_on_identical_front() -> None:
    front = np.array(
        [
            [1.0, 3.0, 0.20],
            [2.0, 2.0, 0.10],
            [3.0, 1.0, 0.05],
        ],
        dtype=np.float64,
    )
    assert igd(front, front) == 0.0



def test_hypervolume_monotonic_with_fixed_reference() -> None:
    ref = np.array([4.0, 4.0, 1.0], dtype=np.float64)
    better = np.array([[1.0, 1.0, 0.10], [1.5, 1.5, 0.08]], dtype=np.float64)
    worse = np.array([[2.0, 2.0, 0.20], [2.5, 2.5, 0.15]], dtype=np.float64)

    assert hypervolume(better, ref) > hypervolume(worse, ref)
