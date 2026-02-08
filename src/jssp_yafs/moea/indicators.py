from __future__ import annotations

import numpy as np
from pymoo.config import Config
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

Config.warnings["not_compiled"] = False



def hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    indicator = HV(ref_point=ref_point)
    return float(indicator(points))



def build_ref_point(points: np.ndarray, margin: float = 0.1) -> np.ndarray:
    worst = points.max(axis=0)
    return worst * (1.0 + margin) + 1e-12


def non_dominated_front(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 0), dtype=np.float64)
    front_idx = NonDominatedSorting().do(points, only_non_dominated_front=True)
    return points[np.array(front_idx, dtype=np.int64)]


def igd(points: np.ndarray, reference_front: np.ndarray) -> float:
    if len(points) == 0:
        return float("inf")
    if len(reference_front) == 0:
        return float("inf")
    distances = np.linalg.norm(reference_front[:, None, :] - points[None, :, :], axis=2)
    return float(np.min(distances, axis=1).mean())
