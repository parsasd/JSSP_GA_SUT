from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_metrics(per_run_csv: str | Path, out_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(per_run_csv)
    grouped = df.groupby(["instance", "algorithm"], as_index=False)

    rows: list[dict[str, float | str]] = []
    metric_cols = [
        "hypervolume",
        "igd",
        "best_makespan",
        "best_energy",
        "best_reliability",
        "runtime_sec",
    ]
    for (instance, algorithm), g in grouped:
        n = len(g)
        row: dict[str, float | str] = {
            "instance": instance,
            "algorithm": algorithm,
            "runs": int(n),
        }
        for col in metric_cols:
            mean = float(g[col].mean())
            std = float(g[col].std(ddof=1)) if n > 1 else 0.0
            ci95 = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95"] = ci95

        # Include BKS reference if available in the data.
        if "bks_makespan" in g.columns:
            bks = g["bks_makespan"].iloc[0]
            row["bks_makespan"] = bks if not (isinstance(bks, float) and np.isnan(bks)) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["instance", "algorithm"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out
