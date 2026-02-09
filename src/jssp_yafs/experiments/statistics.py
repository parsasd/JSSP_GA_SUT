from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def holm_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        corr = min(1.0, p_values[idx] * factor)
        running_max = max(running_max, corr)
        adjusted[idx] = running_max
    return adjusted



def vargha_delaney_a12(a: np.ndarray, b: np.ndarray) -> float:
    """Vargha-Delaney A12 effect size measure.

    Returns the probability that a randomly chosen observation from *a*
    is greater than a randomly chosen observation from *b*.

    Interpretation (for a "larger is better" metric):
      A12 = 0.50  -> no effect
      A12 > 0.56  -> small
      A12 > 0.64  -> medium
      A12 > 0.71  -> large
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return np.nan
    total = 0.0
    for x in a:
        for y in b:
            if x > y:
                total += 1.0
            elif x == y:
                total += 0.5
    return total / (m * n)



def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size measure.

    Returns a value in [-1, 1].

    Interpretation (absolute value):
      |d| < 0.147  -> negligible
      |d| < 0.33   -> small
      |d| < 0.474  -> medium
      |d| >= 0.474 -> large
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return np.nan
    more = 0
    less = 0
    for x in a:
        for y in b:
            if x > y:
                more += 1
            elif x < y:
                less += 1
    return (more - less) / (m * n)



def run_statistics(per_run_csv: str | Path, out_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(per_run_csv)
    metrics = ["hypervolume", "igd", "best_makespan", "best_energy", "best_reliability"]
    rows: list[dict[str, float | str]] = []

    for metric in metrics:
        for instance, g in df.groupby("instance"):
            pivot = g.pivot_table(index="seed", columns="algorithm", values=metric, aggfunc="first")
            pivot = pivot.dropna(axis=1, how="any")
            if pivot.shape[1] < 2 or pivot.shape[0] < 2:
                continue

            cols = list(pivot.columns)
            arrays = [pivot[c].values for c in cols]

            try:
                stat, p = friedmanchisquare(*arrays)
            except ValueError:
                stat, p = np.nan, np.nan

            rows.append(
                {
                    "metric": metric,
                    "instance": instance,
                    "test": "friedman",
                    "comparison": "all",
                    "statistic": stat,
                    "p_value": p,
                    "p_value_adj": p,
                    "a12": np.nan,
                    "cliffs_delta": np.nan,
                }
            )

            pair_results: list[tuple[str, str, float, float, float, float]] = []
            for a, b in combinations(cols, 2):
                a_vals = pivot[a].values
                b_vals = pivot[b].values
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=".*Wilcoxon.*",
                            category=UserWarning,
                        )
                        stat_w, p_w = wilcoxon(
                            a_vals,
                            b_vals,
                            alternative="two-sided",
                            zero_method="wilcox",
                            method="auto",
                        )
                except ValueError:
                    stat_w, p_w = np.nan, np.nan

                a12 = vargha_delaney_a12(a_vals, b_vals)
                cd = cliffs_delta(a_vals, b_vals)
                pair_results.append((a, b, stat_w, p_w, a12, cd))

            pvals = [x[3] if not np.isnan(x[3]) else 1.0 for x in pair_results]
            p_adj = holm_correction(pvals)

            for (a, b, stat_w, p_w, a12, cd), adj in zip(pair_results, p_adj, strict=True):
                rows.append(
                    {
                        "metric": metric,
                        "instance": instance,
                        "test": "wilcoxon",
                        "comparison": f"{a} vs {b}",
                        "statistic": stat_w,
                        "p_value": p_w,
                        "p_value_adj": adj,
                        "a12": a12,
                        "cliffs_delta": cd,
                    }
                )

    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out
