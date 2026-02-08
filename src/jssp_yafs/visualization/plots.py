from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from jssp_yafs.config import ExperimentConfig
from jssp_yafs.simulation.edge_topology import build_edge_fog_cloud_topology

sns.set_theme(style="whitegrid")



def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)



def plot_all(cfg: ExperimentConfig, run_root: Path, mode: str) -> Path:
    fig_root = cfg.run.output_dir / "figures" / mode
    fig_root.mkdir(parents=True, exist_ok=True)

    per_run = pd.read_csv(run_root / "per_run_metrics.csv")
    pareto = pd.read_csv(run_root / "pareto_points.csv")
    conv = pd.read_csv(run_root / "convergence.csv")
    traces = pd.read_csv(run_root / "schedule_traces.csv")

    _plot_pareto_per_instance(pareto, fig_root)
    _plot_pareto_aggregated(pareto, fig_root)
    _plot_convergence(conv, fig_root)
    _plot_box_violin(per_run, fig_root)
    _plot_topology(cfg, fig_root)
    _plot_gantt(per_run, traces, fig_root)
    _plot_correlation_heatmap(pareto, fig_root)
    _write_captions(fig_root)
    return fig_root



def _plot_pareto_per_instance(df: pd.DataFrame, fig_root: Path) -> None:
    for instance, g in df.groupby("instance"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for algo, ga in g.groupby("algorithm"):
            sc = ax.scatter(
                ga["makespan"],
                ga["energy"],
                c=ga["reliability"],
                cmap="viridis",
                alpha=0.75,
                label=algo,
            )
        ax.set_title(f"Pareto Front ({instance})")
        ax.set_xlabel("Makespan (lower is better)")
        ax.set_ylabel("Energy (J, lower is better)")
        ax.legend(loc="best", fontsize=8)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Reliability")
        _save(fig, fig_root / f"pareto_{instance}")



def _plot_pareto_aggregated(df: pd.DataFrame, fig_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo, g in df.groupby("algorithm"):
        ax.scatter(g["makespan"], g["energy"], alpha=0.45, s=20, label=algo)
    ax.set_title("Aggregated Pareto Points (All Instances/Seeds)")
    ax.set_xlabel("Makespan")
    ax.set_ylabel("Energy")
    ax.legend(fontsize=8)
    _save(fig, fig_root / "pareto_aggregated")



def _plot_convergence(df: pd.DataFrame, fig_root: Path) -> None:
    if df.empty:
        return
    agg = (
        df.groupby(["algorithm", "generation"], as_index=False)[
            ["hypervolume", "best_makespan", "best_energy", "best_reliability"]
        ]
        .mean()
        .sort_values("generation")
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ["hypervolume", "best_makespan", "best_energy", "best_reliability"]
    titles = ["Hypervolume", "Best Makespan", "Best Energy", "Best Reliability"]
    for ax, metric, title in zip(axes.flatten(), metrics, titles, strict=True):
        for algo, g in agg.groupby("algorithm"):
            ax.plot(g["generation"], g[metric], label=algo)
        ax.set_title(title)
        ax.set_xlabel("Generation")
    axes[0, 0].legend(fontsize=8)
    _save(fig, fig_root / "convergence_metrics")



def _plot_box_violin(df: pd.DataFrame, fig_root: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(data=df, x="algorithm", y="best_makespan", ax=axes[0])
    sns.violinplot(data=df, x="algorithm", y="best_energy", ax=axes[1], inner="quart")
    sns.boxplot(data=df, x="algorithm", y="best_reliability", ax=axes[2])

    axes[0].set_title("Makespan Distribution")
    axes[1].set_title("Energy Distribution")
    axes[2].set_title("Reliability Distribution")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    _save(fig, fig_root / "box_violin_metrics")



def _plot_topology(cfg: ExperimentConfig, fig_root: Path) -> None:
    topo = build_edge_fog_cloud_topology(cfg.topology)
    g = topo.topology.G

    color_map = []
    for n in g.nodes():
        tier = g.nodes[n]["tier"]
        if tier == "edge":
            color_map.append("#1f77b4")
        elif tier == "fog":
            color_map.append("#ff7f0e")
        else:
            color_map.append("#2ca02c")

    pos = nx.spring_layout(g, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx(g, pos=pos, node_color=color_map, with_labels=True, node_size=650, ax=ax)
    ax.set_title("Edge-Fog-Cloud Topology (YAFS)")
    ax.axis("off")
    _save(fig, fig_root / "topology")



def _plot_gantt(per_run: pd.DataFrame, traces: pd.DataFrame, fig_root: Path) -> None:
    if traces.empty:
        return

    # Representative: enhanced GA with min makespan, fallback to global minimum.
    subset = per_run[per_run["algorithm"] == "enhanced_ga"]
    if subset.empty:
        subset = per_run
    best_row = subset.sort_values("best_makespan", ascending=True).iloc[0]

    filt = (
        (traces["instance"] == best_row["instance"])
        & (traces["seed"] == best_row["seed"])
        & (traces["algorithm"] == best_row["algorithm"])
    )
    g = traces.loc[filt].copy()
    if g.empty:
        return

    machines = sorted(g["machine"].unique().tolist())
    y_pos = {m: i for i, m in enumerate(machines)}

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab20")

    for _, row in g.iterrows():
        m = int(row["machine"])
        y = y_pos[m]
        start = float(row["start"])
        duration = float(row["end"] - row["start"])
        job = int(row["job"])
        ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors=cmap(job % 20), alpha=0.9)

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([f"M{m}" for m in machines])
    ax.set_xlabel("Time")
    ax.set_title(
        f"Representative Gantt: {best_row['instance']} | {best_row['algorithm']} | seed={int(best_row['seed'])}"
    )
    _save(fig, fig_root / "gantt_representative")



def _plot_correlation_heatmap(df: pd.DataFrame, fig_root: Path) -> None:
    corr = df[["makespan", "energy", "reliability"]].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Trade-off Correlation Heatmap")
    _save(fig, fig_root / "tradeoff_correlation_heatmap")



def _write_captions(fig_root: Path) -> None:
    text = """# Figure Captions

1. `pareto_<instance>`: Pareto fronts per benchmark instance (makespan vs energy; color encodes reliability).
2. `pareto_aggregated`: Combined Pareto points across all instances and seeds.
3. `convergence_metrics`: Convergence curves of hypervolume and objective best-values per generation.
4. `box_violin_metrics`: Distribution plots across seeds for makespan, energy, and reliability.
5. `topology`: Edge-fog-cloud topology used in YAFS-coupled simulation.
6. `gantt_representative`: Representative schedule (enhanced GA best run).
7. `tradeoff_correlation_heatmap`: Spearman correlation among makespan, energy, reliability.
"""
    (fig_root / "figure_captions.md").write_text(text, encoding="utf-8")
