from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from jssp_yafs.config import ExperimentConfig
from jssp_yafs.simulation.edge_topology import build_edge_fog_cloud_topology

# ── Publication-quality theme (IEEE / ACM style) ──────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "grid.linestyle": "--",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.5,
    "patch.linewidth": 0.5,
})
sns.set_theme(style="ticks", font="serif", font_scale=1.0, rc={
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

# ── Consistent algorithm styling ────────────────────────────────────────────
_ALGO_ORDER = [
    "enhanced_ga",
    "plain_ga",
    "ablation_no_aos",
    "ablation_no_local_search",
    "heuristic_mwr",
    "heuristic_spt",
]

_ALGO_LABELS = {
    "enhanced_ga": "Enhanced GA",
    "plain_ga": "Plain GA",
    "ablation_no_aos": "w/o AOS",
    "ablation_no_local_search": "w/o Local Search",
    "heuristic_mwr": "MWR Heuristic",
    "heuristic_spt": "SPT Heuristic",
}

_ALGO_COLORS = {
    "enhanced_ga": "#D62728",       # red — protagonist
    "plain_ga": "#1F77B4",          # blue
    "ablation_no_aos": "#FF7F0E",   # orange
    "ablation_no_local_search": "#2CA02C",  # green
    "heuristic_mwr": "#8C564B",     # brown
    "heuristic_spt": "#7F7F7F",     # grey
}

_ALGO_MARKERS = {
    "enhanced_ga": "D",
    "plain_ga": "o",
    "ablation_no_aos": "s",
    "ablation_no_local_search": "^",
    "heuristic_mwr": "P",
    "heuristic_spt": "X",
}

# GA-only subset for plots where heuristics are noise
_GA_ALGOS = [
    "enhanced_ga", "plain_ga",
    "ablation_no_aos", "ablation_no_local_search",
]

def _label(algo: str) -> str:
    return _ALGO_LABELS.get(algo, algo)


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry
# ═══════════════════════════════════════════════════════════════════════════

def plot_all(cfg: ExperimentConfig, run_root: Path, mode: str) -> Path:
    fig_root = cfg.run.output_dir / "figures" / mode
    fig_root.mkdir(parents=True, exist_ok=True)

    per_run = pd.read_csv(run_root / "per_run_metrics.csv")
    pareto = pd.read_csv(run_root / "pareto_points.csv")
    conv = pd.read_csv(run_root / "convergence.csv")
    traces = pd.read_csv(run_root / "schedule_traces.csv")

    stats_path = run_root / "statistical_tests.csv"
    stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()

    # ── Filter out excluded algorithms (e.g. ablation_no_smart_init) ──
    _EXCLUDED_ALGOS = {"ablation_no_smart_init"}
    _allowed = set(_ALGO_ORDER)
    per_run = per_run[~per_run["algorithm"].isin(_EXCLUDED_ALGOS)]
    pareto = pareto[~pareto["algorithm"].isin(_EXCLUDED_ALGOS)]
    conv = conv[~conv["algorithm"].isin(_EXCLUDED_ALGOS)]
    if "algorithm" in traces.columns:
        traces = traces[~traces["algorithm"].isin(_EXCLUDED_ALGOS)]
    if not stats.empty and "algorithm_b" in stats.columns:
        stats = stats[~stats["algorithm_b"].isin(_EXCLUDED_ALGOS)]
    if not stats.empty and "algorithm" in stats.columns:
        stats = stats[~stats["algorithm"].isin(_EXCLUDED_ALGOS)]
    if not stats.empty and "comparison" in stats.columns:
        mask = stats["comparison"].apply(
            lambda c: not any(ex in c for ex in _EXCLUDED_ALGOS)
        )
        stats = stats[mask]

    # Existing (improved)
    _plot_pareto_per_instance(pareto, fig_root)
    _plot_convergence_per_instance(conv, fig_root)
    _plot_topology(cfg, fig_root)
    _plot_gantt(per_run, traces, fig_root)
    _plot_correlation_heatmap(pareto, fig_root)

    # New publication figures
    _plot_hv_boxplot_per_instance(per_run, fig_root)
    _plot_igd_boxplot_per_instance(per_run, fig_root)
    _plot_metric_boxplot_faceted(per_run, fig_root)
    _plot_convergence_global(conv, fig_root)
    _plot_algorithm_rank_heatmap(per_run, fig_root)
    _plot_ablation_effect_sizes(stats, fig_root)
    _plot_statistical_significance_heatmap(stats, fig_root)
    _plot_pareto_grid(pareto, fig_root)
    _plot_runtime_budget(per_run, fig_root)

    # Cumulative / aggregate comparison figures
    _plot_normalized_performance_profile(per_run, fig_root)
    _plot_radar_comparison(per_run, fig_root)
    _plot_critical_difference(per_run, fig_root)
    _plot_win_tie_loss(per_run, fig_root)
    _plot_overall_metric_summary(per_run, fig_root)
    _plot_multi_metric_rank_heatmap(per_run, fig_root)
    _plot_parallel_coordinates(per_run, fig_root)

    _write_captions(fig_root)
    return fig_root


# ═══════════════════════════════════════════════════════════════════════════
#  1. Pareto fronts — per instance (improved)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_pareto_per_instance(df: pd.DataFrame, fig_root: Path) -> None:
    for instance, g in df.groupby("instance"):
        fig, ax = plt.subplots(figsize=(5.5, 4.2))

        draw_order = [a for a in reversed(_ALGO_ORDER) if a in g["algorithm"].unique()]

        for algo in draw_order:
            ga = g[g["algorithm"] == algo]
            ax.scatter(
                ga["makespan"],
                ga["energy"],
                c=_ALGO_COLORS[algo],
                marker=_ALGO_MARKERS[algo],
                s=50 if algo == "enhanced_ga" else 28,
                alpha=0.90 if algo == "enhanced_ga" else 0.55,
                edgecolors="black" if algo == "enhanced_ga" else "none",
                linewidths=0.5 if algo == "enhanced_ga" else 0,
                label=_label(algo),
                zorder=10 if algo == "enhanced_ga" else 3,
            )

        ax.set_xlabel("Makespan")
        ax.set_ylabel("Energy (J)")
        ax.set_title(instance, fontweight="bold")
        ax.legend(loc="best", fontsize=7, handletextpad=0.3, borderpad=0.4)
        fig.tight_layout()
        _save(fig, fig_root / f"pareto_{instance}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. Pareto grid — 4×4 multi-panel (compact paper figure)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_pareto_grid(df: pd.DataFrame, fig_root: Path) -> None:
    instances = sorted(df["instance"].unique())
    n = len(instances)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows))
    axes = axes.flatten()

    for idx, instance in enumerate(instances):
        ax = axes[idx]
        g = df[df["instance"] == instance]

        draw_order = [a for a in reversed(_ALGO_ORDER) if a in g["algorithm"].unique()]
        for algo in draw_order:
            ga = g[g["algorithm"] == algo]
            ax.scatter(
                ga["makespan"], ga["energy"],
                c=_ALGO_COLORS[algo], marker=_ALGO_MARKERS[algo],
                s=25 if algo == "enhanced_ga" else 12,
                alpha=0.85 if algo == "enhanced_ga" else 0.50,
                edgecolors="black" if algo == "enhanced_ga" else "none",
                linewidths=0.3 if algo == "enhanced_ga" else 0,
                label=_label(algo) if idx == 0 else None,
                zorder=10 if algo == "enhanced_ga" else 3,
            )
        ax.set_title(instance, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Makespan", fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel("Energy (J)", fontsize=9)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(_ALGO_ORDER),
               fontsize=7, bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, fig_root / "pareto_grid")


# ═══════════════════════════════════════════════════════════════════════════
#  3. HV boxplot per instance (key paper figure)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_hv_boxplot_per_instance(per_run: pd.DataFrame, fig_root: Path) -> None:
    ga_only = per_run[per_run["algorithm"].isin(_GA_ALGOS)].copy()
    ga_only["algorithm"] = pd.Categorical(ga_only["algorithm"], categories=_GA_ALGOS, ordered=True)

    instances = sorted(ga_only["instance"].unique())
    n = len(instances)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows))
    axes = axes.flatten()

    for idx, instance in enumerate(instances):
        ax = axes[idx]
        inst_data = ga_only[ga_only["instance"] == instance]
        sns.boxplot(
            data=inst_data, x="algorithm", y="hypervolume",
            hue="algorithm", palette=_ALGO_COLORS, ax=ax,
            linewidth=0.6, fliersize=2, order=_GA_ALGOS, legend=False,
            width=0.6, saturation=0.85,
        )
        ax.set_title(instance, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Hypervolume" if idx % ncols == 0 else "")
        labels = [_label(a).replace(" ", "\n") for a in _GA_ALGOS]
        ax.set_xticks(range(len(_GA_ALGOS)))
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.yaxis.get_major_formatter().set_scientific(False)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    _save(fig, fig_root / "hv_boxplot_per_instance")


# ═══════════════════════════════════════════════════════════════════════════
#  4. IGD boxplot per instance
# ═══════════════════════════════════════════════════════════════════════════

def _plot_igd_boxplot_per_instance(per_run: pd.DataFrame, fig_root: Path) -> None:
    ga_only = per_run[per_run["algorithm"].isin(_GA_ALGOS)].copy()
    ga_only["algorithm"] = pd.Categorical(ga_only["algorithm"], categories=_GA_ALGOS, ordered=True)

    instances = sorted(ga_only["instance"].unique())
    n = len(instances)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows))
    axes = axes.flatten()

    for idx, instance in enumerate(instances):
        ax = axes[idx]
        inst_data = ga_only[ga_only["instance"] == instance]
        sns.boxplot(
            data=inst_data, x="algorithm", y="igd",
            hue="algorithm", palette=_ALGO_COLORS, ax=ax,
            linewidth=0.6, fliersize=2, order=_GA_ALGOS, legend=False,
            width=0.6, saturation=0.85,
        )
        ax.set_title(instance, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("IGD" if idx % ncols == 0 else "")
        labels = [_label(a).replace(" ", "\n") for a in _GA_ALGOS]
        ax.set_xticks(range(len(_GA_ALGOS)))
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    _save(fig, fig_root / "igd_boxplot_per_instance")


# ═══════════════════════════════════════════════════════════════════════════
#  5. Metric boxplots — faceted by instance (improved box/violin)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_metric_boxplot_faceted(per_run: pd.DataFrame, fig_root: Path) -> None:
    """Three separate figures: makespan, energy, reliability — each faceted by instance.

    Only GA variants are shown; heuristics operate on a completely different
    scale and would compress the GA boxes into thin lines.
    """
    metrics = [
        ("best_makespan", "Makespan", True),
        ("best_energy", "Energy (J)", True),
        ("best_reliability", "Reliability", False),
    ]
    algos_plot = _GA_ALGOS
    palette = [_ALGO_COLORS[a] for a in algos_plot]

    for col, label, lower_better in metrics:
        sub = per_run[per_run["algorithm"].isin(algos_plot)].copy()
        sub["algorithm"] = pd.Categorical(sub["algorithm"], categories=algos_plot, ordered=True)

        instances = sorted(sub["instance"].unique())
        n = len(instances)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows))
        axes = axes.flatten()

        for idx, instance in enumerate(instances):
            ax = axes[idx]
            inst_data = sub[sub["instance"] == instance]
            sns.boxplot(
                data=inst_data, x="algorithm", y=col,
                hue="algorithm", palette=_ALGO_COLORS, ax=ax,
                linewidth=0.6, fliersize=2, order=algos_plot,
                legend=False, width=0.6, saturation=0.85,
            )
            ax.set_title(instance, fontsize=9, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(label if idx % ncols == 0 else "")
            tick_labels = [_label(a).replace(" ", "\n") for a in algos_plot]
            ax.set_xticks(range(len(algos_plot)))
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=35, ha="right")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.yaxis.get_major_formatter().set_scientific(False)

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
        _save(fig, fig_root / f"boxplot_{col}")


# ═══════════════════════════════════════════════════════════════════════════
#  6. Convergence — per instance (4 selected instances)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_convergence_per_instance(conv: pd.DataFrame, fig_root: Path) -> None:
    if conv.empty:
        return

    # Select representative instances: small, medium, large
    available = sorted(conv["instance"].unique())
    # Pick up to 6 diverse instances
    preferred = ["ft06", "la16", "abz7", "ta01", "ta21", "yn1"]
    selected = [i for i in preferred if i in available]
    if len(selected) < 4:
        selected = available[:6]

    ga_conv = conv[conv["algorithm"].isin(_GA_ALGOS)].copy()

    ncols = min(3, len(selected))
    nrows = int(np.ceil(len(selected) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.5 * nrows))
    if len(selected) == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, instance in enumerate(selected):
        ax = axes[idx]
        inst_data = ga_conv[ga_conv["instance"] == instance]

        for algo in _GA_ALGOS:
            a_data = inst_data[inst_data["algorithm"] == algo]
            if a_data.empty:
                continue
            a_agg = a_data.groupby("generation")["hypervolume"].agg(["mean", "std"]).reset_index()
            ax.plot(
                a_agg["generation"], a_agg["mean"],
                color=_ALGO_COLORS[algo], label=_label(algo),
                linewidth=1.8 if algo == "enhanced_ga" else 1.0,
                zorder=10 if algo == "enhanced_ga" else 3,
            )
            ax.fill_between(
                a_agg["generation"],
                a_agg["mean"] - a_agg["std"],
                a_agg["mean"] + a_agg["std"],
                alpha=0.10, color=_ALGO_COLORS[algo],
            )

        ax.set_title(instance, fontsize=10, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Hypervolume" if idx % ncols == 0 else "")
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right", handletextpad=0.3)

    for idx in range(len(selected), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    _save(fig, fig_root / "convergence_per_instance")


# ═══════════════════════════════════════════════════════════════════════════
#  7. Convergence — global average (improved old plot)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_convergence_global(conv: pd.DataFrame, fig_root: Path) -> None:
    if conv.empty:
        return

    ga_conv = conv[conv["algorithm"].isin(_GA_ALGOS)].copy()
    metrics = [
        ("hypervolume", "Hypervolume"),
        ("best_makespan", "Best Makespan"),
        ("best_energy", "Best Energy (J)"),
        ("best_reliability", "Best Reliability"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for algo in _GA_ALGOS:
            a_data = ga_conv[ga_conv["algorithm"] == algo]
            if a_data.empty:
                continue
            a_agg = a_data.groupby("generation")[metric].mean().reset_index()
            ax.plot(
                a_agg["generation"], a_agg[metric],
                color=_ALGO_COLORS[algo], label=_label(algo),
                linewidth=1.8 if algo == "enhanced_ga" else 1.0,
                zorder=10 if algo == "enhanced_ga" else 3,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Generation")

    axes[0, 0].legend(fontsize=7, loc="lower right", handletextpad=0.3)
    fig.tight_layout()
    _save(fig, fig_root / "convergence_global")


# ═══════════════════════════════════════════════════════════════════════════
#  8. Algorithm rank heatmap
# ═══════════════════════════════════════════════════════════════════════════

def _plot_algorithm_rank_heatmap(per_run: pd.DataFrame, fig_root: Path) -> None:
    """Mean HV rank per instance (1 = best). Classic MOEA comparison table."""
    ga_only = per_run[per_run["algorithm"].isin(_GA_ALGOS)].copy()

    # Compute mean HV per (instance, algorithm)
    mean_hv = ga_only.groupby(["instance", "algorithm"])["hypervolume"].mean().reset_index()

    # Rank within each instance (higher HV = better → ascending=False → rank 1 = best)
    mean_hv["rank"] = mean_hv.groupby("instance")["hypervolume"].rank(ascending=False)

    pivot = mean_hv.pivot(index="instance", columns="algorithm", values="rank")
    pivot = pivot[[a for a in _GA_ALGOS if a in pivot.columns]]
    pivot.columns = [_label(a) for a in pivot.columns]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
        vmin=1, vmax=len(_GA_ALGOS),
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Rank (1 = best)", "shrink": 0.65},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, fig_root / "rank_heatmap")

    avg_rank = pivot.mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    bars = ax2.barh(
        avg_rank.index, avg_rank.values,
        color=[_ALGO_COLORS.get(k, "#999") for k in [
            a for a in _GA_ALGOS if _label(a) in avg_rank.index
        ]],
        edgecolor="black", linewidth=0.4, height=0.55,
    )
    ax2.set_xlabel("Average Rank")
    ax2.invert_yaxis()
    for bar, val in zip(bars, avg_rank.values):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=9)
    ax2.set_xlim(0, len(_GA_ALGOS) + 0.5)
    fig2.tight_layout()
    _save(fig2, fig_root / "rank_average_bar")


# ═══════════════════════════════════════════════════════════════════════════
#  9. Statistical significance heatmap
# ═══════════════════════════════════════════════════════════════════════════

def _plot_statistical_significance_heatmap(stats: pd.DataFrame, fig_root: Path) -> None:
    if stats.empty:
        return

    # Focus on enhanced_ga vs each other algorithm, on HV metric
    for metric in ["hypervolume", "igd", "best_makespan"]:
        wil = stats[
            (stats["metric"] == metric)
            & (stats["test"] == "wilcoxon")
            & (stats["comparison"].str.contains("enhanced_ga"))
        ].copy()
        if wil.empty:
            continue

        # Parse comparison to get the "other" algorithm
        def _other(comp: str) -> str:
            parts = comp.split(" vs ")
            return parts[1] if parts[0] == "enhanced_ga" else parts[0]

        wil["other"] = wil["comparison"].apply(_other)
        wil["other_label"] = wil["other"].map(_ALGO_LABELS)
        wil["significant"] = wil["p_value_adj"] < 0.05

        # Build matrix: instance x other_algorithm → Cliff's delta (or A12)
        effect_col = "cliffs_delta"
        if effect_col not in wil.columns or wil[effect_col].isna().all():
            continue

        # For comparisons where enhanced_ga is second, flip the sign
        def _signed_effect(row):
            parts = row["comparison"].split(" vs ")
            val = row[effect_col]
            if pd.isna(val):
                return 0.0
            if parts[0] == "enhanced_ga":
                return float(val)
            else:
                return -float(val)

        wil["effect"] = wil.apply(_signed_effect, axis=1)

        pivot = wil.pivot(index="instance", columns="other_label", values="effect")
        sig_pivot = wil.pivot(index="instance", columns="other_label", values="significant")

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Cliff's $\\delta$", "shrink": 0.65},
            ax=ax,
        )
        for i in range(sig_pivot.shape[0]):
            for j in range(sig_pivot.shape[1]):
                if sig_pivot.iloc[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                               edgecolor="black", linewidth=2.0))

        metric_label = {"hypervolume": "Hypervolume", "igd": "IGD",
                        "best_makespan": "Makespan"}.get(metric, metric)
        ax.set_title(
            f"Enhanced GA vs Others ({metric_label})",
            fontweight="bold", fontsize=11,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        _save(fig, fig_root / f"significance_heatmap_{metric}")


# ═══════════════════════════════════════════════════════════════════════════
#  10. Ablation effect-size chart
# ═══════════════════════════════════════════════════════════════════════════

def _plot_ablation_effect_sizes(stats: pd.DataFrame, fig_root: Path) -> None:
    if stats.empty:
        return

    ablations = {
        "ablation_no_aos": "w/o AOS",
        "ablation_no_local_search": "w/o Local Search",
    }

    for metric in ["hypervolume"]:
        wil = stats[
            (stats["metric"] == metric)
            & (stats["test"] == "wilcoxon")
        ].copy()

        rows = []
        for abl_key, abl_label in ablations.items():
            for _, row in wil.iterrows():
                comp = row["comparison"]
                if "enhanced_ga" not in comp or abl_key not in comp:
                    continue
                parts = comp.split(" vs ")
                delta = row["cliffs_delta"]
                if pd.isna(delta):
                    continue
                # Make positive = enhanced is better
                if parts[0] == "enhanced_ga":
                    delta = float(delta)
                else:
                    delta = -float(delta)
                rows.append({
                    "instance": row["instance"],
                    "ablation": abl_label,
                    "cliffs_delta": delta,
                    "significant": row["p_value_adj"] < 0.05,
                })

        if not rows:
            continue

        df_abl = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        instances = sorted(df_abl["instance"].unique())
        x = np.arange(len(instances))
        width = 0.35
        abl_labels = list(ablations.values())
        colors = ["#FF7F0E", "#2CA02C"]

        for i, (abl_label, color) in enumerate(zip(abl_labels, colors)):
            sub = df_abl[df_abl["ablation"] == abl_label].set_index("instance")
            vals = [sub.loc[inst, "cliffs_delta"] if inst in sub.index else 0 for inst in instances]
            sigs = [sub.loc[inst, "significant"] if inst in sub.index else False for inst in instances]
            bars = ax.bar(x + i * width, vals, width, label=abl_label, color=color,
                          edgecolor="black", linewidth=0.3)
            for j, (bar, sig) in enumerate(zip(bars, sigs)):
                if sig:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            "*", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.6, linestyle="-")
        ax.axhline(0.147, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.axhline(-0.147, color="grey", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.text(len(instances) - 0.5, 0.17, "negligible", fontsize=7,
                color="grey", ha="right", style="italic")

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(instances, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Cliff's $\\delta$")
        ax.set_title("Ablation Study (Hypervolume)", fontweight="bold")
        ax.legend(fontsize=8, handletextpad=0.3)
        ax.set_ylim(-1.1, 1.1)
        fig.tight_layout()
        _save(fig, fig_root / "ablation_effect_sizes")


# ═══════════════════════════════════════════════════════════════════════════
#  11. Topology (improved with layered layout)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_topology(cfg: ExperimentConfig, fig_root: Path) -> None:
    topo = build_edge_fog_cloud_topology(cfg.topology)
    g = topo.topology.G

    tier_colors = {"edge": "#4C9BE8", "fog": "#FF9B3E", "cloud": "#56B870"}
    tier_labels = {"edge": "Edge", "fog": "Fog", "cloud": "Cloud"}

    # Force layered layout: cloud top, fog middle, edge bottom
    pos = {}
    for tier, nodes, y in [
        ("cloud", topo.cloud_nodes, 2.0),
        ("fog", topo.fog_nodes, 1.0),
        ("edge", topo.edge_nodes, 0.0),
    ]:
        for i, n in enumerate(nodes):
            x = (i - (len(nodes) - 1) / 2) * 1.5
            pos[n] = (x, y)

    fig, ax = plt.subplots(figsize=(7, 5))

    nx.draw_networkx_edges(g, pos=pos, ax=ax, alpha=0.15, edge_color="#888", width=0.6)

    for tier, nodes in [("cloud", topo.cloud_nodes), ("fog", topo.fog_nodes), ("edge", topo.edge_nodes)]:
        node_color = tier_colors[tier]
        mips = g.nodes[nodes[0]]["IPT"]
        lbl = f"{tier_labels[tier]} ({mips:.0f} MIPS, n={len(nodes)})"
        nx.draw_networkx_nodes(g, pos=pos, nodelist=nodes, node_color=node_color,
                               node_size=500, edgecolors="black", linewidths=0.8,
                               label=lbl, ax=ax)

    nx.draw_networkx_labels(g, pos=pos, ax=ax, font_size=8, font_weight="bold",
                            font_family="serif")

    ax.legend(loc="upper left", fontsize=8, handletextpad=0.3)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, fig_root / "topology")


# ═══════════════════════════════════════════════════════════════════════════
#  12. Gantt chart (improved)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_gantt(per_run: pd.DataFrame, traces: pd.DataFrame, fig_root: Path) -> None:
    if traces.empty:
        return

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
    n_jobs = int(g["job"].max()) + 1

    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.6 * len(machines))))
    cmap = plt.get_cmap("tab20")

    for _, row in g.iterrows():
        m = int(row["machine"])
        y = y_pos[m]
        start = float(row["start"])
        duration = float(row["end"] - row["start"])
        job = int(row["job"])
        color = cmap(job % 20)
        ax.broken_barh(
            [(start, duration)], (y - 0.36, 0.72),
            facecolors=color, edgecolors="black", linewidths=0.2, alpha=0.88,
        )
        if duration > (g["end"].max() - g["start"].min()) * 0.025:
            ax.text(start + duration / 2, y, f"J{job}",
                    ha="center", va="center", fontsize=5.5, fontweight="bold")

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([f"M{m}" for m in machines], fontsize=8)
    ax.set_xlabel("Time")
    ax.set_title(
        f"{best_row['instance']} ({_label(best_row['algorithm'])})",
        fontweight="bold",
    )

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=cmap(j % 20), edgecolor="black", linewidth=0.3, label=f"J{j}")
                    for j in range(min(n_jobs, 20))]
    ax.legend(handles=legend_items, loc="upper right", ncol=min(n_jobs, 10),
              fontsize=5.5, handletextpad=0.3, columnspacing=0.5)
    fig.tight_layout()
    _save(fig, fig_root / "gantt_representative")


# ═══════════════════════════════════════════════════════════════════════════
#  13. Correlation heatmap (improved)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_correlation_heatmap(df: pd.DataFrame, fig_root: Path) -> None:
    rename = {"makespan": "Makespan", "energy": "Energy", "reliability": "Reliability"}
    sub = df[["makespan", "energy", "reliability"]].rename(columns=rename)
    corr = sub.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Spearman $\\rho$", "shrink": 0.75},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title("Objective Correlation", fontweight="bold")
    fig.tight_layout()
    _save(fig, fig_root / "tradeoff_correlation_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
#  14. Runtime / evaluation budget comparison
# ═══════════════════════════════════════════════════════════════════════════

def _plot_runtime_budget(per_run: pd.DataFrame, fig_root: Path) -> None:
    ga_only = per_run[per_run["algorithm"].isin(_GA_ALGOS)].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    order = _GA_ALGOS
    sns.boxplot(data=ga_only, x="algorithm", y="runtime_sec", order=order,
                hue="algorithm", palette=_ALGO_COLORS, ax=ax1,
                linewidth=0.6, fliersize=2, legend=False, width=0.6)
    ax1.set_title("Runtime (s)", fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Seconds")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels([_label(a).replace(" ", "\n") for a in order],
                        fontsize=7, rotation=25, ha="right")

    sns.boxplot(data=ga_only, x="algorithm", y="evaluations", order=order,
                hue="algorithm", palette=_ALGO_COLORS, ax=ax2,
                linewidth=0.6, fliersize=2, legend=False, width=0.6)
    ax2.set_title("Evaluations", fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("Count")
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([_label(a).replace(" ", "\n") for a in order],
                        fontsize=7, rotation=25, ha="right")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))

    fig.tight_layout()
    _save(fig, fig_root / "runtime_budget")


# ═══════════════════════════════════════════════════════════════════════════
#  15. Normalized performance profile (cumulative across instances)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_normalized_performance_profile(per_run: pd.DataFrame, fig_root: Path) -> None:
    """Performance profile: for each metric, normalise per-instance values to [0,1]
    and compare aggregate distributions across algorithms.

    Uses min-max normalisation within each instance so all instances
    contribute equally, avoiding large-instance domination.
    """
    metrics = [
        ("hypervolume", "Hypervolume", False),      # higher is better → invert for "cost"
        ("best_makespan", "Makespan", True),         # lower is better
        ("best_energy", "Energy", True),
        ("best_reliability", "Reliability", False),
    ]
    ga_only = per_run[per_run["algorithm"].isin(_GA_ALGOS)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    for ax, (metric, label, lower_better) in zip(axes.flatten(), metrics):
        def _normalise(group: pd.DataFrame) -> pd.DataFrame:
            vals = group[metric].values.astype(float)
            lo, hi = vals.min(), vals.max()
            if hi - lo < 1e-15:
                group["normalised"] = 0.0
            elif lower_better:
                group["normalised"] = (vals - lo) / (hi - lo)
            else:
                group["normalised"] = (hi - vals) / (hi - lo)
            return group

        normed = ga_only.groupby("instance", group_keys=False).apply(_normalise)

        for algo in _GA_ALGOS:
            a_data = normed[normed["algorithm"] == algo]["normalised"].sort_values().values
            n = len(a_data)
            if n == 0:
                continue
            y = np.arange(1, n + 1) / n
            ax.step(
                a_data, y,
                color=_ALGO_COLORS[algo], label=_label(algo),
                linewidth=1.8 if algo == "enhanced_ga" else 1.0,
                zorder=10 if algo == "enhanced_ga" else 3,
                where="post",
            )

        ax.set_xlabel(f"Normalised {label} (0 = best)")
        ax.set_ylabel("Fraction of runs")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.02)

    axes[0, 0].legend(fontsize=7, loc="lower right", handletextpad=0.3)
    fig.tight_layout()
    _save(fig, fig_root / "performance_profile")


# ═══════════════════════════════════════════════════════════════════════════
#  16. Radar/spider chart — aggregate metric comparison
# ═══════════════════════════════════════════════════════════════════════════

def _plot_radar_comparison(per_run: pd.DataFrame, fig_root: Path) -> None:
    """Radar chart showing each algorithm's normalised performance across metrics.

    Each axis represents a metric; values are normalised so 1 = best algorithm
    and 0 = worst. This gives an at-a-glance multi-objective profile.
    """
    metrics = [
        ("hypervolume", "Hypervolume", False),
        ("igd", "IGD", True),
        ("best_makespan", "Makespan", True),
        ("best_energy", "Energy", True),
        ("best_reliability", "Reliability", False),
    ]
    ga_only = per_run[per_run["algorithm"].isin(_GA_ALGOS)].copy()

    # Compute overall mean per algorithm for each metric
    algo_means: dict[str, list[float]] = {}
    metric_labels: list[str] = []
    for metric, label, lower_better in metrics:
        metric_labels.append(label)
        means = ga_only.groupby("algorithm")[metric].mean()
        lo, hi = means.min(), means.max()
        for algo in _GA_ALGOS:
            val = means.get(algo, np.nan)
            if hi - lo < 1e-15:
                normed = 1.0
            elif lower_better:
                normed = 1.0 - (val - lo) / (hi - lo)  # 1 = best (lowest)
            else:
                normed = (val - lo) / (hi - lo)          # 1 = best (highest)
            algo_means.setdefault(algo, []).append(normed)

    n_metrics = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for algo in _GA_ALGOS:
        values = algo_means[algo] + algo_means[algo][:1]
        ax.plot(angles, values, color=_ALGO_COLORS[algo], label=_label(algo),
                linewidth=2.0 if algo == "enhanced_ga" else 1.2,
                zorder=10 if algo == "enhanced_ga" else 3)
        ax.fill(angles, values, alpha=0.10 if algo == "enhanced_ga" else 0.03,
                color=_ALGO_COLORS[algo])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="grey")
    ax.set_title("Algorithm Comparison (1 = best)", fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=8,
              handletextpad=0.3)
    fig.tight_layout()
    _save(fig, fig_root / "radar_comparison")


# ═══════════════════════════════════════════════════════════════════════════
#  17. Critical difference diagram (Nemenyi post-hoc)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_critical_difference(per_run: pd.DataFrame, fig_root: Path) -> None:
    """Critical difference-style diagram: ranks each algorithm per (instance, seed),
    averages ranks, and shows with confidence intervals.

    This is the standard comparison plot used in multi-algorithm benchmarks
    (Demšar, 2006 style).
    """
    metrics = [
        ("hypervolume", "Hypervolume", False),
        ("igd", "IGD", True),
        ("best_makespan", "Makespan", True),
    ]
    ga_only = per_run[per_run["algorithm"].isin(_GA_ALGOS)].copy()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.5 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, label, lower_better) in zip(axes, metrics):
        ga_only[f"rank_{metric}"] = ga_only.groupby(["instance", "seed"])[metric].rank(
            ascending=lower_better, method="average"
        )

        mean_ranks = ga_only.groupby("algorithm")[f"rank_{metric}"].mean().reindex(_GA_ALGOS)
        std_ranks = ga_only.groupby("algorithm")[f"rank_{metric}"].std().reindex(_GA_ALGOS)
        n_obs = ga_only.groupby("algorithm")[f"rank_{metric}"].count().reindex(_GA_ALGOS)
        ci95 = 1.96 * std_ranks / np.sqrt(n_obs)

        sorted_algos = mean_ranks.sort_values().index.tolist()
        y_pos = np.arange(len(sorted_algos))

        for i, algo in enumerate(sorted_algos):
            rank_val = mean_ranks[algo]
            ci = ci95[algo]
            color = _ALGO_COLORS[algo]

            ax.barh(i, rank_val, color=color, edgecolor="black", linewidth=0.4,
                    height=0.55, alpha=0.85, zorder=3)
            ax.errorbar(rank_val, i, xerr=ci, fmt="none", color="black",
                        capsize=3, linewidth=1.0, zorder=5)
            ax.text(rank_val + ci + 0.06, i, f"{rank_val:.2f}",
                    va="center", fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([_label(a) for a in sorted_algos], fontsize=9)
        ax.set_xlabel("Mean Rank")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0, len(_GA_ALGOS) + 0.5)
        ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, fig_root / "critical_difference")


# ═══════════════════════════════════════════════════════════════════════════
#  18. Win/tie/loss matrix
# ═══════════════════════════════════════════════════════════════════════════

def _plot_win_tie_loss(per_run: pd.DataFrame, fig_root: Path) -> None:
    """For each pair of algorithms, count how many (instance, seed) runs one
    beats the other on HV.  Presented as a stacked bar chart.
    """
    ga_only = per_run[per_run["algorithm"].isin(_GA_ALGOS)].copy()

    # Focus on enhanced_ga vs each other
    others = [a for a in _GA_ALGOS if a != "enhanced_ga"]
    results: dict[str, dict[str, int]] = {}

    for other in others:
        wins = ties = losses = 0
        for (inst, seed), grp in ga_only.groupby(["instance", "seed"]):
            enh_vals = grp[grp["algorithm"] == "enhanced_ga"]["hypervolume"].values
            oth_vals = grp[grp["algorithm"] == other]["hypervolume"].values
            if len(enh_vals) == 0 or len(oth_vals) == 0:
                continue
            diff = enh_vals[0] - oth_vals[0]
            if abs(diff) < 1e-10:
                ties += 1
            elif diff > 0:
                wins += 1
            else:
                losses += 1
        results[_label(other)] = {"Win": wins, "Tie": ties, "Loss": losses}

    labels = list(results.keys())
    wins = [results[l]["Win"] for l in labels]
    ties = [results[l]["Tie"] for l in labels]
    losses = [results[l]["Loss"] for l in labels]

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, wins, width, label="Win", color="#2CA02C", edgecolor="black", linewidth=0.3)
    ax.bar(x, ties, width, bottom=wins, label="Tie", color="#FFDD57", edgecolor="black", linewidth=0.3)
    ax.bar(x, losses, width, bottom=[w + t for w, t in zip(wins, ties)],
           label="Loss", color="#D62728", edgecolor="black", linewidth=0.3)

    for i in range(len(labels)):
        if wins[i] > 0:
            ax.text(i, wins[i] / 2, str(wins[i]), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        if ties[i] > 0:
            ax.text(i, wins[i] + ties[i] / 2, str(ties[i]), ha="center", va="center",
                    fontsize=9, fontweight="bold")
        if losses[i] > 0:
            ax.text(i, wins[i] + ties[i] + losses[i] / 2, str(losses[i]),
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Runs (instance $\\times$ seed)")
    ax.set_title("Enhanced GA vs Others (Hypervolume)", fontweight="bold")
    ax.legend(fontsize=8, handletextpad=0.3)
    ax.set_ylim(0, max(w + t + l for w, t, l in zip(wins, ties, losses)) * 1.08)
    fig.tight_layout()
    _save(fig, fig_root / "win_tie_loss")


# ═══════════════════════════════════════════════════════════════════════════
#  19. Overall metric summary — grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════

def _plot_overall_metric_summary(per_run: pd.DataFrame, fig_root: Path) -> None:
    """Grouped bar chart showing each algorithm's normalised mean across metrics.

    Normalisation: for each metric, the best algorithm gets 1.0 and the worst gets 0.0.
    Allows direct cross-metric comparison on a single axis.
    """
    metrics = [
        ("hypervolume", "Hypervolume", False),
        ("igd", "IGD", True),
        ("best_makespan", "Makespan", True),
        ("best_energy", "Energy", True),
        ("best_reliability", "Reliability", False),
    ]

    all_algos = [a for a in _ALGO_ORDER if a in per_run["algorithm"].unique()]

    # Compute normalised scores
    normed_scores: dict[str, dict[str, float]] = {a: {} for a in all_algos}
    metric_labels = []

    for metric, label, lower_better in metrics:
        metric_labels.append(label)
        means = per_run.groupby("algorithm")[metric].mean()
        lo, hi = means.min(), means.max()
        for algo in all_algos:
            val = means.get(algo, np.nan)
            if hi - lo < 1e-15:
                normed_scores[algo][label] = 1.0
            elif lower_better:
                normed_scores[algo][label] = 1.0 - (val - lo) / (hi - lo)
            else:
                normed_scores[algo][label] = (val - lo) / (hi - lo)

    n_metrics = len(metric_labels)
    n_algos = len(all_algos)
    x = np.arange(n_metrics)
    width = 0.8 / n_algos

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, algo in enumerate(all_algos):
        vals = [normed_scores[algo][m] for m in metric_labels]
        ax.bar(
            x + i * width - (n_algos - 1) * width / 2,
            vals, width,
            label=_label(algo), color=_ALGO_COLORS[algo],
            edgecolor="black" if algo == "enhanced_ga" else "none",
            linewidth=0.6 if algo == "enhanced_ga" else 0,
            alpha=0.88, zorder=5 if algo == "enhanced_ga" else 3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Normalised Score (1 = best)")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=len(all_algos),
              fontsize=7, handletextpad=0.3, columnspacing=1.0)
    ax.axhline(1.0, color="grey", linewidth=0.4, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, fig_root / "overall_metric_summary")


# ═══════════════════════════════════════════════════════════════════════════
#  20. Multi-metric rank heatmap (all metrics, all algorithms)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_multi_metric_rank_heatmap(per_run: pd.DataFrame, fig_root: Path) -> None:
    """Heatmap: rows = metrics, columns = algorithms, cell = mean rank.

    Summarises how each algorithm ranks across all metrics and instances.
    """
    metrics = [
        ("hypervolume", "Hypervolume", False),
        ("igd", "IGD", True),
        ("best_makespan", "Makespan", True),
        ("best_energy", "Energy", True),
        ("best_reliability", "Reliability", False),
    ]
    all_algos = [a for a in _ALGO_ORDER if a in per_run["algorithm"].unique()]

    rank_data: dict[str, dict[str, float]] = {}
    for metric, label, lower_better in metrics:
        per_run[f"_rank_{metric}"] = per_run.groupby(["instance", "seed"])[metric].rank(
            ascending=lower_better, method="average"
        )
        mean_ranks = per_run.groupby("algorithm")[f"_rank_{metric}"].mean()
        rank_data[label] = {_label(a): mean_ranks.get(a, np.nan) for a in all_algos}
        per_run.drop(columns=[f"_rank_{metric}"], inplace=True)

    df_ranks = pd.DataFrame(rank_data).T
    # Reorder columns by overall average rank
    avg_per_algo = df_ranks.mean().sort_values()
    df_ranks = df_ranks[avg_per_algo.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        df_ranks, annot=True, fmt=".2f", cmap="RdYlGn_r",
        vmin=1, vmax=len(all_algos),
        linewidths=0.6, linecolor="white",
        cbar_kws={"label": "Mean Rank", "shrink": 0.65},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, fig_root / "multi_metric_rank_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
#  21. Parallel coordinates — algorithm profiles
# ═══════════════════════════════════════════════════════════════════════════

def _plot_parallel_coordinates(per_run: pd.DataFrame, fig_root: Path) -> None:
    """Parallel coordinates plot showing each algorithm's profile across
    normalised metrics.  Each line is an algorithm's mean (averaged over
    all instances and seeds).
    """
    metrics = [
        ("hypervolume", "Hypervolume", False),
        ("igd", "IGD", True),
        ("best_makespan", "Makespan", True),
        ("best_energy", "Energy", True),
        ("best_reliability", "Reliability", False),
    ]
    ga_only = per_run[per_run["algorithm"].isin(_GA_ALGOS)].copy()

    # Normalise each metric to [0,1] where 1 = best
    normed: dict[str, dict[str, float]] = {a: {} for a in _GA_ALGOS}
    metric_labels = []

    for metric, label, lower_better in metrics:
        metric_labels.append(label)
        means = ga_only.groupby("algorithm")[metric].mean()
        lo, hi = means.min(), means.max()
        for algo in _GA_ALGOS:
            val = means.get(algo, np.nan)
            if hi - lo < 1e-15:
                normed[algo][label] = 1.0
            elif lower_better:
                normed[algo][label] = 1.0 - (val - lo) / (hi - lo)
            else:
                normed[algo][label] = (val - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metric_labels))

    for algo in _GA_ALGOS:
        values = [normed[algo][m] for m in metric_labels]
        ax.plot(x, values,
                color=_ALGO_COLORS[algo], label=_label(algo),
                marker=_ALGO_MARKERS[algo], markersize=8,
                linewidth=2.2 if algo == "enhanced_ga" else 1.2,
                zorder=10 if algo == "enhanced_ga" else 3,
                markeredgecolor="black" if algo == "enhanced_ga" else "none",
                markeredgewidth=0.6 if algo == "enhanced_ga" else 0)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Normalised Score (1 = best)")
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=len(_GA_ALGOS),
              fontsize=8, handletextpad=0.3, columnspacing=1.0)
    ax.axhline(1.0, color="grey", linewidth=0.4, linestyle="--", alpha=0.4)
    ax.axhline(0.0, color="grey", linewidth=0.4, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, fig_root / "parallel_coordinates")


# ═══════════════════════════════════════════════════════════════════════════
#  Captions
# ═══════════════════════════════════════════════════════════════════════════

def _write_captions(fig_root: Path) -> None:
    text = """# Figure Captions

1. `pareto_<instance>`: Pareto fronts per benchmark instance (makespan vs energy).

2. `pareto_grid`: Multi-panel overview of Pareto fronts across all benchmark instances.

3. `hv_boxplot_per_instance`: Hypervolume distributions (boxplots) for each instance.

4. `igd_boxplot_per_instance`: IGD distributions per instance (lower is better).

5. `boxplot_best_makespan / best_energy / best_reliability`:
   Per-instance faceted boxplots comparing GA variants on each objective.

6. `convergence_per_instance`: HV convergence curves for representative instances.
   Shaded bands show +/- 1 std across seeds.

7. `convergence_global`: Global average convergence across all instances and seeds.

8. `rank_heatmap`: Mean HV rank per instance (1 = best).

9. `rank_average_bar`: Average rank summary across all instances.

10. `significance_heatmap_<metric>`: Cliff's delta effect size of Enhanced GA vs each
    other algorithm per instance. Bold border = p_adj < 0.05.

11. `ablation_effect_sizes`: Per-instance contribution of each component (AOS, local search).

12. `topology`: Layered Edge-Fog-Cloud network topology.

13. `gantt_representative`: Gantt chart of the best Enhanced GA schedule.

14. `tradeoff_correlation_heatmap`: Spearman correlation among the three objectives.

15. `runtime_budget`: Runtime and evaluation counts comparison.

16. `performance_profile`: Cumulative distribution of normalised performance.

17. `radar_comparison`: Radar chart comparing algorithms across all five metrics.

18. `critical_difference`: Mean rank with 95% CI for HV, IGD, and Makespan.

19. `win_tie_loss`: Win/Tie/Loss of Enhanced GA vs each other algorithm on HV.

20. `overall_metric_summary`: Grouped bar chart with all algorithms on all metrics.

21. `multi_metric_rank_heatmap`: Mean rank per (metric x algorithm).

22. `parallel_coordinates`: Algorithm profiles across normalised metrics.
"""
    (fig_root / "figure_captions.md").write_text(text, encoding="utf-8")
