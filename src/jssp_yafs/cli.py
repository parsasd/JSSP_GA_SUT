from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from jssp_yafs.config import ExperimentConfig, load_config
from jssp_yafs.data.loader import verify_checksums
from jssp_yafs.data.prepare import build_checksum_manifest, prepare_subsets
from jssp_yafs.experiments.runner import RunnerOutput, run_experiments
from jssp_yafs.utils.logging_utils import setup_logging
from jssp_yafs.visualization.plots import plot_all

logger = logging.getLogger(__name__)



def _prepare_data(cfg: ExperimentConfig) -> None:
    prepare_subsets(
        source_file=cfg.dataset.source_file,
        quick_instances=cfg.dataset.quick_instances,
        full_instances=cfg.dataset.full_instances,
        out_root=Path("data/processed"),
    )
    build_checksum_manifest("data/processed", "data/manifests/processed_checksums.csv")



def _run_and_plot(cfg: ExperimentConfig, mode: str, include_ablations: bool, progress: bool) -> RunnerOutput:
    output = run_experiments(
        cfg=cfg,
        instance_mode=mode,
        include_ablations=include_ablations,
        show_progress=progress,
    )
    fig_root = plot_all(cfg, output.run_root, mode=mode)
    logger.info("Figures written to %s", fig_root)
    return output



def _cmd_prepare_data(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.run.output_dir / "pipeline_prepare.log")
    _prepare_data(cfg)
    logger.info("Prepared data subsets and checksums")
    return 0



def _cmd_verify_data(args: argparse.Namespace) -> int:
    df = verify_checksums(args.manifest)
    ok = bool(df["ok"].all())
    print(df)
    return 0 if ok else 1



def _cmd_smoke(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.run.output_dir / "pipeline_smoke.log")
    _prepare_data(cfg)
    output = _run_and_plot(cfg, mode="smoke", include_ablations=False, progress=args.progress)
    summary = pd.read_csv(output.aggregate_csv)
    logger.info("Smoke run complete. Aggregated rows=%d", len(summary))
    return 0



def _cmd_full(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.max_seeds is not None and args.max_seeds > 0:
        cfg.run.random_seeds = cfg.run.random_seeds[: args.max_seeds]
    setup_logging(cfg.run.output_dir / "pipeline_full.log")
    _prepare_data(cfg)
    output = _run_and_plot(
        cfg,
        mode="full",
        include_ablations=not args.no_ablations,
        progress=args.progress,
    )
    summary = pd.read_csv(output.aggregate_csv)
    logger.info("Full run complete. Aggregated rows=%d", len(summary))
    return 0



def _cmd_plots(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.run.output_dir / "pipeline_plots.log")
    run_root = cfg.run.output_dir / "runs" / args.mode
    fig_root = plot_all(cfg, run_root=run_root, mode=args.mode)
    logger.info("Plots regenerated at %s", fig_root)
    return 0



def _cmd_all(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.max_seeds is not None and args.max_seeds > 0:
        cfg.run.random_seeds = cfg.run.random_seeds[: args.max_seeds]
    setup_logging(cfg.run.output_dir / "pipeline_all.log")
    _prepare_data(cfg)
    _run_and_plot(
        cfg,
        mode="full",
        include_ablations=not args.no_ablations,
        progress=args.progress,
    )
    return 0



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jssp-yafs", description="JSSP + YAFS research pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-data", help="Prepare benchmark subsets and checksums")
    p_prepare.add_argument("--config", default="configs/smoke.yaml")
    p_prepare.set_defaults(func=_cmd_prepare_data)

    p_verify = sub.add_parser("verify-data", help="Verify dataset checksums")
    p_verify.add_argument("--manifest", default="data/manifests/processed_checksums.csv")
    p_verify.set_defaults(func=_cmd_verify_data)

    p_smoke = sub.add_parser("smoke", help="Run smoke experiment pipeline")
    p_smoke.add_argument("--config", default="configs/smoke.yaml")
    p_smoke.add_argument("--progress", action="store_true")
    p_smoke.set_defaults(func=_cmd_smoke)

    p_full = sub.add_parser("full", help="Run full experiment pipeline")
    p_full.add_argument("--config", default="configs/full.yaml")
    p_full.add_argument("--progress", action="store_true")
    p_full.add_argument("--no-ablations", action="store_true")
    p_full.add_argument("--max-seeds", type=int, default=None)
    p_full.set_defaults(func=_cmd_full)

    p_plots = sub.add_parser("plots", help="Regenerate plots from existing run outputs")
    p_plots.add_argument("--config", default="configs/full.yaml")
    p_plots.add_argument("--mode", default="full")
    p_plots.set_defaults(func=_cmd_plots)

    p_all = sub.add_parser("all", help="Run full pipeline end-to-end")
    p_all.add_argument("--config", default="configs/full.yaml")
    p_all.add_argument("--progress", action="store_true")
    p_all.add_argument("--no-ablations", action="store_true")
    p_all.add_argument("--max-seeds", type=int, default=None)
    p_all.set_defaults(func=_cmd_all)

    return p



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
