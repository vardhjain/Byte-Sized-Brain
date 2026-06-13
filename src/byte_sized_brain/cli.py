"""``bsb`` — one CLI for the whole project.

    bsb train     <pipeline|all> [--smoke]
    bsb convert   <pipeline|all> [--smoke]
    bsb benchmark <pipeline|all> [--smoke] [--num-samples N]
    bsb run       <pipeline|all> [--smoke]      # train + convert + benchmark
    bsb report                                  # aggregate CSVs -> docs/report.md
    bsb info                                    # system + library versions
    bsb list                                    # available pipelines
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .config import Paths, PipelineConfig, load_config
from .registry import all_names, get_pipeline
from .seeding import seed_everything
from .utils import get_logger

log = get_logger("cli")


def _targets(name: str) -> list[str]:
    return all_names() if name == "all" else [name]


def _load(pipeline: str, args: argparse.Namespace) -> PipelineConfig:
    cfg = load_config(args.config or pipeline, smoke=getattr(args, "smoke", False))
    if getattr(args, "num_samples", None):
        cfg.benchmark.num_samples = args.num_samples
    return cfg


def _do_train(cfg: PipelineConfig, paths: Paths) -> None:
    acc = get_pipeline(cfg.name).train(cfg, paths)
    log.info("[%s] trained - baseline FP32 accuracy: %.4f", cfg.name, acc)


def _do_convert(cfg: PipelineConfig, paths: Paths) -> None:
    variants = get_pipeline(cfg.name).convert(cfg, paths)
    for v in variants:
        log.info("[%s] wrote %s (%s)", cfg.name, v.path, v.quantization)


def _do_benchmark(cfg: PipelineConfig, paths: Paths) -> None:
    from .benchmark import write_results

    rows = get_pipeline(cfg.name).benchmark(cfg, paths)
    out = write_results(rows, paths.results_csv)
    log.info("[%s] wrote %d result rows -> %s", cfg.name, len(rows), out)


def _run_stages(pipeline: str, args: argparse.Namespace, stages: tuple[str, ...]) -> None:
    for name in _targets(pipeline):
        cfg = _load(name, args)
        seed_everything(cfg.seed)
        paths = Paths(cfg.name)
        log.info("=== %s (%s) | stages: %s%s ===", name, cfg.framework, ",".join(stages),
                 " | SMOKE" if getattr(args, "smoke", False) else "")
        if "train" in stages:
            _do_train(cfg, paths)
        if "convert" in stages:
            _do_convert(cfg, paths)
        if "benchmark" in stages:
            _do_benchmark(cfg, paths)


def cmd_info(_: argparse.Namespace) -> int:
    from .utils.sysinfo import collect_sysinfo, library_versions

    info: dict[str, Any] = {"bsb_version": __version__}
    info.update(collect_sysinfo())
    info["libraries"] = library_versions()
    print(json.dumps(info, indent=2))
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    for name in all_names():
        p = get_pipeline(name)
        print(f"{name:18} {p.framework}")
    return 0


def cmd_report(_: argparse.Namespace) -> int:
    from .report import generate_report

    out = generate_report()
    return 0 if out else 1


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import format_rows, run_demo

    try:
        rows = run_demo(args.pipeline, args.text or None, config=args.config)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    print(format_rows(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bsb", description="Byte-Sized Brain CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    stage_map = {
        "train": ("train",),
        "convert": ("convert",),
        "benchmark": ("benchmark",),
        "run": ("train", "convert", "benchmark"),
    }
    for name in stage_map:
        sp = sub.add_parser(name, help=f"{name} a pipeline")
        sp.add_argument("pipeline", choices=[*all_names(), "all"])
        sp.add_argument("--smoke", action="store_true", help="tiny, fast end-to-end run")
        sp.add_argument("--config", default=None, help="explicit config path")
        if name in {"benchmark", "run"}:
            sp.add_argument("--num-samples", type=int, default=None, help="override eval sample count")

    sub.add_parser("report", help="aggregate CSVs into docs/report.md + charts")
    sub.add_parser("info", help="print system + library versions")
    sub.add_parser("list", help="list available pipelines")

    demo = sub.add_parser("demo", help="FP32-vs-INT8 sentiment demo on a review")
    demo.add_argument("pipeline", choices=["distilbert_imdb", "rnn_imdb"])
    demo.add_argument(
        "text", nargs="*", help="review(s) to classify; quote each (default: built-in examples)"
    )
    demo.add_argument("--config", default=None)

    args = parser.parse_args(argv)

    if args.command == "info":
        return cmd_info(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "demo":
        return cmd_demo(args)

    try:
        _run_stages(args.pipeline, args, stage_map[args.command])
    except Exception as exc:  # surface a clean message, non-zero exit for CI
        log.error("%s failed: %s", args.command, exc)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
