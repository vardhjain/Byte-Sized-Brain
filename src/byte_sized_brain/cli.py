"""``bsb``, one CLI for the whole project.

Usage::

    bsb train     <pipeline|all> [--smoke]
    bsb convert   <pipeline|all> [--smoke]
    bsb benchmark <pipeline|all> [--smoke] [--num-samples N] [--repeats N]
    bsb run       <pipeline|all> [--smoke]      # train + convert + benchmark
    bsb convert   cnn_cifar10 --cnn-alternatives   # add 2 opt-in CNN variants
    bsb benchmark cnn_cifar10 --cnn-alternatives   # compare all 4
    bsb report                                  # aggregate CSVs -> docs/report.md
    bsb recommend ffn_mnist --max-size-mb 0.2     # measured, constraint-aware model choice
    bsb demo      <distilbert_imdb|rnn_imdb> [TEXT ...] [--smoke]
    bsb info                                    # system + library versions
    bsb list                                    # available pipelines

``--smoke`` runs keep their artifacts under ``artifacts/smoke/`` and their results
under ``benchmarks/results/smoke/``, so they never replace a full run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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
    if cfg.name != pipeline:
        raise ValueError(
            f"{args.config} configures the {cfg.name!r} pipeline, but the command "
            f"targets {pipeline!r}. Pass the matching pipeline name."
        )
    if getattr(args, "num_samples", None) is not None:
        cfg.benchmark.num_samples = args.num_samples  # validated, must be >= 1
    if getattr(args, "repeats", None) is not None:
        cfg.benchmark.repeats = args.repeats  # validated, 1..50
    if getattr(args, "cnn_alternatives", False):
        if cfg.name != "cnn_cifar10":
            raise ValueError("--cnn-alternatives applies only to cnn_cifar10")
        cfg.convert.cnn_alternatives = ["dynamic_range", "int8_except_depthwise"]
    expected = get_pipeline(cfg.name).quantization
    if cfg.convert.quantization != expected:
        raise ValueError(
            f"{cfg.name} config asks for convert.quantization={cfg.convert.quantization!r}, "
            f"but the {cfg.name} pipeline implements {expected!r}."
        )
    return cfg


def _do_train(cfg: PipelineConfig, paths: Paths) -> None:
    acc = get_pipeline(cfg.name).train(cfg, paths)
    log.info("[%s] trained, baseline FP32 accuracy: %.4f", cfg.name, acc)


def _rerun_hint(stage: str, cfg: PipelineConfig, paths: Paths) -> str:
    return f"Run `bsb {stage} {cfg.name}{' --smoke' if paths.smoke else ''}` first."


def _do_convert(cfg: PipelineConfig, paths: Paths) -> None:
    if not paths.fp32_source.exists():
        raise FileNotFoundError(
            f"No trained {cfg.name} model at {paths.fp32_source}. "
            + _rerun_hint("train", cfg, paths)
        )
    variants = get_pipeline(cfg.name).convert(cfg, paths)
    for v in variants:
        log.info("[%s] wrote %s (%s)", cfg.name, v.path, v.quantization)


def _do_benchmark(cfg: PipelineConfig, paths: Paths) -> None:
    from .benchmark import write_results

    pipeline = get_pipeline(cfg.name)
    # Check before the datasets are loaded, which can take a while.
    missing = [v.path for v in pipeline.variants(cfg, paths) if not v.path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact {missing[0]}. " + _rerun_hint("convert", cfg, paths)
        )
    rows = pipeline.benchmark(cfg, paths)
    out = write_results(rows, paths.results_csv)
    log.info("[%s] wrote %d result rows -> %s", cfg.name, len(rows), out)


def _run_stages(pipeline: str, args: argparse.Namespace, stages: tuple[str, ...]) -> None:
    smoke = getattr(args, "smoke", False)
    for name in _targets(pipeline):
        cfg = _load(name, args)
        seed_everything(cfg.seed, frameworks=(cfg.framework,))
        paths = Paths(cfg.name, smoke=smoke)
        log.info(
            "=== %s (%s) | stages: %s%s ===",
            name,
            cfg.framework,
            ",".join(stages),
            " | SMOKE" if smoke else "",
        )
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
        print(f"{name:18} {p.framework:12} {p.quantization}")
    return 0


def cmd_report(_: argparse.Namespace) -> int:
    from .report import generate_report

    out = generate_report()
    return 0 if out else 1


def cmd_recommend(args: argparse.Namespace) -> int:
    from .recommend import Constraints, recommend_from_results

    constraints = Constraints(
        max_accuracy_drop_pts=args.max_accuracy_drop_pts,
        min_accuracy=args.min_accuracy,
        max_size_mb=args.max_size_mb,
        max_latency_ms=args.max_latency_ms,
        max_p95_ms=args.max_p95_ms,
        max_memory_mb=args.max_memory_mb,
        latency_kind=args.latency_kind,
        objective=args.objective,
    )
    try:
        outcome = recommend_from_results(
            args.pipeline,
            results_dir=args.results_dir,
            arch=args.arch,
            device=args.device,
            run_id=args.run_id,
            constraints=constraints,
        )
    except ValueError as exc:
        log.error("recommend failed: %s", exc)
        return 1
    if args.json:
        print(json.dumps(outcome, indent=2, sort_keys=True))
    else:
        print(
            f"Measured experiment: {outcome['pipeline']} on {outcome['arch']} "
            f"({outcome['device']}) | run_id={outcome['run_id']}"
        )
        print(
            f"Objective: minimize/maximize {args.objective} | latency: {args.latency_kind}"
        )
        for choice in outcome["candidates"]:
            label = "FEASIBLE" if choice["feasible"] else "REJECTED"
            reason = "; ".join(choice["reasons"])
            print(
                f"  {choice['variant']:<24} {label:<8} "
                f"acc={choice['accuracy']} size={choice['size_mb']} MiB "
                f"mean={choice['latency_mean_ms']}ms "
                f"p95={choice['latency_p95_ms']}ms "
                f"RSS={choice['memory_mb']}MiB"
                + (f" | {reason}" if reason else "")
            )
        if outcome["recommended_variant"]:
            print(f"Recommendation: {outcome['recommended_variant']}")
            print("Feasible Pareto variants: " + ", ".join(outcome["pareto_variants"]))
        else:
            print("No measured variant satisfies all constraints. No recommendation made.")
        print(outcome["note"])
    return 0 if outcome["recommended_variant"] else 2


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import format_rows, run_demo

    try:
        rows = run_demo(args.pipeline, args.text or None, config=args.config, smoke=args.smoke)
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
        sp.add_argument(
            "--smoke",
            action="store_true",
            help="tiny, fast end-to-end run (kept under artifacts/smoke and results/smoke)",
        )
        sp.add_argument("--config", default=None, help="explicit config path (one pipeline only)")
        if name in {"convert", "benchmark", "run"}:
            sp.add_argument(
                "--cnn-alternatives",
                action="store_true",
                help="CNN only: include dynamic-range and depthwise-float variants (experimental)",
            )
        if name in {"benchmark", "run"}:
            sp.add_argument(
                "--num-samples", type=int, default=None, help="override eval sample count"
            )
            sp.add_argument(
                "--repeats",
                type=int,
                default=None,
                help="timing passes per variant (1-50; default: 1)",
            )

    sub.add_parser("report", help="aggregate CSVs into docs/report.md + charts")
    recommend = sub.add_parser(
        "recommend", help="choose a benchmarked variant under deployment constraints"
    )
    recommend.add_argument("pipeline", choices=all_names())
    recommend.add_argument("--results-dir", type=Path, default=None)
    recommend.add_argument("--arch", default="x86_64")
    recommend.add_argument("--device", default=None, help="exact CPU label, when needed")
    recommend.add_argument("--run-id", default=None, help="use an explicit paired experiment")
    recommend.add_argument(
        "--max-accuracy-drop-pts", type=float, default=1.0, help="default: 1 percentage point"
    )
    recommend.add_argument("--min-accuracy", type=float, default=None, help="fraction, 0..1")
    recommend.add_argument("--max-size-mb", type=float, default=None)
    recommend.add_argument("--max-latency-ms", type=float, default=None)
    recommend.add_argument("--max-p95-ms", type=float, default=None)
    recommend.add_argument("--max-memory-mb", type=float, default=None)
    recommend.add_argument(
        "--latency-kind", choices=["model", "prepared-input"], default="model"
    )
    recommend.add_argument(
        "--objective", choices=["latency", "size", "memory", "accuracy"], default="latency"
    )
    recommend.add_argument("--json", action="store_true", help="machine-readable audit output")
    sub.add_parser("info", help="print system + library versions")
    sub.add_parser("list", help="list available pipelines")

    demo = sub.add_parser("demo", help="FP32-vs-INT8 sentiment demo on a review")
    demo.add_argument("pipeline", choices=["distilbert_imdb", "rnn_imdb"])
    demo.add_argument(
        "text", nargs="*", help="review(s) to classify; quote each (default: built-in examples)"
    )
    demo.add_argument("--smoke", action="store_true", help="use the artifacts of a --smoke run")
    demo.add_argument("--config", default=None)

    args = parser.parse_args(argv)

    if args.command == "info":
        return cmd_info(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "recommend":
        return cmd_recommend(args)
    if args.command == "demo":
        return cmd_demo(args)

    if args.config and args.pipeline == "all":
        parser.error("--config applies to a single pipeline, not 'all'")

    try:
        _run_stages(args.pipeline, args, stage_map[args.command])
    except (FileNotFoundError, ValueError) as exc:
        # Usage problems (a missing artifact, a bad config or option) get a one-line
        # message. Anything else is a real bug and keeps its traceback.
        log.error("%s failed: %s", args.command, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
