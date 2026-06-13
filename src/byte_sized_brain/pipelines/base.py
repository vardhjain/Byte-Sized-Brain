"""Pipeline contract + shared helpers (SavedModel export, TFLite benchmarking)."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..benchmark import TFLiteRunner, benchmark_inference, build_row
from ..utils import get_logger, size_mb

if TYPE_CHECKING:
    from ..config import Paths, PipelineConfig


@dataclass(frozen=True)
class Variant:
    """One artifact to benchmark."""

    name: str  # fp32 | int8 | dynamic_range
    quantization: str  # none | static_int8 | dynamic_range | dynamic_int8
    runtime: str  # tflite | onnxruntime
    path: Path


class Pipeline(ABC):
    """A model family's full lifecycle. Frameworks are imported lazily in methods."""

    name: str
    framework: str

    @abstractmethod
    def train(self, cfg: PipelineConfig, paths: Paths) -> float:
        """Train and persist the FP32 source model; return baseline accuracy."""

    @abstractmethod
    def variants(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        """The artifacts this pipeline produces (single source of truth)."""

    @abstractmethod
    def convert(self, cfg: PipelineConfig, paths: Paths) -> list[Variant]:
        """Write every variant's artifact file."""

    @abstractmethod
    def benchmark(self, cfg: PipelineConfig, paths: Paths) -> list[dict[str, Any]]:
        """Benchmark every variant; return result rows."""


def export_savedmodel(model, dest: Path) -> Path:
    """Export a Keras model to a TF SavedModel dir (Keras 3: ``model.export``)."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(dest))
    return dest


def tflite_benchmark(
    cfg: PipelineConfig,
    variants: Sequence[Variant],
    samples: Sequence[Any],
    labels: Sequence[int],
    decision_fn: Callable[[Any], int],
) -> list[dict[str, Any]]:
    """Benchmark a set of TFLite variants against shared eval data."""
    log = get_logger(cfg.name)
    rows: list[dict[str, Any]] = []
    for v in variants:
        if not Path(v.path).exists():
            raise FileNotFoundError(f"Missing artifact {v.path}. Run `bsb convert {cfg.name}` first.")
        runner = TFLiteRunner(v.path)
        metrics = benchmark_inference(
            runner.predict,
            samples,
            labels,
            num_samples=cfg.benchmark.num_samples,
            warmup=cfg.benchmark.warmup,
            decision_fn=decision_fn,
        )
        mb = size_mb(v.path)
        rows.append(
            build_row(
                cfg,
                variant=v.name,
                quantization=v.quantization,
                runtime=v.runtime,
                size_mb=mb,
                metrics=metrics,
            )
        )
        log.info(
            "%-6s acc=%.4f lat=%.2fms (p95 %.2f) size=%.2fMB",
            v.name,
            metrics["accuracy"],
            metrics["latency_ms_mean"],
            metrics["latency_ms_p95"],
            mb,
        )
    return rows
