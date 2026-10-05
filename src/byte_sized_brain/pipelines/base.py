"""Pipeline contract + shared helpers (SavedModel export, variant benchmarking)."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..benchmark import OnnxRunner, TFLiteRunner, benchmark_inference, build_row
from ..benchmark.memory import measure_memory
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
    quantization: str  # the technique the quantized variant uses (see ConvertConfig)

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


def export_savedmodel(model, dest: Path, *, input_signature=None) -> Path:
    """Export a Keras model to a TF SavedModel dir (Keras 3: ``model.export``).

    Pass ``input_signature`` to pin a static input shape. The LSTM pipeline needs
    a static batch dimension so the converter can lower its ``TensorListReserve``
    ops to builtins (a ``WHILE`` loop of ordinary ops). With a dynamic batch the
    only route is TF-Select (Flex) ops, which the stock interpreter can't execute.
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if input_signature is None:
        model.export(str(dest))
    else:
        import keras

        archive = keras.export.ExportArchive()
        archive.track(model)
        archive.add_endpoint(name="serve", fn=model.call, input_signature=input_signature)
        archive.write_out(str(dest))
    return dest


def benchmark_variants(
    cfg: PipelineConfig,
    paths: Paths,
    variants: Sequence[Variant],
    samples: Sequence[Any] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    decision_fn: Callable[[Any], int],
) -> list[dict[str, Any]]:
    """Benchmark every variant against the same eval data, one result row each."""
    log = get_logger(cfg.name)
    runners: dict[str, Callable[[Path], Any]] = {
        "tflite": TFLiteRunner,
        "onnxruntime": OnnxRunner,
    }
    rows: list[dict[str, Any]] = []
    baseline: list[int] | None = None  # predictions of the first (FP32) variant
    for v in variants:
        if not Path(v.path).exists():
            smoke = " --smoke" if paths.smoke else ""
            raise FileNotFoundError(
                f"Missing artifact {v.path}. Run `bsb convert {cfg.name}{smoke}` first."
            )
        runner = runners[v.runtime](v.path)
        # Time only the model call. Input preparation (the float to INT8 quantization
        # of a full-integer model) happens before the clock starts.
        metrics = benchmark_inference(
            runner.invoke,
            samples,
            labels,
            num_samples=cfg.benchmark.num_samples,
            warmup=cfg.benchmark.warmup,
            decision_fn=decision_fn,
            prepare_fn=runner.prepare,
        )
        metrics["threads"] = runner.threads
        # Share of samples where this variant predicts what the baseline predicts.
        predictions = metrics.pop("predictions")
        baseline = predictions if baseline is None else baseline
        metrics["agreement"] = sum(
            a == b for a, b in zip(predictions, baseline, strict=True)
        ) / len(predictions)
        del runner
        # Footprint of this variant alone, measured in a fresh process.
        metrics.update(measure_memory(v.runtime, v.path, samples[0]))
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
            "%-13s acc=%.4f lat=%.2fms (p95 %.2f) size=%.2fMB",
            v.name,
            metrics["accuracy"],
            metrics["latency_ms_mean"],
            metrics["latency_ms_p95"],
            mb,
        )
    return rows
