"""The one shared benchmark loop used by all four pipelines.

Given a ``predict_fn`` (prepared sample to raw output) and a ``decision_fn`` (raw
output to predicted label), it measures, identically across runtimes, the accuracy
over ``num_samples`` and the per-inference latency (mean, p50 and p95, in ms).

The historical ``latency_ms_*`` series times only ``predict_fn``, leaving
``prepare_fn`` outside that timer. With ``include_stage_latencies=True``,
the same inference also captures ``prepare + predict`` and the preparation
time separately. All series use the same calls and samples. This "end-to-end"
boundary starts with a *loaded and dataset-preprocessed* example: it excludes
dataset loading, image resizing, raw-text tokenization, decision decoding,
and model initialization.

Memory is not measured here. A process that has already loaded the model shows no
RSS change while it runs inferences, so the loop has nothing meaningful to report.
See :mod:`byte_sized_brain.benchmark.memory`, which measures the footprint of each
variant in a fresh process.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def benchmark_inference(
    predict_fn: Callable[[Any], Any],
    samples: Sequence[Any] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    num_samples: int,
    warmup: int = 5,
    decision_fn: Callable[[Any], int],
    prepare_fn: Callable[[Any], Any] | None = None,
    include_latencies: bool = False,
    include_stage_latencies: bool = False,
) -> dict[str, Any]:
    n = min(num_samples, len(samples), len(labels))
    if n <= 0:
        raise ValueError("No samples to benchmark")

    def prepared(i: int) -> Any:
        sample = samples[i]
        return prepare_fn(sample) if prepare_fn is not None else sample

    for i in range(warmup):
        predict_fn(prepared(i % n))

    gc.collect()
    latencies = np.empty(n, dtype=np.float64)
    prep_latencies = np.empty(n, dtype=np.float64) if include_stage_latencies else None
    e2e_latencies = np.empty(n, dtype=np.float64) if include_stage_latencies else None
    predictions: list[int] = []

    for i in range(n):
        # This outer timer includes runner.prepare + runner.invoke, but *not*
        # the already-loaded dataset or post-inference decision decoding.
        start = time.perf_counter() if include_stage_latencies else None
        sample = prepared(i)
        model_start = time.perf_counter()
        out = predict_fn(sample)
        model_end = time.perf_counter()
        latencies[i] = (model_end - model_start) * 1000.0
        if start is not None and prep_latencies is not None and e2e_latencies is not None:
            prep_latencies[i] = (model_start - start) * 1000.0
            e2e_latencies[i] = (model_end - start) * 1000.0
        predictions.append(decision_fn(out))

    correct = sum(p == int(labels[i]) for i, p in enumerate(predictions))
    result: dict[str, Any] = {
        "accuracy": correct / n,
        "predictions": predictions,
        "latency_ms_mean": float(latencies.mean()),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "num_samples": n,
    }
    if include_latencies:
        result["latencies_ms"] = latencies
    if prep_latencies is not None and e2e_latencies is not None:
        # Exact per-call arrays are retained for the shared repeated-pass
        # estimator; the public CSV stores only aggregates.
        result["prep_latencies_ms"] = prep_latencies
        result["e2e_latencies_ms"] = e2e_latencies
    return result
