"""The one shared benchmark loop used by all four pipelines.

Given a ``predict_fn`` (prepared sample to raw output) and a ``decision_fn`` (raw
output to predicted label), it measures, identically across runtimes, the accuracy
over ``num_samples`` and the per-inference latency (mean, p50 and p95, in ms).

Only the model call is timed. An optional ``prepare_fn`` converts each raw sample
into the tensor the model takes (for example the float to INT8 input quantization
of a full-integer model) and runs *before* the clock starts, so Python-side
preprocessing never ends up in the latency figures.

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
    predictions: list[int] = []

    for i in range(n):
        sample = prepared(i)  # outside the timed block
        t0 = time.perf_counter()
        out = predict_fn(sample)
        latencies[i] = (time.perf_counter() - t0) * 1000.0
        predictions.append(decision_fn(out))

    correct = sum(p == int(labels[i]) for i, p in enumerate(predictions))
    return {
        "accuracy": correct / n,
        "predictions": predictions,
        "latency_ms_mean": float(latencies.mean()),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "num_samples": n,
    }
