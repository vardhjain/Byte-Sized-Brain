"""The one shared benchmark loop used by all four pipelines.

Given a ``predict_fn`` (sample → raw output) and a ``decision_fn`` (raw output →
predicted label), it measures, identically across runtimes:

* accuracy over ``num_samples``
* per-inference latency (mean / p50 / p95, in ms)
* process RSS delta and peak RSS increase over the pre-inference baseline

Memory is measured at the **process** level (``psutil`` RSS), not whole-machine
``virtual_memory()`` as the original scripts did, so the numbers are comparable
across pipelines and far less noisy. RSS is sampled *outside* the timed region so
it never inflates the latency figures.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import psutil


def benchmark_inference(
    predict_fn: Callable[[Any], Any],
    samples: Sequence[Any] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    num_samples: int,
    warmup: int = 5,
    decision_fn: Callable[[Any], int],
) -> dict[str, Any]:
    n = min(num_samples, len(samples))
    if n == 0:
        raise ValueError("No samples to benchmark")

    proc = psutil.Process()

    for i in range(warmup):
        predict_fn(samples[i % n])

    gc.collect()
    rss_before = proc.memory_info().rss
    peak = rss_before
    latencies = np.empty(n, dtype=np.float64)
    correct = 0

    for i in range(n):
        sample = samples[i]
        t0 = time.perf_counter()
        out = predict_fn(sample)
        latencies[i] = (time.perf_counter() - t0) * 1000.0
        if decision_fn(out) == int(labels[i]):
            correct += 1
        rss = proc.memory_info().rss  # outside the timed block
        if rss > peak:
            peak = rss

    rss_after = proc.memory_info().rss
    return {
        "accuracy": correct / n,
        "latency_ms_mean": float(latencies.mean()),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "rss_delta_mb": (rss_after - rss_before) / (1024**2),
        "peak_rss_mb": (peak - rss_before) / (1024**2),
        "num_samples": n,
    }
