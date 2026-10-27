"""Paired accuracy and repeated-pass timing statistics for benchmark runs.

We compare predictions on the exact same evaluation examples. McNemar's exact
two-sided test uses only samples whose correctness changed. A paired bootstrap
resamples *examples*, preserving each example's FP32/quantized correlation.

For latency, the independent unit is a complete timing pass (not each model
invocation): resampling passes avoids pretending adjacent calls are independent
measurements of machine-to-machine variability.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np

_BOOTSTRAP_SAMPLES = 2000


def paired_accuracy(
    fp32_predictions: Sequence[int],
    variant_predictions: Sequence[int],
    labels: Sequence[int] | np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired 95% bootstrap CI for accuracy delta and exact McNemar p-value.

    The accuracy delta is variant minus FP32. A negative value indicates worse
    accuracy under quantization. The p-value is descriptive, not proof of
    generalization to hardware, datasets or unrelated training runs.
    """
    base = np.asarray(fp32_predictions)
    quant = np.asarray(variant_predictions)
    y = np.asarray(labels)
    if base.ndim != 1 or quant.ndim != 1 or y.ndim != 1:
        raise ValueError("Expected one-dimensional predictions and labels")
    if len(base) != len(quant) or len(base) != len(y) or not len(y):
        raise ValueError("Paired predictions and labels must have equal nonzero lengths")

    base_ok = base == y
    quant_ok = quant == y
    lost = int(np.count_nonzero(base_ok & ~quant_ok))
    gained = int(np.count_nonzero(~base_ok & quant_ok))
    discordant = lost + gained
    pvalue = (
        min(1.0, 2.0 * sum(comb(discordant, k) for k in range(min(lost, gained) + 1))
            / (1 << discordant))
        if discordant
        else 1.0
    )
    differences = quant_ok.astype(np.int8) - base_ok.astype(np.int8)
    if not np.any(differences):
        lower = upper = 0.0
    else:
        rng = np.random.default_rng(seed)
        samples = np.empty(_BOOTSTRAP_SAMPLES, dtype=np.float64)
        for i in range(_BOOTSTRAP_SAMPLES):
            samples[i] = float(differences[rng.integers(len(y), size=len(y))].mean())
        lower, upper = (float(x) for x in np.percentile(samples, [2.5, 97.5]))
    return {
        "accuracy_delta_ci95_low": lower,
        "accuracy_delta_ci95_high": upper,
        "mcnemar_pvalue": float(pvalue),
        "fp32_only_correct": lost,
        "variant_only_correct": gained,
    }


def repeated_latency(
    passes: Sequence[Sequence[float] | np.ndarray],
    *,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Pool inference latencies, and bootstrap pass means when repeats >= 2.

    Per-inference mean/p50/p95 reflect all measured calls. CI/SD on independent
    pass means are only reported for >=2 passes. They describe local run
    variability, not confidence in performance on other hardware.
    """
    if not passes:
        raise ValueError("At least one latency pass is required")
    arrays = [np.asarray(p, dtype=np.float64) for p in passes]
    if any(a.ndim != 1 or a.size == 0 for a in arrays):
        raise ValueError("Every latency pass must contain at least one observation")
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Latency passes must use the same evaluation sample count")
    all_latencies = np.concatenate(arrays)
    stats: dict[str, float | int | str] = {
        "latency_ms_mean": float(all_latencies.mean()),
        "latency_ms_p50": float(np.percentile(all_latencies, 50)),
        "latency_ms_p95": float(np.percentile(all_latencies, 95)),
        "latency_passes": len(arrays),
        "latency_pass_std_ms": "",
        "latency_ms_ci95_low": "",
        "latency_ms_ci95_high": "",
    }
    if len(arrays) > 1:
        pass_means = np.array([float(a.mean()) for a in arrays])
        rng = np.random.default_rng(seed)
        draws = rng.choice(pass_means, size=(2000, len(pass_means)), replace=True)
        low, high = np.percentile(draws.mean(axis=1), [2.5, 97.5])
        stats.update(
            latency_pass_std_ms=float(pass_means.std(ddof=1)),
            latency_ms_ci95_low=float(low),
            latency_ms_ci95_high=float(high),
        )
    return stats
