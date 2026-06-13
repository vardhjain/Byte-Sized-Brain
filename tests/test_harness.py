"""The benchmark harness, exercised with fake predictors (no ML frameworks)."""

from __future__ import annotations

import pytest

from byte_sized_brain.benchmark import benchmark_inference


def test_perfect_predictor_scores_one() -> None:
    samples = list(range(20))
    labels = list(range(20))
    m = benchmark_inference(
        predict_fn=lambda s: s,  # echo the sample
        samples=samples,
        labels=labels,
        num_samples=20,
        warmup=2,
        decision_fn=lambda o: int(o),
    )
    assert m["accuracy"] == 1.0
    assert m["num_samples"] == 20
    assert m["latency_ms_mean"] >= 0.0
    assert m["latency_ms_p95"] >= m["latency_ms_p50"]
    assert "rss_delta_mb" in m and "peak_rss_mb" in m


def test_always_wrong_predictor_scores_zero() -> None:
    m = benchmark_inference(
        predict_fn=lambda s: s,
        samples=[0, 1, 2, 3],
        labels=[9, 9, 9, 9],
        num_samples=4,
        warmup=0,
        decision_fn=lambda o: int(o),
    )
    assert m["accuracy"] == 0.0


def test_num_samples_is_capped_to_available() -> None:
    m = benchmark_inference(
        predict_fn=lambda s: s,
        samples=[1, 1, 1],
        labels=[1, 1, 1],
        num_samples=999,  # more than we have
        warmup=1,
        decision_fn=lambda o: int(o),
    )
    assert m["num_samples"] == 3
    assert m["accuracy"] == 1.0


def test_empty_samples_raises() -> None:
    with pytest.raises(ValueError):
        benchmark_inference(
            predict_fn=lambda s: s, samples=[], labels=[],
            num_samples=5, warmup=0, decision_fn=lambda o: 0,
        )
