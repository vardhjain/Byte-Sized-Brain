"""The benchmark harness, exercised with fake predictors (no ML frameworks)."""

from __future__ import annotations

import time

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
            predict_fn=lambda s: s,
            samples=[],
            labels=[],
            num_samples=5,
            warmup=0,
            decision_fn=lambda o: 0,
        )


def test_prepare_fn_runs_outside_the_timed_region() -> None:
    """Slow input preparation must not show up in the latency figures."""

    def slow_prepare(sample: int) -> int:
        time.sleep(0.02)  # 20 ms, far more than the echo "model" takes
        return sample

    m = benchmark_inference(
        predict_fn=lambda s: s,
        samples=[0, 1, 2, 3],
        labels=[0, 1, 2, 3],
        num_samples=4,
        warmup=1,
        decision_fn=lambda o: int(o),
        prepare_fn=slow_prepare,
    )
    assert m["accuracy"] == 1.0
    assert m["latency_ms_mean"] < 5.0


def test_labels_shorter_than_samples_cap_the_run() -> None:
    m = benchmark_inference(
        predict_fn=lambda s: s,
        samples=[1, 1, 1, 1],
        labels=[1, 1],
        num_samples=4,
        warmup=0,
        decision_fn=lambda o: int(o),
    )
    assert m["num_samples"] == 2


def test_harness_exposes_raw_latencies_only_when_requested() -> None:
    kwargs = {
        "predict_fn": lambda sample: sample,
        "samples": [0, 1, 2],
        "labels": [0, 1, 2],
        "num_samples": 3,
        "warmup": 0,
        "decision_fn": int,
    }
    assert "latencies_ms" not in benchmark_inference(**kwargs)
    measured = benchmark_inference(**kwargs, include_latencies=True)
    assert len(measured["latencies_ms"]) == 3
    assert all(ms >= 0 for ms in measured["latencies_ms"])


def test_preparation_and_model_timing_are_from_the_same_calls(monkeypatch) -> None:
    """One prepared sample, one model call, three clock reads per example."""
    from byte_sized_brain.benchmark import harness

    # Example 1: preparation 2 ms, model 5 ms.
    # Example 2: preparation 5 ms, model 1 ms.
    ticks = iter([0.0, 0.002, 0.007, 1.0, 1.005, 1.006])
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(ticks))
    calls = []

    def prepare(sample: int) -> int:
        calls.append(("prepare", sample))
        return sample + 1

    def predict(sample: int) -> int:
        calls.append(("predict", sample))
        return sample

    m = benchmark_inference(
        predict,
        [0, 1],
        [1, 2],
        num_samples=2,
        warmup=0,
        decision_fn=int,
        prepare_fn=prepare,
        include_latencies=True,
        include_stage_latencies=True,
    )
    assert calls == [
        ("prepare", 0),
        ("predict", 1),
        ("prepare", 1),
        ("predict", 2),
    ]
    assert m["predictions"] == [1, 2]
    assert list(m["latencies_ms"]) == pytest.approx([5.0, 1.0])
    assert list(m["prep_latencies_ms"]) == pytest.approx([2.0, 5.0])
    assert list(m["e2e_latencies_ms"]) == pytest.approx([7.0, 6.0])
    assert m["latency_ms_mean"] == pytest.approx(3.0)
    for model, prep, total in zip(
        m["latencies_ms"], m["prep_latencies_ms"], m["e2e_latencies_ms"], strict=True
    ):
        assert total == pytest.approx(model + prep)


def test_stage_timers_are_optional_and_do_not_change_legacy_output() -> None:
    common = dict(
        predict_fn=lambda sample: sample,
        samples=[1, 2],
        labels=[1, 2],
        num_samples=2,
        warmup=0,
        decision_fn=int,
        prepare_fn=lambda sample: sample,
    )
    old = benchmark_inference(**common)
    assert "latencies_ms" not in old
    assert "prep_latencies_ms" not in old
    assert "e2e_latencies_ms" not in old
    timed = benchmark_inference(**common, include_stage_latencies=True)
    assert timed["accuracy"] == old["accuracy"]
    assert "latencies_ms" not in timed
    assert len(timed["prep_latencies_ms"]) == 2
    assert len(timed["e2e_latencies_ms"]) == 2
