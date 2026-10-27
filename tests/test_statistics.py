"""Paired correctness and repeated-pass latency statistics, with known inputs."""

from __future__ import annotations

import numpy as np
import pytest

from byte_sized_brain.benchmark.statistics import paired_accuracy, repeated_latency


def test_mcnemar_exact_pvalue_and_accuracy_direction() -> None:
    # FP32 correct on 4/4; INT8 correct on 1/4. Three lost, none gained.
    result = paired_accuracy([0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0], seed=7)
    assert result["fp32_only_correct"] == 3
    assert result["variant_only_correct"] == 0
    assert result["mcnemar_pvalue"] == pytest.approx(0.25)
    assert result["accuracy_delta_ci95_low"] <= -0.75
    assert result["accuracy_delta_ci95_high"] <= 0.0


def test_mcnemar_balanced_discordance_has_pvalue_one() -> None:
    result = paired_accuracy([0, 1], [1, 0], [0, 0])
    assert result["fp32_only_correct"] == 1
    assert result["variant_only_correct"] == 1
    assert result["mcnemar_pvalue"] == 1.0


def test_equal_correctness_keeps_zero_paired_delta() -> None:
    # Labels disagree in places but correctness never changes between variants.
    result = paired_accuracy([0, 1, 1], [0, 1, 1], [0, 0, 1])
    assert result["accuracy_delta_ci95_low"] == 0
    assert result["accuracy_delta_ci95_high"] == 0
    assert result["mcnemar_pvalue"] == 1


def test_paired_bootstrap_is_seeded() -> None:
    args = ([0, 0, 1, 1, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 0])
    assert paired_accuracy(*args, seed=99) == paired_accuracy(*args, seed=99)


@pytest.mark.parametrize(
    "base,variant,labels",
    [([], [], []), ([0], [0, 1], [0]), ([0], [0], [0, 1])],
)
def test_paired_accuracy_rejects_misaligned_samples(base, variant, labels) -> None:
    with pytest.raises(ValueError, match="equal nonzero lengths"):
        paired_accuracy(base, variant, labels)


def test_latency_repeats_pool_percentiles_and_bootstrap_pass_means() -> None:
    stats = repeated_latency([[1, 2], [3, 4]], seed=33)
    assert stats["latency_passes"] == 2
    assert stats["latency_ms_mean"] == pytest.approx(2.5)
    assert stats["latency_ms_p50"] == pytest.approx(2.5)
    assert stats["latency_ms_p95"] == pytest.approx(3.85)
    assert stats["latency_pass_std_ms"] == pytest.approx(2**0.5)
    assert 1.5 <= stats["latency_ms_ci95_low"] <= stats["latency_ms_ci95_high"] <= 3.5


def test_single_latency_pass_does_not_invent_a_confidence_interval() -> None:
    stats = repeated_latency([np.array([1.0, 2.0, 3.0])])
    assert stats["latency_passes"] == 1
    assert stats["latency_ms_mean"] == 2.0
    assert stats["latency_ms_ci95_low"] == ""
    assert stats["latency_ms_ci95_high"] == ""
    assert stats["latency_pass_std_ms"] == ""


@pytest.mark.parametrize("data", [[], [[]], [[1.0], []], [[1.0], [1.0, 2.0]]])
def test_latency_repeats_validate_pass_lengths(data) -> None:
    with pytest.raises(ValueError):
        repeated_latency(data)
