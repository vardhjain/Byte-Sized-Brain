"""Report aggregation math (size reduction, latency speedup, accuracy delta)."""

from __future__ import annotations

import pandas as pd

from byte_sized_brain.report import summarize


def _df():
    base = {"modality": "vision", "runtime": "tflite", "arch": "x86_64", "emulated": False}
    return pd.DataFrame(
        [
            {"pipeline": "p1", "variant": "fp32", "size_mb": 4.0, "accuracy": 0.90,
             "latency_ms_mean": 10.0, **base},
            {"pipeline": "p1", "variant": "int8", "size_mb": 1.0, "accuracy": 0.88,
             "latency_ms_mean": 5.0, **base},
        ]
    )


def test_summarize_trade_offs() -> None:
    s = summarize(_df())
    int8 = s[s["variant"] == "int8"].iloc[0]
    assert int8["size_reduction_%"] == 75.0
    assert int8["latency_speedup_x"] == 2.0
    assert round(int8["accuracy_delta"], 2) == -0.02

    fp32 = s[s["variant"] == "fp32"].iloc[0]
    assert fp32["size_reduction_%"] == 0.0
    assert fp32["latency_speedup_x"] == 1.0


def test_summarize_empty_is_empty() -> None:
    assert summarize(pd.DataFrame()).empty
