"""Deterministic, network-free TFLite smoke — the CI gate.

Exercises the real export -> TFLite (fp32 + static INT8) -> TFLiteRunner -> harness
path on a tiny in-memory model with random weights. No dataset downloads and no
training, so it is fast and reliable on any platform (unlike the full real-data
``smoke`` suite, which trains models and is kept for local `make smoke`).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.tf

INPUT_DIM = 64
N_CLASSES = 10


def test_tflite_convert_and_run_roundtrip(tmp_path):
    pytest.importorskip("tensorflow")
    from tensorflow import keras

    from byte_sized_brain.benchmark import TFLiteRunner, benchmark_inference
    from byte_sized_brain.convert import tflite
    from byte_sized_brain.pipelines.base import export_savedmodel

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(INPUT_DIM,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(N_CLASSES, activation="softmax"),
        ]
    )
    sm = tmp_path / "sm"
    export_savedmodel(model, sm)

    fp32 = tmp_path / "m_fp32.tflite"
    int8 = tmp_path / "m_int8.tflite"
    rng = np.random.default_rng(0)

    tflite.to_fp32(sm, fp32)

    def rep():
        for _ in range(20):
            yield [rng.standard_normal((1, INPUT_DIM)).astype("float32")]

    tflite.to_static_int8(sm, int8, rep, int8_io=True)

    # Quantization shrinks the model.
    assert int8.stat().st_size < fp32.stat().st_size

    # Both variants load and run through the shared harness.
    samples = [rng.standard_normal(INPUT_DIM).astype("float32") for _ in range(12)]
    labels = [0] * len(samples)
    decision = lambda o: int(np.argmax(o))  # noqa: E731

    for path in (fp32, int8):
        runner = TFLiteRunner(path)
        metrics = benchmark_inference(
            runner.predict, samples, labels, num_samples=12, warmup=2, decision_fn=decision
        )
        assert metrics["num_samples"] == 12
        assert metrics["latency_ms_mean"] >= 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
