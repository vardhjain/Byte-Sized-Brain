"""Deterministic, network-free TFLite smoke test, the CI gate.

Exercises the real export -> TFLite (fp32 + static INT8 + dynamic-range) ->
TFLiteRunner -> harness path on a tiny in-memory model with random weights. No
dataset downloads and no training, so it is fast and reliable on any platform
(unlike the full real-data ``smoke`` suite, which trains models).
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
    from byte_sized_brain.data import representative_dataset
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
    dyn = tmp_path / "m_dynamic_range.tflite"
    rng = np.random.default_rng(0)
    calibration = rng.standard_normal((20, INPUT_DIM)).astype("float32")

    tflite.to_fp32(sm, fp32)
    tflite.to_static_int8(sm, int8, representative_dataset(calibration, 20), int8_io=True)
    tflite.to_dynamic_range(sm, dyn)

    # Quantization shrinks the model.
    assert int8.stat().st_size < fp32.stat().st_size
    assert dyn.stat().st_size < fp32.stat().st_size

    # Each variant has the tensor types its name promises.
    runners = {p: TFLiteRunner(p) for p in (fp32, int8, dyn)}
    assert runners[fp32].input_dtype == np.float32
    assert runners[int8].input_dtype == np.int8  # full-integer, INT8 in and out
    assert runners[int8].output_dtype == np.int8
    assert runners[dyn].input_dtype == np.float32  # dynamic-range keeps float I/O

    # Every variant loads and runs through the shared harness.
    samples = [rng.standard_normal(INPUT_DIM).astype("float32") for _ in range(12)]
    labels = [0] * len(samples)

    for runner in runners.values():
        metrics = benchmark_inference(
            runner.predict,
            samples,
            labels,
            num_samples=12,
            warmup=2,
            decision_fn=lambda o: int(np.argmax(o)),
        )
        assert metrics["num_samples"] == 12
        assert metrics["latency_ms_mean"] >= 0.0
        assert 0.0 <= metrics["accuracy"] <= 1.0

    # Quantized predictions should mostly agree with the float model.
    agree = sum(
        int(np.argmax(runners[fp32].predict(s))) == int(np.argmax(runners[int8].predict(s)))
        for s in samples
    )
    assert agree >= len(samples) * 0.75


def test_int8_input_saturates_instead_of_wrapping(tmp_path):
    """Out-of-range inputs must clamp to the INT8 limits, as the TFLite quantize op does."""
    pytest.importorskip("tensorflow")
    from tensorflow import keras

    from byte_sized_brain.benchmark import TFLiteRunner
    from byte_sized_brain.convert import tflite
    from byte_sized_brain.data import representative_dataset
    from byte_sized_brain.pipelines.base import export_savedmodel

    model = keras.Sequential(
        [keras.layers.Input(shape=(4,)), keras.layers.Dense(3, activation="softmax")]
    )
    sm = tmp_path / "sm"
    export_savedmodel(model, sm)
    calibration = np.random.default_rng(0).uniform(0, 1, (20, 4)).astype("float32")
    int8 = tflite.to_static_int8(sm, tmp_path / "m.tflite", representative_dataset(calibration))

    runner = TFLiteRunner(int8)
    in_range = runner.predict(np.full(4, 1.0, dtype=np.float32))
    far_out = runner.predict(np.full(4, 1e6, dtype=np.float32))
    # A wrapped value would land somewhere arbitrary; a saturated one equals the max input.
    assert np.array_equal(in_range, far_out)
