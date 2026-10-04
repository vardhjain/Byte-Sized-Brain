"""Network-free ONNX smoke test that quantizes a tiny graph and runs it through the harness.

The DistilBERT pipeline is too heavy for CI, so without this test nothing in CI ever
touches ``convert.onnx.quantize_int8`` or ``OnnxRunner``. This builds a two-layer
classifier straight from ``onnx.helper`` (no PyTorch, no downloads), quantizes it the
same way the real pipeline does, and checks both graphs through the shared harness.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.onnx

IN_DIM = 256
HIDDEN = 128
N_CLASSES = 2


def _tiny_classifier(path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((IN_DIM, HIDDEN)).astype(np.float32)
    w2 = rng.standard_normal((HIDDEN, N_CLASSES)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w1"], ["h"]),
            helper.make_node("Relu", ["h"], ["h_relu"]),
            helper.make_node("MatMul", ["h_relu", "w2"], ["logits"]),
        ],
        "tiny_classifier",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", IN_DIM])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", N_CLASSES])],
        initializer=[numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(w2, "w2")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_onnx_quantize_and_run_roundtrip(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    from byte_sized_brain.benchmark import OnnxRunner, benchmark_inference
    from byte_sized_brain.convert.onnx import quantize_int8

    fp32 = tmp_path / "m_fp32.onnx"
    int8 = tmp_path / "m_int8.onnx"
    _tiny_classifier(fp32)
    quantize_int8(fp32, int8)

    # Dynamic INT8 stores the weights as 8-bit integers, so the file must shrink.
    assert int8.stat().st_size < fp32.stat().st_size

    rng = np.random.default_rng(1)
    samples = [{"x": rng.standard_normal((1, IN_DIM)).astype(np.float32)} for _ in range(16)]
    fp32_runner, int8_runner = OnnxRunner(fp32), OnnxRunner(int8)
    assert fp32_runner.input_names == ["x"]

    labels = [int(np.argmax(fp32_runner.predict(s))) for s in samples]
    for runner in (fp32_runner, int8_runner):
        metrics = benchmark_inference(
            runner.predict,
            samples,
            labels,
            num_samples=len(samples),
            warmup=2,
            decision_fn=lambda o: int(np.argmax(o)),
        )
        assert metrics["num_samples"] == len(samples)
        # The quantized graph should mostly agree with the float graph it came from.
        assert metrics["accuracy"] >= 0.75
