"""The per-variant memory probe, exercised on a tiny ONNX graph (no TF, no PyTorch)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.onnx


def test_measure_memory_reports_the_cost_of_loading_a_model(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    from byte_sized_brain.benchmark.memory import measure_memory

    # One 2048 x 2048 float32 weight is 16 MB, so the footprint is easy to see.
    weight = np.random.default_rng(0).standard_normal((2048, 2048)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "one_matmul",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2048])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2048])],
        initializer=[numpy_helper.from_array(weight, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))

    mem = measure_memory("onnxruntime", path, {"x": np.ones((1, 2048), dtype=np.float32)})
    assert set(mem) == {"rss_delta_mb", "peak_rss_mb"}
    assert mem["rss_delta_mb"] > 8.0  # at least half of the 16 MB weight is resident
    assert mem["peak_rss_mb"] >= mem["rss_delta_mb"] - 0.5
