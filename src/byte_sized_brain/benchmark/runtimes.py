"""Thin, uniform inference adapters over TFLite and ONNX Runtime.

Each runner exposes the same tiny surface — ``size_bytes()`` and
``predict(sample)`` returning the raw model output for a single example — so the
harness can time both runtimes with identical code.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np


def _load_tflite_interpreter(model_path: str):
    """tf.lite.Interpreter is deprecated in TF 2.16+; fall back to ai-edge-litert."""
    try:
        import tensorflow as tf

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*Interpreter is deprecated.*")
            return tf.lite.Interpreter(model_path=model_path)
    except Exception:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=model_path)


class TFLiteRunner:
    """Runs a ``.tflite`` model, transparently quantizing the input if needed."""

    def __init__(self, model_path: str | Path) -> None:
        self.path = str(model_path)
        self.interp = _load_tflite_interpreter(self.path)
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]
        self._out = self.interp.get_output_details()[0]
        self._in_dtype = self._in["dtype"]
        qp = self._in.get("quantization_parameters") or {}
        scales = qp.get("scales", [])
        self._scale = float(scales[0]) if len(scales) else None
        zps = qp.get("zero_points", [])
        self._zero_point = int(zps[0]) if len(zps) else 0

    def size_bytes(self) -> int:
        return os.path.getsize(self.path)

    def predict(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if np.issubdtype(self._in_dtype, np.integer) and self._scale:
            arr = np.round(arr / self._scale + self._zero_point).astype(self._in_dtype)
        else:
            arr = arr.astype(self._in_dtype)
        self.interp.set_tensor(self._in["index"], arr[None, ...])
        self.interp.invoke()
        return self.interp.get_tensor(self._out["index"])[0]


class OnnxRunner:
    """Runs an ``.onnx`` model on CPU. ``predict`` takes a dict of named inputs."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        self.path = str(model_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("BSB_ORT_THREADS", "0")) or 0
        self.sess = ort.InferenceSession(self.path, opts, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def size_bytes(self) -> int:
        return os.path.getsize(self.path)

    def predict(self, sample: dict[str, np.ndarray]) -> np.ndarray:
        feed = {n: sample[n] for n in self.input_names}
        outs = self.sess.run(self.output_names, feed)
        return outs[0][0]  # logits for the single example
