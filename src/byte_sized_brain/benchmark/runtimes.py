"""Thin, uniform inference adapters over TFLite and ONNX Runtime.

Each runner exposes the same three calls, so the harness can time both runtimes
with identical code.

* ``prepare(sample)`` turns one raw sample into the exact tensor(s) the model
  takes. For a full-integer TFLite model this is the float to INT8 input
  quantization.
* ``invoke(prepared)`` runs the model and returns the raw output for that sample.
  This is the only call the harness times.
* ``predict(sample)`` is ``invoke(prepare(sample))``, for callers that do not care
  about timing (the demo).

``prepare`` is separate on purpose. Quantizing an input in NumPy takes several
passes over the array, which costs more than the whole forward pass of a small
model. On a device the same step is a handful of integer instructions, or the
sensor already produces integers. Timing it would make INT8 models look slower
than they are.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def _load_tflite_interpreter(model_path: str, num_threads: int | None = None) -> Any:
    """Load with ``tf.lite.Interpreter``, or LiteRT (``ai-edge-litert``) without it.

    ``tf.lite.Interpreter`` is deprecated in favour of LiteRT but still ships with
    the pinned TensorFlow, and it is what the committed results were measured
    with. Only a missing TensorFlow (or a TensorFlow that has dropped the class)
    falls through to LiteRT, so a genuine load error, such as a corrupt model, is
    raised as itself instead of being masked by an unrelated ImportError.
    """
    try:
        import tensorflow as tf

        interpreter_cls = tf.lite.Interpreter
    except (ImportError, AttributeError):
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=model_path, num_threads=num_threads)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*Interpreter is deprecated.*")
        return interpreter_cls(model_path=model_path, num_threads=num_threads)


class TFLiteRunner:
    """Runs a ``.tflite`` model, quantizing the input first if the model needs it."""

    def __init__(self, model_path: str | Path) -> None:
        self.path = str(model_path)
        # The interpreter runs on one thread unless BSB_TFLITE_THREADS says otherwise.
        requested = int(os.environ.get("BSB_TFLITE_THREADS", "0"))
        self.threads = requested if requested > 0 else 1
        self.interp = _load_tflite_interpreter(self.path, requested if requested > 0 else None)
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]
        self._out = self.interp.get_output_details()[0]
        self._in_dtype = self._in["dtype"]
        qp = self._in.get("quantization_parameters") or {}
        scales = qp.get("scales", [])
        self._scale = float(scales[0]) if len(scales) else None
        zps = qp.get("zero_points", [])
        self._zero_point = int(zps[0]) if len(zps) else 0

    @property
    def input_dtype(self) -> type:
        """Tensor type the model expects (``np.int8`` for a full-integer model)."""
        return self._in_dtype

    @property
    def output_dtype(self) -> type:
        return self._out["dtype"]

    def prepare(self, x: np.ndarray) -> np.ndarray:
        """One float sample as the batch-1 input tensor (quantized if needed)."""
        arr = np.asarray(x, dtype=np.float32)
        if np.issubdtype(self._in_dtype, np.integer) and self._scale:
            # Saturate like TFLite's quantize op: clamp(round(r/s)+zp, qmin, qmax).
            # A bare .astype(int8) would *wrap* (e.g. 128 -> -128) on boundary values.
            info = np.iinfo(self._in_dtype)
            arr = np.clip(np.round(arr / self._scale + self._zero_point), info.min, info.max)
        return np.ascontiguousarray(arr[None, ...], dtype=self._in_dtype)

    def invoke(self, tensor: np.ndarray) -> np.ndarray:
        """Run one prepared input tensor; this is what the harness times."""
        self.interp.set_tensor(self._in["index"], tensor)
        self.interp.invoke()
        return self.interp.get_tensor(self._out["index"])[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.invoke(self.prepare(x))


class OnnxRunner:
    """Runs an ``.onnx`` model on CPU. Samples are dicts of named input arrays."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        self.path = str(model_path)
        opts = ort.SessionOptions()
        # 0 lets ONNX Runtime pick, which is one thread per physical core.
        requested = int(os.environ.get("BSB_ORT_THREADS", "0"))
        opts.intra_op_num_threads = requested
        self.threads = requested if requested > 0 else (psutil.cpu_count(logical=False) or 1)
        self.sess = ort.InferenceSession(self.path, opts, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def prepare(self, sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Keep only the inputs the graph declares, in a fresh feed dict."""
        return {n: sample[n] for n in self.input_names}

    def invoke(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        """Run one prepared feed; this is what the harness times."""
        return self.sess.run(self.output_names, feed)[0][0]  # logits for the single example

    def predict(self, sample: dict[str, np.ndarray]) -> np.ndarray:
        return self.invoke(self.prepare(sample))
