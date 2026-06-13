"""TensorFlow SavedModel → TFLite conversions.

Three honest, clearly-named paths:

* :func:`to_fp32` — plain FP32 TFLite (baseline).
* :func:`to_static_int8` — full-integer PTQ using a **real** representative
  dataset (weights *and* activations quantized). Best size/latency win, needs
  ops that support integer inference.
* :func:`to_dynamic_range` — dynamic-range PTQ (weights INT8, activations FP32
  at runtime). The right choice for the LSTM, whose ops need the TF-Select
  (Flex) runtime and don't support full-integer quantization.

``flex=True`` enables ``SELECT_TF_OPS`` so TF-only ops (LSTM / TensorList) work.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path


def _new_converter(saved_model_dir: str | Path, flex: bool):
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if flex:
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        conv._experimental_lower_tensor_list_ops = False
    return conv


def _write(model_bytes: bytes, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(model_bytes)
    return out


def to_fp32(saved_model_dir: str | Path, out_path: str | Path, *, flex: bool = False) -> Path:
    conv = _new_converter(saved_model_dir, flex)
    return _write(conv.convert(), out_path)


def to_static_int8(
    saved_model_dir: str | Path,
    out_path: str | Path,
    representative_dataset: Callable[[], Iterator[list]],
    *,
    flex: bool = False,
    int8_io: bool = True,
) -> Path:
    import tensorflow as tf

    conv = _new_converter(saved_model_dir, flex)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if int8_io:
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return _write(conv.convert(), out_path)


def to_dynamic_range(
    saved_model_dir: str | Path, out_path: str | Path, *, flex: bool = False
) -> Path:
    import tensorflow as tf

    conv = _new_converter(saved_model_dir, flex)
    # Optimize.DEFAULT with no representative dataset == dynamic-range quant.
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return _write(conv.convert(), out_path)
