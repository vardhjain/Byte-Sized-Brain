"""Why does the MobileNetV2 CNN lose accuracy under full-integer quantization?

The headline benchmark shows the CNN dropping about 17 points when it is converted
to static INT8, while the other three models barely move. This script isolates the
cause by converting the same trained model several different ways and scoring every
variant on the same test images.

Usage (needs the trained model, so run ``bsb train cnn_cifar10`` first)::

    python scripts/cnn_int8_ablation.py
    python scripts/cnn_int8_ablation.py --num-samples 300    # quicker, noisier

It writes ``benchmarks/results/ablations/cnn_int8_ablation.csv``. The findings are
discussed in ``docs/methodology.md``.

Two of the variants rely on TensorFlow APIs that are experimental or private
(``QuantizationDebugger`` for selective quantization, and the converter's
``_experimental_disable_per_channel`` switch). That is acceptable for a diagnostic
script, which is why this lives outside the installable package.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from byte_sized_brain.benchmark import TFLiteRunner
from byte_sized_brain.config import Paths, load_config
from byte_sized_brain.data import representative_dataset
from byte_sized_brain.data.cifar10 import load_split
from byte_sized_brain.seeding import seed_everything
from byte_sized_brain.utils import get_logger, size_mb

log = get_logger("ablation")

RepDataset = Callable[[], Iterator[list[np.ndarray]]]
COLUMNS = ["variant", "description", "accuracy", "accuracy_delta", "size_mb", "num_samples"]


def _converter(saved_model: Path, rep: RepDataset | None):
    """A TFLite converter with default optimizations, calibrated if ``rep`` is given."""
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if rep is not None:
        conv.representative_dataset = rep
    return conv


def fp32(saved_model: Path) -> bytes:
    import tensorflow as tf

    return tf.lite.TFLiteConverter.from_saved_model(str(saved_model)).convert()


def full_int8(saved_model: Path, rep: RepDataset, *, per_channel: bool = True) -> bytes:
    """Static INT8 for weights and activations, as the main pipeline produces it."""
    import tensorflow as tf

    conv = _converter(saved_model, rep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    if not per_channel:
        conv._experimental_disable_per_channel = True
    return conv.convert()


def weights_only(saved_model: Path) -> bytes:
    """Dynamic-range quantization, which stores INT8 weights but keeps float activations."""
    return _converter(saved_model, None).convert()


def int8_except(saved_model: Path, rep: RepDataset, float_ops: list[str]) -> bytes:
    """Static INT8 everywhere except the listed op types, which stay in float."""
    import tensorflow as tf

    options = tf.lite.experimental.QuantizationDebugOptions(denylisted_ops=float_ops)
    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=_converter(saved_model, rep), debug_dataset=rep, debug_options=options
    )
    return debugger.get_nondebug_quantized_model()


def accuracy(model: bytes, x: np.ndarray, y: np.ndarray, workdir: Path) -> tuple[float, float]:
    """Top-1 accuracy and file size (MB) of a serialized TFLite model."""
    path = workdir / "variant.tflite"
    path.write_bytes(model)
    runner = TFLiteRunner(path)
    correct = sum(
        int(np.argmax(runner.predict(sample))) == int(label)
        for sample, label in zip(x, y, strict=True)
    )
    return correct / len(y), size_mb(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-samples", type=int, default=None, help="test images to score")
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/results/ablations/cnn_int8_ablation.csv")
    )
    args = parser.parse_args()

    cfg = load_config("cnn_cifar10")
    seed_everything(cfg.seed, frameworks=(cfg.framework,))
    saved_model = Paths(cfg.name).fp32_source
    if not saved_model.exists():
        log.error("No trained model at %s. Run `bsb train cnn_cifar10` first.", saved_model)
        return 1

    img_size = cfg.data.img_size or 96
    n = args.num_samples or cfg.benchmark.num_samples
    x_test, y_test = load_split("test", img_size, subset=n)
    # 1000 calibration images, so the "more calibration data" variant has a real superset.
    x_cal, _ = load_split("train", img_size, subset=1000)
    rep = representative_dataset(x_cal, cfg.convert.rep_samples)
    rep_large = representative_dataset(x_cal, len(x_cal))

    variants: list[tuple[str, str, Callable[[], bytes]]] = [
        ("fp32", "Full precision baseline", lambda: fp32(saved_model)),
        (
            "int8",
            "Static INT8, weights and activations (the benchmarked variant)",
            lambda: full_int8(saved_model, rep),
        ),
        (
            "int8_more_calibration",
            f"Static INT8 calibrated on {len(x_cal)} images instead of {cfg.convert.rep_samples}",
            lambda: full_int8(saved_model, rep_large),
        ),
        (
            "int8_per_tensor_weights",
            "Static INT8 with one scale per weight tensor instead of one per channel",
            lambda: full_int8(saved_model, rep, per_channel=False),
        ),
        (
            "int8_weights_float_activations",
            "INT8 weights with float activations (dynamic-range)",
            lambda: weights_only(saved_model),
        ),
        (
            "int8_except_depthwise",
            "Static INT8 except the depthwise convolutions, kept in float",
            lambda: int8_except(saved_model, rep, ["DEPTHWISE_CONV_2D"]),
        ),
        (
            "int8_except_conv2d",
            "Static INT8 except the standard and 1x1 convolutions, kept in float",
            lambda: int8_except(saved_model, rep, ["CONV_2D"]),
        ),
        (
            "int8_except_residual_add",
            "Static INT8 except the residual additions, kept in float",
            lambda: int8_except(saved_model, rep, ["ADD"]),
        ),
        (
            "int8_except_head",
            "Static INT8 except pooling, the dense head and softmax, kept in float",
            lambda: int8_except(saved_model, rep, ["MEAN", "FULLY_CONNECTED", "SOFTMAX"]),
        ),
    ]

    rows: list[dict[str, object]] = []
    baseline: float | None = None
    with tempfile.TemporaryDirectory() as tmp:
        for name, description, build in variants:
            acc, mb = accuracy(build(), x_test, y_test, Path(tmp))
            baseline = acc if baseline is None else baseline
            rows.append(
                {
                    "variant": name,
                    "description": description,
                    "accuracy": round(acc, 4),
                    "accuracy_delta": round(acc - baseline, 4),
                    "size_mb": round(mb, 3),
                    "num_samples": len(y_test),
                }
            )
            log.info("%-32s acc=%.4f (%+.4f) size=%.2fMB", name, acc, acc - baseline, mb)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
