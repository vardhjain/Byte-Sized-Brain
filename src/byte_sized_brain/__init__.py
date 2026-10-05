"""Byte-Sized Brain: post-training quantization trade-off benchmarks.

A small, reproducible toolkit that trains four model families across three
modalities, applies post-training quantization (TFLite static/dynamic-range,
ONNX dynamic INT8), and benchmarks the accuracy / size / latency / memory
trade-offs on x86 and ARM64 with a single device-agnostic harness.
"""

import os

# This project uses Hugging Face transformers with PyTorch only. Without this,
# transformers also probes the installed TensorFlow, and with Keras 3 it then
# refuses to import its Trainer unless the separate tf-keras package is present.
os.environ.setdefault("USE_TF", "0")

__version__ = "0.3.2"

__all__ = ["__version__"]
