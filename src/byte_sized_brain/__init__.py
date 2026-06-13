"""Byte-Sized Brain — post-training quantization trade-off benchmarks.

A small, reproducible toolkit that trains four model families across three
modalities, applies post-training quantization (TFLite static/dynamic-range,
ONNX dynamic INT8), and benchmarks the accuracy / size / latency / memory
trade-offs on x86 and ARM64 with a single device-agnostic harness.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
