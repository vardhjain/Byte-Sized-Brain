"""Memory footprint of one model variant, measured in a fresh Python process.

Two things make an in-process measurement useless for comparing variants. Once a
model is loaded, running inferences barely moves the process RSS, so a delta taken
around the benchmark loop is a few pages of noise. And when two variants are
benchmarked one after the other in the same process, the second one reuses memory
the first one freed, so its number is understated.

So each variant gets its own child process. The child imports the runtime first
(that cost is the same for every variant and is excluded), records its RSS, loads
the model, runs a few inferences, and reports how much the RSS grew and how high
it peaked over that baseline.

:func:`measure_memory` runs this module as a script, in the form
``python -m byte_sized_brain.benchmark.memory <runtime> <model> <sample.npz> <repeats>``.
"""

from __future__ import annotations

import contextlib
import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import psutil

_MB = 1024**2
_ARRAY_KEY = "__sample__"  # npz key used when the sample is a bare array (TFLite)


def _peak_rss_bytes(proc: psutil.Process) -> int:
    """Highest RSS this process has reached so far."""
    if sys.platform == "win32":
        return int(proc.memory_info().peak_wset)
    import resource

    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maxrss if sys.platform == "darwin" else maxrss * 1024)  # macOS reports bytes


def _preload_runtime(runtime: str) -> None:
    """Import the inference library so its own footprint is in the baseline."""
    if runtime == "onnxruntime":
        import onnxruntime  # noqa: F401

        return
    try:
        import tensorflow as tf

        with contextlib.suppress(AttributeError):
            _ = tf.lite.Interpreter  # resolve TensorFlow's lazy tf.lite module now
    except ImportError:
        import ai_edge_litert.interpreter  # type: ignore  # noqa: F401


def _probe(runtime: str, model_path: str, sample_path: str, repeats: int) -> dict[str, float]:
    from .runtimes import OnnxRunner, TFLiteRunner

    with np.load(sample_path) as data:
        loaded = {key: data[key] for key in data.files}
    sample: Any = loaded.get(_ARRAY_KEY, loaded)

    _preload_runtime(runtime)
    proc = psutil.Process()
    gc.collect()
    baseline = proc.memory_info().rss

    runner: Any = TFLiteRunner(model_path) if runtime == "tflite" else OnnxRunner(model_path)
    for _ in range(max(repeats, 1)):
        runner.predict(sample)

    return {
        "rss_delta_mb": (proc.memory_info().rss - baseline) / _MB,
        "peak_rss_mb": max(_peak_rss_bytes(proc) - baseline, 0) / _MB,
    }


def measure_memory(
    runtime: str, model_path: str | Path, sample: Any, *, repeats: int = 10
) -> dict[str, float]:
    """RSS growth and peak (MB) from loading ``model_path`` and running ``sample``.

    ``sample`` is one raw example in the form the runner's ``predict`` takes, which
    is an array for TFLite and a dict of named arrays for ONNX Runtime.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sample_path = Path(tmp) / "sample.npz"
        if isinstance(sample, dict):
            np.savez(sample_path, **sample)
        else:
            np.savez(sample_path, **{_ARRAY_KEY: np.asarray(sample)})
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                __name__,
                runtime,
                str(model_path),
                str(sample_path),
                str(repeats),
            ],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        tail = result.stderr.strip()[-2000:]
        raise RuntimeError(f"Memory probe failed for {model_path}:\n{tail}")
    # The runtimes may print their own start-up lines; ours is the last one.
    return json.loads(result.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    print(json.dumps(_probe(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))))
