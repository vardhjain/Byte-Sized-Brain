"""Memory footprint of one model variant, measured in a fresh Python process.

Two things make an in-process measurement useless for comparing variants. Once a
model is loaded, running inferences barely moves the process RSS, so a delta taken
around the benchmark loop is a few pages of noise. And when two variants are
benchmarked one after the other in the same process, the second one reuses memory
the first one freed, so its number is understated.

So each variant gets its own child process. The child imports the runtime first
(that cost is the same for every variant and is excluded), records its RSS,
loads the model and runs a few inferences. On Linux, the kernel peak-RSS counter
is reset after imports to exclude older peaks. On other platforms (or when
Linux blocks the reset), RSS is sampled while the model runs.

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
import threading
from pathlib import Path
from typing import Any

import numpy as np
import psutil

_MB = 1024**2
_ARRAY_KEY = "__sample__"  # npz key used when the sample is a bare array (TFLite)
_SAMPLE_INTERVAL_SECONDS = 0.005


def _reset_linux_peak_rss() -> bool:
    """Reset VmHWM to current RSS, discarding peaks from earlier imports.

    Linux exposes this through /proc/self/clear_refs (write 5). Some containers
    restrict this interface; callers must then sample RSS instead of trusting
    the historical process-wide high-water mark.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        Path("/proc/self/clear_refs").write_text("5\n", encoding="ascii")
    except OSError:
        return False
    return True


class _PeakRSSMonitor:
    """Measure the RSS peak *after* the runtime-import baseline.

    Linux offers an exact high-water mark after resetting it. On other systems,
    and on restricted Linux containers, sample RSS periodically. Sampling can
    miss very short-lived allocations, so it is a best-effort fallback.
    """

    def __init__(self, proc: psutil.Process) -> None:
        self._proc = proc
        self._peak_bytes = int(proc.memory_info().rss)
        self._linux_hwm_reset = _reset_linux_peak_rss()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if not self._linux_hwm_reset:
            self._thread = threading.Thread(target=self._sample_rss, daemon=True)
            self._thread.start()

    def _sample_rss(self) -> None:
        while not self._stop.wait(_SAMPLE_INTERVAL_SECONDS):
            with contextlib.suppress(psutil.Error):
                self._peak_bytes = max(self._peak_bytes, int(self._proc.memory_info().rss))

    def stop(self) -> int:
        """Finish sampling and return the highest RSS since the measurement began."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._peak_bytes = max(self._peak_bytes, int(self._proc.memory_info().rss))
        if self._linux_hwm_reset:
            import resource

            self._peak_bytes = max(
                self._peak_bytes, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            )
        return self._peak_bytes


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

    monitor = _PeakRSSMonitor(proc)
    try:
        runner: Any = TFLiteRunner(model_path) if runtime == "tflite" else OnnxRunner(model_path)
        for _ in range(max(repeats, 1)):
            runner.predict(sample)
    finally:
        peak = monitor.stop()

    return {
        "rss_delta_mb": (proc.memory_info().rss - baseline) / _MB,
        "peak_rss_mb": max(peak - baseline, 0) / _MB,
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
