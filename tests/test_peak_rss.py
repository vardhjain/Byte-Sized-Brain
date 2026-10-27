"""Regression tests for peak RSS collected after the runtime import baseline."""

from __future__ import annotations

import mmap
import sys
import threading
from types import SimpleNamespace

import psutil
import pytest

from byte_sized_brain.benchmark import memory


def test_sampling_fallback_records_post_baseline_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "_reset_linux_peak_rss", lambda: False)

    class FakeProcess:
        rss = 100
        sampled_peak = threading.Event()

        def memory_info(self):
            value = self.rss
            if value == 300:
                self.sampled_peak.set()
            return SimpleNamespace(rss=value)

    proc = FakeProcess()
    monitor = memory._PeakRSSMonitor(proc)
    try:
        proc.rss = 300
        assert proc.sampled_peak.wait(timeout=1.0)
        proc.rss = 125
    finally:
        measured_peak = monitor.stop()

    assert measured_peak == 300


def test_reset_falls_back_when_proc_is_inaccessible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory.sys, "platform", "linux")

    def inaccessible(*args, **kwargs):
        raise PermissionError("/proc restricted")

    monkeypatch.setattr(memory.Path, "write_text", inaccessible)
    assert memory._reset_linux_peak_rss() is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only VmHWM reset")
def test_linux_peak_excludes_memory_allocated_before_baseline() -> None:
    # Allocate and release resident memory before starting the measurement. A
    # process-lifetime high-water mark would incorrectly report it as model RSS.
    with mmap.mmap(-1, 64 * 1024**2) as allocation:
        for offset in range(0, len(allocation), 4096):
            allocation[offset] = 1

    proc = psutil.Process()
    baseline = proc.memory_info().rss
    import resource

    old_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert old_peak - baseline > 32 * 1024**2

    monitor = memory._PeakRSSMonitor(proc)
    if not monitor._linux_hwm_reset:
        monitor.stop()
        pytest.skip("Kernel doesn't permit resetting VmHWM")
    measured_peak = monitor.stop()
    assert measured_peak - baseline < 8 * 1024**2
