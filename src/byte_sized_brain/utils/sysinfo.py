"""Capture the execution environment so every benchmark row is self-describing.

The whole point of the project is honest cross-architecture benchmarking, so we
stamp each result with the device, CPU architecture, OS and an ``emulated`` flag.
On emulated/real ARM runs set ``BSB_EMULATED`` and ``BSB_DEVICE`` (the Docker and
cloud recipes do this) so the CSV makes the distinction explicit.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def collect_sysinfo() -> dict[str, object]:
    return {
        "device": os.environ.get("BSB_DEVICE") or platform.node() or "unknown",
        "arch": platform.machine() or "unknown",
        "os": f"{platform.system()} {platform.release()}".strip(),
        "python": platform.python_version(),
        "emulated": _truthy(os.environ.get("BSB_EMULATED")),
    }


_VERSION_PKGS = (
    "tensorflow",
    "torch",
    "transformers",
    "datasets",
    "onnxruntime",
    "onnx",
    "numpy",
)


def library_versions() -> dict[str, str]:
    """Best-effort version strings for the libraries that affect results."""
    out: dict[str, str] = {}
    for pkg in _VERSION_PKGS:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            try:  # some packages report a different dist name than import name
                out[pkg] = importlib.import_module(pkg).__version__
            except Exception:
                continue
    return out
