"""Capture the execution environment so every benchmark row is self-describing.

The whole point of the project is honest cross-architecture benchmarking, so we
stamp each result with the device, CPU architecture, OS and an ``emulated`` flag.
On emulated/real ARM runs set ``BSB_EMULATED`` and ``BSB_DEVICE`` (the Docker and
cloud recipes do this) so the CSV makes the distinction explicit.

The ``device`` label is the CPU model (for example ``Intel Core i7-10710U``), never
the machine's hostname, because result CSVs and ``bsb info`` output get published.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform
import re
import subprocess
import sys


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


# platform.machine() spells the same ISA differently per OS (Windows says AMD64,
# Linux says x86_64, macOS says arm64, Linux says aarch64). Normalize so results
# from one architecture always group together in the report.
_ARCH_ALIASES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "x64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
    "armv8": "aarch64",
}


def normalize_arch(machine: object) -> str:
    """Canonical architecture name, e.g. ``AMD64`` and ``x86_64`` both become ``x86_64``."""
    # Anything that is not text (a missing CSV cell read as NaN, say) is unknown.
    raw = machine.strip() if isinstance(machine, str) else ""
    if not raw:
        return "unknown"
    return _ARCH_ALIASES.get(raw.lower(), raw.lower())


def _clean_cpu_name(raw: str) -> str:
    """Drop trademark marks and the nominal clock so the label stays short."""
    name = re.sub(r"\((?:R|TM)\)|\bCPU\b|@.*$", "", raw, flags=re.IGNORECASE)
    return " ".join(name.split())


def parse_cpuinfo(text: str) -> str:
    """Best CPU label in a Linux ``/proc/cpuinfo`` dump, or ``""``.

    x86 kernels report ``model name``. Raspberry Pi kernels report a board
    ``Model`` line instead (x86 also has a numeric ``model`` line, which is skipped).
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if sep and value and key not in fields:
            fields[key] = value
    for key in ("model name", "model", "hardware"):
        value = fields.get(key, "")
        if value and not value.isdigit():
            return _clean_cpu_name(value)
    return ""


def cpu_name() -> str:
    """Human-readable CPU model, or ``""`` when it cannot be determined."""
    # sys.platform (not platform.system()) so mypy skips the other OSes' branches.
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return _clean_cpu_name(str(winreg.QueryValueEx(key, "ProcessorNameString")[0]))
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                return parse_cpuinfo(f.read())
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return _clean_cpu_name(out.stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def os_name() -> str:
    """``platform.system()`` plus release, reporting Windows 11 correctly.

    Before Python 3.12, ``platform.release()`` returns ``"10"`` on Windows 11, so
    the build number (22000 or later means Windows 11) decides instead.
    """
    system, release = platform.system(), platform.release()
    if system == "Windows" and release == "10":
        try:
            if int(platform.version().split(".")[2]) >= 22000:
                release = "11"
        except (IndexError, ValueError):
            pass
    return f"{system} {release}".strip()


def collect_sysinfo() -> dict[str, object]:
    return {
        "device": os.environ.get("BSB_DEVICE") or cpu_name() or "unknown",
        "arch": normalize_arch(platform.machine()),
        "os": os_name(),
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
