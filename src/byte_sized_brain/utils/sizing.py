"""Measure on-disk size of a file or a directory tree (e.g. a SavedModel)."""

from __future__ import annotations

import os
from pathlib import Path


def size_bytes(path: str | os.PathLike) -> int:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    raise FileNotFoundError(f"No such file or directory: {p}")


def size_mb(path: str | os.PathLike) -> float:
    """Size in megabytes (MiB, base-1024)."""
    return size_bytes(path) / (1024**2)
