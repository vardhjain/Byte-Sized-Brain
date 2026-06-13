"""Result-row schema and CSV IO.

Every benchmark produces one row per (pipeline, variant). Each row is fully
self-describing — it carries the device, architecture, OS and the ``emulated``
flag — so x86, emulated-ARM and real-ARM results can be concatenated and
compared honestly.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.sysinfo import collect_sysinfo, library_versions

if TYPE_CHECKING:
    from ..config import PipelineConfig

RESULT_COLUMNS = [
    "timestamp",
    "pipeline",
    "modality",
    "framework",
    "runtime",
    "variant",
    "quantization",
    "size_mb",
    "accuracy",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "rss_delta_mb",
    "peak_rss_mb",
    "num_samples",
    "device",
    "arch",
    "os",
    "python",
    "emulated",
    "lib_versions",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_row(
    cfg: PipelineConfig,
    *,
    variant: str,
    quantization: str,
    runtime: str,
    size_mb: float,
    metrics: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    sysinfo = collect_sysinfo()
    row: dict[str, Any] = {
        "timestamp": timestamp or utc_timestamp(),
        "pipeline": cfg.name,
        "modality": cfg.modality,
        "framework": cfg.framework,
        "runtime": runtime,
        "variant": variant,
        "quantization": quantization,
        "size_mb": round(float(size_mb), 4),
        "accuracy": round(float(metrics["accuracy"]), 4),
        "latency_ms_mean": round(float(metrics["latency_ms_mean"]), 4),
        "latency_ms_p50": round(float(metrics["latency_ms_p50"]), 4),
        "latency_ms_p95": round(float(metrics["latency_ms_p95"]), 4),
        "rss_delta_mb": round(float(metrics["rss_delta_mb"]), 3),
        "peak_rss_mb": round(float(metrics["peak_rss_mb"]), 3),
        "num_samples": int(metrics["num_samples"]),
        "lib_versions": json.dumps(library_versions(), sort_keys=True),
    }
    row.update(sysinfo)
    return row


def write_results(rows: list[dict[str, Any]], csv_path: str | Path) -> Path:
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out
