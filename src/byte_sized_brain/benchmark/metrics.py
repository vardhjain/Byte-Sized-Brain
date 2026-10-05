"""Result-row schema and CSV IO.

Every benchmark produces one row per (pipeline, variant). Each row is fully
self-describing (it carries the device, architecture, OS and the ``emulated``
flag), so x86, emulated-ARM and real-ARM results can live in one CSV and be
compared honestly. :func:`write_results` merges rather than overwrites, so a run
on one architecture never erases the rows another architecture produced.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.sysinfo import collect_sysinfo, library_versions, normalize_arch

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
    "agreement",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "rss_delta_mb",
    "peak_rss_mb",
    "num_samples",
    "threads",
    "device",
    "arch",
    "os",
    "python",
    "emulated",
    "lib_versions",
]


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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
        # Optional, so rows can still be built from the core metrics alone.
        "agreement": round(float(metrics["agreement"]), 4) if "agreement" in metrics else "",
        "threads": int(metrics["threads"]) if "threads" in metrics else "",
        "lib_versions": json.dumps(library_versions(), sort_keys=True),
    }
    row.update(sysinfo)
    return row


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def result_key(row: dict[str, Any]) -> tuple[str, str, bool]:
    """The (pipeline, arch, emulated) group a row belongs to."""
    return (str(row["pipeline"]), normalize_arch(str(row["arch"])), _is_true(row["emulated"]))


def write_results(rows: list[dict[str, Any]], csv_path: str | Path) -> Path:
    """Write ``rows`` to ``csv_path``, keeping rows from other architectures.

    A new run replaces every existing row in the same (pipeline, arch, emulated)
    group and leaves the others alone, so an x86 run and a real-ARM run of the
    same pipeline can share one CSV.
    """
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    replaced = {result_key(r) for r in rows}
    kept: list[dict[str, Any]] = []
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            absent = {"pipeline", "arch", "emulated"} - set(reader.fieldnames or [])
            if absent:
                raise ValueError(
                    f"{out} is not a results file this version can merge into "
                    f"(missing columns: {sorted(absent)}). Move it aside and rerun."
                )
            kept = [r for r in reader if result_key(r) not in replaced]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=RESULT_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows([*kept, *rows])
    return out
