"""Result-row schema and CSV IO.

Every benchmark produces one row per (run_id, pipeline, variant). A paired
FP32/quantized benchmark shares a run ID. Results carry machine metadata and
are preserved across repeated runs on the same architecture or device.
:func:`write_results` also migrates pre-run-ID CSVs transparently.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

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
    "run_id",
    "latency_passes",
    "latency_pass_std_ms",
    "latency_ms_ci95_low",
    "latency_ms_ci95_high",
    "accuracy_delta_ci95_low",
    "accuracy_delta_ci95_high",
    "mcnemar_pvalue",
    "fp32_only_correct",
    "variant_only_correct",
]


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _optional_float(metrics: dict[str, Any], key: str, decimals: int) -> float | str:
    """Leave unmeasured statistics blank, rather than inventing precision."""
    value = metrics.get(key, "")
    return round(float(value), decimals) if value != "" and value is not None else ""


def build_row(
    cfg: PipelineConfig,
    *,
    variant: str,
    quantization: str,
    runtime: str,
    size_mb: float,
    metrics: dict[str, Any],
    timestamp: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    sysinfo = collect_sysinfo()
    row: dict[str, Any] = {
        "timestamp": timestamp or utc_timestamp(),
        "run_id": run_id or uuid4().hex,
        "latency_passes": int(metrics["latency_passes"])
        if "latency_passes" in metrics
        else "",
        "latency_pass_std_ms": _optional_float(metrics, "latency_pass_std_ms", 4),
        "latency_ms_ci95_low": _optional_float(metrics, "latency_ms_ci95_low", 4),
        "latency_ms_ci95_high": _optional_float(metrics, "latency_ms_ci95_high", 4),
        "accuracy_delta_ci95_low": _optional_float(metrics, "accuracy_delta_ci95_low", 4),
        "accuracy_delta_ci95_high": _optional_float(metrics, "accuracy_delta_ci95_high", 4),
        "mcnemar_pvalue": _optional_float(metrics, "mcnemar_pvalue", 9),
        "fp32_only_correct": int(metrics["fp32_only_correct"])
        if "fp32_only_correct" in metrics
        else "",
        "variant_only_correct": int(metrics["variant_only_correct"])
        if "variant_only_correct" in metrics
        else "",
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


def effective_run_id(row: dict[str, Any]) -> str:
    """Run identifier; infer a deterministic group for pre-0.5 CSV rows.

    Legacy CSVs recorded a single FP32/INT8 pair per device and architecture,
    but didn't contain a run ID. Both variants therefore receive the same
    stable synthetic ID. History already overwritten by older versions cannot
    be recovered.
    """
    explicit = str(row.get("run_id") or "").strip()
    if explicit and explicit.lower() != "nan":
        return explicit
    identity = (
        str(row["pipeline"]),
        normalize_arch(str(row["arch"])),
        str(_is_true(row["emulated"])),
        str(row.get("device", "")),
        str(row.get("os", "")),
        str(row.get("python", "")),
    )
    digest = hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()[:16]
    return f"legacy-{digest}"


def result_key(row: dict[str, Any]) -> tuple[str, str]:
    """A benchmark run is immutable unless its own ID is explicitly reused."""
    return str(row["pipeline"]), effective_run_id(row)


def write_results(rows: list[dict[str, Any]], csv_path: str | Path) -> Path:
    """Append independent runs; replace only rows bearing the same run ID.

    Older CSVs without ``run_id`` are migrated in memory, preserving the
    historical FP32/quantized pair under a deterministic legacy ID.
    """
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    incoming = [{**r, "run_id": effective_run_id(r)} for r in rows]
    replaced = {result_key(r) for r in incoming}
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
            kept = [
                {**r, "run_id": effective_run_id(r)}
                for r in reader
                if result_key(r) not in replaced
            ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=RESULT_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows([*kept, *incoming])
    return out
