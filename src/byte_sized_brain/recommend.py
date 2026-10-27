"""Pick a measured model variant that satisfies explicit deployment constraints.

This is a rule-based, explainable decision helper, not a trained predictor.
It never mixes architectures, devices, experiment run IDs, or smoke results,
and never invents latency or accuracy for strategies that were not benchmarked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .report import _with_run_ids, load_results
from .utils import normalize_arch

LatencyKind = Literal["model", "prepared-input"]
Objective = Literal["latency", "size", "memory", "accuracy"]


@dataclass(frozen=True)
class Constraints:
    """Accuracy loss is in percentage points; min_accuracy is a fraction (0..1)."""

    max_accuracy_drop_pts: float = 1.0
    min_accuracy: float | None = None
    max_size_mb: float | None = None
    max_latency_ms: float | None = None
    max_p95_ms: float | None = None
    max_memory_mb: float | None = None
    latency_kind: LatencyKind = "model"
    objective: Objective = "latency"
    require_paired_ci: bool = False

    def __post_init__(self) -> None:
        if self.latency_kind not in {"model", "prepared-input"}:
            raise ValueError("latency_kind must be model or prepared-input")
        if self.objective not in {"latency", "size", "memory", "accuracy"}:
            raise ValueError("objective must be latency, size, memory or accuracy")
        if not isfinite(self.max_accuracy_drop_pts) or not 0 <= self.max_accuracy_drop_pts <= 100:
            raise ValueError("max_accuracy_drop_pts must be between 0 and 100")
        if self.min_accuracy is not None and (
            not isfinite(self.min_accuracy) or not 0 <= self.min_accuracy <= 1
        ):
            raise ValueError("min_accuracy must be a fraction from 0 to 1")
        for name in ("max_size_mb", "max_latency_ms", "max_p95_ms", "max_memory_mb"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be greater than zero")


def _number(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _choose_run(
    df: pd.DataFrame,
    pipeline: str,
    arch: str,
    device: str | None,
    run_id: str | None,
) -> pd.DataFrame:
    """Choose the newest *complete native* experiment, never a mixture of runs."""
    if df.empty:
        raise ValueError("No benchmark results found. Run bsb benchmark first.")
    df = _with_run_ids(df)
    df = df[(df["pipeline"] == pipeline) & (df["arch"] == arch)]
    df = df[~df["emulated"].astype(str).str.strip().str.lower().isin({"true", "1"})]
    if device is not None:
        df = df[df["device"] == device]
    if run_id is not None:
        df = df[df["run_id"] == run_id]
    if df.empty:
        raise ValueError("No native benchmark rows match the selected pipeline/arch/device/run.")
    options: list[tuple[str, str, pd.DataFrame]] = []
    for rid, rows in df.groupby("run_id", sort=False):
        variants = rows["variant"].tolist()
        if "fp32" not in variants or len(variants) < 2:
            continue
        if len(set(variants)) != len(variants):
            raise ValueError(f"Duplicate variants in run {rid}; cannot safely recommend.")
        # A run ID must not silently combine different hardware or sample counts.
        for metadata in ("device", "arch", "os", "num_samples"):
            if metadata in rows and rows[metadata].nunique(dropna=False) != 1:
                raise ValueError(f"Inconsistent {metadata} in run {rid}")
        latest = max(rows["timestamp"].astype(str)) if "timestamp" in rows else ""
        options.append((latest, str(rid), rows))
    if not options:
        raise ValueError("No complete FP32 + quantized experiment matches the selection.")
    return max(options, key=lambda item: (item[0], item[1]))[2]


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # Four simultaneous objectives: maximize accuracy, minimize the other three.
    a = (
        -left["accuracy"],
        left["latency_mean_ms"],
        left["size_mb"],
        left["memory_mb"],
    )
    b = (
        -right["accuracy"],
        right["latency_mean_ms"],
        right["size_mb"],
        right["memory_mb"],
    )
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(
        x < y for x, y in zip(a, b, strict=True)
    )


def recommend(
    df: pd.DataFrame,
    *,
    pipeline: str,
    arch: str = "x86_64",
    device: str | None = None,
    run_id: str | None = None,
    constraints: Constraints | None = None,
) -> dict[str, Any]:
    """Return a traceable selection and rejected-variant explanations.

    Feasible variants include FP32, so the defensible answer can be "do not
    quantize." The most recent complete run is selected *before* applying any
    thresholds; a failed constraint never causes a silent search for better
    numbers in an older experiment.
    """
    limits = constraints or Constraints()
    arch = normalize_arch(arch)
    rows = _choose_run(df, pipeline, arch, device, run_id)
    baseline = rows[rows["variant"] == "fp32"].iloc[0]
    base_accuracy = _number(baseline, "accuracy")
    if base_accuracy is None or not 0 <= base_accuracy <= 1:
        raise ValueError("The selected experiment has no valid FP32 accuracy")

    prefix = "e2e_latency" if limits.latency_kind == "prepared-input" else "latency"
    latency_key, p95_key = f"{prefix}_ms_mean", f"{prefix}_ms_p95"
    if _number(baseline, latency_key) is None:
        raise ValueError(
            f"No {limits.latency_kind} latency for FP32 in this run. "
            "Re-benchmark with current code; historical results contain only model-only timings."
        )

    candidates: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        metrics = {
            "accuracy": _number(row, "accuracy"),
            "latency_mean_ms": _number(row, latency_key),
            "latency_p95_ms": _number(row, p95_key),
            "size_mb": _number(row, "size_mb"),
            "memory_mb": _number(row, "rss_delta_mb"),
        }
        reasons: list[str] = []
        for name, val in metrics.items():
            if val is None:
                reasons.append(f"unmeasured: {name}")
        accuracy = metrics["accuracy"]
        if limits.require_paired_ci:
            lower = _number(row, "accuracy_delta_ci95_low")
            if lower is None:
                reasons.append("unmeasured: paired accuracy CI (rerun the benchmark)")
            elif -100 * lower > limits.max_accuracy_drop_pts + 1e-9:
                reasons.append(
                    f"paired 95% lower accuracy bound {lower * 100:+.2f} pts "
                    f"exceeds allowed loss {limits.max_accuracy_drop_pts:g} pts"
                )
        if accuracy is not None:
            loss_pts = (base_accuracy - accuracy) * 100
            if loss_pts > limits.max_accuracy_drop_pts + 1e-9:
                reasons.append(
                    f"accuracy drop {loss_pts:.2f} pts exceeds {limits.max_accuracy_drop_pts:g} pts"
                )
            if limits.min_accuracy is not None and accuracy + 1e-12 < limits.min_accuracy:
                reasons.append(f"accuracy {accuracy:.3f} below {limits.min_accuracy:.3f}")

        threshold_fields = (
            ("max_size_mb", "size_mb"),
            ("max_latency_ms", "latency_mean_ms"),
            ("max_p95_ms", "latency_p95_ms"),
            ("max_memory_mb", "memory_mb"),
        )
        for threshold, key in threshold_fields:
            maximum = getattr(limits, threshold)
            value = metrics[key]
            if maximum is not None and value is not None and value > maximum + 1e-9:
                reasons.append(f"{key} {value:.4f} exceeds {maximum:g}")

        candidates.append(
            {
                "variant": str(row["variant"]),
                "quantization": str(row.get("quantization", "")),
                **metrics,
                "accuracy_delta_pts": round((accuracy - base_accuracy) * 100, 3)
                if accuracy is not None
                else None,
                "feasible": not reasons,
                "reasons": reasons,
                "pareto": False,
            }
        )

    feasible = [c for c in candidates if c["feasible"]]
    for candidate in feasible:
        candidate["pareto"] = not any(
            _dominates(other, candidate) for other in feasible if other is not candidate
        )
    objective_keys = {
        "latency": "latency_mean_ms",
        "size": "size_mb",
        "memory": "memory_mb",
        "accuracy": "accuracy",
    }
    target = objective_keys[limits.objective]

    def rank(candidate: dict[str, Any]) -> tuple[float, float, str]:
        value = candidate[target]
        primary = -value if limits.objective == "accuracy" else value
        # Break ties consistently without any hidden global weighted score.
        return (primary, candidate["latency_mean_ms"], candidate["variant"])

    chosen = min(feasible, key=rank) if feasible else None
    return {
        "pipeline": pipeline,
        "arch": arch,
        "device": str(baseline.get("device", "")),
        "run_id": str(baseline["run_id"]),
        "timestamp": str(max(rows["timestamp"].astype(str))) if "timestamp" in rows else "",
        "num_samples": int(baseline["num_samples"]),
        "baseline_accuracy": base_accuracy,
        "constraints": asdict(limits),
        "recommended_variant": chosen["variant"] if chosen else None,
        "pareto_variants": sorted(c["variant"] for c in feasible if c["pareto"]),
        "candidates": candidates,
        "historical": str(baseline["run_id"]).startswith("legacy-"),
        "note": (
            "Selection uses observed measurements from one native benchmark run. "
            "Accuracy thresholds use observed point estimates by default; "
            "require_paired_ci also checks the paired 95% lower bound. "
            "A single pass is not evidence of stable latency, and historical "
            "per-model accuracy intervals do not test paired accuracy loss. "
            "Results from other devices or unmeasured quantizers are excluded."
        ),
    }


def recommend_from_results(
    pipeline: str,
    *,
    results_dir: Path | None = None,
    arch: str = "x86_64",
    device: str | None = None,
    run_id: str | None = None,
    constraints: Constraints | None = None,
) -> dict[str, Any]:
    """Read existing CSVs; never train, convert, mutate, or download anything."""
    return recommend(
        load_results(results_dir),
        pipeline=pipeline,
        arch=arch,
        device=device,
        run_id=run_id,
        constraints=constraints,
    )
