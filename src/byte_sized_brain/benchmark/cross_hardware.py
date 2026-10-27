"""Verify that x86 and ARM64 result files used byte-identical model artifacts.

The two benchmark jobs have distinct run IDs and hardware environments. Only
model file digests and the evaluation sample count are required to match.
The same kernels can produce slightly different predictions across CPUs, so
accuracy is reported but must not be forced to be numerically identical.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _rows(directory: Path, pipeline: str, expected_arch: str) -> dict[str, dict[str, str]]:
    file = directory / f"{pipeline}.csv"
    with file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No benchmark rows in {file}")
    variants: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["pipeline"] != pipeline or row["arch"] != expected_arch:
            raise ValueError(f"Wrong pipeline/architecture in {file}")
        variant = row["variant"]
        if variant in variants:
            raise ValueError(f"Duplicate variant {variant} in {file}")
        if not row.get("model_sha256"):
            raise ValueError(f"Missing model_sha256 for {variant} in {file}")
        variants[variant] = row
    if len({row["run_id"] for row in variants.values()}) != 1:
        raise ValueError(f"Results in {file} came from more than one run")
    return variants


def compare_results(x86: Path, arm: Path) -> list[dict[str, Any]]:
    """Ensure both machines' rows are backed by the manifest's identical SHA-256s."""
    m1 = json.loads((x86 / "model-manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((arm / "model-manifest.json").read_text(encoding="utf-8"))
    if m1 != m2:
        raise ValueError("Different model manifests were used by the two architectures")

    summary: list[dict[str, Any]] = []
    for pipeline in m1["pipelines"]:
        hashes = {
            entry["variant"]: entry["sha256"]
            for entry in m1["files"]
            if entry["pipeline"] == pipeline and entry["variant"] != "tokenizer"
        }
        a = _rows(x86, pipeline, "x86_64")
        b = _rows(arm, pipeline, "aarch64")
        if set(a) != set(b) or set(a) != set(hashes):
            raise ValueError(f"Variant sets do not agree for {pipeline}")
        for variant, expected in hashes.items():
            left, right = a[variant], b[variant]
            if left["model_sha256"] != expected or right["model_sha256"] != expected:
                raise ValueError(f"Model bytes differ across architectures for {pipeline}/{variant}")
            if left["num_samples"] != right["num_samples"]:
                raise ValueError(f"Test sample counts differ for {pipeline}/{variant}")
            summary.append(
                {
                    "pipeline": pipeline,
                    "variant": variant,
                    "sha256": expected,
                    "samples": int(left["num_samples"]),
                    "x86_latency_ms": float(left["latency_ms_mean"]),
                    "arm64_latency_ms": float(right["latency_ms_mean"]),
                    "x86_accuracy": float(left["accuracy"]),
                    "arm64_accuracy": float(right["accuracy"]),
                }
            )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("x86", type=Path)
    parser.add_argument("arm64", type=Path)
    args = parser.parse_args(argv)
    summary = compare_results(args.x86, args.arm64)
    print("Both machines used identical model SHA-256 digests:")
    for result in summary:
        print(
            f"{result['pipeline']}/{result['variant']}  {result['sha256'][:16]} "
            f"x86={result['x86_latency_ms']:.4f}ms "
            f"ARM64={result['arm64_latency_ms']:.4f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
