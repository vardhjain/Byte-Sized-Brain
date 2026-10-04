"""Result-row schema, CSV round-trip, and merge-by-architecture semantics."""

from __future__ import annotations

import csv
import json

from byte_sized_brain.benchmark.metrics import RESULT_COLUMNS, build_row, write_results
from byte_sized_brain.config import load_config

_METRICS = {
    "accuracy": 0.9123,
    "latency_ms_mean": 1.234,
    "latency_ms_p50": 1.0,
    "latency_ms_p95": 2.5,
    "rss_delta_mb": 3.0,
    "peak_rss_mb": 4.0,
    "num_samples": 100,
}


def _row(variant: str = "int8", **overrides):
    cfg = load_config("ffn_mnist")
    row = build_row(
        cfg,
        variant=variant,
        quantization="none" if variant == "fp32" else "static_int8",
        runtime="tflite",
        size_mb=0.1,
        metrics=_METRICS,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    row.update(overrides)
    return row


def _read(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_row_has_exactly_the_schema() -> None:
    row = _row()
    assert set(row.keys()) == set(RESULT_COLUMNS)


def test_row_values_and_versions() -> None:
    row = _row()
    assert row["pipeline"] == "ffn_mnist"
    assert row["variant"] == "int8"
    assert row["accuracy"] == 0.9123
    assert isinstance(row["emulated"], bool)
    # lib_versions is JSON-serialisable text
    json.loads(row["lib_versions"])


def test_csv_round_trip(tmp_path) -> None:
    out = write_results([_row("fp32"), _row("int8")], tmp_path / "r.csv")
    rows = _read(out)
    assert len(rows) == 2
    assert list(rows[0].keys()) == RESULT_COLUMNS
    assert rows[0]["pipeline"] == "ffn_mnist"


def test_rerun_on_the_same_arch_replaces_its_rows(tmp_path) -> None:
    out = tmp_path / "r.csv"
    write_results([_row("fp32", arch="x86_64"), _row("int8", arch="x86_64")], out)
    write_results([_row("fp32", arch="x86_64", accuracy=0.5)], out)
    rows = _read(out)
    # The whole (pipeline, arch, emulated) group is replaced, stale variants included.
    assert [(r["variant"], r["accuracy"]) for r in rows] == [("fp32", "0.5")]


def test_other_architectures_survive_a_rerun(tmp_path) -> None:
    out = tmp_path / "r.csv"
    write_results([_row("fp32", arch="x86_64"), _row("int8", arch="x86_64")], out)
    write_results([_row("fp32", arch="aarch64"), _row("int8", arch="aarch64")], out)
    write_results([_row("fp32", arch="aarch64", emulated=True)], out)
    groups = sorted((r["arch"], r["emulated"]) for r in _read(out))
    assert groups == [
        ("aarch64", "False"),
        ("aarch64", "False"),
        ("aarch64", "True"),
        ("x86_64", "False"),
        ("x86_64", "False"),
    ]


def test_windows_and_linux_spellings_count_as_one_arch(tmp_path) -> None:
    out = tmp_path / "r.csv"
    write_results([_row("fp32", arch="AMD64")], out)  # a row written on Windows
    write_results([_row("fp32", arch="x86_64")], out)  # the same ISA, written on Linux
    assert len(_read(out)) == 1
