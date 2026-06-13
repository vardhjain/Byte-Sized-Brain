"""Result-row schema and CSV round-trip."""

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


def _row():
    cfg = load_config("ffn_mnist")
    return build_row(
        cfg, variant="int8", quantization="static_int8", runtime="tflite",
        size_mb=0.1, metrics=_METRICS, timestamp="2026-01-01T00:00:00+00:00",
    )


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
    out = write_results([_row(), _row()], tmp_path / "r.csv")
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert list(rows[0].keys()) == RESULT_COLUMNS
    assert rows[0]["pipeline"] == "ffn_mnist"
