"""Exact checks on the numbers the project publishes.

The harness, the merge of result files and the report all get a fake clock or
hand-built rows here, so a wrong percentile, a dropped filter or a broken merge
changes an asserted value instead of slipping through.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from byte_sized_brain import report
from byte_sized_brain.benchmark import benchmark_inference, harness, write_results
from byte_sized_brain.benchmark.metrics import build_row
from byte_sized_brain.config import _deep_merge, load_config
from byte_sized_brain.demo import DemoRow, format_rows, run_demo


class _Clock:
    """A perf_counter whose every start/stop pair measures the next scripted duration."""

    def __init__(self, durations_ms: list[float]) -> None:
        self._ticks: list[float] = []
        now = 0.0
        for d in durations_ms:
            self._ticks += [now, now + d / 1000.0]
            now += 1.0
        self._i = 0

    def __call__(self) -> float:
        self._i += 1
        return self._ticks[self._i - 1]


def test_latency_statistics_are_exact(monkeypatch) -> None:
    durations = [1.0, 2.0, 3.0, 4.0, 100.0]
    monkeypatch.setattr(harness.time, "perf_counter", _Clock(durations))
    m = benchmark_inference(
        lambda s: s, list(range(5)), list(range(5)), num_samples=5, warmup=0, decision_fn=int
    )
    assert m["latency_ms_mean"] == pytest.approx(22.0)
    assert m["latency_ms_p50"] == pytest.approx(3.0)
    assert m["latency_ms_p95"] == pytest.approx(80.8)  # numpy's linear interpolation
    assert m["predictions"] == [0, 1, 2, 3, 4]


def test_warmup_longer_than_the_data_wraps_around() -> None:
    seen: list[int] = []
    benchmark_inference(
        lambda s: seen.append(s) or s, [0, 1], [0, 1], num_samples=2, warmup=5, decision_fn=int
    )
    assert seen == [0, 1, 0, 1, 0, 0, 1]  # five warm-up calls, then the two timed ones


def test_deep_merge_keeps_sibling_keys() -> None:
    base = {"train": {"epochs": 5, "batch_size": 128}, "seed": 42}
    merged = _deep_merge(base, {"train": {"epochs": 1}})
    assert merged == {"train": {"epochs": 1, "batch_size": 128}, "seed": 42}
    assert base["train"]["epochs"] == 5  # the input is not modified


def test_smoke_profile_overrides_only_what_it_names() -> None:
    full, smoke = load_config("ffn_mnist"), load_config("ffn_mnist", smoke=True)
    assert smoke.train.epochs < full.train.epochs
    assert smoke.train.batch_size == full.train.batch_size
    assert smoke.convert.quantization == full.convert.quantization


def _result_rows(arch: str, emulated: bool, latency: float) -> list[dict]:
    cfg = load_config("ffn_mnist")
    rows = []
    for variant, quantization, size, lat in (
        ("fp32", "none", 4.0, latency),
        ("int8", "static_int8", 1.0, latency / 2),
    ):
        metrics = {
            "accuracy": 0.9,
            "latency_ms_mean": lat,
            "latency_ms_p50": lat,
            "latency_ms_p95": lat,
            "rss_delta_mb": size * 2,
            "peak_rss_mb": size * 2,
            "num_samples": 1000,
            "agreement": 1.0,
            "threads": 1,
        }
        row = build_row(
            cfg,
            variant=variant,
            quantization=quantization,
            runtime="tflite",
            size_mb=size,
            metrics=metrics,
        )
        row.update(arch=arch, emulated=emulated)
        rows.append(row)
    return rows


def test_generate_report_end_to_end(tmp_path, monkeypatch) -> None:
    """Runs the real report on a native and an emulated group and checks every output."""
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    write_results(_result_rows("x86_64", False, 10.0), results / "ffn_mnist.csv")
    write_results(_result_rows("aarch64", True, 777.0), results / "ffn_mnist.csv")
    (tmp_path / "README.md").write_text(
        "top\n<!-- RESULTS_TABLE -->\nstale\n<!-- /RESULTS_TABLE -->\nbottom\n", encoding="utf-8"
    )

    out = report.generate_report(results)

    assert (
        out == tmp_path / "docs" / "report.md" or out.resolve() == tmp_path / "docs" / "report.md"
    )
    assert (tmp_path / "docs" / "images" / "results_dashboard.png").stat().st_size > 1000
    text = (tmp_path / "docs" / "report.md").read_text(encoding="utf-8")
    assert "aarch64" in text and "x86_64" in text  # the detail table keeps both groups
    assert "nan" not in text.lower()

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "stale" not in readme and readme.startswith("top\n") and readme.endswith("bottom\n")
    assert "4.00 → 1.00 MB (75% smaller)" in readme
    assert "10.0 → 5.00 ms (2.0× faster)" in readme
    assert "777" not in readme  # emulated rows never reach the headline table
    assert "native x86_64" in readme


def test_speed_wording_treats_small_differences_as_noise() -> None:
    assert report._speed_words(10.0, 9.0) == "about the same"
    assert report._speed_words(10.0, 11.0) == "about the same"
    assert report._speed_words(10.0, 5.0) == "2.0× faster"
    assert report._speed_words(5.0, 10.0) == "2.0× slower"


def test_wilson_interval_matches_the_textbook_value() -> None:
    # 85% of 1000 gives roughly plus or minus 2.2 points.
    assert 100 * report.wilson_halfwidth(0.85, 1000) == pytest.approx(2.2, abs=0.05)
    assert pd.isna(report.wilson_halfwidth(0.5, 0))


def test_merging_into_a_foreign_csv_is_refused_clearly(tmp_path) -> None:
    out = tmp_path / "r.csv"
    out.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        write_results(_result_rows("x86_64", False, 1.0), out)
    assert out.read_text(encoding="utf-8") == "a,b\n1,2\n"  # left untouched


def test_seeding_without_frameworks_imports_none() -> None:
    """Checked in a fresh interpreter, since this process may already have them loaded."""
    code = (
        "import sys; from byte_sized_brain.seeding import seed_everything; "
        "seed_everything(1, frameworks=()); "
        "print(sorted(m for m in ('tensorflow', 'torch') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_format_rows_layout_is_stable() -> None:
    out = format_rows([DemoRow("great movie", "int8", "POSITIVE", 0.973, 25.0, 64.27)])
    assert out.splitlines()[-1] == "  int8          POSITIVE    97.3%    25.00ms   64.27MB"


def test_format_rows_shortens_long_reviews() -> None:
    out = format_rows([DemoRow("x" * 100, "fp32", "NEGATIVE", 0.5, 1.0, 1.0)])
    assert '"' + "x" * 70 + '..."' in out


def test_demo_rejects_a_config_for_another_pipeline(isolated_artifacts) -> None:
    with pytest.raises(ValueError, match="distilbert_imdb"):
        run_demo("rnn_imdb", ["x"], config="configs/distilbert_imdb.yaml")


def test_demo_without_models_leaves_no_folders_behind(isolated_artifacts) -> None:
    with pytest.raises(FileNotFoundError):
        run_demo("rnn_imdb", ["x"], smoke=True)
    assert not (isolated_artifacts / "artifacts").exists()
