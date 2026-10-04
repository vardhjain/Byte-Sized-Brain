"""Report aggregation math, result loading, and README table injection."""

from __future__ import annotations

import pandas as pd

from byte_sized_brain.report import inject_readme_table, load_results, summarize


def _rows(arch: str = "x86_64", emulated: bool = False, latency: float = 10.0) -> list[dict]:
    base = {"modality": "vision", "runtime": "tflite", "arch": arch, "emulated": emulated}
    return [
        {
            "pipeline": "p1",
            "variant": "fp32",
            "size_mb": 4.0,
            "accuracy": 0.90,
            "latency_ms_mean": latency,
            "rss_delta_mb": 20.0,
            **base,
        },
        {
            "pipeline": "p1",
            "variant": "int8",
            "size_mb": 1.0,
            "accuracy": 0.88,
            "latency_ms_mean": latency / 2,
            "rss_delta_mb": 5.0,
            **base,
        },
    ]


def _df() -> pd.DataFrame:
    return pd.DataFrame(_rows())


def test_summarize_trade_offs() -> None:
    s = summarize(_df())
    int8 = s[s["variant"] == "int8"].iloc[0]
    assert int8["size_reduction_%"] == 75.0
    assert int8["latency_speedup_x"] == 2.0
    assert round(int8["accuracy_delta"], 2) == -0.02

    fp32 = s[s["variant"] == "fp32"].iloc[0]
    assert fp32["size_reduction_%"] == 0.0
    assert fp32["latency_speedup_x"] == 1.0


def test_summarize_empty_is_empty() -> None:
    assert summarize(pd.DataFrame()).empty


def test_each_architecture_is_compared_to_its_own_baseline() -> None:
    """An emulated-ARM row must never be scored against the native x86 baseline."""
    df = pd.DataFrame(_rows() + _rows(arch="aarch64", emulated=True, latency=400.0))
    s = summarize(df)
    arm_int8 = s[(s["arch"] == "aarch64") & (s["variant"] == "int8")].iloc[0]
    assert arm_int8["latency_speedup_x"] == 2.0  # 400 / 200, not 10 / 200
    assert bool(arm_int8["emulated"]) is True


def test_load_results_normalizes_arch_and_skips_smoke(tmp_path) -> None:
    pd.DataFrame(_rows(arch="AMD64")).to_csv(tmp_path / "p1.csv", index=False)
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    pd.DataFrame(_rows(arch="AMD64")).to_csv(smoke / "p1.csv", index=False)

    df = load_results(tmp_path)
    assert len(df) == 2  # the smoke/ copy is ignored
    assert set(df["arch"]) == {"x86_64"}
    assert df["emulated"].dtype == bool


def test_load_results_empty_dir(tmp_path) -> None:
    assert load_results(tmp_path).empty


def test_inject_readme_table_replaces_only_the_marked_block(tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n<!-- RESULTS_TABLE -->\nold table\n<!-- /RESULTS_TABLE -->\noutro\n",
        encoding="utf-8",
    )
    assert inject_readme_table("| new |", "x86_64", "docs/images/c.png", readme=readme)
    text = readme.read_text(encoding="utf-8")
    assert "old table" not in text
    assert "| new |" in text and "docs/images/c.png" in text
    assert text.startswith("intro\n") and text.endswith("outro\n")


def test_inject_readme_table_without_markers_is_a_noop(tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n", encoding="utf-8")
    assert inject_readme_table("| new |", "x86_64", readme=readme) is False
    assert readme.read_text(encoding="utf-8") == "no markers here\n"


def test_readme_table_is_plain_language_one_row_per_model() -> None:
    """The README table pairs each model's original and quantized variant in words."""
    from byte_sized_brain.report import _readme_table

    rows = _rows()
    rows[1]["quantization"] = "static_int8"
    rows[0]["quantization"] = "none"
    slow = _rows(latency=1.0)
    for r in slow:
        r["pipeline"] = "ffn_mnist"
    slow[1]["latency_ms_mean"] = 1.5  # quantized variant 1.5x slower
    slow[1]["quantization"] = "static_int8"
    table = _readme_table(pd.DataFrame(rows + slow), "x86_64")
    lines = table.splitlines()
    assert lines[0].startswith("| Model (dataset) | Method |")
    assert "size_mb" not in table and "accuracy_delta" not in table
    assert len(lines) == 4  # header, separator, one row per model
    assert lines[2].startswith("| Digit reader (MNIST) | Static INT8 |")  # known models first
    assert "4.00 → 1.00 MB (75% smaller)" in table
    assert "90.0% → 88.0% (-2.0 pts)" in table
    assert "(2.0× faster)" in table and "(1.5× slower)" in table
