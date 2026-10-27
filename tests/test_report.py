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


def test_run_id_blocks_cross_device_baseline_pairing() -> None:
    """Identical architectures can still refer to unrelated experiments."""
    first = _rows(latency=10.0)
    second = _rows(latency=100.0)
    for r in first:
        r.update(run_id="pi-run", device="Raspberry Pi", timestamp="2026-10-01T10:00:00Z")
    for r in second:
        r.update(run_id="graviton-run", device="Graviton", timestamp="2026-10-02T10:00:00Z")
    # Deliberately interleave FP32/INT8 rows from different devices.
    df = pd.DataFrame([first[0], second[0], first[1], second[1]])
    summary = summarize(df)
    assert list(summary[summary["variant"] == "int8"]["latency_speedup_x"]) == [2.0, 2.0]
    assert set(summary["run_id"]) == {"pi-run", "graviton-run"}


def test_featured_run_uses_latest_complete_pair_and_skips_partial() -> None:
    from byte_sized_brain.report import _featured_pairs, _readme_table

    old = _rows(latency=10.0)
    recent = _rows(latency=100.0)
    partial = _rows(latency=2000.0)[:1]
    for rows, run_id, stamp in (
        (old, "old", "2026-10-01"),
        (recent, "recent", "2026-10-02"),
        (partial, "partial", "2026-10-03"),
    ):
        for r in rows:
            r.update(run_id=run_id, timestamp=stamp, device=run_id)
    df = pd.DataFrame([old[0], recent[0], partial[0], old[1], recent[1]])
    pairs = _featured_pairs(df, "x86_64")
    assert len(pairs) == 1
    _, fp32, int8 = pairs[0]
    assert fp32["run_id"] == int8["run_id"] == "recent"
    table = _readme_table(df, "x86_64")
    assert "100.0 → 50.0 ms (2.0× faster)" in table
    assert "2000" not in table


def test_featured_architecture_does_not_depend_on_run_count() -> None:
    from byte_sized_brain.report import headline_arch

    x86 = _rows(arch="x86_64")
    arm = _rows(arch="aarch64")
    reruns = []
    for i in range(20):
        for row in arm:
            reruns.append({**row, "run_id": f"arm-{i}"})
    assert headline_arch(pd.DataFrame(x86 + arm + reruns)) == "x86_64"


def test_cnn_alternatives_do_not_replace_static_int8_in_headline() -> None:
    from byte_sized_brain.report import _featured_pairs, _paired, _readme_table

    rows = _rows(latency=10.0)
    for row in rows:
        row.update(pipeline="cnn_cifar10", run_id="cnn-experiment", timestamp="2026-10-26")
    rows[1]["quantization"] = "static_int8"
    dynamic = {
        **rows[1],
        "variant": "dynamic_range",
        "quantization": "dynamic_range",
        "accuracy": 0.91,
        "latency_ms_mean": 2.0,
    }
    selective = {
        **rows[1],
        "variant": "int8_except_depthwise",
        "quantization": "selective_int8",
        "accuracy": 0.89,
        "latency_ms_mean": 3.0,
    }
    # Extra quantized variants precede the default INT8 row on purpose.
    df = pd.DataFrame([rows[0], dynamic, selective, rows[1]])
    summary = summarize(df)
    assert set(summary["variant"]) == {
        "fp32",
        "int8",
        "dynamic_range",
        "int8_except_depthwise",
    }
    assert len({r["run_id"] for _, r in summary.iterrows()}) == 1
    featured = _featured_pairs(df, "x86_64")
    assert featured[0][2]["variant"] == "int8"
    assert _paired(df, "x86_64").iloc[0]["q_lat"] == 5.0
    assert "10.0 → 5.00 ms" in _readme_table(df, "x86_64")


def test_e2e_latency_speedup_requires_same_run_and_both_measurements() -> None:
    df_rows = _rows(latency=10.0)
    for row in df_rows:
        row.update(run_id="paired-run")
    df_rows[0].update(
        e2e_latency_ms_mean=16.0,
        input_prep_latency_ms_mean=6.0,
        e2e_latency_ms_p50=16.0,
        e2e_latency_ms_p95=17.0,
    )
    df_rows[1].update(
        e2e_latency_ms_mean=8.0,
        input_prep_latency_ms_mean=3.0,
        e2e_latency_ms_p50=8.0,
        e2e_latency_ms_p95=9.0,
    )
    summary = summarize(pd.DataFrame(df_rows))
    quant = summary.loc[summary["variant"] == "int8"].iloc[0]
    assert quant["latency_speedup_x"] == 2.0
    assert quant["e2e_latency_speedup_x"] == 2.0
    assert quant["e2e_latency_ms_mean"] == 8.0
    assert quant["input_prep_latency_ms_mean"] == 3.0
    assert quant["e2e_latency_ms_ci95_low"] == ""

    # A newer variant from another run must not use the prior FP32 e2e baseline.
    foreign = {**df_rows[1], "run_id": "different-run"}
    extra = summarize(pd.DataFrame([df_rows[0], foreign]))
    assert extra.loc[extra["variant"] == "int8", "e2e_latency_speedup_x"].iloc[0] == ""


def test_historical_rows_do_not_claim_end_to_end_latency() -> None:
    summary = summarize(_df())
    assert all(v == "" for v in summary["e2e_latency_ms_mean"])
    assert all(v == "" for v in summary["e2e_latency_speedup_x"])
