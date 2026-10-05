"""Pipeline contracts and the shared variant benchmark loop, without any ML framework."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from byte_sized_brain.config import Paths, load_config
from byte_sized_brain.pipelines import base
from byte_sized_brain.pipelines.base import Variant, benchmark_variants
from byte_sized_brain.registry import all_names, get_pipeline


class _EchoRunner:
    """Replaces TFLiteRunner so the loop can be exercised on dummy files."""

    def __init__(self, path) -> None:
        self.path = path

    threads = 1

    def prepare(self, sample):
        return sample

    def invoke(self, sample):
        return sample


@pytest.mark.parametrize("name", all_names())
def test_variants_are_an_fp32_baseline_plus_the_advertised_quantization(
    name: str, tmp_path: Path
) -> None:
    pipeline = get_pipeline(name)
    paths = Paths(name, root=tmp_path)
    variants = pipeline.variants(load_config(name), paths)

    assert (variants[0].name, variants[0].quantization) == ("fp32", "none")
    assert [v.quantization for v in variants[1:]] == [pipeline.quantization]
    assert len({v.path for v in variants}) == len(variants)
    assert all(v.path.parent == paths.dir for v in variants)
    expected_runtime = "onnxruntime" if pipeline.framework == "pytorch" else "tflite"
    assert {v.runtime for v in variants} == {expected_runtime}


def test_unknown_pipeline_error_lists_the_choices() -> None:
    with pytest.raises(KeyError, match="ffn_mnist"):
        get_pipeline("nope")


def test_benchmark_variants_produces_one_scored_row_per_variant(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(base, "TFLiteRunner", _EchoRunner)
    monkeypatch.setattr(
        base, "measure_memory", lambda *a, **k: {"rss_delta_mb": 1.0, "peak_rss_mb": 2.0}
    )
    cfg = load_config("ffn_mnist")
    cfg.benchmark.num_samples, cfg.benchmark.warmup = 4, 1
    paths = Paths("ffn_mnist", root=tmp_path)
    variants = get_pipeline("ffn_mnist").variants(cfg, paths)
    for variant, size in zip(variants, (4096, 1024), strict=True):
        variant.path.write_bytes(b"\0" * size)

    rows = benchmark_variants(cfg, paths, variants, [0, 1, 2, 3], [0, 1, 9, 9], decision_fn=int)

    assert [r["variant"] for r in rows] == ["fp32", "int8"]
    assert [r["accuracy"] for r in rows] == [0.5, 0.5]
    assert [r["size_mb"] for r in rows] == [round(4096 / 1024**2, 4), round(1024 / 1024**2, 4)]
    assert all(r["num_samples"] == 4 for r in rows)
    assert [r["agreement"] for r in rows] == [1.0, 1.0]
    assert all(r["threads"] == 1 and r["rss_delta_mb"] == 1.0 for r in rows)


@pytest.mark.parametrize(
    ("smoke", "command"),
    [(False, "bsb convert ffn_mnist`"), (True, "bsb convert ffn_mnist --smoke`")],
)
def test_missing_artifact_names_the_command_that_builds_it(
    tmp_path: Path, smoke: bool, command: str
) -> None:
    cfg = load_config("ffn_mnist", smoke=smoke)
    paths = Paths("ffn_mnist", root=tmp_path, smoke=smoke)
    missing = [Variant("fp32", "none", "tflite", paths.tflite("fp32"))]
    with pytest.raises(FileNotFoundError, match=re.escape(command)):
        benchmark_variants(cfg, paths, missing, [0], [0], decision_fn=int)
