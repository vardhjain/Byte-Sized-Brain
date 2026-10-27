"""CNN optional quantization strategies, exercised without heavyweight ML imports."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from byte_sized_brain.cli import main
from byte_sized_brain.config import Paths, load_config
from byte_sized_brain.pipelines.cnn_cifar10 import CNNCifar10


def _opted_in():
    cfg = load_config("cnn_cifar10")
    cfg.convert.cnn_alternatives = ["dynamic_range", "int8_except_depthwise"]
    return cfg


def test_cnn_opt_in_has_four_distinct_named_artifacts(tmp_path: Path) -> None:
    pipeline = CNNCifar10()
    paths = Paths("cnn_cifar10", root=tmp_path)
    assert [v.name for v in pipeline.variants(load_config("cnn_cifar10"), paths)] == [
        "fp32", "int8"
    ]

    variants = pipeline.variants(_opted_in(), paths)
    assert [v.name for v in variants] == [
        "fp32", "int8", "dynamic_range", "int8_except_depthwise"
    ]
    assert [v.quantization for v in variants] == [
        "none", "static_int8", "dynamic_range", "selective_int8"
    ]
    assert len({v.path for v in variants}) == len(variants)
    assert all(v.runtime == "tflite" for v in variants)


def test_cnn_convert_dispatches_opt_in_variants(monkeypatch, tmp_path: Path) -> None:
    from byte_sized_brain.convert import tflite
    from byte_sized_brain.data import cifar10

    calls = []

    def fake_convert(kind, source, path, *args, **kwargs):
        assert source == paths.fp32_source
        Path(path).write_bytes(kind.encode("ascii"))
        calls.append((kind, Path(path).name))
        return Path(path)

    monkeypatch.setattr(
        cifar10,
        "load_split",
        lambda split, img_size, subset: (
            np.zeros((subset, 2, 2, 3), dtype=np.float32),
            np.zeros(subset, dtype=np.int64),
        ),
    )
    for name in ("to_fp32", "to_static_int8", "to_dynamic_range", "to_int8_except_depthwise"):
        monkeypatch.setattr(
            tflite,
            name,
            lambda source, path, *a, _name=name, **k: fake_convert(
                _name, source, path, *a, **k
            ),
        )

    cfg = _opted_in()
    paths = Paths("cnn_cifar10", root=tmp_path)
    variants = CNNCifar10().convert(cfg, paths)

    assert [kind for kind, _ in calls] == [
        "to_fp32", "to_static_int8", "to_dynamic_range", "to_int8_except_depthwise"
    ]
    assert {v.path for v in variants} == {paths.tflite(v.name) for v in variants}
    assert all(v.path.exists() for v in variants)


def test_cli_cnn_alternatives_apply_to_convert_and_benchmark(fake_pipelines, isolated_artifacts) -> None:
    assert main(["run", "cnn_cifar10", "--cnn-alternatives", "--smoke"]) == 0
    assert [kind for name, kind in fake_pipelines.calls if name == "cnn_cifar10"] == [
        "train", "convert", "benchmark"
    ]
    assert all(
        c.convert.cnn_alternatives == ["dynamic_range", "int8_except_depthwise"]
        for c in fake_pipelines.configs
    )


@pytest.mark.parametrize("command", ["convert", "benchmark", "run"])
def test_cli_rejects_cnn_variants_for_other_models(
    command: str, fake_pipelines, cli_errors
) -> None:
    assert main([command, "ffn_mnist", "--cnn-alternatives"]) == 1
    assert fake_pipelines.calls == []
    assert "only to cnn_cifar10" in cli_errors[0]
