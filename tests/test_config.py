"""Config loading, smoke-override merge, validation, and path derivation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from byte_sized_brain.config import Paths, PipelineConfig, load_config
from byte_sized_brain.registry import all_names

PIPELINES = all_names()


@pytest.mark.parametrize("name", PIPELINES)
def test_every_config_loads_and_validates(name: str) -> None:
    cfg = load_config(name)
    assert cfg.name == name
    assert cfg.framework in {"tensorflow", "pytorch"}
    assert cfg.modality in {"vision", "sequence", "nlp"}
    assert cfg.benchmark.num_samples > 0


@pytest.mark.parametrize("name", PIPELINES)
def test_smoke_overrides_shrink_the_run(name: str) -> None:
    base = load_config(name)
    smoke = load_config(name, smoke=True)
    # Smoke must never be larger than the base run.
    assert smoke.benchmark.num_samples <= base.benchmark.num_samples
    # The `smoke` key itself must never leak into the validated model.
    assert not hasattr(smoke, "smoke")


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(
            name="x", framework="tensorflow", modality="vision",
            dataset="d", model="m", not_a_field=1,
        )


def test_bad_enum_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(name="x", framework="mxnet", modality="vision", dataset="d", model="m")


def test_paths_layout(tmp_path) -> None:
    p = Paths("demo", root=tmp_path)
    assert p.dir == tmp_path / "demo"
    assert p.dir.is_dir()  # created on init
    assert p.tflite("int8").name == "demo_int8.tflite"
    assert p.onnx("fp32").name == "demo_fp32.onnx"
    assert p.fp32_source.name == "fp32_source"


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist_pipeline")
