"""End-to-end smoke tests: tiny train -> convert -> benchmark for each pipeline.

Marked ``smoke`` (and skipped unless the needed framework is installed). Run with:
    pytest -m smoke
These download datasets and train for one epoch, so they are slow — the default
``pytest -m "not smoke"`` lane skips them.
"""

from __future__ import annotations

import pytest

from byte_sized_brain.config import Paths, load_config
from byte_sized_brain.registry import get_pipeline
from byte_sized_brain.seeding import seed_everything

TF_PIPELINES = ["ffn_mnist", "rnn_imdb", "cnn_cifar10"]
TORCH_PIPELINES = ["distilbert_imdb"]


@pytest.mark.smoke
@pytest.mark.parametrize("name", TF_PIPELINES + TORCH_PIPELINES)
def test_pipeline_end_to_end(name: str, isolated_artifacts) -> None:
    cfg = load_config(name, smoke=True)
    if cfg.framework == "tensorflow":
        pytest.importorskip("tensorflow")
    else:
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

    seed_everything(cfg.seed)
    paths = Paths(cfg.name)
    pipeline = get_pipeline(name)

    pipeline.train(cfg, paths)
    variants = pipeline.convert(cfg, paths)
    for v in variants:
        assert v.path.exists(), f"{v.name} artifact missing"

    rows = pipeline.benchmark(cfg, paths)
    assert len(rows) == len(variants)

    sizes = {r["variant"]: r["size_mb"] for r in rows}
    for r in rows:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert r["size_mb"] > 0
        assert r["latency_ms_mean"] >= 0.0
        assert r["num_samples"] == cfg.benchmark.num_samples

    # Quantization must actually shrink the model.
    for variant, size in sizes.items():
        if variant != "fp32":
            assert size < sizes["fp32"], f"{variant} ({size}MB) not smaller than fp32"
