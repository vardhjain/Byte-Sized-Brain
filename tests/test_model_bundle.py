"""No-ML tests for building, transferring and validating identical model bundles."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from byte_sized_brain.benchmark.bundle import (
    file_sha256,
    install_bundle,
    pack_bundle,
    verify_bundle,
)
from byte_sized_brain.benchmark.cross_hardware import compare_results
from byte_sized_brain.config import Paths, load_config
from byte_sized_brain.registry import get_pipeline


def _fake_models(source: Path, pipeline: str = "ffn_mnist") -> None:
    cfg = load_config(pipeline)
    paths = Paths(pipeline, root=source)
    for v in get_pipeline(pipeline).variants(cfg, paths):
        v.path.write_bytes((v.name + "-" + pipeline).encode("utf-8"))
    if pipeline == "distilbert_imdb":
        paths.fp32_source.mkdir()
        (paths.fp32_source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (paths.fp32_source / "tokenizer.json").write_text("{}", encoding="utf-8")
        # Don't upload a large model checkpoint when only a tokenizer is required.
        (paths.fp32_source / "model.safetensors").write_bytes(b"not-in-bundle")


def test_pack_verify_and_install_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fake_models(source)
    bundle = tmp_path / "bundle"
    manifest = pack_bundle("ffn_mnist", bundle, source=source)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(data["files"]) == 2
    assert data["pipelines"] == ["ffn_mnist"]
    assert set(entry["variant"] for entry in data["files"]) == {"fp32", "int8"}
    assert verify_bundle("ffn_mnist", bundle) == data

    target = tmp_path / "destination" / "artifacts"
    install_bundle("ffn_mnist", bundle, destination=target)
    for entry in data["files"]:
        copied = target / Path(entry["path"]).relative_to("artifacts")
        assert copied.is_file()
        assert copied.stat().st_size == entry["bytes"]
        assert file_sha256(copied) == entry["sha256"]


def test_sha256_detects_same_size_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fake_models(source)
    bundle = tmp_path / "bundle"
    pack_bundle("ffn_mnist", bundle, source=source)
    target = bundle / "artifacts" / "ffn_mnist" / "ffn_mnist_fp32.tflite"
    target.write_bytes(b"xxxx")
    with pytest.raises(ValueError, match="SHA-256 mismatch|wrong-size"):
        verify_bundle("ffn_mnist", bundle)


def test_bundle_rejects_pipeline_mismatch_and_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fake_models(source)
    bundle = tmp_path / "bundle"
    path = pack_bundle("ffn_mnist", bundle, source=source)
    with pytest.raises(ValueError, match="pipeline selection"):
        verify_bundle("rnn_imdb", bundle)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "artifacts/../escaped.tflite"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong pipeline|Unsafe"):
        verify_bundle("ffn_mnist", bundle)


def test_pack_rejects_missing_models(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing"):
        pack_bundle("ffn_mnist", tmp_path / "bundle", source=tmp_path / "empty")


def test_distilbert_bundle_contains_tokenizer_but_not_training_weights(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fake_models(source, "distilbert_imdb")
    bundle = tmp_path / "bundle"
    manifest = json.loads(
        pack_bundle("distilbert_imdb", bundle, source=source).read_text(encoding="utf-8")
    )
    files = {entry["path"] for entry in manifest["files"]}
    assert "artifacts/distilbert_imdb/fp32_source/tokenizer.json" in files
    assert not any("safetensors" in path for path in files)
    verify_bundle("distilbert_imdb", bundle)


def _write_cross_results(folder: Path, manifest: dict, arch: str) -> None:
    folder.mkdir()
    (folder / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (folder / "ffn_mnist.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pipeline",
                "arch",
                "variant",
                "model_sha256",
                "num_samples",
                "run_id",
                "accuracy",
                "latency_ms_mean",
            ],
        )
        writer.writeheader()
        for entry in manifest["files"]:
            writer.writerow(
                {
                    "pipeline": "ffn_mnist",
                    "arch": arch,
                    "variant": entry["variant"],
                    "model_sha256": entry["sha256"],
                    "num_samples": "10",
                    "run_id": arch + "-run",
                    "accuracy": "0.9",
                    "latency_ms_mean": "1.0",
                }
            )


def test_cross_hardware_comparison_rejects_changed_model(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fake_models(source)
    manifest = json.loads(
        pack_bundle("ffn_mnist", tmp_path / "bundle", source=source).read_text(
            encoding="utf-8"
        )
    )
    x86, arm = tmp_path / "x86", tmp_path / "arm"
    _write_cross_results(x86, manifest, "x86_64")
    _write_cross_results(arm, manifest, "aarch64")
    assert len(compare_results(x86, arm)) == 2

    rows = list(csv.DictReader((arm / "ffn_mnist.csv").open(encoding="utf-8")))
    rows[0]["model_sha256"] = "0" * 64
    with (arm / "ffn_mnist.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Model bytes differ"):
        compare_results(x86, arm)
