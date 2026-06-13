"""Typed, validated, YAML-driven configuration for every pipeline.

A single :class:`PipelineConfig` describes how to train, convert and benchmark
one model family. Configs live in ``configs/<name>.yaml`` and are validated with
pydantic so a typo fails loudly instead of silently doing the wrong thing.

Each YAML may carry a ``smoke:`` block whose keys are deep-merged over the base
config when ``--smoke`` is passed — that is how CI runs a 1-epoch, few-sample
version of every pipeline end-to-end in seconds.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Framework = Literal["tensorflow", "pytorch"]
Modality = Literal["vision", "sequence", "nlp"]
Quantization = Literal["static_int8", "dynamic_range", "dynamic_int8"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown keys → catch typos


class DataConfig(_Base):
    num_words: int | None = None  # IMDB vocabulary cap
    max_len: int | None = None  # sequence length (IMDB / DistilBERT)
    img_size: int | None = None  # CIFAR resize target


class TrainConfig(_Base):
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-3
    validation_split: float = 0.1
    # Optional second, lower-LR fine-tuning stage (CNN unfreezes its backbone).
    fine_tune_epochs: int = 0
    fine_tune_lr: float = 1e-4
    # Cap the number of samples (None = use the full split). Keeps CPU runs sane.
    train_subset: int | None = None
    eval_subset: int | None = None
    num_proc: int = 1  # dataset .map() workers (1 is Windows-safe)


class ConvertConfig(_Base):
    quantization: Quantization = "static_int8"
    rep_samples: int = 100  # representative-dataset size for static PTQ
    flex_ops: bool = False  # enable SELECT_TF_OPS (needed for the LSTM)
    int8_io: bool = True  # int8 input/output tensors for static PTQ
    opset: int = 14  # ONNX opset (DistilBERT export)


class BenchmarkConfig(_Base):
    num_samples: int = 1000
    warmup: int = 5


class PipelineConfig(_Base):
    name: str
    framework: Framework
    modality: Modality
    dataset: str
    model: str
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    convert: ConvertConfig = Field(default_factory=ConvertConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


# ─────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────
def _config_dirs() -> list[Path]:
    """Directories searched for ``<name>.yaml``, most specific first."""
    dirs: list[Path] = []
    if env := os.environ.get("BSB_CONFIG_DIR"):
        dirs.append(Path(env))
    dirs.append(Path.cwd() / "configs")
    # Repo root relative to this file: src/byte_sized_brain/config.py -> repo/
    dirs.append(Path(__file__).resolve().parents[2] / "configs")
    return dirs


def resolve_config_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    stem = p.stem if p.suffix else name_or_path
    for d in _config_dirs():
        cand = d / f"{stem}.yaml"
        if cand.exists():
            return cand
    searched = ", ".join(str(d) for d in _config_dirs())
    raise FileNotFoundError(
        f"No config '{stem}.yaml' found. Searched: {searched}"
    )


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name_or_path: str, *, smoke: bool = False) -> PipelineConfig:
    """Load and validate a pipeline config, applying the ``smoke`` overrides."""
    path = resolve_config_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    smoke_overrides = raw.pop("smoke", {}) or {}
    if smoke:
        raw = _deep_merge(raw, smoke_overrides)
    return PipelineConfig(**raw)


# ─────────────────────────────────────────────────────────────────────────
# Artifact paths (everything regenerable lives under artifacts/<name>/)
# ─────────────────────────────────────────────────────────────────────────
def artifacts_root() -> Path:
    return Path(os.environ.get("BSB_ARTIFACTS", "artifacts"))


def results_root() -> Path:
    return Path(os.environ.get("BSB_RESULTS", "benchmarks/results"))


class Paths:
    """Canonical artifact locations for one pipeline."""

    def __init__(self, name: str, root: Path | None = None) -> None:
        self.name = name
        self.dir = (root or artifacts_root()) / name
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def fp32_source(self) -> Path:
        """TF SavedModel dir, or HuggingFace model dir, the converters read from."""
        return self.dir / "fp32_source"

    def tflite(self, variant: str) -> Path:
        return self.dir / f"{self.name}_{variant}.tflite"

    def onnx(self, variant: str) -> Path:
        return self.dir / f"{self.name}_{variant}.onnx"

    @property
    def results_csv(self) -> Path:
        out = results_root()
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{self.name}.csv"
