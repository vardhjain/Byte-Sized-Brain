"""Typed, validated, YAML-driven configuration for every pipeline.

A single :class:`PipelineConfig` describes how to train, convert and benchmark
one model family. Configs live in ``configs/<name>.yaml`` and are validated with
pydantic so a typo fails loudly instead of silently doing the wrong thing.

Each YAML may carry a ``smoke:`` block whose keys are deep-merged over the base
config when ``--smoke`` is passed. That is how CI runs a 1-epoch, few-sample
version of every pipeline end-to-end in seconds. Smoke runs keep their artifacts
and results in separate ``smoke/`` folders (see :class:`Paths`), so they can never
overwrite a full run.
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
    # Reject unknown keys and re-validate on assignment (the CLI overrides
    # ``benchmark.num_samples``), so a typo or a bad value fails loudly.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataConfig(_Base):
    num_words: int | None = Field(default=None, ge=1)  # IMDB vocabulary cap
    max_len: int | None = Field(default=None, ge=1)  # sequence length (IMDB / DistilBERT)
    img_size: int | None = Field(default=None, ge=32)  # CIFAR resize target (MobileNetV2 minimum)


class TrainConfig(_Base):
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    validation_split: float = Field(default=0.1, ge=0, lt=1)
    # Optional second, lower-LR fine-tuning stage (CNN unfreezes its backbone).
    fine_tune_epochs: int = Field(default=0, ge=0)
    fine_tune_lr: float = Field(default=1e-4, gt=0)
    # Cap the number of samples (None = use the full split). Keeps CPU runs sane.
    train_subset: int | None = Field(default=None, ge=1)
    eval_subset: int | None = Field(default=None, ge=1)
    num_proc: int = Field(default=1, ge=1)  # dataset .map() workers (1 is Windows-safe)


class ConvertConfig(_Base):
    # Must match the technique the pipeline implements (checked by the CLI), so a
    # config can never claim one quantization while the code applies another.
    quantization: Quantization = "static_int8"
    rep_samples: int = Field(default=100, ge=1)  # representative-dataset size for static PTQ
    int8_io: bool = True  # int8 input/output tensors for static PTQ
    opset: int = 14  # ONNX opset (DistilBERT export)


class BenchmarkConfig(_Base):
    num_samples: int = Field(default=1000, ge=1)
    warmup: int = Field(default=5, ge=0)


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
    """Resolve a bare pipeline name (``ffn_mnist``) or an explicit YAML path.

    Anything that looks like a path (it has a directory part or a YAML suffix)
    must exist as given. It never falls back to a bundled config that merely
    shares its file name, because that would silently run a different config
    than the one asked for.
    """
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} or len(p.parts) > 1:
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p
    for d in _config_dirs():
        for suffix in (".yaml", ".yml"):
            cand = d / f"{name_or_path}{suffix}"
            if cand.is_file():
                return cand
    searched = ", ".join(str(d) for d in _config_dirs())
    raise FileNotFoundError(f"No config named '{name_or_path}' found. Searched: {searched}")


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
    """Canonical artifact and result locations for one pipeline.

    ``smoke=True`` nests everything under a ``smoke/`` folder (for example
    ``artifacts/smoke/ffn_mnist/`` and ``benchmarks/results/smoke/ffn_mnist.csv``)
    so a quick smoke run never replaces fully trained models or committed results.
    """

    def __init__(self, name: str, root: Path | None = None, *, smoke: bool = False) -> None:
        self.name = name
        self.smoke = smoke
        base = root or artifacts_root()
        self.dir = (base / "smoke" if smoke else base) / name
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
        out = results_root() / "smoke" if self.smoke else results_root()
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{self.name}.csv"
