"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any

import pytest

_BSB_ENV = (
    "BSB_EMULATED",
    "BSB_DEVICE",
    "BSB_ORT_THREADS",
    "BSB_TFLITE_THREADS",
    "BSB_CONFIG_DIR",
)


@pytest.fixture(autouse=True)
def clean_bsb_environment(monkeypatch):
    """Tests must not depend on BSB_* variables set outside (the ARM recipe exports some)."""
    for name in _BSB_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    """Point artifacts/ and results/ at a temp dir so tests don't touch the repo."""
    monkeypatch.setenv("BSB_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BSB_RESULTS", str(tmp_path / "results"))
    return tmp_path


class FakePipelines:
    """Stands in for the registry so CLI tests run the stage logic without any training.

    Each fake keeps the real pipeline's name, quantization and variant list, records
    the stages it was asked to run, and writes empty files where the real one would
    write models.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.configs: list[Any] = []

    def get(self, name: str):
        from byte_sized_brain.benchmark import build_row
        from byte_sized_brain.registry import get_pipeline

        real = get_pipeline(name)
        outer = self

        class _Fake:
            framework = real.framework
            quantization = real.quantization

            def variants(self, cfg, paths):
                return real.variants(cfg, paths)

            def train(self, cfg, paths) -> float:
                outer.calls.append((name, "train"))
                outer.configs.append(cfg)
                paths.fp32_source.mkdir(parents=True, exist_ok=True)
                return 0.5

            def convert(self, cfg, paths):
                outer.calls.append((name, "convert"))
                for v in real.variants(cfg, paths):
                    v.path.write_bytes(b"\0" * 16)
                return real.variants(cfg, paths)

            def benchmark(self, cfg, paths):
                outer.calls.append((name, "benchmark"))
                outer.configs.append(cfg)
                metrics = {
                    "accuracy": 0.9,
                    "latency_ms_mean": 1.0,
                    "latency_ms_p50": 1.0,
                    "latency_ms_p95": 2.0,
                    "rss_delta_mb": 1.0,
                    "peak_rss_mb": 1.0,
                    "num_samples": cfg.benchmark.num_samples,
                }
                return [
                    build_row(
                        cfg,
                        variant=v.name,
                        quantization=v.quantization,
                        runtime=v.runtime,
                        size_mb=0.1,
                        metrics=metrics,
                    )
                    for v in real.variants(cfg, paths)
                ]

        _Fake.name = name
        return _Fake()


@pytest.fixture
def fake_pipelines(monkeypatch) -> FakePipelines:
    from byte_sized_brain import cli

    fakes = FakePipelines()
    monkeypatch.setattr(cli, "get_pipeline", fakes.get)
    monkeypatch.setattr(cli, "seed_everything", lambda *a, **k: 0)  # no framework imports
    return fakes


@pytest.fixture
def cli_errors(monkeypatch) -> list[str]:
    """The one-line messages the CLI logged at error level."""
    from byte_sized_brain import cli

    messages: list[str] = []
    monkeypatch.setattr(cli.log, "error", lambda fmt, *args: messages.append(fmt % args))
    return messages
