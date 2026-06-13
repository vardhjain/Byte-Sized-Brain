"""Shared pytest fixtures / helpers."""

from __future__ import annotations

import importlib.util

import pytest


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


requires_tf = pytest.mark.skipif(not _have("tensorflow"), reason="TensorFlow not installed")
requires_torch = pytest.mark.skipif(
    not (_have("torch") and _have("transformers")), reason="torch/transformers not installed"
)


@pytest.fixture
def isolated_artifacts(tmp_path, monkeypatch):
    """Point artifacts/ and results/ at a temp dir so tests don't touch the repo."""
    monkeypatch.setenv("BSB_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BSB_RESULTS", str(tmp_path / "results"))
    return tmp_path
