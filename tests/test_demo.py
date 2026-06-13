"""Framework-free unit tests for the demo helpers."""

from __future__ import annotations

import numpy as np
import pytest

from byte_sized_brain.demo import SUPPORTED, DemoRow, _softmax, format_rows, run_demo


def test_softmax_is_a_distribution() -> None:
    p = _softmax(np.array([2.0, 1.0, 0.1]))
    assert pytest.approx(p.sum()) == 1.0
    assert np.argmax(p) == 0
    assert (p >= 0).all()


def test_format_rows_groups_by_text_and_shows_fields() -> None:
    rows = [
        DemoRow("great movie", "fp32", "POSITIVE", 0.97, 50.0, 255.5),
        DemoRow("great movie", "int8", "POSITIVE", 0.96, 25.0, 64.3),
    ]
    out = format_rows(rows)
    assert "great movie" in out
    assert out.count("POSITIVE") == 2
    assert "fp32" in out and "int8" in out
    assert "97" in out  # confidence rendered


def test_run_demo_rejects_unknown_pipeline() -> None:
    assert "distilbert_imdb" in SUPPORTED and "rnn_imdb" in SUPPORTED
    with pytest.raises(ValueError):
        run_demo("cnn_cifar10", ["x"])
