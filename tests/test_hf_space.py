"""Offline checks for the hosted Gradio demo in demo/hf_space/app.py.

The Hub download, the tokenizer and ONNX Runtime are replaced with small fakes, so the
test needs no network and no trained model. It catches Gradio API changes (the app is
built for real) and checks the input guards and the real-file-size column.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

gr = pytest.importorskip("gradio")
hub = pytest.importorskip("huggingface_hub")
ort = pytest.importorskip("onnxruntime")
transformers = pytest.importorskip("transformers")

APP = Path(__file__).resolve().parents[1] / "demo" / "hf_space" / "app.py"
MB = 1024**2


class _FakeTokenizer:
    def __call__(self, text, *, truncation, padding, max_length, return_tensors):
        n = min(len(text.split()) + 2, max_length)
        mask = np.zeros((1, max_length), dtype=np.int64)
        mask[0, :n] = 1
        return {"input_ids": np.zeros((1, max_length), dtype=np.int64), "attention_mask": mask}


class _FakeSession:
    def __init__(self, path, providers=None):
        self.path = path

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in ("input_ids", "attention_mask")]

    def run(self, _outputs, feed):
        assert all(v.dtype == np.int64 for v in feed.values())
        return [np.array([[0.1, 2.0]], dtype=np.float32)]


@pytest.fixture
def app(monkeypatch, tmp_path):
    def fake_download(repo_id, filename, **_):
        path = tmp_path / filename
        path.write_bytes(b"\0" * (4 * MB if "fp32" in filename else MB))
        return str(path)

    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer()
    )
    spec = importlib.util.spec_from_file_location("hf_space_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_builds(app):
    assert isinstance(app.demo, gr.Blocks)
    names = [d.get("api_name") for d in app.demo.get_config_file()["dependencies"]]
    assert "classify" in names


def test_sizes_come_from_the_downloaded_files(app):
    rows, summary = app.classify(app.EXAMPLES[0])
    assert [r[-1] for r in rows] == ["4.0 MB", "1.0 MB"]
    assert "4.0 times smaller" in summary


@pytest.mark.parametrize("text", ["", "   \n ", None])
def test_empty_review_is_rejected(app, text):
    with pytest.raises(gr.Error):
        app.classify(text)


def test_long_review_is_flagged_as_cut(app):
    _, summary = app.classify("a moving and beautiful film " * 400)
    assert "first 100 words" in summary
