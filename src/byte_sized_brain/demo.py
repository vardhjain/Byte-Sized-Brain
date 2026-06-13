"""Interactive FP32-vs-INT8 sentiment demo.

Runs the same review text through both the full-precision and the quantized
artifact of an IMDB sentiment model and shows, side by side, the prediction,
confidence, per-inference latency and on-disk size — making the quantization
trade-off tangible. Used by ``bsb demo`` and by the optional Streamlit app.

Requires the artifacts to exist (run ``bsb run <pipeline>`` first).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import Paths, PipelineConfig, load_config
from .utils import size_mb

SENTIMENT = {0: "NEGATIVE", 1: "POSITIVE"}
SUPPORTED = ("distilbert_imdb", "rnn_imdb")
# Full-paragraph reviews — the models trained on long IMDB reviews, so a single
# short sentence is out-of-distribution and reads as low-confidence.
DEFAULT_TEXTS = (
    "This is hands down one of the best films I have seen all year. The performances "
    "were stunning, the cinematography gorgeous, and the story kept me engaged from "
    "start to finish. I would happily watch it again and recommend it to everyone.",
    "What a complete waste of time. The plot made no sense, the acting was wooden, the "
    "dialogue was painful, and I found myself checking my watch every five minutes. "
    "Easily the most boring film I have sat through in years. Avoid at all costs.",
)


@dataclass
class DemoRow:
    text: str
    variant: str
    prediction: str
    confidence: float
    latency_ms: float
    size_mb: float


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _timed(fn, *, repeats: int = 5) -> tuple[object, float]:
    fn()  # warm up
    out = None
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return out, best


def _encode_imdb(text: str, word_index: dict[str, int], num_words: int, max_len: int) -> np.ndarray:
    """Encode raw text the way keras.datasets.imdb encodes its sequences."""
    try:
        from tensorflow.keras.utils import pad_sequences
    except ImportError:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

    index_from = 3  # keras reserves 0=pad, 1=start, 2=oov
    tokens = [1]
    for raw in text.lower().split():
        word = "".join(ch for ch in raw if ch.isalnum())
        if not word:
            continue
        idx = word_index.get(word)
        tokens.append(idx + index_from if idx is not None and idx + index_from < num_words else 2)
    return pad_sequences([tokens], maxlen=max_len).astype(np.float32)[0]


def _demo_distilbert(cfg: PipelineConfig, paths: Paths, texts: list[str]) -> list[DemoRow]:
    from transformers import AutoTokenizer

    from .benchmark import OnnxRunner

    tokenizer = AutoTokenizer.from_pretrained(paths.fp32_source)
    seq_len = cfg.data.max_len or 128
    variants = [("fp32", paths.onnx("fp32")), ("int8", paths.onnx("int8"))]
    runners = {name: (OnnxRunner(p), size_mb(p)) for name, p in variants if p.exists()}

    rows: list[DemoRow] = []
    for text in texts:
        enc = tokenizer(
            text, truncation=True, padding="max_length", max_length=seq_len, return_tensors="np"
        )
        sample = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        for variant, (runner, mb) in runners.items():
            logits, lat = _timed(lambda r=runner, s=sample: r.predict(s))
            probs = _softmax(np.asarray(logits, dtype=np.float64))
            pred = int(np.argmax(probs))
            rows.append(DemoRow(text, variant, SENTIMENT[pred], float(probs[pred]), lat, mb))
    return rows


def _demo_rnn(cfg: PipelineConfig, paths: Paths, texts: list[str]) -> list[DemoRow]:
    from tensorflow.keras.datasets import imdb

    from .benchmark import TFLiteRunner

    num_words = cfg.data.num_words or 10000
    max_len = cfg.data.max_len or 200
    word_index = imdb.get_word_index()

    variants = [("fp32", paths.tflite("fp32")), ("dynamic_range", paths.tflite("dynamic_range"))]
    runners = {name: (TFLiteRunner(p), size_mb(p)) for name, p in variants if p.exists()}

    rows: list[DemoRow] = []
    for text in texts:
        x = _encode_imdb(text, word_index, num_words, max_len)
        for variant, (runner, mb) in runners.items():
            out, lat = _timed(lambda r=runner, xx=x: r.predict(xx))
            p_pos = float(np.asarray(out).ravel()[0])
            pred = 1 if p_pos > 0.5 else 0
            conf = p_pos if pred == 1 else 1.0 - p_pos
            rows.append(DemoRow(text, variant, SENTIMENT[pred], conf, lat, mb))
    return rows


def run_demo(pipeline: str, texts: list[str] | None = None, *, config: str | None = None) -> list[DemoRow]:
    if pipeline not in SUPPORTED:
        raise ValueError(f"demo supports {SUPPORTED}, got {pipeline!r}")
    cfg = load_config(config or pipeline)
    paths = Paths(cfg.name)
    texts = list(texts) if texts else list(DEFAULT_TEXTS)

    expected = paths.onnx("fp32") if pipeline == "distilbert_imdb" else paths.tflite("fp32")
    if not expected.exists():
        raise FileNotFoundError(
            f"No artifacts for '{pipeline}'. Run `bsb run {pipeline}` (or `--smoke`) first."
        )

    if pipeline == "distilbert_imdb":
        return _demo_distilbert(cfg, paths, texts)
    return _demo_rnn(cfg, paths, texts)


def format_rows(rows: list[DemoRow]) -> str:
    lines: list[str] = []
    current: str | None = None
    for r in rows:
        if r.text != current:
            current = r.text
            preview = (r.text[:70] + "...") if len(r.text) > 70 else r.text
            lines.append(f'\n"{preview}"')
            lines.append(f"  {'variant':<14}{'prediction':<11}{'conf':>7}{'latency':>11}{'size':>10}")
            lines.append("  " + "-" * 52)
        lines.append(
            f"  {r.variant:<14}{r.prediction:<11}{r.confidence:>6.1%}"
            f"{r.latency_ms:>9.2f}ms{r.size_mb:>8.2f}MB"
        )
    return "\n".join(lines)
