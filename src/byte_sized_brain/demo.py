"""Interactive FP32-vs-INT8 sentiment demo.

Runs the same review text through both the full-precision and the quantized
artifact of an IMDB sentiment model and shows, side by side, the prediction,
confidence, per-inference latency and on-disk size, which makes the quantization
trade-off tangible. Used by ``bsb demo`` and by the optional Streamlit app.

Requires the artifacts to exist (run ``bsb run <pipeline>`` first, or
``bsb run <pipeline> --smoke`` and then pass ``smoke=True``).
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass

import numpy as np

from .config import Paths, PipelineConfig, load_config
from .utils import size_mb

SENTIMENT = {0: "NEGATIVE", 1: "POSITIVE"}
SUPPORTED = ("distilbert_imdb", "rnn_imdb")
# Full-paragraph reviews. The models trained on long IMDB reviews, so a single
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


def _timed(fn, *, repeats: int = 15) -> tuple[object, float]:
    # Report the best of several runs: a single timing is dominated by CPU
    # contention/cold-start noise, which can even make INT8 look slower than FP32.
    for _ in range(3):  # warm up (kernels, dynamic-quant scales, caches)
        fn()
    out = None
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return out, best


# The punctuation the Keras tokenizer replaced with spaces when the IMDB vocabulary
# was built. The apostrophe is deliberately absent, so "don't" and "wasn't" stay
# single vocabulary words, and a hyphen splits "well-made" into "well" and "made".
_KERAS_FILTERS = str.maketrans(dict.fromkeys('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', " "))


def _imdb_token_ids(text: str, word_index: dict[str, int], num_words: int) -> list[int]:
    """Token ids for raw text, matching how keras.datasets.imdb encodes reviews."""
    index_from = 3  # keras reserves 0=pad, 1=start, 2=oov
    tokens = [1]
    for word in text.lower().translate(_KERAS_FILTERS).split():
        idx = word_index.get(word)
        tokens.append(idx + index_from if idx is not None and idx + index_from < num_words else 2)
    return tokens


def _encode_imdb(text: str, word_index: dict[str, int], num_words: int, max_len: int) -> np.ndarray:
    """Padded float32 model input for one review (see :func:`_imdb_token_ids`)."""
    from .data.imdb import pad_sequences

    tokens = _imdb_token_ids(text, word_index, num_words)
    return pad_sequences([tokens], maxlen=max_len).astype(np.float32)[0]


@functools.lru_cache(maxsize=8)
def _onnx_runner(path: str):
    from .benchmark import OnnxRunner

    return OnnxRunner(path)


@functools.lru_cache(maxsize=8)
def _tflite_runner(path: str):
    from .benchmark import TFLiteRunner

    return TFLiteRunner(path)


@functools.lru_cache(maxsize=4)
def _tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


@functools.lru_cache(maxsize=1)
def _word_index() -> dict[str, int]:
    from tensorflow.keras.datasets import imdb

    return imdb.get_word_index()


def _demo_distilbert(cfg: PipelineConfig, paths: Paths, texts: list[str]) -> list[DemoRow]:
    # Loaders are cached, so the web demo does not reload the models on every click.
    tokenizer = _tokenizer(str(paths.fp32_source))
    seq_len = cfg.data.max_len or 128
    variants = [("fp32", paths.onnx("fp32")), ("int8", paths.onnx("int8"))]
    runners = {name: (_onnx_runner(str(p)), size_mb(p)) for name, p in variants if p.exists()}

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
    num_words = cfg.data.num_words or 10000
    max_len = cfg.data.max_len or 200
    word_index = _word_index()

    variants = [("fp32", paths.tflite("fp32")), ("dynamic_range", paths.tflite("dynamic_range"))]
    runners = {name: (_tflite_runner(str(p)), size_mb(p)) for name, p in variants if p.exists()}

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


def run_demo(
    pipeline: str,
    texts: list[str] | None = None,
    *,
    config: str | None = None,
    smoke: bool = False,
) -> list[DemoRow]:
    """Classify ``texts`` with every available variant of ``pipeline``.

    ``smoke=True`` loads the smoke config and the ``artifacts/smoke/`` models, so
    the text is encoded with the same sequence length and vocabulary the smoke
    model was trained with.
    """
    if pipeline not in SUPPORTED:
        raise ValueError(f"demo supports {SUPPORTED}, got {pipeline!r}")
    cfg = load_config(config or pipeline, smoke=smoke)
    if cfg.name != pipeline:
        raise ValueError(
            f"{config} configures the {cfg.name!r} pipeline, but the demo was asked "
            f"for {pipeline!r}."
        )
    paths = Paths(cfg.name, smoke=smoke, create=False)
    texts = list(texts) if texts else list(DEFAULT_TEXTS)

    expected = paths.onnx("fp32") if pipeline == "distilbert_imdb" else paths.tflite("fp32")
    if not expected.exists():
        flag = " --smoke" if smoke else ""
        raise FileNotFoundError(
            f"No artifacts for '{pipeline}' in {paths.dir}. Run `bsb run {pipeline}{flag}` first."
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
            lines.append(
                f"  {'variant':<14}{'prediction':<11}{'conf':>7}{'latency':>11}{'size':>10}"
            )
            lines.append("  " + "-" * 52)
        lines.append(
            f"  {r.variant:<14}{r.prediction:<11}{r.confidence:>6.1%}"
            f"{r.latency_ms:>9.2f}ms{r.size_mb:>8.2f}MB"
        )
    return "\n".join(lines)
