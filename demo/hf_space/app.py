"""Hugging Face Space for Byte-Sized Brain that judges one movie review at two precisions.

The app is self-contained, so it does not import ``byte_sized_brain``. At startup it
downloads the two DistilBERT ONNX graphs and the tokenizer from a companion Hugging Face
model repo. Each review then runs through the full-precision (FP32) model and the 8-bit
(INT8) model, and the page shows the prediction, confidence, speed and file size of both.

Deploy it with ``demo/hf_space/deploy_hf.py``. That script sets the ``BSB_MODEL_REPO``
Space variable, so a copy deployed under another account loads its own models.

Run it locally with ``pip install -r demo/hf_space/requirements.txt`` and then
``python demo/hf_space/app.py``.
"""

from __future__ import annotations

import os
import time

import gradio as gr
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_REPO = os.environ.get("BSB_MODEL_REPO", "vardhjain20/byte-sized-brain-distilbert-imdb")
PROJECT_URL = "https://github.com/vardhjain/Byte-Sized-Brain"

# The sequence length the model was fine-tuned and benchmarked with
# (configs/distilbert_imdb.yaml). Every review is padded or cut to this length,
# so both models always do the same amount of work.
MAX_TOKENS = 128
# Hard cap on the input text. The model only reads MAX_TOKENS tokens anyway, and the
# cap keeps a pasted novel from slowing the tokenizer down.
MAX_CHARS = 5000
# Below this many words a review is far shorter than the IMDB reviews the model
# learned from, so its answer is less reliable and the page says so.
SHORT_REVIEW_WORDS = 8
# Latency on shared hardware is noisy, so report the fastest of several timed runs.
WARMUP_RUNS = 2
TIMED_RUNS = 10

VARIANTS = {
    "fp32": "Full precision (FP32)",
    "int8": "Quantized (INT8)",
}
LABELS = ("Negative", "Positive")
COLUMNS = ["Model", "Prediction", "Confidence", "Time per review", "File size"]

EXAMPLES = [
    "This is hands down one of the best films I have seen all year. The performances "
    "were stunning, the cinematography gorgeous, and the story kept me engaged from start "
    "to finish. I would happily watch it again and recommend it to everyone.",
    "What a complete waste of time. The plot made no sense, the acting was wooden, the "
    "dialogue was painful, and I found myself checking my watch every five minutes. "
    "Easily the most boring film I have sat through in years.",
    "The first hour is beautiful and the lead actor is wonderful, but the second half "
    "drags, the twist is easy to guess, and the ending undoes a lot of the goodwill. "
    "Worth a look if you are patient.",
]


def _load(variant: str) -> tuple[ort.InferenceSession, float]:
    """Download one ONNX graph and return its session and its real size in MB."""
    path = hf_hub_download(MODEL_REPO, f"distilbert_imdb_{variant}.onnx")
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session, os.path.getsize(path) / 1024**2


tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
MODELS = {variant: _load(variant) for variant in VARIANTS}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _predict(
    session: ort.InferenceSession, feed: dict[str, np.ndarray]
) -> tuple[int, float, float]:
    """Return (label index, confidence, fastest time in ms) for one encoded review."""
    inputs = {i.name: feed[i.name] for i in session.get_inputs()}
    for _ in range(WARMUP_RUNS):
        session.run(None, inputs)
    best_ms = float("inf")
    logits = None
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        logits = session.run(None, inputs)[0]
        best_ms = min(best_ms, (time.perf_counter() - start) * 1000.0)
    probs = _softmax(np.asarray(logits, dtype=np.float64)[0])
    label = int(np.argmax(probs))
    return label, float(probs[label]), best_ms


def _summary(labels: list[int], times: list[float], sizes: list[float], notes: list[str]) -> str:
    fp32_label, int8_label = labels
    if fp32_label == int8_label:
        verdict = f"Both models say **{LABELS[fp32_label].lower()}**."
    else:
        verdict = (
            "The two models **disagree** on this review. Rounding to 8 bits shifts the "
            "scores a little, which can tip a borderline review either way."
        )
    smaller = sizes[0] / sizes[1]
    speedup = times[0] / times[1]
    if speedup >= 1.1:
        speed = f"ran **{speedup:.1f} times faster** on this review"
    elif speedup > 0.9:
        speed = "ran at about the same speed this time"
    else:
        speed = "ran slower this time, which happens on busy shared hardware, so try again"
    parts = [verdict, f"The quantized model is **{smaller:.1f} times smaller** and {speed}."]
    return " ".join(parts + notes)


def classify(review: str) -> tuple[list[list[str]], str]:
    text = (review or "").strip()
    if not text:
        raise gr.Error("Type or paste a movie review first.", title="Nothing to classify")
    text = text[:MAX_CHARS]

    enc = tokenizer(
        text, truncation=True, padding="max_length", max_length=MAX_TOKENS, return_tensors="np"
    )
    feed = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    rows: list[list[str]] = []
    labels: list[int] = []
    times: list[float] = []
    sizes: list[float] = []
    for variant, name in VARIANTS.items():
        session, size = MODELS[variant]
        label, confidence, ms = _predict(session, feed)
        rows.append([name, LABELS[label], f"{confidence:.0%}", f"{ms:.0f} ms", f"{size:.1f} MB"])
        labels.append(label)
        times.append(ms)
        sizes.append(size)

    notes: list[str] = []
    if int(enc["attention_mask"].sum()) >= MAX_TOKENS:
        notes.append(
            f"This review fills the model's {MAX_TOKENS}-token window, so only roughly "
            "the first 100 words were read."
        )
    if len(text.split()) < SHORT_REVIEW_WORDS:
        notes.append(
            "Very short reviews are harder to judge, because the model learned from "
            "full-length IMDB reviews."
        )
    return rows, _summary(labels, times, sizes, notes)


INITIAL_ROWS, INITIAL_SUMMARY = classify(EXAMPLES[0])

INTRO = """
# 🧠 Byte-Sized Brain
### The same AI model at full size and at a quarter of the size

Paste a movie review and two versions of one sentiment model will judge it. The first is
the full-precision original. The second was **quantized**, which means its numbers were
rounded from 32-bit to 8-bit values, so the file is about four times smaller. In the
project's benchmark on 500 test reviews the two versions gave the same answer 95% of the
time and scored 84.6% and 84.0% accuracy, so very little is lost.
"""

FOOTER = f"""
Times are the fastest of {TIMED_RUNS} runs on the free shared CPU this Space runs on, so
they change a little from click to click. The model reads English movie reviews and only
looks at the first {MAX_TOKENS} tokens (roughly 100 words).

This demo is part of [Byte-Sized Brain]({PROJECT_URL}), a project that measures what
quantization does to the size, accuracy, speed and memory of four kinds of neural network.
"""

with gr.Blocks(title="Byte-Sized Brain, FP32 vs INT8") as demo:
    gr.Markdown(INTRO)
    review = gr.Textbox(
        label="Movie review",
        value=EXAMPLES[0],
        placeholder="Paste or type a movie review in English",
        lines=5,
        max_lines=12,
        max_length=MAX_CHARS,
    )
    button = gr.Button("Compare the two models", variant="primary")
    table = gr.Dataframe(
        value=INITIAL_ROWS,
        headers=COLUMNS,
        datatype="str",
        label="Results",
        interactive=False,
    )
    summary = gr.Markdown(INITIAL_SUMMARY)
    # The button is the only event that runs the models, and an event serves one
    # request at a time (the Gradio default), so two visitors never share the CPU
    # while their timings are measured.
    button.click(classify, inputs=review, outputs=[table, summary], api_name="classify")
    # The examples only fill the text box. Without an fn there is nothing to cache,
    # so Spaces never replays stale timings from an example cache.
    gr.Examples(
        examples=[[text] for text in EXAMPLES],
        example_labels=["A rave", "A pan", "Mixed feelings"],
        inputs=review,
        label="Or pick one of these, then press the button",
    )
    gr.Markdown(FOOTER)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
