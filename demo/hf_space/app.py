"""HuggingFace Space (Gradio): FP32 vs INT8 DistilBERT sentiment demo for Byte-Sized Brain.

Self-contained (no byte_sized_brain dependency). It downloads the two ONNX graphs and the
tokenizer from a companion HuggingFace model repo, then runs a movie review through both
precisions side by side. Deploy with demo/hf_space/deploy_hf.py.
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
SENTIMENT = {0: "NEGATIVE", 1: "POSITIVE"}
SIZES_MB = {"fp32": 255.5, "int8": 64.3}
MAX_LEN = 128
DEFAULT_REVIEW = (
    "This is hands down one of the best films I have seen all year. The performances "
    "were stunning and the story kept me engaged from start to finish."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
sessions = {
    name: ort.InferenceSession(
        hf_hub_download(MODEL_REPO, f"distilbert_imdb_{name}.onnx"),
        providers=["CPUExecutionProvider"],
    )
    for name in ("fp32", "int8")
}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def classify(review: str):
    enc = tokenizer(
        review or "", truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="np"
    )
    feed = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    rows = []
    for name, sess in sessions.items():
        inputs = {i.name: feed[i.name] for i in sess.get_inputs()}
        for _ in range(3):  # warm up
            sess.run(None, inputs)
        best, logits = float("inf"), None
        for _ in range(12):  # best of several, latency is noisy
            t0 = time.perf_counter()
            logits = sess.run(None, inputs)
            best = min(best, (time.perf_counter() - t0) * 1000.0)
        probs = _softmax(np.asarray(logits[0][0], dtype=np.float64))
        pred = int(np.argmax(probs))
        rows.append(
            [name.upper(), SENTIMENT[pred], f"{probs[pred]:.0%}", f"{best:.0f} ms", f"{SIZES_MB[name]:.0f} MB"]
        )
    return rows


with gr.Blocks(title="Byte-Sized Brain FP32 vs INT8", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🧠 Byte-Sized Brain\n"
        "### FP32 vs INT8 sentiment, same review at two precisions\n"
        "Quantizing this DistilBERT model makes it about 4 times smaller and noticeably "
        "faster, with the same prediction. Part of "
        "[Byte-Sized Brain](https://github.com/vardhjain/Byte-Sized-Brain)."
    )
    review = gr.Textbox(label="Movie review", value=DEFAULT_REVIEW, lines=4)
    button = gr.Button("Classify", variant="primary")
    table = gr.Dataframe(
        headers=["variant", "prediction", "confidence", "latency", "size"],
        datatype=["str", "str", "str", "str", "str"],
        label="FP32 vs INT8",
        interactive=False,
    )
    button.click(classify, inputs=review, outputs=table)
    gr.Examples(
        examples=[
            [DEFAULT_REVIEW],
            ["A boring, predictable mess. The acting was wooden and I kept checking my watch."],
        ],
        inputs=review,
    )

if __name__ == "__main__":
    demo.launch()
