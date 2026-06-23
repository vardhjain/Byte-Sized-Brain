"""HuggingFace Space: FP32 vs INT8 DistilBERT sentiment demo for Byte-Sized Brain.

Self-contained (no byte_sized_brain dependency). It downloads the two ONNX graphs
and the tokenizer from a companion HuggingFace model repo, then runs a movie review
through both precisions side by side. Deploy with demo/hf_space/deploy_hf.py.
"""

from __future__ import annotations

import os
import time

import numpy as np
import onnxruntime as ort
import streamlit as st
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_REPO = os.environ.get("BSB_MODEL_REPO", "vardhjain/byte-sized-brain-distilbert-imdb")
SENTIMENT = {0: "NEGATIVE", 1: "POSITIVE"}
MAX_LEN = 128
DEFAULT_REVIEW = (
    "This is hands down one of the best films I have seen all year. The performances "
    "were stunning and the story kept me engaged from start to finish."
)


@st.cache_resource(show_spinner="Downloading the models (one time)...")
def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    variants = {}
    for name, size_mb in (("fp32", 255.5), ("int8", 64.3)):
        path = hf_hub_download(MODEL_REPO, f"distilbert_imdb_{name}.onnx")
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        variants[name] = (sess, size_mb)
    return tokenizer, variants


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def run_variant(sess, encoding):
    feed = {i.name: encoding[i.name] for i in sess.get_inputs()}
    for _ in range(3):  # warm up
        sess.run(None, feed)
    best, logits = float("inf"), None
    for _ in range(12):  # best of several, latency is noisy
        t0 = time.perf_counter()
        logits = sess.run(None, feed)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    probs = _softmax(np.asarray(logits[0][0], dtype=np.float64))
    return int(np.argmax(probs)), float(np.max(probs)), best


st.set_page_config(page_title="Byte-Sized Brain FP32 vs INT8", page_icon="🧠")
st.title("🧠 Byte-Sized Brain")
st.subheader("FP32 vs INT8 sentiment, same review at two precisions")
st.caption(
    "Quantizing this DistilBERT model makes it about 4 times smaller and noticeably "
    "faster, with the same prediction. Part of "
    "[Byte-Sized Brain](https://github.com/vardhjain/Byte-Sized-Brain)."
)

review = st.text_area("Movie review", DEFAULT_REVIEW, height=140)

if st.button("Classify", type="primary"):
    tokenizer, variants = load()
    enc = tokenizer(
        review, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="np"
    )
    enc = {"input_ids": enc["input_ids"].astype(np.int64),
           "attention_mask": enc["attention_mask"].astype(np.int64)}
    cols = st.columns(len(variants))
    for col, (name, (sess, size_mb)) in zip(cols, variants.items(), strict=False):
        pred, conf, latency = run_variant(sess, enc)
        with col:
            st.metric(name.upper(), SENTIMENT[pred], f"{conf:.0%} confidence")
            st.write(f"latency about **{latency:.0f} ms**")
            st.write(f"size **{size_mb:.0f} MB**")
