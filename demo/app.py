"""Streamlit demo: one movie review, classified by a model and its quantized version.

Usage::

    pip install -e ".[all,demo]"
    bsb run distilbert_imdb          # produce the models the demo loads
    streamlit run demo/app.py
"""

from __future__ import annotations

import streamlit as st

from byte_sized_brain.demo import DEFAULT_TEXTS, SUPPORTED, DemoRow, run_demo

MODEL_NAMES = {
    "distilbert_imdb": "DistilBERT (ONNX Runtime)",
    "rnn_imdb": "LSTM (TensorFlow Lite)",
}
VARIANT_NAMES = {
    "fp32": "Original (FP32)",
    "int8": "Quantized (INT8)",
    "dynamic_range": "Quantized (INT8 weights)",
}


def summarize(rows: list[DemoRow]) -> str:
    """One sentence describing what this run actually showed."""
    if len(rows) < 2:
        return "Only one version of this model was found, so there is nothing to compare."
    original, quantized = rows[0], rows[1]
    verdict = (
        f"Both versions say **{original.prediction.lower()}**."
        if original.prediction == quantized.prediction
        else f"The versions disagree. The original says **{original.prediction.lower()}** "
        f"and the quantized one says **{quantized.prediction.lower()}**, which can happen "
        "on a borderline review."
    )
    smaller = original.size_mb / quantized.size_mb
    speed = original.latency_ms / quantized.latency_ms
    if speed >= 1.15:
        pace = f"ran **{speed:.1f} times faster**"
    elif speed <= 0.85:
        pace = f"ran **{1 / speed:.1f} times slower**"
    else:
        pace = "took **about the same time**"
    return f"{verdict} The quantized model is **{smaller:.1f} times smaller** and {pace} here."


st.set_page_config(page_title="Byte-Sized Brain demo", page_icon="🧠")
st.title("🧠 Byte-Sized Brain")
st.subheader("One review, two precisions")
st.caption(
    "The same movie review runs through the full-precision model and its quantized "
    "twin. Run `bsb run <pipeline>` first so the models exist."
)

pipeline = st.selectbox("Model", SUPPORTED, index=0, format_func=lambda p: MODEL_NAMES.get(p, p))
smoke = st.checkbox(
    "Use the quick `--smoke` models",
    value=False,
    help="Tick this if you built the models with `bsb run <pipeline> --smoke`.",
)
text = st.text_area("Movie review", DEFAULT_TEXTS[0], height=140)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Type a review first.")
        st.stop()
    try:
        with st.spinner("Running both versions..."):
            rows = run_demo(pipeline, [text], smoke=smoke)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    for col, row in zip(st.columns(len(rows)), rows, strict=True):
        with col:
            st.metric(VARIANT_NAMES.get(row.variant, row.variant), row.prediction)
            st.write(f"🎯 **{row.confidence:.0%}** confident")
            st.write(f"⏱ **{row.latency_ms:.1f} ms** per inference")
            st.write(f"💾 **{row.size_mb:.1f} MB** on disk")
    st.markdown(summarize(rows))
    st.caption("Timings are the fastest of several runs and vary with machine load.")
