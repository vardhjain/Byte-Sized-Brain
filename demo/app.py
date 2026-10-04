"""Streamlit FP32-vs-INT8 sentiment demo.

Usage::

    pip install -e ".[all,demo]"
    bsb run distilbert_imdb          # produce the artifacts the demo loads
    streamlit run demo/app.py
"""

from __future__ import annotations

import streamlit as st

from byte_sized_brain.demo import DEFAULT_TEXTS, SUPPORTED, run_demo

st.set_page_config(page_title="Byte-Sized Brain: FP32 vs INT8", page_icon="🧠")
st.title("🧠 Byte-Sized Brain")
st.subheader("One review, two precisions")
st.caption(
    "The same movie review runs through the full-precision model and its quantized "
    "twin. Run `bsb run <pipeline>` first so the models exist."
)

pipeline = st.selectbox("Model", SUPPORTED, index=0)
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
        rows = run_demo(pipeline, [text], smoke=smoke)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    for col, row in zip(st.columns(len(rows)), rows, strict=True):
        with col:
            st.metric(row.variant.upper(), row.prediction, f"{row.confidence:.0%} confidence")
            st.write(f"⏱ **{row.latency_ms:.1f} ms** per inference")
            st.write(f"💾 **{row.size_mb:.1f} MB** on disk")
    st.caption(
        "Quantization keeps the prediction while shrinking the model by about 75%. "
        "On the transformer it also runs noticeably faster."
    )
