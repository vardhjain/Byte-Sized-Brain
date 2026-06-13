"""Streamlit FP32-vs-INT8 sentiment demo.

    pip install -e ".[all,demo]"
    bsb run distilbert_imdb          # produce the artifacts the demo loads
    streamlit run demo/app.py
"""

from __future__ import annotations

import streamlit as st

from byte_sized_brain.demo import SUPPORTED, run_demo

st.set_page_config(page_title="Byte-Sized Brain — FP32 vs INT8", page_icon="🧠")
st.title("🧠 Byte-Sized Brain")
st.subheader("FP32 vs INT8 sentiment — same review, two precisions")
st.caption("Run `bsb run <pipeline>` first so the artifacts exist.")

pipeline = st.selectbox("Model", SUPPORTED, index=0)
text = st.text_area(
    "Movie review",
    "This is hands down one of the best films I have seen all year. The performances "
    "were stunning and the story kept me engaged from start to finish.",
    height=140,
)

if st.button("Classify", type="primary"):
    try:
        rows = run_demo(pipeline, [text])
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    for col, row in zip(st.columns(len(rows)), rows, strict=False):
        with col:
            st.metric(row.variant.upper(), row.prediction, f"{row.confidence:.0%} confidence")
            st.write(f"⏱ **{row.latency_ms:.1f} ms** per inference")
            st.write(f"💾 **{row.size_mb:.1f} MB** on disk")
    st.caption(
        "Quantization keeps the prediction while shrinking the model ~75% — and, on "
        "the transformer, running noticeably faster."
    )
