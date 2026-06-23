---
title: Byte-Sized Brain FP32 vs INT8
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Byte-Sized Brain, FP32 vs INT8

A one-screen demo that classifies a movie review through both the full-precision and the
INT8-quantized DistilBERT model, showing the prediction, confidence, latency, and size for
each. Quantization keeps the prediction while making the model about 4 times smaller and
noticeably faster.

This Space is the hosted demo for [Byte-Sized Brain](https://github.com/vardhjain/Byte-Sized-Brain),
a project that measures the accuracy, size, latency, and memory trade-offs of post-training
quantization across four model families.
