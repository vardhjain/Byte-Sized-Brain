---
title: Byte-Sized Brain FP32 vs INT8
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: A full-size and an 8-bit DistilBERT, side by side
models:
  - vardhjain20/byte-sized-brain-distilbert-imdb
preload_from_hub:
  - vardhjain20/byte-sized-brain-distilbert-imdb
tags:
  - quantization
  - onnx
  - onnxruntime
  - distilbert
  - sentiment-analysis
---

# Byte-Sized Brain, FP32 vs INT8

Paste a movie review and two versions of the same AI model will decide whether it is
positive or negative. The first version is the original, full-precision model (FP32,
32-bit numbers). The second is a **quantized** copy (INT8) whose numbers were rounded to
8 bits, which makes the file about four times smaller. The page shows each model's
answer, how sure it is, how long it took and how big its file is, so you can see the
trade-off for yourself.

## What the project found

These numbers come from the project's benchmark on 500 IMDB test reviews, run one
review at a time on a desktop CPU.

| Model                 | File size | Accuracy | Time per review |
|-----------------------|----------:|---------:|----------------:|
| Full precision (FP32) |  255.5 MB |    84.6% |           72 ms |
| Quantized (INT8)      |   64.3 MB |    84.0% |           43 ms |

The quantized model is four times smaller and about 1.7 times faster, and it gives up
less than one point of accuracy. The two versions gave the same answer on 477 of the 500
reviews (95%). The other 23 were close calls where the small rounding changes tipped the
result, in both directions.

## Good to know

- The times on this page are the fastest of 10 runs, measured live on the free, shared
  CPU this Space runs on. They differ from the benchmark averages above and change a
  little from click to click.
- The model reads English movie reviews and looks at the first 128 tokens, which is
  roughly the first 100 words. Very short or off-topic text gives less reliable answers.
- On a borderline review the two versions can disagree, and the page tells you when
  that happens.
- The Space goes to sleep when nobody has used it for a while. The first visit after
  that can take a minute or two while it starts up again.

## How it works

The model is DistilBERT, fine-tuned on 3,000 IMDB reviews and exported to ONNX. The INT8
copy was made with ONNX Runtime dynamic quantization, so no extra training was needed.
Both files live in the
[byte-sized-brain-distilbert-imdb](https://huggingface.co/vardhjain20/byte-sized-brain-distilbert-imdb)
model repo, and this app runs them with ONNX Runtime on the CPU.

This Space is the hosted demo for
[Byte-Sized Brain](https://github.com/vardhjain/Byte-Sized-Brain), a project that
measures what quantization does to the size, accuracy, speed and memory of four kinds of
neural network. The source for this page is in `demo/hf_space/` of that repository.
