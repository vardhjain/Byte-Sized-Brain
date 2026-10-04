# Byte-Sized Brain

Shrinking AI models to a quarter of their size, and measuring exactly what it costs.

Quantization stores a model's numbers in 8 bits instead of 32, which makes the model
much smaller. Byte-Sized Brain trains four models, quantizes each one, and measures
size, accuracy, speed and memory the same way before and after.

[Get the code on GitHub](https://github.com/vardhjain/Byte-Sized-Brain){ .md-button .md-button--primary }
[Try the live demo](https://huggingface.co/spaces/vardhjain20/byte-sized-brain-demo){ .md-button }
[Read the methodology](methodology.md){ .md-button }

## The headline

![What 8-bit quantization changed, model by model](images/results_dashboard.png)

Every model came out 69 to 75 percent smaller. Three of the four kept their accuracy,
with the largest change being 3 of 500 test reviews. The MobileNetV2 photo classifier is
the exception. It dropped 17 points, and an ablation traced the loss to 8-bit
activations in its depthwise layers, not to its weights. Speed depended on the model.
DistilBERT ran about 1.8 times faster, while the photo classifier ran slightly slower.

See the [full results](report.md) for the per-model breakdown, and the
[methodology](methodology.md) for exactly how every number is measured and what its
limits are.

## What it covers

- Static INT8 with real calibration data and dynamic INT8, across TensorFlow Lite and
  ONNX Runtime.
- Four models, from a tiny digit reader to a DistilBERT language model.
- One benchmark harness that times only the model call and measures each model's memory
  in a fresh process.
- Config files, fixed seeds, pinned dependencies, a command-line tool, tests, and CI on
  both x86 and ARM64.
