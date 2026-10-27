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

Every model came out 69 to 75 percent smaller. Three of the four changed by
no more than 0.6 accuracy points, though the historical data do not include
paired uncertainty estimates. The MobileNetV2 photo classifier is
the exception. It dropped 17 points, and an ablation traced the loss to 8-bit
activations in its depthwise layers, not to its weights. Speed depended on the model.
On an x86 laptop DistilBERT ran about 1.8 times faster and the photo classifier no
faster at all. On an ARM64 server CPU every quantized model was faster, up to 3.6
times. Those historical models were separately trained, not identical weights
benchmarked on both machines.

See the [full results](report.md) for the per-model breakdown, the
[methodology](methodology.md) for measurement boundaries, and
[Reproduce and verify](reproducibility.md) for no-training checks and controlled
benchmark commands.

## What it covers

- Static INT8 with real calibration data and dynamic INT8, across TensorFlow Lite and
  ONNX Runtime.
- Four models, from a tiny digit reader to a DistilBERT language model.
- One benchmark harness that measures **both** model-only latency and runtime
  input-preparation-inclusive latency; memory uses a fresh process per model.
- Config files, fixed seeds, pinned dependencies, paired accuracy statistics,
  repeatable latency passes, and CI on x86 and native ARM64.
- An opt-in workflow for measuring **identical** SHA-256 verified artifacts on both
  CPU architectures. It has not yet replaced the historical benchmark numbers.
