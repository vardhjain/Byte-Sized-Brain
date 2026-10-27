# MNIST: same model files, different CPUs

The first controlled native x86/ARM64 run of Byte-Sized Brain was completed
on **October 27, 2026**, using the **same two inference model files** on both
GitHub-hosted machines. This removes the separately trained model weights
as a confound in this *single* hardware comparison.

**The finding:** INT8 model execution was about **2.1× faster on x86** and
**2.5× faster on ARM64**. But when runtime input preparation is included,
INT8 was **slower on both**. For a tiny network, quantizing each input from
float to INT8 in the Python/TFLite input adapter costs more than the faster
integer inference saves.

## Measured results

| Measurement | x86_64 (AMD EPYC 7763) | ARM64 (Neoverse-N2) |
| --- | ---: | ---: |
| FP32 file size | 0.3899 MiB | 0.3899 MiB |
| INT8 file size | 0.1025 MiB | 0.1025 MiB |
| FP32 accuracy | 97.0% | 97.0% |
| INT8 accuracy | 97.1% | 97.1% |
| FP32 model-only mean latency | 0.0120 ms | 0.0109 ms |
| INT8 model-only mean latency | 0.0058 ms | 0.0043 ms |
| **INT8 model-only speedup** | **2.07×** | **2.53×** |
| FP32 runtime input preparation | 0.0014 ms | 0.0012 ms |
| INT8 runtime input preparation | 0.0150 ms | 0.0107 ms |
| FP32 prepare + model mean latency | 0.0134 ms | 0.0121 ms |
| INT8 prepare + model mean latency | 0.0208 ms | 0.0150 ms |
| **INT8 prepared-input total versus FP32** | **55% slower** | **24% slower** |

Each CPU received **1,000 identical test examples** and used **three latency
passes** per variant, batch size 1, one TensorFlow Lite inference thread, and
the pinned Python 3.11/TensorFlow 2.21 environment. Accuracy changed by
**one correctly classified image** out of 1,000; its paired McNemar p-value
was **1.0**. This is not evidence of a meaningful accuracy improvement.

### What was held constant

The producer trained once on x86 and wrote the FP32 and static INT8 TFLite
files. The two benchmark jobs downloaded a common manifest, checked
SHA-256 before loading, and benchmarked without training or converting.

| Artifact | SHA-256 (full digest) |
| --- | --- |
| FP32 `.tflite` | `8d45a4fcf62081fd285ae72c8607c516342edfeb2f363a163a98d92143022ae9` |
| INT8 `.tflite` | `97b9ede664c2e7da8e40faf0b7b5659421a267a64c7c6e3c2eeed30fbe56aecf` |

Both sets of result rows contain **these exact digests**, and both archived
manifests have identical content. The original run is
[GitHub Actions #40457518952](https://github.com/vardhjain/Byte-Sized-Brain/actions/runs/40457518952).
The raw files and manifests are committed under
[`benchmarks/results/cross_hardware/2026-10-27-ffn-mnist/`](https://github.com/vardhjain/Byte-Sized-Brain/tree/main/benchmarks/results/cross_hardware/2026-10-27-ffn-mnist).

The two producer and benchmark jobs **succeeded**, but the first workflow's
*final comparison job* failed on a missing NumPy import in its minimal
verification environment. We recovered the generated files, compared their
recorded hashes against their original manifests, and preserved the
complete CSVs; the fix runs that verifier as a standalone standard-library
script. **The original Actions run remains marked failed**. Its failure is
not a failed benchmark or a failed checksum comparison, but the original
final gate did not execute. The committed archive is regression-tested in CI.

## Independently verify the archived experiment

From the repository root, using only Python 3.11 or later; **no ML frameworks,
model downloads, or retraining**:

```bash
python -I -S src/byte_sized_brain/benchmark/cross_hardware.py \
  benchmarks/results/cross_hardware/2026-10-27-ffn-mnist/x86 \
  benchmarks/results/cross_hardware/2026-10-27-ffn-mnist/arm
```

This checks that all variants have the same model SHA-256 on both processors
and against the original model manifest, and that their evaluation sample
counts match. It fails if either archived CSV or manifest is altered.
The repository's unit tests also check the archived numbers and this
zero-dependency entrypoint.

## Why model-only and prepared-input numbers disagree

The model-only timer starts **after** `runner.prepare()`. INT8's small
matmul executes quickly, but the quantized runtime adapter must calculate
and copy the INT8 input tensors. The inclusive timer adds that work, on
**the same inference calls**. On this tiny MNIST network, it dominates the
total, turning an apparent 2× model-level improvement into slower
prepared-input inference.

This measurement still **does not include** loading raw images, resizing,
networking, loading the model, or decoding class labels. These are hosted
server CPUs, not a Raspberry Pi or a phone. The absolute times are just
microseconds and sensitive to timer noise, CPU frequency, and cloud
scheduling. Repeating on dedicated devices and increasing inference
batch/passes would strengthen the conclusion. Do not generalize this
specific FP32-versus-INT8 trade-off to the other three model families.

The earlier README headline table uses *separately trained* x86 and ARM64
models and continues to report their historical model-only numbers; this
new controlled MNIST experiment is intentionally kept in its own archive
rather than silently rewriting that older evidence.

See [measurement methodology](methodology.md) and the
[reproducibility guide](reproducibility.md) for more details.
