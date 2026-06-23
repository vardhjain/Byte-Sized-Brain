# Methodology

How Byte-Sized Brain measures the quantization trade-offs, and how it replaces
the original Raspberry Pi with a reproducible cross-architecture benchmark.

## 1. What we measure

For every model variant the shared harness
([`benchmark/harness.py`](https://github.com/vardhjain/Byte-Sized-Brain/blob/Byte-Sized-Brain/src/byte_sized_brain/benchmark/harness.py)) records,
identically across TFLite and ONNX Runtime:

| Metric | Definition |
|---|---|
| **Size (MB)** | On-disk size of the artifact (file, or whole SavedModel dir). |
| **Accuracy** | Top-1 / threshold accuracy over `num_samples` real test examples. |
| **Latency mean / p50 / p95 (ms)** | Per-example, single-sample inference latency. p95 captures tail behaviour the mean hides. |
| **RSS Δ (MB)** | Process resident-set-size growth across the benchmark loop. |
| **Peak RSS (MB)** | Peak RSS increase over the pre-inference baseline. |

Each row is **self-describing** — it is stamped with `device`, `arch`, `os`,
`python`, `emulated`, and the exact library versions — so results from x86,
emulated ARM and real ARM can be concatenated into one CSV and compared honestly.

### Why these choices

- **Process RSS, not whole-machine memory.** The original scripts used
  `psutil.virtual_memory().used`, which measures the entire machine and is
  dominated by unrelated processes. We measure `psutil.Process().memory_info().rss`
  instead — attributable and far less noisy.
- **Warm-up runs.** The first few inferences pay one-off costs (lazy kernel
  init, XNNPACK delegate creation, cache warming). We discard `warmup` iterations
  before timing.
- **RSS sampled outside the timed region.** Memory polling is a syscall; sampling
  it inside the timed block would inflate latency. It is read *after* each timed
  inference.
- **Single-sample latency.** Edge inference is typically one request at a time;
  batch-1 latency is the relevant number, and it is what the static-batch export
  enables for the LSTM.

## 2. Quantization techniques per pipeline

| Pipeline | Runtime | FP32 baseline | Quantized variant | Technique |
|---|---|---|---|---|
| FFN / MNIST | TFLite | FP32 TFLite | INT8 | **Static PTQ** (weights + activations) with a real MNIST representative dataset; INT8 input/output. |
| CNN / CIFAR-10 | TFLite | FP32 TFLite | INT8 | **Static PTQ** with real CIFAR images for calibration. |
| RNN / IMDB | TFLite | FP32 TFLite | dynamic-range | **Dynamic-range PTQ** (INT8 weights, FP32 activations). Full-integer PTQ is not well supported for LSTM. |
| DistilBERT / IMDB | ONNX Runtime | FP32 ONNX | INT8 | **Dynamic INT8** weight quantization (`quantize_dynamic`). |

Static PTQ needs a **representative dataset** to calibrate activation ranges. The
original RNN script fed `np.random.randint(...)` noise — calibration against a
distribution the model never sees, which is a no-op. Here representative datasets
are always **real samples** (`data/*.py::representative_dataset`).

### The LSTM static-batch trick

A Keras LSTM exported as a generic SavedModel lowers to a `tf.while` loop with a
dynamic-shape `TensorListReserve`, which the TFLite converter can neither lower to
a builtin nor run in the Python/edge interpreter (it falls back to TF-Select/Flex
ops that need a delegate). Exporting with a **static batch dimension**
(`Input` signature `(1, max_len)`) makes the element shape static, so the LSTM
lowers to the native `UnidirectionalSequenceLSTM` builtin and runs on the plain
TFLite/LiteRT interpreter — including on ARM, with **no Flex delegate**.

## 3. Replacing the Raspberry Pi

The original latency/RAM numbers came from a Raspberry Pi acting as the "edge
device." That hardware is no longer available. Rather than fabricate numbers or
drop the edge angle, we use a **two-layer, fully reproducible** replacement.

### Layer A — Emulated ARM64 (QEMU in Docker) — primary, free, CI-able

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx build --platform linux/arm64 -f docker/Dockerfile.arm64 -t bsb:arm64 --load .
docker run --rm --platform linux/arm64 -e BSB_EMULATED=1 -e BSB_DEVICE=qemu-arm64 \
    -v "$PWD/benchmarks/results:/app/benchmarks/results" \
    bsb:arm64 bsb run all --smoke
# or:  make benchmark-arm
```

This proves the artifacts genuinely **load and run on aarch64** (the Pi's ISA) and
is reproducible by anyone, including in CI.

> **Caveat — emulated latency is not real latency.** QEMU emulates the ARM
> instruction set on x86; it inflates absolute latency unpredictably. For emulated
> runs we report **functional correctness + size + accuracy + *relative* latency
> ratios only**, and every such row is stamped `emulated = true`. `bsb report`
> keeps emulated and native rows separate.

### Layer B — Real ARM hardware via free cloud — for real latency

A free **Oracle Cloud Ampere A1** instance (or **AWS Graviton**) is real ARM64
silicon:

```bash
# on the ARM VM
git clone https://github.com/vardhjain/Byte-Sized-Brain && cd Byte-Sized-Brain
pip install -r requirements.txt && pip install -e .
BSB_DEVICE=oracle-ampere-a1 bsb run all          # emulated defaults to false
git add benchmarks/results && git commit -m "real ARM64 results"
```

This swaps the Pi's hardware story for "I benchmarked on real ARM silicon in the
cloud" — a more current, deployment-relevant signal (Graviton/Ampere is what edge
and cost-efficient inference actually deploy to).

### Appendix — Run on a real Raspberry Pi (64-bit OS)

The original path still works:

```bash
sudo apt install -y python3-pip libatlas-base-dev
pip install -r requirements.txt && pip install -e .
BSB_DEVICE=rpi4 bsb benchmark ffn_mnist          # train/convert on a beefier box, copy artifacts/ over
```

On 32-bit Pi OS use `tflite-runtime` / `onnxruntime` wheels and benchmark the
converted artifacts directly.

## 4. Reproducibility

- **Determinism** — `seed_everything` seeds Python/NumPy/TF/Torch; configs pin a
  `seed`.
- **Pinned environment** — `requirements.txt` pins exact versions; the committed
  CSVs carry their own `lib_versions` so a result can never be silently attributed
  to the wrong stack.
- **Smoke vs full** — every config has a `smoke:` block; `--smoke` runs a
  1-epoch, few-sample version end-to-end in seconds (used by CI). Full runs use
  the base config.
- **Clean history** — the original repo committed large binaries (a 67 MB ONNX,
  SavedModels, HuggingFace checkpoints, TensorBoard `runs/`). These were purged
  from git history with `git filter-repo` and force-pushed, shrinking the clone
  from ~78 MB to ~1 MB. A full pre-rewrite backup bundle is kept outside the repo.

## 5. Limitations (stated honestly)

- The committed x86 results were produced on a **CPU-only Windows box**. TF has no
  native-Windows GPU support, and CIFAR/MobileNetV2 backbone fine-tuning is
  CPU-prohibitive, so the committed CNN config is a **frozen-backbone
  feature-extraction baseline** (`fine_tune_epochs: 0`, `train_subset: 12000`).
  Higher absolute CNN accuracy is available by unfreezing the backbone on a GPU —
  the *size* story is unaffected by absolute accuracy.
- **MobileNetV2 is PTQ-sensitive.** Unlike the other three models (which retain
  accuracy within ~2% after quantization), the CNN's static-INT8 variant shows a
  notable accuracy drop. This is a real, documented characteristic — depthwise
  separable convolutions quantize poorly under per-tensor full-integer PTQ — not a
  bug. It is exactly the kind of architecture-dependent trade-off this project
  exists to surface; see the per-pipeline `accuracy_delta` in the results.
- Emulated-ARM latency is indicative of correctness and relative behaviour, not
  absolute speed (see §3, Layer A).
- TensorFlow aarch64 wheel availability varies by version; if `tensorflow` won't
  install on a given ARM target, the ONNX (DistilBERT) pipeline still runs and
  demonstrates the cross-architecture story.
