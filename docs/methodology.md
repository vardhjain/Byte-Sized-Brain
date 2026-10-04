# Methodology

How Byte-Sized Brain measures the quantization trade-offs, what each number does and
does not mean, and how the same benchmark can be run on ARM64.

## 1. What is measured

For every model variant, the shared harness
([`benchmark/harness.py`](https://github.com/vardhjain/Byte-Sized-Brain/blob/Byte-Sized-Brain/src/byte_sized_brain/benchmark/harness.py))
records the same metrics in the same way for TFLite and ONNX Runtime.

| Metric | Definition |
|---|---|
| **Size (MB)** | On-disk size of the `.tflite` or `.onnx` file that is benchmarked, in MiB (1,048,576 bytes). |
| **Accuracy** | Top-1 accuracy (argmax, or a 0.5 threshold for the LSTM) over `num_samples` test examples. Both variants of a model see exactly the same examples. |
| **Latency mean / p50 / p95 (ms)** | Wall-clock time of one model call on a single example. p95 shows tail behaviour the mean hides. |
| **RSS delta (MB)** | How much the resident memory of a fresh process grows when it loads the model and runs it. |
| **Peak RSS (MB)** | The highest resident memory that process reached, minus the level before the model was loaded. |

Every row is **self-describing**. It is stamped with `device` (the CPU model, or the
`BSB_DEVICE` label), `arch`, `os`, `python`, `emulated` and the exact library versions,
so results from different machines can share one CSV and still be told apart.
Architecture names are normalized, so Windows `AMD64` and Linux `x86_64` count as the
same architecture.

### Why these choices

- **Only the model call is timed.** A full-integer TFLite model takes INT8 input, so
  each float sample has to be quantized first. That conversion happens in NumPy, before
  the clock starts. An earlier version of the harness timed it, and for the smallest
  model the NumPy work cost more than the model itself, which made INT8 look slower
  than FP32 when the interpreter was in fact faster.
- **Memory is measured in a fresh process per variant.** Once a model is loaded, running
  it barely changes the process's memory, so a measurement taken around the benchmark
  loop records only noise. Instead, a child process imports the runtime, notes its
  memory, then loads the model and runs it a few times. The growth is the model's own
  footprint. A separate process per variant also stops the second variant from reusing
  memory the first one freed.
- **Warm-up runs.** The first few inferences pay one-off costs (lazy kernel setup,
  delegate creation, cache warming). The harness runs `warmup` untimed inferences before
  it starts timing.
- **Single-sample latency.** Edge inference usually serves one request at a time, so
  batch-1 latency is the relevant number.

### Threads and kernels

TFLite models run on the interpreter's default XNNPACK delegate, for both float and
integer models, on one thread. ONNX Runtime runs on its CPU execution provider and uses
all physical cores by default (set `BSB_ORT_THREADS` to change that). The DistilBERT
latency is therefore not comparable with the TFLite latencies, only with its own FP32
baseline. All committed numbers come from an Intel Core i7-10710U laptop CPU (6 cores,
AVX2, no VNNI). Integer kernels gain most on CPUs with VNNI or on ARM, so the speedups
here are specific to this machine.

### Sample sizes and noise

Accuracy is measured on the first 1,000 test images for MNIST and CIFAR-10, the first
500 test reviews for the LSTM, and a seeded random 500 for DistilBERT. With 1,000
samples the 95 percent interval on an accuracy near 85 percent is about plus or minus
2 points, and with 500 samples about plus or minus 3. Differences smaller than that
between a model and its quantized version are not meaningful. The CNN's 17-point drop is
far outside this range.

Equal accuracy can still hide changed predictions. The FP32 and INT8 DistilBERT models
agree on 477 of the 500 reviews, and the 23 disagreements go in both directions.

Latency comes from one run on a laptop. Run-to-run differences of 10 to 20 percent are
normal, and sub-millisecond figures are close to the resolution of the timer, so read
the speed column as "faster", "slower" or "about the same", not to the last digit.

## 2. Quantization technique per pipeline

| Pipeline | Runtime | Quantized variant | Technique |
|---|---|---|---|
| FFN / MNIST | TFLite | INT8 | **Static (full-integer) PTQ** of weights and activations, calibrated on 100 real MNIST training images, with INT8 input and output. |
| CNN / CIFAR-10 | TFLite | INT8 | **Static (full-integer) PTQ**, calibrated on 250 real CIFAR-10 training images, with INT8 input and output. |
| RNN / IMDB | TFLite | dynamic-range | **Dynamic-range PTQ** (INT8 weights, float activations). It needs no calibration data. Most of the saving is the embedding table. |
| DistilBERT / IMDB | ONNX Runtime | INT8 | **Dynamic INT8** with ONNX Runtime's `quantize_dynamic`. Weights are stored as INT8 and activations are quantized on the fly at inference time. |

Static PTQ needs a **representative dataset** to calibrate activation ranges. Feeding it
random noise sets those ranges from inputs the model never sees, so calibration here
always uses real training samples, through the one shared helper
`byte_sized_brain.data.representative_dataset`.

### Why the LSTM is exported with a fixed batch size

With a dynamic batch dimension the TFLite converter stops with
`'tf.TensorListReserve' op requires element_shape to be static`. The only way through is
Select TF ops, which need the Flex delegate at run time. Exporting with a fixed input
shape of `(1, max_len)` lets the converter lower the tensor-list ops to builtins. The
result is a builtin `WHILE` loop whose body is the LSTM cell written out as ordinary ops
(`FULLY_CONNECTED`, `LOGISTIC`, `TANH`, `MUL`, `ADD`). It is not the fused
`UnidirectionalSequenceLSTM` op, but it runs on the stock TFLite interpreter with no
Flex delegate.

## 3. Running on ARM64

All committed results are from x86. The project offers three ways to exercise ARM64,
and none of them needs special hardware.

### Continuous integration on native ARM64

On every push, CI runs the fast tests and the TensorFlow Lite conversion and smoke
pipelines on GitHub's native ARM64 runners as well as on x86. This shows the code and
the conversions work on ARM. It produces no published benchmark numbers.

### Emulated ARM64 (QEMU in Docker)

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
make docker-build-arm
make benchmark-arm
```

This trains, converts and benchmarks all four pipelines on their smoke configs inside
an aarch64 container. Because it is a smoke run, its results land in
`benchmarks/results/smoke/`, which is gitignored and never read by `bsb report`.

> **Emulated latency is not real latency.** QEMU translates ARM instructions on an x86
> host, which slows everything down by an unpredictable amount. Use emulated runs to
> check that things work. Every row from such a run is stamped `emulated = true`.

### Real ARM hardware

An Oracle Cloud Always Free Ampere A1 VM, an AWS Graviton instance or a 64-bit
Raspberry Pi is real ARM64 silicon. The pinned requirements need Python 3.11 or 3.12.

```bash
sudo apt install -y python3-venv git
git clone https://github.com/vardhjain/Byte-Sized-Brain && cd Byte-Sized-Brain
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
BSB_DEVICE=oracle-ampere-a1 bsb run all     # BSB_EMULATED is unset, so rows say emulated=false
```

On a Raspberry Pi, train and convert on a faster machine, copy the `artifacts/` folder
across, and run only `bsb benchmark <pipeline>` on the Pi.

`bsb benchmark` merges rows by architecture, so ARM rows are added next to the x86 rows
in each `benchmarks/results/<pipeline>.csv` instead of replacing them, and `bsb report`
compares each row only with the FP32 baseline from the same architecture.

## 4. Reproducibility

- **Seeding.** `seed_everything` seeds Python, NumPy and whichever framework the
  pipeline uses, and every config pins a `seed`. Op-level determinism is left off for
  speed, so a retrained model is close to the committed one but not guaranteed to be
  bit-identical.
- **Pinned environment.** `requirements.txt` pins the versions behind the committed
  results (Python 3.11 or 3.12), and CI installs against those pins. Each CSV row also
  carries its own `lib_versions`.
- **Smoke versus full runs.** Every config has a `smoke:` block. `--smoke` runs one
  epoch on a small subset and keeps its models in `artifacts/smoke/` and its results in
  `benchmarks/results/smoke/`, so it never overwrites a full run.
- **Regenerating the report.** Run `bsb report` from the repository root. It reads only
  the top-level `benchmarks/results/*.csv` files and rewrites `docs/report.md`,
  `docs/images/results_dashboard.png` and the results table in `README.md`.

## 5. Limitations

- **The CNN is a quick baseline.** The committed results come from a CPU-only Windows
  machine, so the CNN config trains only the classification head on a frozen ImageNet
  backbone (`fine_tune_epochs: 0`, `train_subset: 12000`). Unfreezing the backbone on a
  GPU would raise the absolute accuracy. The size results do not depend on it.
- **MobileNetV2 loses accuracy under full-integer PTQ.** The other three models stay
  within 0.6 percentage points of FP32. The CNN drops from 85.4% to 68.1%. An ablation
  on the same 1,000 test images
  ([`scripts/cnn_int8_ablation.py`](https://github.com/vardhjain/Byte-Sized-Brain/blob/Byte-Sized-Brain/scripts/cnn_int8_ablation.py),
  results in `benchmarks/results/ablations/cnn_int8_ablation.csv`) shows where the loss
  comes from.

    | Variant | Accuracy |
    |---|---|
    | FP32 | 85.4% |
    | Static INT8 (benchmarked) | 68.1% |
    | Static INT8, 1,000 calibration images instead of 250 | 68.6% |
    | Static INT8 with per-tensor weight scales | 8.8% |
    | INT8 weights, float activations (dynamic-range) | 86.0% |
    | Static INT8 except `DEPTHWISE_CONV_2D` | 81.2% |
    | Static INT8 except `CONV_2D` | 74.4% |
    | Static INT8 except the residual `ADD` ops | 67.5% |
    | Static INT8 except pooling, dense head and softmax | 68.7% |

    Weight quantization is not the cause. TFLite already quantizes convolution weights
    per channel (per-tensor scales collapse to chance), and INT8 weights with float
    activations lose nothing. More calibration data does not help. The loss comes from
    storing activations as 8-bit integers, mostly in the depthwise convolutions. The
    usual fixes are quantization-aware training, keeping the depthwise layers in float,
    or using dynamic-range quantization for this model.
- **DistilBERT is also a reduced-budget model.** It is fine-tuned on 3,000 reviews for
  2 epochs on a CPU, with inputs truncated to 128 tokens, which cuts off most of the
  evaluation reviews. That is why it scores 84.6%, no better than the LSTM. The
  comparison between its FP32 and INT8 versions does not depend on this. Latency is
  measured at a fixed 128 tokens. The LSTM and DistilBERT rows use different test
  reviews and tokenization, so they should not be compared with each other.
- **Dynamic INT8 in ONNX Runtime computes activation ranges per batch**, so its
  accuracy can differ slightly at other batch sizes. Everything here is batch size 1.
- **One machine.** All committed numbers are from one x86 laptop CPU.
