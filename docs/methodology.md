# Methodology

How Byte-Sized Brain measures the quantization trade-offs, what each number does and
does not mean, and how the same benchmark can be run on ARM64.

## 1. What is measured

For every model variant, the shared harness
([`benchmark/harness.py`](https://github.com/vardhjain/Byte-Sized-Brain/blob/main/src/byte_sized_brain/benchmark/harness.py))
records the same metrics in the same way for TFLite and ONNX Runtime.

| Metric | Definition |
|---|---|
| **Size (MB)** | On-disk size of the `.tflite` or `.onnx` file that is benchmarked, in MiB (1,048,576 bytes). |
| **Accuracy** | Top-1 accuracy (argmax, or a 0.5 threshold for the LSTM) over `num_samples` test examples. Both variants of a model see exactly the same examples. |
| **Agreement** | Share of those examples where the variant predicts the same label as its FP32 baseline. |
| **Latency mean / p50 / p95 (ms)** | Wall-clock time of one model call on a single example, pooled across timed passes. p95 shows tail behaviour the mean hides. |
| **Latency uncertainty** | Optional full-pass repeats; bootstrap interval and standard deviation are calculated from mean latency **per pass**, not treating each call as an independent machine benchmark. Blank when only one pass is run. |
| **Paired accuracy delta** | Seeded paired bootstrap 95% interval for quantized accuracy minus FP32 accuracy, resampling matching held-out examples. |
| **McNemar exact p-value** | Two-sided exact test from samples where exactly one variant is correct; also records how many examples each variant alone gets right. |
| **RSS delta (MB)** | How much the resident memory of a fresh process grows when it loads the model and runs it. |
| **Peak RSS (MB)** | The highest resident memory reached *after the runtime-import baseline*, minus that baseline. On Linux, the kernel high-water mark is reset first; a 5 ms RSS sampler is the fallback. |

Every row is **self-describing**. It is stamped with `device` (the CPU model, or the
`BSB_DEVICE` label), `arch`, `os`, `python`, `emulated` and the exact library versions.
A new benchmark invocation creates one `run_id` shared by the FP32 and quantized rows.
This identifies a paired experiment regardless of architecture, so different devices
and repeated runs on the same device remain separate. Older committed rows (October
2026) lack this field: the reader assigns stable `legacy-*` IDs using their
pipeline/device/environment fields. Runs overwritten by older versions cannot be recovered.
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
- **Peak RSS excludes earlier allocations.** The probe resets Linux's historical
  resident-memory high-water mark by writing `5` to `/proc/self/clear_refs` after
  importing the runtime and before loading the model. Where the reset is unavailable,
  a 5 ms RSS sampler provides a best-effort peak and can miss brief spikes. The
  previously committed ARM64 `peak_rss_mb` values (October 5, 2026) were measured
  before this fix and are not valid estimates of model-loading peaks. They must be
  regenerated before being cited; the README's memory figures use `rss_delta_mb`,
  which is unaffected by this specific error.
- **Warm-up runs.** The first few inferences pay one-off costs (lazy kernel setup,
  delegate creation, cache warming). The harness runs `warmup` untimed inferences before
  it starts timing.
- **Single-sample latency.** Edge inference usually serves one request at a time, so
  batch-1 latency is the relevant number.

### Threads and kernels

TFLite models run on the interpreter's default XNNPACK delegate, for both float and
integer models, on one thread. ONNX Runtime runs on its CPU execution provider and uses
all physical cores by default. `BSB_TFLITE_THREADS` and `BSB_ORT_THREADS` change
these, and every row records the count it used in its `threads` column. The DistilBERT
latency is therefore not comparable with the TFLite latencies, only with its own FP32
baseline. The x86 numbers come from an Intel Core i7-10710U laptop CPU (6 cores,
AVX2, no VNNI), and the ARM64 numbers from an Arm Neoverse-N2 with 4 virtual CPUs.
Integer kernels gain most on CPUs with VNNI or on ARM, which is what the results
show. Static INT8 gives no speedup for the CNN on the laptop and 2.5 times on the
Neoverse-N2.

### Sample sizes and noise

Accuracy uses the first 1,000 test images for MNIST and CIFAR-10, the first
500 test reviews for the LSTM, and a seeded random 500 for DistilBERT. The
published October 2026 CSV rows record per-model Wilson intervals, which describe
sampling variability of each accuracy **separately**. They are not a valid test
of differences between FP32 and INT8: both variants predict on exactly the same
examples, so their errors are correlated.

**New benchmark runs** additionally report a paired bootstrap interval for the
*accuracy difference*, and an exact two-sided McNemar p-value from discordant
correctness (FP32-only correct versus quantized-only correct). The paired bootstrap
uses 2,000 resamples and the configured random seed. Do not treat the p-value as
a claim about other datasets, separately trained models or deployment hardware.
Both variants can have the same overall accuracy while predicting differently;
`agreement` reports the fraction of identical predicted labels. Historical CSVs
do not have paired prediction statistics; those fields remain blank until rerun.

The published latency figures are one pass on a laptop, with typical run-to-run
variation of 10–20 percent. The new opt-in timing repeat option runs the exact same
examples and warm-up per pass, recording per-call mean/p50/p95 across all passes.
A seeded 2,000-resample bootstrap interval and sample SD of **per-pass mean
latency** are available for two or more passes, not for a single pass.
Resampling passes (rather than individual calls) avoids claiming neighboring
inferences are independent timing experiments. These intervals describe variation
during this benchmark session, *not* across machines or thermal states.

```bash
bsb benchmark cnn_cifar10 --cnn-alternatives --repeats 5
bsb report
```

For a quick alternative, use `bsb benchmark ffn_mnist --repeats 5`.
Repeats are intentionally opt-in (`benchmark.repeats: 1` by default) so normal
CI and smoke runs are not slowed down. Values in previously published CSVs and
the README headline have **not** been backfilled with synthetic statistics.

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

**Historical results:** The committed October 2026 x86 and native ARM64 rows
were produced from separately trained models. Their model graphs and weights
are **not confirmed byte-identical**, so differences in training are a confound
when comparing their absolute accuracy or speed across processors. A hosted
runner is a shared virtual machine, so its timings are noisier than a dedicated
device.

**New reproducible workflow:** The manual GitHub Actions workflow
[`Cross-hardware benchmark (identical artifacts)`](https://github.com/vardhjain/Byte-Sized-Brain/actions/workflows/arm-benchmark.yml)
trains and converts one source model on x86 only, packages the exact FP32 and
quantized inference files (plus the DistilBERT tokenizer), and uploads a SHA-256
manifest. Native x86 and ARM64 runners download and **verify** the same bytes,
then execute `bsb benchmark` only, on the same dataset selection and pinned
software versions. Their result rows contain `model_sha256`. A final job
compares those hashes with each other and against the uploaded manifest, and
requires equal evaluation sample counts. If either checksum differs the
workflow fails instead of publishing a misleading comparison.

Start it from **Actions → Cross-hardware benchmark (identical artifacts) →
Run workflow**. It defaults to `ffn_mnist`, the least expensive model; select
one other pipeline or explicitly choose `all` to run all four. This workflow is
manual because training full models can consume substantial runner time; usage
is subject to GitHub Actions minutes and artifact storage limits on the account.

Outputs are `results-x86_64` and `results-aarch64` download artifacts (with
a copy of `model-manifest.json`), retained for 30 days. The intermediate
models are kept for 7 days. New results are **not** committed automatically
or silently substituted into the published README. Review and import them
explicitly before updating the report. A different CPU can legitimately
produce small numerical prediction differences even with identical model bytes,
so the final gate tests *artifact identity*, not identical accuracy.

Beyond that workflow, the project offers three more ways to exercise ARM64.

### Continuous integration on native ARM64

On every push to `main` and every pull request, CI runs the fast tests and the TensorFlow Lite conversion and smoke
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

`bsb benchmark` appends a new pair of result rows identified by one run ID. Neither
other architectures nor earlier runs on the same architecture are overwritten.
`bsb report` compares a quantized row only to FP32 from the **same run ID** and
selects the latest complete native pair per pipeline for the README and dashboard.
A newer incomplete pair is skipped. The full history remains in the CSV and detailed report.

## 4. Reproducibility

- **Seeding.** `seed_everything` seeds Python, NumPy and whichever framework the
  pipeline uses, and every config pins a `seed`. Op-level determinism is left off for
  speed, so a retrained model is close to the committed one but not guaranteed to be
  bit-identical.
- **Pinned environment.** `requirements.txt` pins the versions behind the committed
  results (Python 3.11 or 3.12), and CI installs against those pins. Each CSV row also
  carries its own `lib_versions`.
- **Model provenance.** New result rows carry `model_sha256`, computed from the exact
  benchmarked TFLite or ONNX file. The manual cross-hardware workflow additionally
  verifies a bundle manifest before running either architecture and enforces SHA-256
  identity in a final comparison job. Historical rows have blank model hashes.
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
  ([`scripts/cnn_int8_ablation.py`](https://github.com/vardhjain/Byte-Sized-Brain/blob/main/scripts/cnn_int8_ablation.py),
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

  Both alternatives can now be benchmarked with the standard `bsb` harness (same
  holdout images, thread settings, memory isolation and run ID):

  ```bash
  bsb train cnn_cifar10
  bsb convert cnn_cifar10 --cnn-alternatives
  bsb benchmark cnn_cifar10 --cnn-alternatives
  bsb report
  ```

  - `dynamic_range` is a supported, weights-focused TFLite converter path;
    quantized weights do **not** guarantee fully integer activation execution.
  - `int8_except_depthwise` uses TensorFlow's **experimental**
    `QuantizationDebugger` denylist for `DEPTHWISE_CONV_2D`, so the resulting
    graph has mixed precision and its converter support is version-dependent.
  - These produce separately named `.tflite` files and appear in the detailed
    results alongside `fp32` and `int8`; the README headline continues to
    compare full static INT8. `--smoke` keeps results out of the published CSV.
  - The ablation values shown above are **historical accuracy and size only**;
    they are not new main-pipeline latency or memory measurements. Run the commands
    above to obtain those metrics on your machine.
- **DistilBERT is also a reduced-budget model.** It is fine-tuned on 3,000 reviews for
  2 epochs on a CPU, with inputs truncated to 128 tokens, which cuts off most of the
  evaluation reviews. That is why it scores 84.6%, no better than the LSTM. The
  comparison between its FP32 and INT8 versions does not depend on this. Latency is
  measured at a fixed 128 tokens. The LSTM and DistilBERT rows use different test
  reviews and tokenization, so they should not be compared with each other.
- **Dynamic INT8 in ONNX Runtime computes activation ranges per batch**, so its
  accuracy can differ slightly at other batch sizes. Everything here is batch size 1.
- **Two machines, one run each.** The committed numbers are from one x86 laptop CPU
  and one shared ARM64 cloud runner.
