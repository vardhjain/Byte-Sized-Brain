# Reproduce and verify the results

Byte-Sized Brain intentionally separates **verifying the benchmark code** from
**retraining models**. The former is inexpensive and automated; the latter is
resource-intensive and produces new experiments, not guaranteed identical
weights.

## 1. Fast verification — no model downloads

Use Python 3.11 or 3.12, from the repository root:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -c requirements.txt -e ".[dev]"
bsb list
pytest -m "not smoke" tests
```

This installs the core, test, and reporting dependencies, **not** TensorFlow,
PyTorch or the training datasets. Framework-specific integration tests may
skip when those frameworks are absent; the GitHub Actions [CI workflow](
https://github.com/vardhjain/Byte-Sized-Brain/actions/workflows/ci.yml)
installs them in separate jobs and exercises x86_64/ARM64 and Python 3.11/3.12.

Especially useful tests to inspect:

| Claim | Reproducible evidence |
| --- | --- |
| Model-only, input-preparation and combined latency use the same calls | [Deterministic clock tests](../tests/test_harness.py) |
| Accuracy deltas account for paired FP32/INT8 predictions | [McNemar and paired bootstrap tests](../tests/test_statistics.py) |
| Duplicate experiments do not overwrite unrelated runs | [CSV/run-ID tests](../tests/test_metrics.py) |
| Architecture comparisons use byte-identical model files | [Bundle integrity tests](../tests/test_model_bundle.py) |
| Published headline matches committed CSV values | [Committed-result tests](../tests/test_committed_results.py) |

The tests check **implementation correctness**, not new hardware performance.

## 2. Inspect the historical evidence

The benchmark inputs for the documentation are the committed files under
[`benchmarks/results/`](../benchmarks/results/). The CNN quantization
investigation is separately recorded in
[`benchmarks/results/ablations/cnn_int8_ablation.csv`](../benchmarks/results/ablations/cnn_int8_ablation.csv).

```bash
# With the core dependencies installed:
bsb report
git diff -- README.md docs/report.md docs/images/results_dashboard.png
```

`bsb report` regenerates the detailed Markdown table, dashboard image and the
README's marked results block **from the committed CSV files**. It does not
train models or change the CSVs. Review the diff before committing generated
files. Older results lack `run_id`, `model_sha256`, paired accuracy intervals,
timing-pass intervals and input-preparation-inclusive latency; absent
measurements must stay blank, **not be guessed**. The report generator can
reformat old results using the current schema without creating new evidence.

The historical x86 and ARM64 runs were each single-session measurements
using separately trained model artifacts. They are useful case studies, but
**not** a same-weights, repeated hardware comparison.

## 3. Train, convert and benchmark one small model

This path installs CPU ML frameworks and downloads datasets. It takes more
disk space and time than the fast test path. No paid GPU is required.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
bsb train ffn_mnist
bsb convert ffn_mnist
bsb benchmark ffn_mnist --repeats 5
bsb report
```

Run IDs keep experimental sessions separate. The output includes:
`latency_ms_*` (model-only inference),
`input_prep_latency_ms_*` (tensor/feed preparation),
`e2e_latency_ms_*` (preparation + runtime inference),
`model_sha256` (model bytes), the accuracy delta's paired bootstrap interval,
and McNemar's exact test. Pass-level latency intervals require
`--repeats 2` or more.

**Timing boundary:** "end-to-end" here means a *preprocessed evaluation
example* passed through `runner.prepare()` and `runner.invoke()`.
Image loading/resizing, raw text tokenization, model initialization,
network overhead and class-label decoding are excluded. See the
[methodology](methodology.md#why-these-choices).

The CNN's extra quantization alternatives are **opt-in**, and one selective
variant uses an experimental TensorFlow API:

```bash
bsb train cnn_cifar10
bsb convert cnn_cifar10 --cnn-alternatives
bsb benchmark cnn_cifar10 --cnn-alternatives --repeats 5
```

These run on the same trained model and evaluation examples. They do **not**
magically reproduce the historical ablation without training the original
model and preparing the same environment.

## 4. Controlled x86 vs native ARM64

Go to [Actions → Cross-hardware benchmark (identical artifacts)](
https://github.com/vardhjain/Byte-Sized-Brain/actions/workflows/arm-benchmark.yml)
and select **Run workflow**. It defaults to `ffn_mnist`; other pipelines,
especially full DistilBERT training, require considerably more time.

The workflow trains and converts once on x86, packages the inference files
and manifest, verifies SHA-256 digests on both native architectures, runs
benchmark-only jobs and **fails** if the resulting model hashes or
evaluation sample counts disagree. Download `results-x86_64` and
`results-aarch64` after a successful run. This workflow has been
implemented and tested in CI, but its **full empirical comparison is not
included** in the committed historical numbers.

It uses GitHub-hosted runners and is manual-only. GitHub Actions minutes,
storage allowances, runner availability and any applicable billing depend
on repository/account settings; inspect the usage limits before triggering
an expensive run.

## 5. What the evidence does not show

The current results do not establish performance across all processors,
all thread counts, repeated model retrainings, thermal conditions or
unseen datasets. The historical per-model Wilson intervals do not prove
a small FP32/INT8 difference is statistically insignificant. The new
paired-accuracy statistics fix that comparison **for newly evaluated
examples**, but cannot be reconstructed from aggregate historical accuracy
alone. Neither a smoke test nor a best-of-N demo measurement should be
substituted for full held-out evaluation.

For architecture trade-offs and quantization limitations, read the
[methodology](methodology.md) and [recorded results](report.md).
