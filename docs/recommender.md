# Constraint-based quantization recommender

A tiny, explainable deployment decision tool on top of the project's **measured**
benchmark CSVs. It does not train a predictor, invent a speed estimate or search
hardware specifications. It answers: *which of the models actually measured on
this CPU satisfies my accuracy, latency, size and RSS budget?*

## Examples: no ML dependencies or retraining

Install the [core plus development environment](reproducibility.md#1-fast-verification--no-model-downloads)
and use the historical CSVs already in the repository:

```bash
# Fastest measured MNIST variant with <= 1 point observed accuracy loss
# and at most 0.2 MiB on disk:
bsb recommend ffn_mnist --arch x86_64 --max-size-mb 0.2

# Require a CNN no larger than 3 MiB and no more than 1 point accuracy drop.
# With the *historical static-INT8-only* results, no variant qualifies.
bsb recommend cnn_cifar10 --max-size-mb 3 --max-accuracy-drop-pts 1

# Optimize footprint instead of speed, with custom limits:
bsb recommend ffn_mnist --objective size --max-memory-mb 2

# Machine-readable selection, feasibility reasons and Pareto set:
bsb recommend ffn_mnist --max-size-mb 0.2 --json
```

The recommender reads only the current top-level `benchmarks/results/*.csv`
files; it does **not** read smoke runs, quantization ablation snapshots or
model binaries. It does not mutate the CSV or README. It includes the FP32
baseline as an option: if all quantized variants lose unacceptable accuracy,
the correct recommendation may be **keep FP32**. If no variant meets every
constraint, the command reports *no feasible candidate* and returns exit
code 2 instead of pretending a model meets the budget.

## Available controls

| CLI option | Meaning |
| --- | --- |
| `--max-accuracy-drop-pts` | Max **observed** percentage-point accuracy loss relative to FP32, default 1.0 |
| `--min-accuracy` | Minimum absolute accuracy as a **fraction** (e.g. `0.90`, not 90) |
| `--max-size-mb` | Largest permitted model file, in MiB |
| `--max-latency-ms` | Max mean inference time in milliseconds |
| `--max-p95-ms` | Max p95 inference time in milliseconds |
| `--max-memory-mb` | Max model-related RSS increase in MiB |
| `--objective` | Among feasible variants, optimize `latency` (default), `size`, `memory`, or `accuracy` |
| `--latency-kind` | `model` (default, same as historical results) or `prepared-input` (includes runtime tensor preparation) |
| `--require-paired-ci` | Only accept candidates with a paired accuracy-difference 95% CI whose lower bound meets the accuracy-loss budget |
| `--arch` | Architecture of the measurement, default `x86_64` (use `aarch64` for native ARM) |
| `--device`, `--run-id` | Optional exact hardware label or experiment ID to remove ambiguity |
| `--results-dir` | Directory of real benchmark CSVs (defaults to `benchmarks/results`) |
| `--json` | JSON output for downstream pipelines and audit logs |

The accuracy requirement is **observational** by default, not a claim of
statistical significance. For a more conservative gate, run the modern paired
benchmark first, then pass `--require-paired-ci`:

```bash
bsb benchmark cnn_cifar10 --cnn-alternatives --repeats 5
bsb recommend cnn_cifar10 --max-size-mb 3 --max-accuracy-drop-pts 1 --require-paired-ci
```

The optional `--cnn-alternatives` conversion/benchmark must have been run to
make those variants eligible. On the historical October 2026 CSVs, there
are no stored paired accuracy confidence intervals; with the CI requirement
enabled, this tool **will not guess them** or claim a feasible quantizer.

Similarly, `--latency-kind prepared-input` requires newer measured CSVs
with `e2e_latency_ms_*`; it fails clearly on old results instead of
substituting model-only latency.

## How a selection is made

1. Filter to one named pipeline, native (not QEMU) CPU architecture, optional
   device and experiment run ID. Choose the **newest complete run containing
   FP32 plus at least one quantized variant**, never mixing different runs
   or switching to an older run because the newest fails a constraint.
2. Require the candidate's observed accuracy, size, latency mean/p95 and RSS
   delta. Missing measurements disqualify the candidate.
3. Apply every threshold independently and preserve human-readable rejection
   reasons. FP32 is treated fairly as an ordinary measured candidate.
4. Mark Pareto-efficient feasible candidates over **accuracy (maximize)**,
   **mean latency**, **model size**, and **RSS memory** (minimize). Pick the
   feasible candidate with the requested single objective. No hidden weighted
   score or unrecorded performance extrapolation.
5. Show the chosen run ID, CPU name, evaluated sample count and all rejected
   options. The command returns exit code 0 for a recommendation, 2 when
   constraints have no feasible measured option and 1 for invalid inputs.

## Limits

A recommendation is **conditional on this exact measured experiment**,
runtime, CPU, thread setting, dataset and score definition. It does not
prove a result transfers to a phone or another processor. The historical
x86/ARM measurements were made with independently trained models; the
recommender never mixes those rows. RSS delta is *not* total application RAM,
and prepared-input time is *not* raw-request end-to-end latency. A single
timing pass is not evidence of repeatability.

This tool is a transparent decision aid, not an automated search over
unmeasured architectures or a trained meta-model. Refer to the
[measurement methodology](methodology.md) and [reproduction guide](reproducibility.md)
before using the result for a real deployment.
