# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [Unreleased]

### Added
- A manually started workflow that runs the full benchmark on GitHub's native ARM64
  runner and uploads the result files.
- Coverage upload to Codecov and a coverage badge in the README.

### Fixed
- DistilBERT training works on a fresh install of the pinned requirements. It used to
  fail unless the `tf-keras` package happened to be installed.

## [0.3.2] - 2026-10-05

### Added
- A coverage report for the fast tests, shown on each CI run's summary page and
  available locally through `make coverage`.

## [0.3.1] - 2026-10-05

### Changed
- The README table calls a timing difference under 15 percent "about the same",
  which matches the run-to-run variation of the measurements.
- The Streamlit demo describes the result it actually got, keeps the models loaded
  between clicks and shows readable model names.
- CI now runs the ONNX round trip, memory probe and hosted-demo tests, which were
  previously skipped there.
- Running a stage out of order, a bad option or a bad config now ends with a one-line
  message and exit code 1 (for example a hint to run `bsb train` first) instead of a
  traceback.
- The fast test suite checks exact values: latency statistics on a fake clock, the full
  report on hand-built results, and the CLI stage logic with the pipelines faked.

### Removed
- The unused `flex` option of the TFLite converters.

### Fixed
- The documentation site deploys from `main` again. The deploy step was still tied
  to the old branch name, so the site had not updated since the rename.
- The README shows the hosted demo next to its link, and the local Streamlit app
  next to its own instructions.

## [0.3.0] - 2026-10-05

### Added
- An `agreement` column (how often a variant predicts the same label as its FP32
  baseline) and a `threads` column in the result files. Benchmarks were re-run.
- 95 percent confidence intervals for accuracy in `docs/report.md`.
- `BSB_TFLITE_THREADS`, to set the TFLite interpreter's thread count.
- Tests that check the committed result files against the configs and the README table.

## [0.2.1] - 2026-10-04

### Changed
- The default branch is now `main`.
- Dependabot proposes updates for the dev tools only. The ML packages stay pinned to
  the versions behind the committed results.
- Updated the pinned dev tools (pytest, ruff, mypy, mkdocs-material).

### Fixed
- The hosted demo limits ONNX Runtime to the CPUs its container may use, which stops
  the full-precision timing from jumping between runs.
- The deploy script no longer tries to create a Space that already exists, which
  Hugging Face rejects on free accounts.
- The `onnx` pytest marker is declared, so newer pytest releases accept the test suite.

## [0.2.0] - 2026-10-04

A measurement-correctness release. The benchmark numbers were regenerated with the
fixed harness, so latency and memory figures differ from 0.1.0. Sizes and accuracies are
unchanged.

### Added
- A memory probe that measures each model variant in a fresh process, so the memory
  columns now show the real footprint of loading and running the model.
- `scripts/cnn_int8_ablation.py`, which converts the CNN nine ways to show where its
  INT8 accuracy loss comes from, with results in `benchmarks/results/ablations/`.
- `bsb demo --smoke`, `make check`, `make smoke-all`, `make ablation-cnn` and `make docs`.
- A plain-language results table in the README, including memory.
- Tests for the CLI, the memory probe, the ONNX round trip, CNN fine-tuning, the hosted
  demo and the README results table.
- `.editorconfig`, a pre-commit configuration, Dependabot, `CITATION.cff` and explicit
  line-ending rules.

### Changed
- Latency now times only the model call. Input preparation, such as the float to INT8
  conversion for full-integer models, happens before the clock starts.
- Smoke runs keep their models in `artifacts/smoke/` and their results in
  `benchmarks/results/smoke/`, so they can no longer overwrite a full run.
- Writing results merges by architecture. A rerun replaces only the rows of the same
  pipeline, architecture and `emulated` flag.
- Architecture names are normalized (`AMD64` becomes `x86_64`), the `device` column
  records the CPU model instead of the machine hostname, and Windows 11 is reported
  correctly.
- The CLI rejects a config that names a different pipeline or a different quantization
  technique than the pipeline implements, and rejects `--config` together with `all`.
- The FFN and LSTM pipelines honor `train.learning_rate` from their configs.
- CIFAR-10 loading preprocesses only the images a stage needs, and DistilBERT picks its
  training subset before tokenizing.
- The supported Python versions are 3.11 and 3.12, which is what the pinned
  requirements can install on.
- CI installs the pinned versions, checks formatting and types, and runs on native x86
  and ARM64 runners.
- The hosted demo pins its Gradio version, handles empty input, reports real file sizes
  and says when the two models disagree.

### Fixed
- Fine-tuning the CNN backbone no longer updates its BatchNorm statistics.
- The ONNX export pins the TorchScript exporter, keeping the graph identical across
  torch versions.
- The TFLite loader only falls back to LiteRT when TensorFlow's interpreter is
  unavailable, so a real model-loading error is no longer hidden.
- The documentation no longer claims the LSTM converts to the fused LSTM op (it becomes
  a loop of builtin ops), and no longer blames the CNN accuracy drop on per-tensor
  weight quantization.

### Removed
- The unused `convert.flex_ops` config field.

## [0.1.0]

The first packaged version, a full rebuild of an earlier collection of loose scripts
into a reproducible toolkit.

### Added
- An installable `byte_sized_brain` package with a single `bsb` CLI covering train,
  convert, benchmark, run, report, info, list, and demo.
- Four pipelines (MNIST, CIFAR-10 and two IMDB sentiment models) across TensorFlow and
  PyTorch, with TFLite and ONNX Runtime conversion.
- One benchmark harness that records size, accuracy and latency, with every result row
  stamped by device, architecture, and the exact library versions.
- YAML configs with a fast `smoke` profile, seeding, Docker images for x86 and ARM64, a
  methodology write-up, a results dashboard, and an FP32-versus-INT8 demo.
- A pytest suite and GitHub Actions CI.

### Changed
- Removed large model binaries from git history, shrinking the repository from about
  78 MB to under 1 MB.

### Fixed
- Representative datasets for static quantization use real samples instead of random
  noise.
- The DistilBERT FP32 ONNX graph is exported and kept, so the comparison has both graphs.
- TFLite INT8 inputs are saturated rather than wrapped, matching the runtime.
- The LSTM exports with a static batch, so it converts to plain TFLite builtin ops and
  needs no Flex delegate.
