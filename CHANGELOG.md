# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

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
  demo and report generation.
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
