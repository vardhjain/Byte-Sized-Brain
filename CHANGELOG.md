# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [0.1.0]

The first packaged release, a full rebuild of an earlier collection of loose scripts into
a reproducible toolkit.

### Added
- An installable `byte_sized_brain` package with a single `bsb` CLI covering train,
  convert, benchmark, run, report, info, list, and demo.
- Four pipelines spanning vision, sequence, and NLP across TensorFlow and PyTorch, with
  TFLite and ONNX Runtime conversion.
- One device-agnostic benchmark harness that records size, accuracy, latency, and
  process-level memory, with every result row stamped by device, architecture, and the
  exact library versions.
- YAML configs with a fast `smoke` profile, deterministic seeding, Docker images for x86
  and ARM64, a methodology write-up, a results dashboard, and an FP32-versus-INT8 demo
  (CLI and Streamlit).
- A pytest suite and GitHub Actions CI covering lint, types, unit tests, a real
  end-to-end pipeline smoke, and ARM64 emulation.

### Fixed
- Representative datasets for static quantization now use real samples instead of random
  noise.
- The DistilBERT FP32 ONNX graph is exported and kept, so the comparison has both graphs.
- TFLite INT8 inputs are saturated rather than wrapped, matching the runtime.
- The LSTM exports with a static batch so it lowers to a native builtin and needs no Flex
  delegate.

### Changed
- Removed large model binaries from git history, shrinking the clone from about 78 MB to
  roughly 1 MB.
