# Contributing

Thanks for taking a look. This is a small, focused project, so contributions are easy to
reason about. Bug reports, result reproductions, and new pipelines are all welcome.

## Getting set up

```bash
git clone https://github.com/vardhjain/Byte-Sized-Brain
cd Byte-Sized-Brain
pip install -e ".[all,dev]"     # package + every framework + dev tools
```

If you only care about one side of the project, the lighter extras `".[tf,dev]"` (the
TensorFlow pipelines) or `".[torch,dev]"` (the DistilBERT pipeline) install faster.

## Before you open a pull request

Run the same checks CI runs.

```bash
make lint        # ruff
make typecheck   # mypy
make test        # fast unit tests (no heavy frameworks)
make smoke       # full train -> convert -> benchmark on a tiny subset (slower)
```

Keep the unit tests fast and framework-free so the lint lane stays quick, and gate any
test that needs TensorFlow or PyTorch behind the existing `tf`, `torch`, or `smoke`
markers so it skips cleanly where that framework is absent.

## Adding a new pipeline

A pipeline is a small, self-contained unit. To add one, drop a loader in `data/`, a model
builder in `models/`, and a `Pipeline` subclass in `pipelines/` that wires train, convert,
and benchmark together, then register it in `registry.py` and add a `configs/<name>.yaml`
with a `smoke` block. The shared harness handles the measurement, so you only describe the
model and how it converts.

## A note on results

The committed CSVs under `benchmarks/results/` are produced by the committed code, and each
row records the exact library versions it came from. If you change a pipeline in a way that
moves the numbers, regenerate the affected CSV with `bsb run <name>` and `bsb report` so the
README table and the report stay in sync.
