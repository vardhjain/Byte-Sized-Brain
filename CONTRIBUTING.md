# Contributing

Thanks for taking a look. This is a small, focused project, so contributions are easy to
reason about. Bug reports, result reproductions, and new pipelines are all welcome.

## Getting set up

You need Python 3.11 or 3.12.

```bash
git clone https://github.com/vardhjain/Byte-Sized-Brain
cd Byte-Sized-Brain
python -m venv .venv && . .venv/bin/activate   # Windows PowerShell: .venv\Scripts\activate
pip install -c requirements.txt -e ".[all,dev]"   # every framework plus the dev tools, pinned
```

If you only care about one side of the project, the lighter extras `".[tf,dev]"` (the
TensorFlow pipelines) or `".[torch,dev]"` (the DistilBERT pipeline) install faster. The
`-c requirements.txt` part pins every package to the versions behind the committed
results.

Optionally, `pip install pre-commit && pre-commit install` runs the lint and formatting
checks on every commit.

## Before you open a pull request

Run the same checks CI runs. `make check` runs the first three in one go.

```bash
make lint        # ruff check + ruff format --check on the whole repository
make typecheck   # mypy on src
make test        # every test except the slow smoke ones (no training, no downloads)
make smoke       # train -> convert -> benchmark all four pipelines on a tiny subset (slow,
                 # downloads about 550 MB of datasets and weights the first time)
```

Without `make` (for example on Windows), run the commands from the Makefile recipes
directly, such as `pytest -m "not smoke" tests`.

Keep fast tests free of dataset downloads and training. A test that needs TensorFlow or
PyTorch should call `pytest.importorskip("tensorflow")` (or `"torch"`) so it skips
cleanly where that framework is not installed, and should carry the matching `tf` or
`torch` marker. Anything that downloads data or trains belongs under the `smoke` marker.

## Adding a new pipeline

A pipeline is a small, self-contained unit. To add one, follow these steps.

1. Add a data loader in `data/` and a model builder in `models/`.
2. Subclass `Pipeline` in `pipelines/`, set its `name`, `framework` and `quantization`,
   and implement `train`, `variants`, `convert` and `benchmark` (the last one usually
   just calls `benchmark_variants`).
3. Register it in `registry.py`.
4. Add `configs/<name>.yaml` with a `smoke` block. Its `convert.quantization` must match
   the pipeline's `quantization`, which the CLI and the tests check.

The shared harness handles the measurement, so you only describe the model and how it
converts.

## A note on results

The committed CSVs under `benchmarks/results/` are produced by the committed code, and
each row records the exact library versions it came from. If you change a pipeline in a
way that moves the numbers, rerun it with `bsb run <name>`, then run `bsb report` from the
repository root so the README table, `docs/report.md` and the dashboard image stay in
sync. Smoke runs write to `benchmarks/results/smoke/`, which is gitignored and never
reaches the report.
