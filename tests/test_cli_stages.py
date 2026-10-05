"""The train, convert and benchmark stage logic of the CLI, with the pipelines faked.

These guard the behaviour that protects the committed results: smoke runs stay in
their own folders, the smoke profile and ``--num-samples`` really reach the
pipeline, and stages run out of order fail with a one-line hint.
"""

from __future__ import annotations

import csv

from byte_sized_brain.cli import main
from byte_sized_brain.config import load_config
from byte_sized_brain.registry import all_names


def _rows(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_run_executes_the_three_stages_in_order(isolated_artifacts, fake_pipelines) -> None:
    assert main(["run", "ffn_mnist"]) == 0
    assert fake_pipelines.calls == [
        ("ffn_mnist", "train"),
        ("ffn_mnist", "convert"),
        ("ffn_mnist", "benchmark"),
    ]
    rows = _rows(isolated_artifacts / "results" / "ffn_mnist.csv")
    assert [r["variant"] for r in rows] == ["fp32", "int8"]


def test_smoke_run_never_touches_the_full_run_folders(isolated_artifacts, fake_pipelines) -> None:
    assert main(["run", "ffn_mnist"]) == 0
    full_csv = isolated_artifacts / "results" / "ffn_mnist.csv"
    full_before = full_csv.read_bytes()

    assert main(["run", "ffn_mnist", "--smoke"]) == 0

    assert full_csv.read_bytes() == full_before
    assert (isolated_artifacts / "results" / "smoke" / "ffn_mnist.csv").exists()
    assert (isolated_artifacts / "artifacts" / "smoke" / "ffn_mnist" / "fp32_source").is_dir()


def test_smoke_flag_applies_the_smoke_profile(isolated_artifacts, fake_pipelines) -> None:
    assert main(["run", "ffn_mnist", "--smoke"]) == 0
    smoke = load_config("ffn_mnist", smoke=True)
    full = load_config("ffn_mnist")
    assert smoke.benchmark.num_samples != full.benchmark.num_samples  # the test means something
    assert {c.benchmark.num_samples for c in fake_pipelines.configs} == {
        smoke.benchmark.num_samples
    }
    assert fake_pipelines.configs[0].train.epochs == smoke.train.epochs


def test_num_samples_option_reaches_the_benchmark(isolated_artifacts, fake_pipelines) -> None:
    assert main(["run", "ffn_mnist", "--num-samples", "7"]) == 0
    rows = _rows(isolated_artifacts / "results" / "ffn_mnist.csv")
    assert {r["num_samples"] for r in rows} == {"7"}


def test_all_runs_every_pipeline(isolated_artifacts, fake_pipelines) -> None:
    assert main(["train", "all"]) == 0
    assert fake_pipelines.calls == [(name, "train") for name in all_names()]


def test_convert_before_train_gives_a_hint(isolated_artifacts, fake_pipelines, cli_errors) -> None:
    assert main(["convert", "ffn_mnist", "--smoke"]) == 1
    assert fake_pipelines.calls == []
    assert "bsb train ffn_mnist --smoke" in cli_errors[0]


def test_benchmark_before_convert_gives_a_hint(
    isolated_artifacts, fake_pipelines, cli_errors
) -> None:
    assert main(["train", "ffn_mnist"]) == 0
    assert main(["benchmark", "ffn_mnist"]) == 1
    assert ("ffn_mnist", "benchmark") not in fake_pipelines.calls
    assert "bsb convert ffn_mnist`" in cli_errors[0]
