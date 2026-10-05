"""The ``bsb`` command line, exercised without training anything."""

from __future__ import annotations

import json

import pytest

from byte_sized_brain import __version__
from byte_sized_brain.cli import main
from byte_sized_brain.registry import all_names


def test_list_prints_every_pipeline(capsys) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for name in all_names():
        assert name in out


def test_info_is_valid_json(capsys) -> None:
    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["bsb_version"] == __version__
    assert {"device", "arch", "emulated", "libraries"} <= set(info)


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_unknown_pipeline_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["train", "not_a_pipeline"])
    assert exc.value.code == 2


def test_config_flag_cannot_be_combined_with_all() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["benchmark", "all", "--config", "configs/ffn_mnist.yaml"])
    assert exc.value.code == 2


def test_config_claiming_the_wrong_quantization_is_rejected(
    tmp_path, isolated_artifacts, cli_errors
) -> None:
    """The LSTM pipeline is dynamic-range only; a config can't relabel it INT8."""
    bad = tmp_path / "rnn_imdb.yaml"
    bad.write_text(
        "name: rnn_imdb\nframework: tensorflow\nmodality: sequence\n"
        "dataset: imdb\nmodel: lstm\nconvert:\n  quantization: static_int8\n",
        encoding="utf-8",
    )
    assert main(["convert", "rnn_imdb", "--config", str(bad)]) == 1
    assert "dynamic_range" in cli_errors[0]


def test_report_without_results_fails_cleanly(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BSB_RESULTS", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    assert main(["report"]) == 1


def test_demo_without_artifacts_fails_cleanly(isolated_artifacts) -> None:
    assert main(["demo", "rnn_imdb", "--smoke"]) == 1


def test_config_for_a_different_pipeline_is_rejected(isolated_artifacts, fake_pipelines) -> None:
    """`bsb train ffn_mnist --config rnn_imdb.yaml` must not quietly train the LSTM."""
    assert main(["train", "ffn_mnist", "--config", "configs/rnn_imdb.yaml"]) == 1
    assert fake_pipelines.calls == []


def test_num_samples_must_be_positive(isolated_artifacts, fake_pipelines) -> None:
    assert main(["benchmark", "ffn_mnist", "--num-samples", "0"]) == 1
    assert fake_pipelines.calls == []
