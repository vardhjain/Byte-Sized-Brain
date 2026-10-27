"""Constraint-based recommendations use only paired native *measured* rows."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from byte_sized_brain.cli import main
from byte_sized_brain.recommend import Constraints, recommend, recommend_from_results


def _run(
    *,
    name: str = "experiment-1",
    stamp: str = "2026-10-26T12:00:00Z",
    device: str = "CPU A",
    arch: str = "x86_64",
) -> list[dict]:
    fields = dict(
        pipeline="cnn_cifar10",
        arch=arch,
        emulated=False,
        run_id=name,
        timestamp=stamp,
        device=device,
        os="Linux",
        num_samples=1000,
    )
    return [
        dict(
            fields,
            variant="fp32",
            quantization="none",
            accuracy=0.90,
            size_mb=4.0,
            latency_ms_mean=10.0,
            latency_ms_p95=13.0,
            rss_delta_mb=20.0,
            e2e_latency_ms_mean=14.0,
            e2e_latency_ms_p95=16.0,
            accuracy_delta_ci95_low=0.0,
        ),
        dict(
            fields,
            variant="int8",
            quantization="static_int8",
            accuracy=0.895,
            size_mb=2.0,
            latency_ms_mean=6.0,
            latency_ms_p95=9.0,
            rss_delta_mb=9.0,
            e2e_latency_ms_mean=12.0,
            e2e_latency_ms_p95=14.0,
            accuracy_delta_ci95_low=-0.025,
        ),
        dict(
            fields,
            variant="dynamic_range",
            quantization="dynamic_range",
            accuracy=0.90,
            size_mb=1.0,
            latency_ms_mean=8.0,
            latency_ms_p95=10.0,
            rss_delta_mb=11.0,
            e2e_latency_ms_mean=9.0,
            e2e_latency_ms_p95=11.0,
            accuracy_delta_ci95_low=-0.005,
        ),
    ]


def _pick(rows: list[dict], constraints: Constraints | None = None, **kwargs) -> dict:
    return recommend(
        pd.DataFrame(rows), pipeline="cnn_cifar10", constraints=constraints, **kwargs
    )


def test_speed_and_size_objectives_disagree_and_both_are_pareto() -> None:
    measured = _pick(_run())
    assert measured["recommended_variant"] == "int8"
    assert measured["pareto_variants"] == ["dynamic_range", "int8"]
    assert measured["baseline_accuracy"] == 0.9
    assert measured["device"] == "CPU A"

    small = _pick(_run(), Constraints(objective="size"))
    assert small["recommended_variant"] == "dynamic_range"
    assert small["candidates"][0]["pareto"] is False


def test_unmet_constraints_cannot_fabricate_a_better_quantizer() -> None:
    result = _pick(
        _run(), Constraints(max_size_mb=1.5, max_latency_ms=7.0, max_accuracy_drop_pts=1)
    )
    assert result["recommended_variant"] is None
    assert all(not c["feasible"] for c in result["candidates"])
    assert "latency_mean_ms" in " ".join(result["candidates"][2]["reasons"])


def test_recommender_can_safely_choose_unquantized_fp32() -> None:
    rows = _run()
    rows[1]["accuracy"] = 0.70
    rows[2]["accuracy"] = 0.80
    result = _pick(rows)
    assert result["recommended_variant"] == "fp32"
    assert len([c for c in result["candidates"] if c["feasible"]]) == 1


def test_newest_complete_native_pair_skips_partial_or_emulated_data() -> None:
    first = _run(name="old", stamp="2026-10-24")
    new = _run(name="new", stamp="2026-10-25", device="CPU B")
    partial = _run(name="partial", stamp="2026-10-27")[:1]
    emulated = [{**r, "emulated": True, "run_id": "emulated", "timestamp": "2026-10-29"} for r in new]
    result = _pick(first + new + partial + emulated)
    assert result["run_id"] == "new"
    assert result["device"] == "CPU B"
    assert _pick(first + new, device="CPU A")["run_id"] == "old"
    assert _pick(first + new, run_id="old")["run_id"] == "old"


def test_never_pairs_different_hardware_or_evaluation_sizes() -> None:
    rows = _run()
    rows[2]["device"] = "CPU B"
    with pytest.raises(ValueError, match="Inconsistent device"):
        _pick(rows)
    rows = _run()
    rows[2]["num_samples"] = 500
    with pytest.raises(ValueError, match="Inconsistent num_samples"):
        _pick(rows)


def test_prepared_input_selection_requires_actual_inclusive_timings() -> None:
    constraints = Constraints(latency_kind="prepared-input")
    measured = _pick(_run(), constraints)
    assert measured["recommended_variant"] == "dynamic_range"
    old = _run()
    for row in old:
        del row["e2e_latency_ms_mean"]
        del row["e2e_latency_ms_p95"]
    with pytest.raises(ValueError, match="Re-benchmark"):
        _pick(old, constraints)


def test_paired_ci_gate_refuses_legacy_data_but_accepts_supported_option() -> None:
    conservative = Constraints(require_paired_ci=True)
    rows = _run()
    result = _pick(rows, conservative)
    # INT8's -2.5 point lower bound does not meet the 1-point allowed loss.
    assert result["recommended_variant"] == "dynamic_range"
    assert not result["candidates"][1]["feasible"]
    assert "paired 95%" in " ".join(result["candidates"][1]["reasons"])

    for row in rows:
        del row["accuracy_delta_ci95_low"]
    result = _pick(rows, conservative)
    assert result["recommended_variant"] is None
    assert all("unmeasured: paired" in str(c["reasons"]) for c in result["candidates"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_accuracy_drop_pts", -1),
        ("max_accuracy_drop_pts", 101),
        ("max_size_mb", 0),
        ("max_latency_ms", -1),
        ("max_p95_ms", 0),
        ("max_memory_mb", float("nan")),
        ("min_accuracy", 95),
    ],
)
def test_invalid_thresholds_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(Constraints(), **{field: value})


def test_cli_json_uses_custom_results_directory_without_model_downloads(
    tmp_path, capsys
) -> None:
    rows = _run()
    pd.DataFrame(rows).to_csv(tmp_path / "cnn_cifar10.csv", index=False)
    assert main(
        [
            "recommend",
            "cnn_cifar10",
            "--results-dir",
            str(tmp_path),
            "--max-accuracy-drop-pts",
            "1",
            "--max-size-mb",
            "2.1",
            "--json",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["recommended_variant"] == "int8"
    assert result["candidates"][0]["feasible"] is False
    assert result["constraints"]["max_size_mb"] == 2.1

    assert main(
        [
            "recommend",
            "cnn_cifar10",
            "--results-dir",
            str(tmp_path),
            "--max-size-mb",
            "0.1",
        ]
    ) == 2
    assert "No measured variant" in capsys.readouterr().out


def test_recommender_reads_committed_historical_results() -> None:
    # The real MNIST x86 results have 0.97 FP32 and INT8 accuracy, with a
    # 0.1 MiB INT8 model. This asserts the CLI can use history with no models.
    result = recommend_from_results(
        "ffn_mnist",
        constraints=Constraints(max_size_mb=0.2, max_accuracy_drop_pts=1),
    )
    assert result["recommended_variant"] == "int8"
    assert result["historical"] is True
    assert result["arch"] == "x86_64"
