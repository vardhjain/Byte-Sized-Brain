"""The committed benchmark CSVs and the README table agree with the code.

These tests only read files from the repository. They guard the numbers a
reader sees on the front page, for example against a smoke run overwriting a
full run or a README table that was not regenerated after a re-benchmark.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from byte_sized_brain.benchmark import RESULT_COLUMNS
from byte_sized_brain.config import Paths, load_config
from byte_sized_brain.registry import all_names, get_pipeline
from byte_sized_brain.report import _readme_table, load_results

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
COMMITTED = sorted(RESULTS.glob("*.csv"))


def test_every_pipeline_has_committed_results() -> None:
    assert {p.stem for p in COMMITTED} == set(all_names())


@pytest.mark.parametrize("csv_path", COMMITTED, ids=lambda p: p.stem)
def test_committed_results_are_full_runs_in_the_current_schema(
    csv_path: Path, tmp_path: Path
) -> None:
    name = csv_path.stem
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == RESULT_COLUMNS

    full = load_config(name)
    pipeline = get_pipeline(name)
    expected = {v.name: v.quantization for v in pipeline.variants(full, Paths(name, root=tmp_path))}

    groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        assert row["pipeline"] == name
        assert int(row["num_samples"]) == full.benchmark.num_samples, "smoke rows were committed"
        groups[(row["arch"], row["emulated"])][row["variant"]] = row["quantization"]
    for variants in groups.values():
        assert variants == expected


def test_readme_table_matches_the_committed_results() -> None:
    results = load_results(RESULTS)
    arch = results[~results["emulated"]]["arch"].mode().iloc[0]
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert _readme_table(results, arch) in readme, "run `bsb report` to refresh the README"
