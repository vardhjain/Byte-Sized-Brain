"""Fast checks that portfolio evidence and reproduction documentation stay truthful."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    Path("README.md"),
    Path("docs/index.md"),
    Path("docs/methodology.md"),
    Path("docs/reproducibility.md"),
    Path("docs/recommender.md"),
    Path("docs/report.md"),
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


@pytest.mark.parametrize("page", PAGES)
def test_all_local_page_links_resolve(page: Path) -> None:
    """Catch renamed files and broken portfolio entry points without network access."""
    content = (ROOT / page).read_text(encoding="utf-8")
    bad = []
    for href in MARKDOWN_LINK.findall(content):
        destination = href.split("#", 1)[0].strip()
        if not destination or "://" in destination or destination.startswith(("mailto:", "#")):
            continue
        if not (ROOT / page.parent / destination).exists():
            bad.append(destination)
    assert not bad, f"Broken links in {page}: {bad}"


def test_readme_links_actual_ci_reproducibility_and_analysis() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## The engineering work" in readme
    assert "docs/reproducibility.md" in readme
    assert "src/byte_sized_brain/benchmark/statistics.py" in readme
    assert ".github/workflows/arm-benchmark.yml" in readme
    assert "scripts/cnn_int8_ablation.py" in readme
    assert "separately trained" in readme
    assert "model-only" in readme
    assert 'the "ARM64 benchmark" workflow' not in readme


def test_results_do_not_claim_old_wilson_intervals_test_paired_accuracy() -> None:
    text = (ROOT / "docs/report.md").read_text(encoding="utf-8")
    assert "Historical results snapshot" in text
    assert "cannot test a paired FP32/INT8 accuracy difference" in text
    assert "a difference smaller than it is within noise" not in text
    assert "not byte-identical" in text


def test_quick_verification_avoids_training_and_includes_real_tests() -> None:
    guide = (ROOT / "docs/reproducibility.md").read_text(encoding="utf-8")
    quick = guide.split("## 1. Fast verification", 1)[1].split("## 2.", 1)[0]
    assert 'python -m pip install -c requirements.txt -e ".[dev]"' in quick
    assert 'pytest -m "not smoke" tests' in quick
    assert "bsb train" not in quick
    for test in (
        "tests/test_harness.py",
        "tests/test_statistics.py",
        "tests/test_model_bundle.py",
        "tests/test_committed_results.py",
    ):
        assert test in quick


def test_mkdocs_exposes_the_reproducibility_guide() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Reproduce and verify: reproducibility.md" in config
    assert "Choose a variant: recommender.md" in config
