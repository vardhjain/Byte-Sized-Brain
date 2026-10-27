"""Regression tests for the required-suite CI wrapper.

Fake pytest exercises the shell's actual pipefail behavior without needing any
TensorFlow, ONNX or other heavy integration-test dependency.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_WRAPPER = _REPO / "scripts" / "ci_pytest.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="The integration job runs on Bash")
@pytest.mark.parametrize(
    ("pytest_output", "pytest_exit", "expected_exit"),
    [
        ("3 passed", 0, 0),
        ("1 failed", 7, 7),
        ("SKIPPED [1] tests/test_memory.py", 0, 1),
        ("SKIPPED [1] tests/test_memory.py", 2, 2),
    ],
)
def test_ci_pytest_propagates_status_and_rejects_skips(
    tmp_path: Path, pytest_output: str, pytest_exit: int, expected_exit: int
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_pytest = bindir / "pytest"
    fake_pytest.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" > "$MOCK_ARG_FILE"\n'
        'printf "%s\\n" "$MOCK_OUTPUT"\n'
        'exit "$MOCK_EXIT"\n',
        encoding="utf-8",
    )
    fake_pytest.chmod(0o755)

    args_file = tmp_path / "pytest_args.txt"
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(bindir) + os.pathsep + env.get("PATH", ""),
            "MOCK_OUTPUT": pytest_output,
            "MOCK_EXIT": str(pytest_exit),
            "MOCK_ARG_FILE": str(args_file),
        }
    )
    result = subprocess.run(
        ["bash", str(_WRAPPER), "tests/test_memory.py", "tests/test_hf_space.py"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == expected_exit
    assert pytest_output in result.stdout
    assert args_file.read_text(encoding="utf-8").strip() == (
        "-rs tests/test_memory.py tests/test_hf_space.py"
    )
    if pytest_exit == 0 and expected_exit != 0:
        assert "Required pytest tests were skipped" in result.stderr


def test_ci_workflow_uses_checked_pytest_wrapper() -> None:
    workflow = (_REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert (
        "bash scripts/ci_pytest.sh tests/test_onnx_roundtrip.py "
        "tests/test_memory.py tests/test_hf_space.py"
    ) in workflow
