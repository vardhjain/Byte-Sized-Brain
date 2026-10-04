"""seeding, sizing, sysinfo."""

from __future__ import annotations

import sys

import pytest

from byte_sized_brain.seeding import seed_everything
from byte_sized_brain.utils import (
    collect_sysinfo,
    library_versions,
    normalize_arch,
    size_bytes,
    size_mb,
)


def test_seed_is_returned_and_numpy_is_reproducible() -> None:
    import numpy as np

    # frameworks=() keeps this unit test from importing TensorFlow / PyTorch.
    assert seed_everything(123, frameworks=()) == 123
    a = np.random.rand(5)
    seed_everything(123, frameworks=())
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_seeding_only_imports_the_requested_frameworks() -> None:
    before = {m for m in ("tensorflow", "torch") if m in sys.modules}
    seed_everything(7, frameworks=())
    after = {m for m in ("tensorflow", "torch") if m in sys.modules}
    assert after == before


def test_size_of_file_and_dir(tmp_path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 2048)
    assert size_bytes(f) == 2048
    assert size_mb(f) == 2048 / (1024**2)

    d = tmp_path / "d"
    d.mkdir()
    (d / "1.bin").write_bytes(b"y" * 1000)
    (d / "2.bin").write_bytes(b"z" * 1500)
    assert size_bytes(d) == 2500


def test_size_of_missing_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        size_bytes(tmp_path / "nope")


def test_sysinfo_keys_and_emulated_flag(monkeypatch) -> None:
    # The ARM Docker recipe exports these, so clear them before asserting defaults.
    monkeypatch.delenv("BSB_EMULATED", raising=False)
    monkeypatch.delenv("BSB_DEVICE", raising=False)
    info = collect_sysinfo()
    assert {"device", "arch", "os", "python", "emulated"} <= set(info)
    assert info["emulated"] is False

    monkeypatch.setenv("BSB_EMULATED", "1")
    monkeypatch.setenv("BSB_DEVICE", "qemu-arm64")
    info2 = collect_sysinfo()
    assert info2["emulated"] is True
    assert info2["device"] == "qemu-arm64"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AMD64", "x86_64"),
        ("x86_64", "x86_64"),
        ("arm64", "aarch64"),
        ("aarch64", "aarch64"),
        ("riscv64", "riscv64"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_normalize_arch(raw, expected) -> None:
    assert normalize_arch(raw) == expected


def test_library_versions_is_a_dict() -> None:
    assert isinstance(library_versions(), dict)
