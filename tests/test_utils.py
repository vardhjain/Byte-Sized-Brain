"""seeding, sizing, sysinfo."""

from __future__ import annotations

from byte_sized_brain.seeding import seed_everything
from byte_sized_brain.utils import collect_sysinfo, library_versions, size_bytes, size_mb


def test_seed_is_returned_and_numpy_is_reproducible() -> None:
    import numpy as np

    assert seed_everything(123) == 123
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


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


def test_sysinfo_keys_and_emulated_flag(monkeypatch) -> None:
    info = collect_sysinfo()
    assert {"device", "arch", "os", "python", "emulated"} <= set(info)
    assert info["emulated"] is False

    monkeypatch.setenv("BSB_EMULATED", "1")
    monkeypatch.setenv("BSB_DEVICE", "qemu-arm64")
    info2 = collect_sysinfo()
    assert info2["emulated"] is True
    assert info2["device"] == "qemu-arm64"


def test_library_versions_is_a_dict() -> None:
    assert isinstance(library_versions(), dict)
