"""Package and verify identical model artifacts for x86/ARM64 benchmarking.

Only inference artifacts are transferred. For DistilBERT, include the tokenizer
files needed by the benchmark, not the large training checkpoint. Every file
and the relevant YAML config is SHA-256 verified before any benchmark starts.

Usage:
    python -m byte_sized_brain.benchmark.bundle pack ffn_mnist /tmp/model-bundle
    python -m byte_sized_brain.benchmark.bundle install ffn_mnist /tmp/model-bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

_FORMAT_VERSION = 1
_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
)


def file_sha256(path: Path) -> str:
    """Hash an artifact in chunks without loading a large ONNX model in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names(selected: str) -> list[str]:
    from ..registry import all_names

    choices = all_names()
    if selected == "all":
        return choices
    if selected not in choices:
        raise ValueError(f"Unknown pipeline {selected!r}; expected one of: {', '.join(choices)}")
    return [selected]


def _config_hash(name: str) -> str:
    from ..config import resolve_config_path

    return file_sha256(resolve_config_path(name))


def _relative_path(root: Path, relative: str) -> Path:
    """Reject absolute/traversal paths and links in an untrusted bundle."""
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not relative.startswith("artifacts/"):
        raise ValueError(f"Unsafe bundle path: {relative}")
    item = root / candidate
    if any(p.is_symlink() for p in (item, *item.parents) if p != root and root in p.parents):
        raise ValueError(f"Bundle symlink is not allowed: {relative}")
    if not item.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes its root: {relative}")
    return item


def _files_for(name: str, source: Path) -> list[tuple[str, Path, str]]:
    from ..config import Paths, load_config
    from ..registry import get_pipeline

    cfg = load_config(name)
    paths = Paths(name, root=source, create=False)
    files: list[tuple[str, Path, str]] = []
    for v in get_pipeline(name).variants(cfg, paths):
        files.append((f"artifacts/{name}/{v.path.name}", v.path, v.name))
    if name == "distilbert_imdb":
        source_dir = paths.fp32_source
        for filename in _TOKENIZER_FILES:
            target = source_dir / filename
            if target.is_file():
                files.append((f"artifacts/{name}/fp32_source/{filename}", target, "tokenizer"))
        if not any(f[1].name in {"tokenizer.json", "vocab.txt"} for f in files):
            raise FileNotFoundError("DistilBERT tokenizer files are missing from fp32_source")
    return files


def pack_bundle(selected: str, bundle: Path, source: Path = Path("artifacts")) -> Path:
    """Copy just the files needed for inference, then write the hash manifest."""
    bundle = Path(bundle)
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError(f"Bundle destination must be empty: {bundle}")
    bundle.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    names = _names(selected)
    for name in names:
        for relative, original, variant in _files_for(name, source):
            if not original.is_file() or original.is_symlink():
                raise FileNotFoundError(f"Missing or unsafe model artifact: {original}")
            target = _relative_path(bundle, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            entries.append(
                {
                    "path": relative,
                    "pipeline": name,
                    "variant": variant,
                    "bytes": target.stat().st_size,
                    "sha256": file_sha256(target),
                }
            )
    manifest = {
        "format_version": _FORMAT_VERSION,
        "code_revision": os.environ.get("GITHUB_SHA", ""),
        "pipelines": names,
        "config_sha256": {name: _config_hash(name) for name in names},
        "files": entries,
    }
    path = bundle / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_bundle(selected, bundle)
    return path


def verify_bundle(selected: str, bundle: Path) -> dict[str, Any]:
    """Fail closed on missing, altered, extra-path, or wrong-pipeline models."""
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    names = _names(selected)
    if manifest.get("format_version") != _FORMAT_VERSION or manifest.get("pipelines") != names:
        raise ValueError("Incompatible manifest version or pipeline selection")
    if manifest.get("config_sha256") != {name: _config_hash(name) for name in names}:
        raise ValueError("Model bundle config does not match the checked-out source")

    expected: dict[str, set[str]] = {}
    from ..config import Paths, load_config
    from ..registry import get_pipeline

    for name in names:
        paths = Paths(name, root=Path("artifacts"), create=False)
        expected[name] = {v.name for v in get_pipeline(name).variants(load_config(name), paths)}

    seen: set[str] = set()
    models: dict[str, set[str]] = {name: set() for name in names}
    for entry in manifest["files"]:
        relative = entry["path"]
        pipeline = entry["pipeline"]
        variant = entry["variant"]
        if not isinstance(relative, str) or relative in seen or pipeline not in names:
            raise ValueError("Duplicate path or unknown pipeline in bundle")
        seen.add(relative)
        if not relative.startswith(f"artifacts/{pipeline}/"):
            raise ValueError(f"Artifact belongs to wrong pipeline: {relative}")
        if variant == "tokenizer":
            if pipeline != "distilbert_imdb" or Path(relative).name not in _TOKENIZER_FILES:
                raise ValueError(f"Unexpected tokenizer artifact: {relative}")
        else:
            if variant not in expected[pipeline]:
                raise ValueError(f"Unexpected model variant: {variant}")
            if Path(relative).name != f"{pipeline}_{variant}." + (
                "onnx" if pipeline == "distilbert_imdb" else "tflite"
            ):
                raise ValueError(f"Wrong artifact file for variant: {relative}")
            models[pipeline].add(variant)
        artifact = _relative_path(bundle, relative)
        if not artifact.is_file() or artifact.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or wrong-size artifact: {relative}")
        if file_sha256(artifact) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative}")
    if models != expected:
        raise ValueError(f"Missing variants: expected {expected}, got {models}")
    return manifest


def install_bundle(
    selected: str, bundle: Path, destination: Path = Path("artifacts")
) -> dict[str, Any]:
    """Verify downloaded bytes, copy into standard paths, then recheck hashes."""
    manifest = verify_bundle(selected, bundle)
    destination = Path(destination)
    for entry in manifest["files"]:
        relative = entry["path"]
        target = _relative_path(destination.parent, f"{destination.name}/" + relative.removeprefix("artifacts/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_relative_path(bundle, relative), target)
        if file_sha256(target) != entry["sha256"]:
            raise ValueError(f"Installed artifact checksum mismatch: {relative}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("pack", "verify", "install"))
    parser.add_argument("pipeline", help="A single pipeline from bsb list, or all")
    parser.add_argument("bundle", type=Path, help="Directory containing manifest.json")
    args = parser.parse_args(argv)
    if args.action == "pack":
        pack_bundle(args.pipeline, args.bundle)
    elif args.action == "verify":
        verify_bundle(args.pipeline, args.bundle)
    else:
        install_bundle(args.pipeline, args.bundle)
    print(f"{args.action} verified for {args.pipeline}: {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
