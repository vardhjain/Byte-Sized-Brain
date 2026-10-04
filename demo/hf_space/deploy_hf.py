"""Publish the hosted demo to Hugging Face, as a model repo plus a Gradio Space.

The script makes two repos under your Hugging Face account, or updates them if they
already exist.

1. A model repo holding the two DistilBERT ONNX graphs, the tokenizer files and a short
   model card.
2. A Gradio Space built from this folder. The script sets the Space variable
   ``BSB_MODEL_REPO`` so the app loads the models from the repo in step 1.

Build the models locally first. You only need to do this once.

    bsb run distilbert_imdb

Then log in once with a write token from https://huggingface.co/settings/tokens, either
with ``hf auth login`` or by setting the ``HF_TOKEN`` environment variable, and deploy.

    python demo/hf_space/deploy_hf.py --dry-run   # show the plan, upload nothing
    python demo/hf_space/deploy_hf.py

There is deliberately no ``--token`` option. A token typed on the command line is saved
in your shell history and can be read by other programs from the process list, while
``HF_TOKEN`` and the cached login keep it out of both.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PIPELINE = "distilbert_imdb"
# The repo that app.py and README.md name by default. When you deploy under another
# name, README.md is rewritten on upload and the Space variable points app.py at it.
DEFAULT_MODEL_REPO = "vardhjain20/byte-sized-brain-distilbert-imdb"
SPACE_FILES = ("app.py", "requirements.txt", "README.md")
TOKENIZER_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)

# A string.Template, so the braces in the code sample need no escaping.
MODEL_CARD = Template("""---
license: mit
language: en
pipeline_tag: text-classification
base_model: distilbert/distilbert-base-uncased
datasets:
  - stanfordnlp/imdb
tags:
  - onnx
  - quantization
  - int8
  - sentiment-analysis
---

# DistilBERT movie-review sentiment, FP32 and INT8 ONNX

Two ONNX versions of one DistilBERT model that reads an English movie review and says
whether it is positive or negative. They were made by
[Byte-Sized Brain](https://github.com/vardhjain/Byte-Sized-Brain), a project that measures
what quantization does to the size, accuracy, speed and memory of neural networks.

- `distilbert_imdb_fp32.onnx` is the full-precision model, fine-tuned on 3,000 IMDB reviews.
- `distilbert_imdb_int8.onnx` is the same model after ONNX Runtime dynamic INT8
  quantization, which stores the weights as 8-bit numbers.

Try both side by side in the [hosted demo](https://huggingface.co/spaces/$space_repo).
$results
## Use it

The model reads at most 128 tokens. Label 0 is negative and label 1 is positive. Both
inputs must be int64, which the tokenizer does not guarantee on every platform.

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

repo = "$model_repo"
tokenizer = AutoTokenizer.from_pretrained(repo)
session = ort.InferenceSession(hf_hub_download(repo, "distilbert_imdb_int8.onnx"))
enc = tokenizer("A moving, beautifully acted film.", truncation=True, max_length=128,
                return_tensors="np")
feed = {name: enc[name].astype(np.int64) for name in ("input_ids", "attention_mask")}
logits = session.run(None, feed)[0]
print(["negative", "positive"][int(np.argmax(logits))])
```
""")


def artifacts_dir() -> Path:
    """Where ``bsb run`` saved the models. Honors BSB_ARTIFACTS like the package does."""
    root = os.environ.get("BSB_ARTIFACTS")
    return (Path(root) if root else REPO_ROOT / "artifacts") / PIPELINE


def human_size(path: Path) -> str:
    size = path.stat().st_size
    return f"{size / 1024**2:.1f} MB" if size >= 1024**2 else f"{size / 1024:.0f} KB"


def results_csv() -> Path:
    root = os.environ.get("BSB_RESULTS")
    return (Path(root) if root else REPO_ROOT / "benchmarks" / "results") / f"{PIPELINE}.csv"


def gradio_pins() -> tuple[str | None, str | None]:
    """Return the Gradio version in README.md (sdk_version) and in requirements.txt."""
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    reqs = (HERE / "requirements.txt").read_text(encoding="utf-8")
    sdk = re.search(r"^sdk_version:\s*\"?([\w.]+)\"?\s*$", readme, re.MULTILINE)
    pin = re.search(r"^gradio==([\w.]+)\s*$", reqs, re.MULTILINE)
    return (sdk.group(1) if sdk else None, pin.group(1) if pin else None)


def results_table() -> str:
    """A small markdown table from the committed benchmark CSV, or "" if it is missing."""
    path = results_csv()
    if not path.exists():
        return ""
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("pipeline") == PIPELINE and row.get("emulated") != "True":
                rows[row["variant"]] = row  # later rows win, so the newest run is kept
    if not {"fp32", "int8"} <= rows.keys():
        return ""
    lines = [
        "",
        f"## Benchmark ({rows['fp32']['num_samples']} IMDB test reviews, CPU)",
        "",
        "| File | Size | Accuracy | Mean time per review |",
        "|------|-----:|---------:|---------------------:|",
    ]
    for variant in ("fp32", "int8"):
        r = rows[variant]
        lines.append(
            f"| `distilbert_imdb_{variant}.onnx` | {float(r['size_mb']):.1f} MB "
            f"| {float(r['accuracy']):.1%} | {float(r['latency_ms_mean']):.0f} ms |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the FP32 vs INT8 demo to a Hugging Face model repo and Space.",
    )
    parser.add_argument("--user", help="Hugging Face user or org, by default the logged-in user")
    parser.add_argument("--model-name", default="byte-sized-brain-distilbert-imdb")
    parser.add_argument("--space-name", default="byte-sized-brain-demo")
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would be uploaded and stop"
    )
    args = parser.parse_args()

    sdk_version, gradio_pin = gradio_pins()
    if sdk_version is None or sdk_version != gradio_pin:
        print(
            f"README.md sets sdk_version {sdk_version} but requirements.txt pins gradio "
            f"{gradio_pin}. Make them the same version before deploying.",
            file=sys.stderr,
        )
        return 1

    art = artifacts_dir()
    models = [art / f"{PIPELINE}_{variant}.onnx" for variant in ("fp32", "int8")]
    tokenizer = [art / "fp32_source" / name for name in TOKENIZER_FILES]
    missing = [p for p in models + tokenizer if not p.exists()]
    if missing:
        print(f"Missing {', '.join(p.name for p in missing)} in {art}.", file=sys.stderr)
        print(f"Build them first with `bsb run {PIPELINE}`.", file=sys.stderr)
        return 1

    from huggingface_hub import CommitOperationAdd, HfApi, get_token

    api = HfApi()
    user = args.user
    if not args.dry_run:
        if get_token() is None:
            print(
                "Not logged in to Hugging Face. Run `hf auth login`, or set HF_TOKEN to a "
                "write token from https://huggingface.co/settings/tokens.",
                file=sys.stderr,
            )
            return 2
        user = user or api.whoami()["name"]
    user = user or "YOUR_HF_USERNAME"

    model_repo = f"{user}/{args.model_name}"
    space_repo = f"{user}/{args.space_name}"
    card = MODEL_CARD.substitute(
        model_repo=model_repo, space_repo=space_repo, results=results_table()
    )
    readme = (
        (HERE / "README.md").read_text(encoding="utf-8").replace(DEFAULT_MODEL_REPO, model_repo)
    )

    model_ops = [CommitOperationAdd(p.name, str(p)) for p in models + tokenizer]
    model_ops.append(CommitOperationAdd("README.md", card.encode("utf-8")))
    space_ops = [
        CommitOperationAdd(
            name, readme.encode("utf-8") if name == "README.md" else str(HERE / name)
        )
        for name in SPACE_FILES
    ]

    print(f"Model repo  https://huggingface.co/{model_repo}")
    for p in models + tokenizer:
        print(f"  {p.name:<30} {human_size(p):>10}")
    print(f"  {'README.md':<30} (model card, {len(card)} characters)")
    print(f"Space       https://huggingface.co/spaces/{space_repo} (Gradio {sdk_version})")
    for name in SPACE_FILES:
        print(f"  {name}")
    print(f"  variable BSB_MODEL_REPO={model_repo}")
    if args.dry_run:
        print("\nDry run, nothing was uploaded.")
        return 0

    print("\nUploading the model repo. The FP32 graph is large, so this can take a while.")
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    api.create_commit(
        model_repo,
        operations=model_ops,
        commit_message="Upload FP32 and INT8 DistilBERT ONNX models",
        repo_type="model",
    )

    print("Uploading the Space.")
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(
        space_repo,
        "BSB_MODEL_REPO",
        model_repo,
        description="Model repo the demo downloads its ONNX graphs and tokenizer from",
    )
    # One commit for all three files, so the Space rebuilds once instead of three times.
    api.create_commit(
        space_repo,
        operations=space_ops,
        commit_message="Deploy the FP32 vs INT8 demo",
        repo_type="space",
    )

    print("\nDone. The Space builds for a few minutes, then it is live.")
    print(f"  model  https://huggingface.co/{model_repo}")
    print(f"  space  https://huggingface.co/spaces/{space_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
