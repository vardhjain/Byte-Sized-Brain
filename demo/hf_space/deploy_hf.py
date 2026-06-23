"""Deploy the Byte-Sized Brain demo to HuggingFace, for real.

This does two things with your HuggingFace account. It creates a model repo holding the
two DistilBERT ONNX graphs and the tokenizer, and it creates a Streamlit Space that
downloads those models and runs the FP32-versus-INT8 demo.

First produce the artifacts locally (once):

    bsb run distilbert_imdb

Then deploy, supplying a write token (https://huggingface.co/settings/tokens):

    HF_TOKEN=hf_xxx python demo/hf_space/deploy_hf.py --user YOUR_HF_USERNAME

The Space will be live at https://huggingface.co/spaces/YOUR_HF_USERNAME/byte-sized-brain-demo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ARTIFACTS = Path("artifacts/distilbert_imdb")
SOURCE = ARTIFACTS / "fp32_source"
SPACE_DIR = Path("demo/hf_space")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "config.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="your HuggingFace username")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token")
    parser.add_argument("--model-name", default="byte-sized-brain-distilbert-imdb")
    parser.add_argument("--space-name", default="byte-sized-brain-demo")
    args = parser.parse_args()

    if not args.token:
        print("No token. Set HF_TOKEN or pass --token (a write token).", file=sys.stderr)
        return 2

    fp32 = ARTIFACTS / "distilbert_imdb_fp32.onnx"
    int8 = ARTIFACTS / "distilbert_imdb_int8.onnx"
    if not fp32.exists() or not int8.exists():
        print("Missing ONNX artifacts. Run `bsb run distilbert_imdb` first.", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    model_repo = f"{args.user}/{args.model_name}"
    space_repo = f"{args.user}/{args.space_name}"

    print(f"Creating model repo {model_repo} ...")
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    for path in (fp32, int8):
        print(f"  uploading {path.name} ...")
        api.upload_file(
            path_or_fileobj=str(path), path_in_repo=path.name, repo_id=model_repo, repo_type="model"
        )
    for name in TOKENIZER_FILES:
        f = SOURCE / name
        if f.exists():
            api.upload_file(
                path_or_fileobj=str(f), path_in_repo=name, repo_id=model_repo, repo_type="model"
            )

    print(f"Creating Space {space_repo} ...")
    api.create_repo(space_repo, repo_type="space", space_sdk="streamlit", exist_ok=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        api.upload_file(
            path_or_fileobj=str(SPACE_DIR / name),
            path_in_repo=name,
            repo_id=space_repo,
            repo_type="space",
        )

    print("\nDone.")
    print(f"  model: https://huggingface.co/{model_repo}")
    print(f"  space: https://huggingface.co/spaces/{space_repo}")
    print("The Space builds for a minute or two, then it is live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
