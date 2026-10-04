# x86_64 image that reproduces train, convert and benchmark for every pipeline.
#
#   docker build -f docker/Dockerfile -t bsb:x86 .
#   docker run --rm -v "$PWD/benchmarks/results:/app/benchmarks/results" bsb:x86
#
# The default command is a quick smoke run of every pipeline, whose CSVs land in
# benchmarks/results/smoke/. Append another command for a full run, for example
# `docker run ... bsb:x86 bsb run ffn_mnist`.
#
# The container runs as the unprivileged user `bsb` (uid 1000). Docker Desktop
# on Windows and macOS maps file ownership for you. On a Linux host whose user
# id is not 1000, make the mounted results folder writable for uid 1000.
#
# docker/Dockerfile.arm64 is the same recipe for aarch64, so keep the two in step.

# A fixed Python patch release and Debian release keep rebuilds comparable.
# Dependabot proposes updates (see .github/dependabot.yml).
FROM python:3.11.17-slim-trixie

LABEL org.opencontainers.image.title="byte-sized-brain" \
      org.opencontainers.image.description="Post-training quantization benchmarks (x86_64)" \
      org.opencontainers.image.source="https://github.com/vardhjain/Byte-Sized-Brain" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TOKENIZERS_PARALLELISM=false

RUN useradd --create-home --uid 1000 bsb

WORKDIR /app

# Dependencies come first, so editing the source does not reinstall them.
# Every pinned package ships a prebuilt wheel for this platform, so no compiler
# is needed. torch is installed from PyTorch's CPU-only index, because the PyPI
# wheel for x86_64 Linux pulls in several GB of CUDA libraries. Its version is
# read from requirements.txt (tr drops a Windows line ending), so the pin lives
# in one place, and the second install then accepts 2.7.0+cpu as torch==2.7.0.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "$(grep -E '^torch==' requirements.txt | tr -d '\r')" \
 && pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
# --no-deps keeps exactly the versions installed from requirements.txt.
RUN pip install --no-deps -e . \
 && mkdir -p artifacts benchmarks/results \
 && chown -R bsb:bsb /app

USER bsb

# By default, prove the whole thing works end to end on a tiny subset.
CMD ["bsb", "run", "all", "--smoke"]
