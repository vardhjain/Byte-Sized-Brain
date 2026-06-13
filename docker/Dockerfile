# x86_64 image: reproducible train / convert / benchmark for every pipeline.
#   docker build -f docker/Dockerfile -t bsb:x86 .
#   docker run --rm -v "$PWD/benchmarks/results:/app/benchmarks/results" bsb:x86
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first so the rest of requirements don't pull CUDA wheels.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0 && \
    pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY Makefile ./
RUN pip install -e .

# Default: prove the whole thing works end-to-end on a tiny subset.
CMD ["bsb", "run", "all", "--smoke"]
