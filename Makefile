# Byte-Sized Brain developer entry points.
# Every target is a thin wrapper around the `bsb` CLI, pytest or docker, so the
# commands in the README are copy-pasteable. On Windows without `make`, run the
# underlying `bsb ...` / `python -m ...` commands shown in each recipe directly.

PIPELINES := ffn_mnist cnn_cifar10 rnn_imdb distilbert_imdb
PLATFORM_ARM := linux/arm64

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_%-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────
.PHONY: install install-dev
install: ## Install the pinned environment plus the package
	pip install -r requirements.txt && pip install -e .

install-dev: ## Editable install with every framework, dev and docs tooling (pinned)
	pip install -c requirements.txt -e ".[all,dev,docs]"

# ── Quality ──────────────────────────────────────────────────────────────
.PHONY: lint fmt typecheck test coverage smoke check
lint: ## Ruff lint + formatting check (what CI runs)
	ruff check .
	ruff format --check .

fmt: ## Ruff autofix + autoformat
	ruff check --fix .
	ruff format .

typecheck: ## mypy
	mypy

test: ## Fast tests (no training, no dataset downloads)
	pytest -m "not smoke" tests

coverage: ## Fast tests with a coverage report
	pytest -m "not smoke" --cov --cov-report=term tests

smoke: ## End-to-end tiny train -> convert -> benchmark test for every pipeline
	pytest -m smoke tests

check: lint typecheck test ## Everything CI gates on, in one command

# ── Per-pipeline lifecycle (e.g. `make run-ffn_mnist`) ───────────────────
.PHONY: $(addprefix train-,$(PIPELINES)) $(addprefix convert-,$(PIPELINES)) \
        $(addprefix benchmark-,$(PIPELINES)) $(addprefix run-,$(PIPELINES))
train-%: ## Train one pipeline (artifacts/<name>/)
	bsb train $*
convert-%: ## Convert one pipeline to fp32 + quantized
	bsb convert $*
benchmark-%: ## Benchmark one pipeline -> benchmarks/results/<name>.csv
	bsb benchmark $*
run-%: ## train + convert + benchmark one pipeline
	bsb run $*

.PHONY: run-all smoke-all report ablation-cnn docs
run-all: ## Full lifecycle for every pipeline (slow, real training)
	bsb run all
smoke-all: ## Tiny lifecycle for every pipeline (results go to benchmarks/results/smoke/)
	bsb run all --smoke
report: ## Aggregate all CSVs -> docs/report.md + docs/images/ charts
	bsb report
ablation-cnn: ## Re-run the CNN INT8 ablation -> benchmarks/results/ablations/
	python scripts/cnn_int8_ablation.py
docs: ## Build the documentation site into site/ (strict)
	mkdocs build --strict

# ── Edge / ARM (Pi replacement) ──────────────────────────────────────────
.PHONY: docker-build docker-build-arm benchmark-arm
docker-build: ## Build the x86 image
	docker build -f docker/Dockerfile -t bsb:x86 .

docker-build-arm: ## Build the aarch64 image (needs buildx + QEMU)
	docker buildx build --platform $(PLATFORM_ARM) -f docker/Dockerfile.arm64 -t bsb:arm64 --load .

benchmark-arm: ## Functional check: tiny run of every pipeline under emulated ARM64 (QEMU)
	docker run --rm --platform $(PLATFORM_ARM) \
		-e BSB_EMULATED=1 -e BSB_DEVICE=qemu-arm64 \
		-v "$(CURDIR)/benchmarks/results:/app/benchmarks/results" \
		bsb:arm64 bsb run all --smoke

# ── Housekeeping ─────────────────────────────────────────────────────────
.PHONY: clean clean-artifacts
clean: ## Remove caches, smoke outputs and the built docs site (keeps trained models)
	rm -rf artifacts/smoke benchmarks/results/smoke site .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-artifacts: clean ## Also delete the trained models (full retraining takes hours)
	rm -rf artifacts
