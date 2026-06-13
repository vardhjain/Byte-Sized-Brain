# Byte-Sized Brain — developer entry points.
# Every target is a thin wrapper around the `bsb` CLI / pytest / docker so the
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
install: ## Install the package + all framework extras (pinned)
	pip install -r requirements.txt && pip install -e .

install-dev: ## Editable install with dev tooling
	pip install -e ".[all,dev]"

# ── Quality ──────────────────────────────────────────────────────────────
.PHONY: lint fmt typecheck test smoke
lint: ## Ruff lint
	ruff check src tests

fmt: ## Ruff autoformat + import sort
	ruff check --fix src tests
	ruff format src tests

typecheck: ## mypy (light)
	mypy src

test: ## Fast unit tests (no heavy frameworks)
	pytest -m "not smoke" tests

smoke: ## End-to-end tiny train→convert→benchmark for every pipeline
	pytest -m smoke tests

# ── Per-pipeline lifecycle (e.g. `make run-ffn_mnist`) ───────────────────
.PHONY: $(addprefix train-,$(PIPELINES)) $(addprefix convert-,$(PIPELINES)) \
        $(addprefix benchmark-,$(PIPELINES)) $(addprefix run-,$(PIPELINES))
train-%: ## Train one pipeline (artifacts/<name>/)
	bsb train $*
convert-%: ## Convert one pipeline to fp32 + quantized
	bsb convert $*
benchmark-%: ## Benchmark one pipeline → benchmarks/results/<name>.csv
	bsb benchmark $*
run-%: ## train + convert + benchmark one pipeline
	bsb run $*

.PHONY: run-all smoke-all report
run-all: ## Full lifecycle for every pipeline
	bsb run all
report: ## Aggregate all CSVs → docs/report.md + docs/images/ charts
	bsb report

# ── Edge / ARM (Pi replacement) ──────────────────────────────────────────
.PHONY: docker-build docker-build-arm benchmark-arm
docker-build: ## Build the x86 image
	docker build -f docker/Dockerfile -t bsb:x86 .

docker-build-arm: ## Build the aarch64 image (needs buildx + QEMU)
	docker buildx build --platform $(PLATFORM_ARM) -f docker/Dockerfile.arm64 -t bsb:arm64 --load .

benchmark-arm: ## Run the full smoke benchmark under emulated ARM64 (QEMU)
	docker run --rm --platform $(PLATFORM_ARM) \
		-e BSB_EMULATED=1 -e BSB_DEVICE=qemu-arm64 \
		-v "$(CURDIR)/benchmarks/results:/app/benchmarks/results" \
		bsb:arm64 bsb run all --smoke

# ── Housekeeping ─────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove regenerable artifacts and caches
	rm -rf artifacts .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
