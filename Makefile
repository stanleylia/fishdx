.PHONY: help install dev lint type-check test test-unit test-integration test-e2e \
       fmt clean docker-up docker-down run-gateway run-reasoning run-learning \
       organize-data organize-data-dry validate-data validate-data-quick validate-data-full \
       run-pipeline run-pipeline-stage1 run-unified

# ─── Variables ──────────────────────────────────────
PYTHON := python3
PIP := pip
PYTEST := pytest
RUFF := ruff
MYPY := mypy
DOCKER := docker compose

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Setup ──────────────────────────────────────────
install: ## Install production dependencies
	$(PIP) install -e .

dev: ## Install with dev dependencies
	$(PIP) install -e ".[dev]"

# ─── Quality ────────────────────────────────────────
lint: ## Run ruff linter
	$(RUFF) check .

fmt: ## Auto-format code
	$(RUFF) format .
	$(RUFF) check --fix .

type-check: ## Run mypy type checker
	$(MYPY) core/ proto/ services/

# ─── Testing ────────────────────────────────────────
test: ## Run all tests
	$(PYTEST) tests/ -v

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v -m unit

test-integration: ## Run integration tests
	$(PYTEST) tests/integration/ -v -m integration

test-e2e: ## Run end-to-end tests
	$(PYTEST) tests/e2e/ -v -m e2e

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ --cov=core --cov=services --cov-report=html --cov-report=term

# ─── Docker ─────────────────────────────────────────
docker-up: ## Start GPU services (perception, embedding, chromadb)
	$(DOCKER) up -d

docker-down: ## Stop all Docker services
	$(DOCKER) down

docker-build: ## Build Docker images
	$(DOCKER) build

# ─── Host Services ──────────────────────────────────
run-gateway: ## Run gateway orchestrator (Port 8000)
	$(PYTHON) -m uvicorn services.gateway.app:app --host 0.0.0.0 --port 8000 --reload

run-reasoning: ## Run reasoning service (Port 8004)
	$(PYTHON) -m uvicorn services.reasoning.app:app --host 0.0.0.0 --port 8004 --reload

run-learning: ## Run learning service (Port 8005)
	$(PYTHON) -m uvicorn services.learning.app:app --host 0.0.0.0 --port 8005 --reload

# ─── Utilities ──────────────────────────────────────
clean: ## Clean build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info htmlcov/
	rm -rf /tmp/multimodal_rag/

organize-data: ## Organize lab_dateset into scene-based structure
	$(PYTHON) scripts/organize_dataset.py

organize-data-dry: ## Preview dataset organization (dry run)
	$(PYTHON) scripts/organize_dataset.py --dry-run

validate-data: ## Run Level 1 dataset validation (offline)
	$(PYTHON) scripts/validate_with_dataset.py --level 1

validate-data-quick: ## Quick validation with representative images only
	$(PYTHON) scripts/validate_with_dataset.py --level 1 --quick

validate-data-full: ## Run all validation levels (needs services)
	$(PYTHON) scripts/validate_with_dataset.py --level all

seed-db: ## Import seed data into ChromaDB
	$(PYTHON) scripts/import_seed_data.py

evaluate: ## Run evaluation suite
	$(PYTHON) scripts/run_evaluation.py

benchmark: ## Run latency benchmarks
	$(PYTHON) scripts/benchmark_latency.py

# ─── Unified Pipeline ─────────────────────────────
run-pipeline: ## Run unified pipeline on image (IMAGE=path/to/img.jpg)
	$(PYTHON) __main__.py $(IMAGE) --json

run-pipeline-stage1: ## Run Stage 1 only (IMAGE=path/to/img.jpg)
	$(PYTHON) __main__.py $(IMAGE) --stage1-only --json

run-unified: ## Run unified FastAPI server (Port 8000)
	$(PYTHON) -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
