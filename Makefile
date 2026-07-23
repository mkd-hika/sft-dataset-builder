# Convenience targets. On Windows, run these under Git Bash / WSL, or invoke the
# underlying `python -m ...` commands directly.
.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install install-dev lint format typecheck test cov check build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with all runtime extras
	$(PYTHON) -m pip install -e ".[all]"

install-dev: ## Install the package with dev + all extras
	$(PYTHON) -m pip install -e ".[all,dev]"

lint: ## Lint with ruff
	$(PYTHON) -m ruff check .

format: ## Auto-format and auto-fix with ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck: ## Type-check with mypy
	$(PYTHON) -m mypy

test: ## Run the test suite
	$(PYTHON) -m pytest

cov: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov=sft_dataset_builder --cov-report=term-missing

check: lint typecheck test ## Run lint + typecheck + tests

build: ## Build sdist and wheel
	$(PYTHON) -m build

clean: ## Remove build/test artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
