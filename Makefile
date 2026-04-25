# Makefile — Cardio ML Project
#
# Main commands:
#   make install     Create local venv and install dependencies
#   make train       Train models (low CPU priority)
#   make train-fast  Train without limiting priority (idle machine)
#   make experiment  Run the full experiment suite (base + PCA + LDA)
#   make select      Select and register the champion model in MLflow Registry
#   make serve       Start the FastAPI server locally
#   make mlflow-ui   Open the MLflow UI at http://localhost:5001
#   make test        Run tests
#   make lint        Run ruff
#   make drift       Simulate drift and generate Evidently report
#   make clean       Remove generated artifacts
#
# All dependencies live in .venv/ locally (never global).
# `uv run` finds the .venv automatically — no activation needed.

SHELL := /bin/bash
UNAME_S := $(shell uname -s 2>/dev/null)

ifeq ($(UNAME_S),Darwin)
    THROTTLE := taskpolicy -c background
else ifeq ($(UNAME_S),Linux)
    THROTTLE := nice -n 19 ionice -c 3
else
    THROTTLE :=
endif

export OMP_NUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4
export MKL_NUM_THREADS ?= 4
export VECLIB_MAXIMUM_THREADS ?= 4
export NUMEXPR_NUM_THREADS ?= 4

MLFLOW_URI ?= file://$(PWD)/mlruns

.PHONY: help install train train-fast experiment select serve mlflow-ui test lint format drift clean all

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*##/ —/' || true
	@echo ""
	@echo "  THROTTLE=$(THROTTLE)  OMP_NUM_THREADS=$(OMP_NUM_THREADS)"

install: ## Create local venv and install dependencies (requires uv)
	uv venv
	uv pip install -e ".[dev]"

train: ## Train model (low priority, limited cores)
	$(THROTTLE) uv run python -m cardio_ml.training.train

train-fast: ## Train without limiting priority (idle machine)
	uv run python -m cardio_ml.training.train

experiment: ## Run the full experiment suite (base + PCA + LDA)
	$(THROTTLE) uv run python scripts/run_full_experiment.py

experiment-quick: ## Run a single experiment (LogReg, full CPU) for demo
	CARDIO_N_JOBS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
	uv run python scripts/run_full_experiment.py --models logistic_regression --dims none

select: ## Select the champion model and register in MLflow Registry
	uv run python scripts/select_final_model.py

serve: ## Start FastAPI at http://localhost:8000/ui
	uv run uvicorn cardio_ml.serving.api:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui: ## Open MLflow UI at http://localhost:5001
	uv run mlflow ui --backend-store-uri $(MLFLOW_URI) --port 5001 --gunicorn-opts "--limit-request-field_size 0"

test: ## Run the test suite
	uv run pytest

lint: ## Check style with ruff
	uv run ruff check src scripts tests

format: ## Format code with ruff
	uv run ruff format src scripts tests
	uv run ruff check --fix src scripts tests

drift: ## Simulate drift and generate Evidently report
	uv run python scripts/simulate_drift.py

clean: ## Remove generated artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: install lint test experiment select ## Full reproduction pipeline
