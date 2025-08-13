.PHONY: help install test test-unit test-integration test-all coverage lint format clean

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install the package and development dependencies
	pip install -e ".[dev]"

test: ## Run all tests with coverage (enforces 90% threshold)
	python scripts/run_tests.py --type all

test-unit: ## Run unit tests only
	python scripts/run_tests.py --type unit

test-integration: ## Run integration tests only
	python scripts/run_tests.py --type integration

test-fast: ## Run tests without linting
	python scripts/run_tests.py --type all --no-lint

test-verbose: ## Run tests in verbose mode
	python scripts/run_tests.py --type all --verbose

coverage: ## Generate coverage report from last test run
	python scripts/run_tests.py --report-only

coverage-html: ## Open HTML coverage report in browser
	@if exist htmlcov\index.html (start htmlcov\index.html) else (echo "No coverage report found. Run 'make test' first.")

lint: ## Run code quality checks (black, isort, flake8, mypy)
	black --check core api tests
	isort --check-only core api tests
	flake8 core api tests --max-line-length=88 --extend-ignore=E203,W503
	mypy core api

format: ## Format code with black and isort
	black core api tests
	isort core api tests

clean: ## Clean up generated files
	@if exist __pycache__ rd /s /q __pycache__
	@if exist .pytest_cache rd /s /q .pytest_cache
	@if exist htmlcov rd /s /q htmlcov
	@if exist .coverage del .coverage
	@if exist coverage.xml del coverage.xml
	@for /r %%i in (*.pyc) do @del "%%i"
	@for /d /r %%i in (__pycache__) do @rd /s /q "%%i"

test-ci: ## Run tests as they would run in CI
	pytest tests/test_neurons.py tests/test_synapses.py tests/test_learning.py -v --cov=core --cov-report=xml --cov-report=term
	pytest tests/test_integration.py -v -m "not slow" --cov=core --cov-append --cov-report=xml --cov-report=term
	coverage report --fail-under=90

test-slow: ## Run all tests including slow ones
	pytest tests/ -v --cov=core --cov=api --cov-report=term-missing --cov-report=html --cov-fail-under=90

test-gpu: ## Run GPU tests (requires GPU)
	pytest tests/ -v -m "gpu" --cov=core --cov-report=xml

test-watch: ## Run tests in watch mode (requires pytest-watch)
	pip install pytest-watch
	ptw -- --cov=core --cov-report=term-missing

benchmark: ## Run performance benchmarks
	pytest tests/test_integration.py::TestLearningPerformance -v --benchmark-only

profile: ## Profile test execution
	pytest tests/ --profile --profile-svg
	@echo "Profile saved to prof/combined.svg"

# Docker commands
.PHONY: docker-build docker-up docker-down docker-clean docker-jupyter docker-test docker-shell

docker-build: ## Build all Docker images
	docker-compose build

docker-build-runtime: ## Build runtime Docker image
	docker build --target runtime -t ghcr.io/neuromorphic-system/neuron:latest .

docker-build-dev: ## Build development Docker image
	docker build --target development -t ghcr.io/neuromorphic-system/neuron:dev .

docker-build-gpu: ## Build GPU-enabled Docker image
	docker build --target gpu -t ghcr.io/neuromorphic-system/neuron:gpu .

docker-up: ## Start all Docker services in detached mode
	docker-compose up -d

docker-down: ## Stop and remove all Docker services
	docker-compose down

docker-clean: docker-down ## Stop services and remove Docker volumes
	docker-compose down -v
	docker system prune -f

docker-jupyter: ## Start Jupyter Lab in Docker
	docker-compose up -d jupyter
	@echo "Jupyter Lab is running at http://localhost:8888"
	@echo "Token: neuron2024 (or check logs with 'docker-compose logs jupyter')"

docker-test: ## Run tests in Docker
	docker-compose run --rm test

docker-lint: ## Run linting in Docker
	docker-compose run --rm lint

docker-shell: ## Open a shell in the development container
	docker-compose run --rm jupyter /bin/bash

docker-demo: ## Run sensorimotor demo in Docker
	docker-compose run --rm demo-sensorimotor

docker-logs: ## Show logs from all Docker services
	docker-compose logs -f

docker-push: ## Push images to GitHub Container Registry
	docker push ghcr.io/neuromorphic-system/neuron:latest
	docker push ghcr.io/neuromorphic-system/neuron:dev
	docker push ghcr.io/neuromorphic-system/neuron:gpu
