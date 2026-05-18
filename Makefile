.PHONY: help install install-dev test lint format clean docker-build docker-train docker-serve

help:
	@echo "Carbon Intensity Forecast - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install the package"
	@echo "  make install-dev    Install package + dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run test suite"
	@echo "  make lint           Run linters (flake8, mypy)"
	@echo "  make format         Format code (black, isort)"
	@echo "  make clean          Remove build artifacts and cache files"
	@echo ""
	@echo "Scripts:"
	@echo "  make train          Run training pipeline"
	@echo "  make fetch-data     Fetch data from APIs"
	@echo "  make evaluate       Run evaluation pipeline"
	@echo "  make agent          Run LLM failure analysis agent"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-train   Run training in Docker"
	@echo "  make docker-serve   Run inference server in Docker"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

pre-commit-install:
	pre-commit install

# Development
test:
	pytest

test-cov:
	pytest --cov=src/carbon_forecast --cov-report=html --cov-report=term

lint:
	flake8 src tests
	mypy src

format:
	black src tests
	isort src tests

format-check:
	black --check src tests
	isort --check-only src tests

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/

# Scripts
fetch-data:
	python scripts/fetch_data.py

train:
	python scripts/run_training.py

evaluate:
	python scripts/run_evaluation.py

agent:
	python scripts/run_agent.py

# Docker
docker-build:
	docker-compose -f docker/docker-compose.yaml build

docker-train:
	docker-compose -f docker/docker-compose.yaml run --rm training

docker-serve:
	docker-compose -f docker/docker-compose.yaml up serving

docker-down:
	docker-compose -f docker/docker-compose.yaml down
