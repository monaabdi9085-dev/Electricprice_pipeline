.PHONY: install lint typecheck test test-live check mlflow-ui

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy

test:
	uv run pytest

test-live:
	uv run pytest -m live -v

check: lint typecheck test

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
