.PHONY: install run debug clean lint

install:
	uv sync

run:
	uv run python -m src --help

debug:
	uv run python -X dev -m src --help

clean:
	find src tests -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

test:
	uv run python -m pytest
