.PHONY: install dev test cov lint fmt type demo validate scan diffusion backtest clean all

PY ?= python

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev,plots]"

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest --cov --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

type:
	$(PY) -m mypy src/kalshi_alpha

demo:
	$(PY) -m kalshi_alpha.cli demo --out artifacts

validate:
	$(PY) -m kalshi_alpha.cli validate

scan:
	$(PY) -m kalshi_alpha.cli scan --offline

diffusion:
	$(PY) -m kalshi_alpha.cli diffusion --out artifacts

backtest:
	$(PY) -m kalshi_alpha.cli backtest --strategy ladder_arb --out artifacts

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

all: lint test validate demo
