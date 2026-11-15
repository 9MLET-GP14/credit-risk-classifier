UV ?= uv
RUN := $(UV) run
LINT_TARGETS := src tests

.PHONY: help install lock lint format test clean

help:
	@echo "Comandos disponiveis:"
	@echo "  make install  - cria .venv e instala dependencias (inclui dev) a partir do uv.lock"
	@echo "  make lock     - regenera o uv.lock a partir do pyproject.toml"
	@echo "  make lint     - verifica regras e formatacao com ruff"
	@echo "  make format   - formata e corrige problemas simples com ruff"
	@echo "  make test     - executa os testes com pytest e cobertura"
	@echo "  make clean    - remove caches locais e artefatos temporarios"

install:
	$(UV) sync --extra dev

lock:
	$(UV) lock

lint:
	$(RUN) ruff check $(LINT_TARGETS)
	$(RUN) ruff format --check $(LINT_TARGETS)

format:
	$(RUN) ruff format $(LINT_TARGETS)
	$(RUN) ruff check --fix $(LINT_TARGETS)

test:
	$(RUN) pytest

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -not -path "./.venv/*" -exec rm -rf {} +

