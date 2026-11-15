UV ?= uv
RUN := $(UV) run
LINT_TARGETS := src tests scripts

.PHONY: help install lock lint format test download validate validate-corrupted train predict drift clean

help:
	@echo "Comandos disponiveis:"
	@echo "  make install            - cria .venv e instala dependencias (inclui dev) a partir do uv.lock"
	@echo "  make lock               - regenera o uv.lock a partir do pyproject.toml"
	@echo "  make lint               - verifica regras e formatacao com ruff"
	@echo "  make format             - formata e corrige problemas simples com ruff"
	@echo "  make test               - executa os testes com pytest e cobertura"
	@echo "  make download           - baixa e prepara o dataset de referencia em Parquet"
	@echo "  make validate           - valida contratos de dados (Pandera) nas particoes de referencia"
	@echo "  make validate-corrupted - simula lote corrompido e demonstra bloqueio pelo Pandera"
	@echo "  make train              - treina o modelo baseline configurado"
	@echo "  make predict            - gera predicoes em lote para o conjunto de teste"
	@echo "  make drift              - simula choque economico e gera relatorio HTML Evidently AI"
	@echo "  make clean              - remove caches locais e artefatos temporarios"

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

download:
	$(RUN) python scripts/download_data.py

validate:
	$(RUN) python scripts/validate_contracts.py

validate-corrupted:
	$(RUN) python scripts/validate_contracts.py --simulate-corruption

train:
	$(RUN) python scripts/train.py

predict:
	$(RUN) python scripts/predict.py

drift:
	$(RUN) python scripts/run_drift_analysis.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -not -path "./.venv/*" -exec rm -rf {} +
