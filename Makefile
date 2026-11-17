UV ?= uv
RUN := $(UV) run
LINT_TARGETS := src tests scripts
SCENARIO ?= drift

.PHONY: help install lock lint format test download validate validate-corrupted train predict drift fairness-ablation pipeline pipeline-clean pipeline-drift pipeline-corrupted mlflow-ui clean

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
	@echo "  make drift              - simula data/concept drift e gera relatorios de drift, performance e fairness"
	@echo "  make fairness-ablation  - compara AUC e disparate impact removendo atributos ligados a idade"
	@echo "  make pipeline           - executa fluxo ponta a ponta (SCENARIO=clean|drift|corrupted, padrao: drift)"
	@echo "  make pipeline-clean     - executa pipeline sem drift (cenario estavel)"
	@echo "  make pipeline-drift     - executa pipeline com choque economico simulado"
	@echo "  make pipeline-corrupted - executa pipeline com lote corrompido para testar bloqueio e logs"
	@echo "  make mlflow-ui          - inicia a interface web local do MLflow (http://127.0.0.1:5000)"
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

fairness-ablation:
	$(RUN) python scripts/run_fairness_ablation.py

pipeline:
	$(RUN) python scripts/run_pipeline.py --scenario $(SCENARIO)

pipeline-clean:
	$(RUN) python scripts/run_pipeline.py --scenario clean

pipeline-drift:
	$(RUN) python scripts/run_pipeline.py --scenario drift

pipeline-corrupted:
	$(RUN) python scripts/run_pipeline.py --scenario corrupted

mlflow-ui:
	$(RUN) mlflow ui

reports-ui:
	python3 -m http.server 8080 --directory reports

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-pipeline:
	docker compose run --rm pipeline python scripts/run_pipeline.py --scenario $(SCENARIO)

docker-pipeline-clean:
	docker compose run --rm pipeline python scripts/run_pipeline.py --scenario clean

docker-pipeline-drift:
	docker compose run --rm pipeline python scripts/run_pipeline.py --scenario drift

docker-pipeline-corrupted:
	-docker compose run --rm pipeline python scripts/run_pipeline.py --scenario corrupted

docker-validate-corrupted:
	docker compose run --rm pipeline python scripts/validate_contracts.py --simulate-corruption

prometheus-metrics:
	$(RUN) python scripts/serve_metrics.py --port 8000

monitoring-up:
	docker compose up -d prometheus grafana

monitoring-down:
	docker compose stop prometheus grafana && docker compose rm -f prometheus grafana

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -not -path "./.venv/*" -exec rm -rf {} +
