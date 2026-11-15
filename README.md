# Credit Risk Classifier — Observabilidade, Detecção de Drift e Governança

Projeto desenvolvido para o **Tech Challenge da Fase 4** da pós-graduação em Machine Learning Engineering (FIAP MLET).

O foco deste projeto é a construção de uma **camada de sustentação e confiabilidade** para um modelo de concessão de crédito (*Credit Scoring*). Em um cenário macroeconômico dinâmico (inflação, alterações nos perfis de renda e crédito), modelos de pontuação podem sofrer **degradação silenciosa**. Este repositório implementa contratos rígidos de dados, detecção estatística de *Data Drift* e *Concept Drift*, observabilidade de ponta a ponta e governança sob a LGPD.

---

## Estrutura Atual do Projeto

```text
credit-risk-classifier/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Pipeline CI (GitHub Actions): setup-uv, lint e testes
├── context/                       # Documentos de apoio, instruções e resumos (ignorado no git)
├── data/
│   ├── raw/                       # Dataset bruto baixado (application_train.csv)
│   ├── reference/                 # Partições limpas em Parquet (treino e teste)
│   └── production/                # Partições simuladas de produção (drift)
├── docs/
│   └── adr/                       # Registros de Decisões Arquiteturais (ADRs)
│       └── 0001-use-configurable-dataset-subset.md
├── models/                        # Artefatos serializados do modelo (.joblib)
├── reports/                       # Relatórios estatísticos de drift (HTML do Evidently AI)
├── scripts/
│   └── download_data.py           # Ingestão, amostragem e particionamento dos dados
├── src/
│   └── credit_risk/               # Pacote Python principal
│       ├── __init__.py
│       ├── config.py              # Configurações dinâmicas via Pydantic Settings
│       └── data.py                # Download com streaming, amostragem e Parquet
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py              # Teste inicial de ambiente e versão
│   └── test_data.py               # Testes de carga, amostragem e particionamento
├── .gitignore
├── .python-version                # 3.12
├── Makefile                       # Automação de tarefas (install, lint, test, download)
├── pyproject.toml                 # Configurações de empacotamento, Ruff e Pytest
└── README.md
```

---

## Quickstart

### Pré-requisitos
* [Python 3.12](https://www.python.org/)
* [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes ultrarrápido)
* [Make](https://www.gnu.org/software/make/) (opcional, para atalhos via terminal)

### 1. Configuração do Ambiente
Clone o repositório e crie o ambiente virtual sincronizando as dependências:

```bash
# Via Makefile
make install

# Ou diretamente via uv
uv sync --extra dev
```

### 2. Download e Preparação do Dataset de Referência
Para baixar o dataset *Home Credit Default Risk* e gerar automaticamente as partições de treino e teste em formato Parquet:

```bash
# Via Makefile
make download

# Ou via CLI com argumentos customizados
uv run python scripts/download_data.py --sample-size 20000
```

Os arquivos serão organizados em:
* `data/raw/application_train.csv`: arquivo bruto original em streaming.
* `data/reference/reference_train.parquet`: partição de treino limpa (~16k registros).
* `data/reference/reference_test.parquet`: partição de teste limpa (~4k registros).

### 3. Validação do Código (Lint e Testes)
```bash
# Executar verificação de formatação e lint com Ruff
make lint

# Executar suite de testes unitários com cobertura
make test
```

---

## Decisões Arquiteturais (ADRs)

Decisões de engenharia e modelagem são documentadas na pasta [`docs/adr/`](docs/adr/):

* **[ADR 0001: Uso de Subconjunto Configurável do Dataset Home Credit Default Risk](docs/adr/0001-use-configurable-dataset-subset.md)** — Justifica o uso de uma fatia estratificada configurável (20.000 amostras e 9 atributos essenciais) para garantir agilidade na experimentação, redução no tempo de CI e baixo consumo de recursos de hardware sem perda de validade estatística.
