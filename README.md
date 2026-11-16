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
│   ├── adr/                       # Registros de Decisões Arquiteturais (ADRs)
│   │   └── 0001-use-configurable-dataset-subset.md
│   ├── images/                    # Evidências visuais (prints do MLflow UI e logs)
│   ├── model_card.md              # Model Card comparativo e especificação do modelo vencedor
│   ├── observability_evidence.md  # Detalhamento e prints do tracking integrado no MLflow
│   └── observability_metrics.md   # Métricas operacionais, SLOs e planos de ação (Runbooks)
├── models/                        # Artefatos serializados do modelo (.joblib)
├── reports/
│   └── drift_report.html          # Relatório estatístico de drift gerado pelo Evidently AI
├── scripts/
│   ├── download_data.py           # Ingestão, amostragem e particionamento dos dados
│   ├── predict.py                 # Inferência em lote com geração de score de risco
│   ├── run_drift_analysis.py      # Simulação de choque econômico e relatório de drift (Evidently AI)
│   ├── run_pipeline.py            # Orquestração ponta a ponta com tracking no MLflow
│   ├── train.py                   # Treinamento e avaliação do modelo baseline
│   └── validate_contracts.py      # Demonstração do contrato Pandera (válido vs. corrompido)
├── src/
│   └── credit_risk/               # Pacote Python principal
│       ├── __init__.py
│       ├── config.py              # Configurações dinâmicas via Pydantic Settings
│       ├── contracts.py           # Contratos de dados estritos com Pandera
│       ├── data.py                # Download com streaming, amostragem e Parquet
│       ├── drift.py               # Simulação de drift e geração de relatórios Evidently AI
│       ├── model.py               # Pipeline Scikit-Learn (treino, avaliação e inferência)
│       └── observability.py       # Logging centralizado, saúde de contratos e MLflow Tracking
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py              # Teste inicial de ambiente e versão
│   ├── test_contracts.py          # Testes dos contratos Pandera e bloqueio de anomalias
│   ├── test_data.py               # Testes de carga, amostragem e particionamento
│   ├── test_drift.py              # Testes de simulação de drift e relatório Evidently AI
│   ├── test_model.py              # Testes de treino, avaliação, inferência e joblib
│   └── test_observability.py      # Testes de logging, auditoria de contratos e MLflow
├── .gitignore
├── .python-version                # 3.12
├── Makefile                       # Automação (install, lint, test, download, validate, train, predict, drift, pipeline)
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

### 3. Validação de Contratos de Dados (Pandera)
Os pipelines de treino (`scripts/train.py`) e inferência (`scripts/predict.py`) aplicam automaticamente o contrato estrito do **Pandera** (`CreditFeaturesContract` / `CreditTrainingContract`) barrando idade < 18, rendas/créditos nulos ou negativos, IDs duplicados e categorias inválidas.

Para validar as partições limpas de referência:
```bash
# Validação das partições íntegras (esperado: APROVADO)
make validate
```

Para demonstrar a barreira de segurança interceptando um lote intencionalmente corrompido (exigência da Etapa 1 do Tech Challenge):
```bash
# Simulação com anomalias propositais (esperado: BLOQUEADO COM SUCESSO)
make validate-corrupted
```

### 4. Treinamento do Modelo Baseline
Para treinar o pipeline de classificação (configurável em `src/credit_risk/config.py` entre `hist_gradient_boosting` e `logistic_regression`, ambos com `class_weight="balanced"`, `SimpleImputer`, `StandardScaler` e `OneHotEncoder`), avaliar as métricas de crédito (`ROC-AUC`, `Recall`, `Precision`, `F1`, `Accuracy`) e salvar o artefato em `models/baseline_model.joblib`:

```bash
# Via Makefile (usa o model_type configurado em Settings)
make train

# Ou diretamente via uv especificando o algoritmo
uv run python scripts/train.py --model-type hist_gradient_boosting
```

### 5. Inferência em Lote
Para gerar predições (`TARGET_PRED`) e scores de risco de inadimplência (`RISK_SCORE`) em lote (por padrão utilizando `data/reference/reference_test.parquet` sem vazamento de `TARGET`):

```bash
# Via Makefile
make predict

# Ou via CLI apontando um arquivo customizado (.parquet ou .csv)
uv run python scripts/predict.py --input-path data/reference/reference_test.parquet --output-path data/production/predictions.parquet
```

### 6. Simulação de Drift e Relatório Evidently AI
```bash
# Simular choque econômico em produção e gerar relatório HTML de drift (reports/drift_report.html)
make drift
```

### 7. Pipeline Ponta a Ponta e Observabilidade (MLflow)
```bash
# Executar todo o fluxo (padrão: cenário 'drift'; gera logs/pipeline.log e run no MLflow)
make pipeline

# Testar cenários específicos no pipeline e nos logs centralizados:
make pipeline-clean       # Produção estável (sem drift, status SUCCESS)
make pipeline-drift       # Choque econômico simulado (alertas WARNING por coluna)
make pipeline-corrupted   # Lote corrompido (bloqueio Pandera com log ERROR e run FAILED)

# Abrir a interface web do MLflow para inspecionar métricas, logs e artefatos (http://127.0.0.1:5000)
make mlflow-ui
```

### 8. Validação do Código (Lint e Testes)
```bash
# Executar verificação de formatação e lint com Ruff
make lint

# Executar suite de testes unitários com cobertura
make test
```

---

## Decisões Arquiteturais e Documentação do Modelo

* **[Model Card](docs/model_card.md)** — Comparativo de desempenho (`LogisticRegression` vs `HistGradientBoostingClassifier`), hiperparâmetros do modelo vencedor, escopo de uso e limitações.
* **[Métricas de Observabilidade e Runbooks](docs/observability_metrics.md)** — Catálogo de métricas operacionais (status, latência, volume), qualidade (Pandera), drift (Evidently AI), SLOs e planos de ação.
* **[Evidências de Observabilidade e Tracking](docs/observability_evidence.md)** — Passo a passo e capturas de tela do MLflow UI demonstrando os cenários `clean`, `drift` e `corrupted` com logs integrados.

Decisões de engenharia e modelagem são documentadas na pasta [`docs/adr/`](docs/adr/):

* **[ADR 0001: Uso de Subconjunto Configurável do Dataset Home Credit Default Risk](docs/adr/0001-use-configurable-dataset-subset.md)** — Justifica o uso de uma fatia estratificada configurável (20.000 amostras e 9 atributos essenciais) para garantir agilidade na experimentação, redução no tempo de CI e baixo consumo de recursos de hardware sem perda de validade estatística.

---

## Observabilidade e Tracking Integrado

O pipeline unificado (`make pipeline`) integra em um único fluxo o **logging estruturado** (`logs/pipeline.log`), as validações do **Pandera**, os testes estatísticos do **Evidently AI** e o **MLflow Tracking**. A cada execução, um run com nome versionado por cenário e timestamp (`credit-risk-<cenário>-<timestamp>`) centraliza parâmetros, métricas de saúde/latência, o modelo `.joblib`, o relatório `drift_report.html` e o próprio log da execução como artefato.

Veja o detalhamento completo e as evidências visuais dos três cenários (`clean`, `drift` e `corrupted`) no MLflow UI em **[docs/observability_evidence.md](docs/observability_evidence.md)**.

---

## Governança, Privacy by Design e Conformidade com a LGPD

A camada de sustentação do modelo foi projetada segundo os fundamentos de **Privacy by Design (PbD)**, **Privacy by Default (PbDf)** e os princípios da **Lei Geral de Proteção de Dados (LGPD — Lei nº 13.709/2018)**, garantindo que proteção de dados e prevenção de vieses façam parte da arquitetura desde a ingestão até o monitoramento contínuo.

### 1. Tratamento de PII, Minimização (*Privacy by Default*) e Dados Sintéticos
* **Minimização na origem (Princípio da Necessidade):** Dos ~122 atributos originais do dataset *Home Credit Default Risk*, o sistema ingere por padrão apenas **9 variáveis estritamente necessárias** (`DEFAULT_COLUMNS` em [`src/credit_risk/config.py`](src/credit_risk/config.py)), descartando antes mesmo da persistência em Parquet qualquer dado pessoal sensível (origem racial/étnica, convicção religiosa, saúde, biometria) e identificadores nominais diretos.
* **Pseudonimização técnica:** O campo `SK_ID_CURR` atua exclusivamente como chave técnica pseudonimizada para validar unicidade no contrato de entrada (Pandera), sendo **removido** antes do treinamento, da inferência (`RISK_SCORE`) e da análise estatística de *drift* (`src/credit_risk/drift.py`).
* **Prevenção de discriminação algorítmica:** Ao excluir atributos sensíveis e aplicar validação contínua de domínios categóricos, o pipeline mitiga o risco de reprodução de vieses discriminatórios na concessão de crédito (Princípio da Não Discriminação da LGPD).
* **Uso de dados sintéticos e simulação baseada em regras:** A simulação de choque econômico (`simulate_economic_shock`) e os lotes corrompidos de teste (`build_corrupted_batch` e suíte `tests/`) operam sobre perturbações estatísticas e regras de negócio sem expor dados pessoais reais em ambientes de validação.

### 2. Mapeamento dos 7 Princípios de Privacy by Design na Arquitetura

| Princípio de PbD (Ann Cavoukian) | Aplicação Concreta neste Repositório |
| :--- | :--- |
| **1. Proativo, não reativo; preventivo, não corretivo** | Contratos **Pandera** (`src/credit_risk/contracts.py`) bloqueiam anomalias antes da inferência; **Evidently AI** e `logs/pipeline.log` emitem alertas `WARNING`/`ERROR` antes da degradação silenciosa. |
| **2. Privacidade como padrão (*Privacy by Default*)** | Subconjunto mínimo de 9 colunas ativado por padrão em `Settings` e exclusão automática de `SK_ID_CURR` e `TARGET` nas etapas analíticas. |
| **3. Privacidade incorporada ao design** | Validação de schema, minimização e rastreabilidade fazem parte do fluxo obrigatório (`scripts/run_pipeline.py`), não de um controle externo opcional. |
| **4. Funcionalidade total (soma positiva)** | A minimização e a pseudonimização preservam o poder discriminativo do modelo baseline (`HistGradientBoostingClassifier`) sem expor PII desnecessária. |
| **5. Segurança de ponta a ponta (ciclo de vida)** | Controle explícito desde a ingestão (`data/raw/`), partições tratadas (`data/reference/`), inferência (`data/production/`) até o descarte de artefatos temporários. |
| **6. Visibilidade e transparência** | Documentação aberta em [`docs/model_card.md`](docs/model_card.md), [`docs/observability_metrics.md`](docs/observability_metrics.md), `reports/drift_report.html` e runs auditáveis no **MLflow**. |
| **7. Respeito pela privacidade do usuário** | Saída estruturada (`RISK_SCORE` + `TARGET_PRED`) e logs auditáveis viabilizam explicabilidade e revisão humana de decisões automatizadas. |

### 3. Base Legal e Direitos dos Titulares (LGPD)
* **Base Legal Principal — Proteção ao Crédito:** O tratamento de dados financeiros e contratuais (renda, valor do crédito, anuidade, vínculo de renda, escolaridade e faixa etária) fundamenta-se na hipótese legal de **proteção ao crédito**, que autoriza a análise de risco financeiro, a concessão responsável e o combate à inadimplência.
* **Bases Complementares:** Quando aplicável ao fluxo de contratação pelo titular, apoia-se em **execução de contrato ou diligências preliminares** e, para monitoramento interno de qualidade e prevenção a fraudes/degradação, no **legítimo interesse do controlador** (observado o princípio da necessidade e o balanceamento com os direitos do titular).
* **Qualidade dos Dados e *Accountability*:** Os contratos estritos do Pandera garantem exatidão, clareza e integridade na entrada, enquanto o MLflow (`mlruns/`) e os logs centralizados (`logs/pipeline.log`) mantêm trilha de auditoria para prestação de contas a reguladores.
* **Decisões Automatizadas e Direitos dos Titulares:** A geração de `RISK_SCORE` contínuo (probabilidade calibrável) acompanhado do *Model Card* evita decisões opacas do tipo "caixa-preta", viabilizando o atendimento aos direitos de confirmação de tratamento, acesso, correção de dados inexatos, eliminação de dados excessivos e solicitação de revisão de decisões automatizadas.

### 4. Ciclo de Vida, Retenção e Descarte de Dados

| Ativo / Diretório | Natureza | Finalidade | Política de Retenção e Descarte |
| :--- | :--- | :--- | :--- |
| `data/raw/` | Dados brutos de origem | Extração inicial e amostragem estratificada | Retenção temporária apenas durante a preparação; **excluído do controle de versão** (`.gitignore`) e passível de descarte imediato após geração dos Parquets. |
| `data/reference/` | Dados tratados e minimizados (Parquet) | Treino, validação e baseline estatístico de referência | Mantido durante o ciclo de vigência da versão do modelo; substituído/eliminado de forma segura ao expirar a janela histórica ou na troca de baseline. |
| `data/production/` | Lotes de inferência e simulação | Escoragem em lote e comparação de *drift* | Retenção de curto prazo (apenas janela operacional de auditoria); eliminação periódica após consolidação das métricas no MLflow. |
| `logs/pipeline.log` e `mlruns/` | Telemetria operacional e auditoria | Rastreabilidade de falhas, latência, rejeições e métricas | Retenção alinhada à política de auditoria de modelos; **não versionado no Git** e restrito a métricas agregadas/artefatos sem exposição de registros individuais em texto plano. |
| `models/` e `reports/` | Artefatos derivados (`.joblib`, `.html`) | Inferência e relatório estatístico agregado | Contêm apenas parâmetros aprendidos e distribuições agregadas (sem PII individual); versionados por ciclo de *release* e arquivados após obsolescência. |

---

## Análise Causal do Drift e Plano Prescritivo de Sustentação

### 1. Diagnóstico Causal do Drift Observado
Na simulação de choque macroeconômico (`make drift` / `make pipeline-drift`), o **Evidently AI** detecta *Dataset Drift* global em **6 de 9 colunas monitoradas (66,7%)**:

* **Causas econômicas e estruturais simuladas:**
  * **Perda de poder aquisitivo (`AMT_INCOME_TOTAL`, K-S `p = 0.0`):** redução média de 25% na renda declarada, comprimindo a capacidade de pagamento da base.
  * **Aumento de alavancagem (`AMT_CREDIT` e `AMT_ANNUITY`, K-S `p = 0.0`):** elevação média de 35% no valor financiado e de 30% na parcela anual, ampliando o comprometimento de renda.
  * **Rejuvenescimento e mudança de perfil laboral (`AGE_YEARS` K-S `p = 0.0` e `NAME_INCOME_TYPE` PSI `= 0.5735`):** migração de 25% dos proponentes para categorias de maior vulnerabilidade econômica (*Commercial associate* / *Unemployed*).
* **Impacto nas predições (`RISK_SCORE`, K-S `p = 0.0097`):** embora a taxa agregada de classificação binária (`TARGET_PRED`) varie pouco no corte fixo (`0.5`), a distribuição contínua das probabilidades (`RISK_SCORE`) sofre deslocamento estatisticamente significativo — sinal clássico de **degradação silenciosa** antes mesmo de os rótulos reais de inadimplência (*ground truth*) chegarem com atraso.

### 2. Plano Prescritivo de Sustentação
As ações operacionais vinculam diretamente os alertas definidos em [`docs/observability_metrics.md`](docs/observability_metrics.md) à tomada de decisão de engenharia e risco:

1. **Bloqueio preventivo na ingestão (`rejection_rate > 0%`):**
   * *Ação:* O contrato Pandera interrompe automaticamente o pipeline (`pipeline_status = 0`), grava os detalhes no log centralizado (`ERROR`) e registra a falha no MLflow, impedindo que dados nulos, negativos ou fora de domínio contaminem o score de crédito.
2. **Alarme de *Drift* Moderado (`drift_share` entre 25% e 50%):**
   * *Ação:* Inspecionar `reports/drift_report.html`, recalibrar o ponto de corte (*threshold*) sobre `RISK_SCORE` junto à área de política de crédito e aumentar a frequência de monitoramento dos atrasos reais de pagamento (FPD/SPD).
3. **Gatilho de Retreinamento e Recalibração (`drift_share >= 0.50` ou `roc_auc < 0.60`):**
   * *Ação:* Acionar *retreino programado* com janela temporal mais recente (pós-choque), comparar o novo modelo candidato no MLflow contra o baseline atual (`ROC-AUC` e `Recall`) e validar a ausência de viés antes de promover o novo artefato `.joblib` para produção.
