# Credit Risk Classifier — Observabilidade, Detecção de Drift e Governança

Projeto desenvolvido para o **Tech Challenge da Fase 4** da pós-graduação em Machine Learning Engineering (FIAP MLET).

O foco deste projeto é a construção de uma **camada de sustentação e confiabilidade** para um modelo de concessão de crédito (*Credit Scoring*). Em um cenário macroeconômico dinâmico (inflação, alterações nos perfis de renda e crédito), modelos de pontuação podem sofrer **degradação silenciosa**. Este repositório implementa contratos rígidos de dados, detecção estatística de *Data Drift* e *Concept Drift*, observabilidade de ponta a ponta, auditoria de vieses e governança sob a LGPD.

## 🎥 Vídeo de Apresentação (método STAR, ≤ 5 min)

**Link:** https://drive.google.com/drive/folders/1gV0i1hOxrZ49ekvuWRRLzS6FLpSRSj2t?usp=drive_link

| Etapa STAR | Conteúdo demonstrado |
| :--- | :--- |
| **Situation** | Risco de degradação silenciosa de modelos de crédito após choques macroeconômicos (inflação, novos perfis de clientes). |
| **Task** | Interceptar dados ruins (contratos Pandera), detectar Data/Concept Drift (Evidently AI) e garantir compliance (LGPD, auditoria de vieses). |
| **Action** | `make pipeline-corrupted` (bloqueio), `make pipeline-drift` (simulação do choque), MLflow UI e plano de governança. |
| **Result** | Relatórios em `reports/`: 6/9 colunas com drift, ROC-AUC 0,62 → 0,51 em produção, alertas no log/MLflow e lições aprendidas. |

## Mapa de Requisitos do Tech Challenge

| Requisito | Onde está |
| :--- | :--- |
| Modelo base (Scikit-Learn) treinado no dataset de Referência | [`src/credit_risk/model.py`](src/credit_risk/model.py), `make train`, [Model Card](docs/model_card.md) |
| Data contracts (Pandera) com ≥ 3 regras rígidas e lote corrompido bloqueado | [`src/credit_risk/contracts.py`](src/credit_risk/contracts.py), `make validate-corrupted`, `make pipeline-corrupted` |
| Separação Referência × Produção | `data/reference/` × `data/production/` |
| Simulação de drift em ≥ 2 variáveis + predições | `simulate_economic_shock` e `simulate_concept_drift` em [`src/credit_risk/drift.py`](src/credit_risk/drift.py) |
| Relatório estatístico (K-S e PSI) com Evidently AI | [`reports/drift_report.html`](reports/drift_report.html) (Data Drift) e [`reports/model_performance_report.html`](reports/model_performance_report.html) (Concept Drift) |
| Logs centralizados, MLflow e métricas de saúde documentadas | [`src/credit_risk/observability.py`](src/credit_risk/observability.py), [Métricas e Runbooks](docs/observability_metrics.md), [Evidências](docs/observability_evidence.md) |
| Governança, LGPD, base legal, retenção, mitigação de vieses e análise causal | Seções [Governança](#governança-privacy-by-design-e-conformidade-com-a-lgpd) e [Análise Causal](#análise-causal-do-drift-e-plano-prescritivo-de-sustentação) deste README, [`reports/fairness_report.html`](reports/fairness_report.html) |

---

## Estrutura Atual do Projeto

```text
credit-risk-classifier/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Pipeline CI (GitHub Actions): setup-uv, lint e testes
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
│   ├── drift_report.html          # Data Drift (K-S/PSI) gerado pelo Evidently AI
│   ├── model_performance_report.html # Concept Drift: qualidade de classificação e drift do alvo (Evidently AI)
│   └── fairness_report.html       # Auditoria de vieses por faixa etária, escolaridade e tipo de renda
├── scripts/
│   ├── download_data.py           # Ingestão, amostragem e particionamento dos dados
│   ├── predict.py                 # Inferência em lote com geração de score de risco
│   ├── run_drift_analysis.py      # Simulação de Data/Concept Drift e relatórios (drift, performance, fairness)
│   ├── run_fairness_ablation.py   # Experimento de mitigação de viés (retreino sem atributos de idade)
│   ├── run_pipeline.py            # Orquestração ponta a ponta com tracking no MLflow
│   ├── train.py                   # Treinamento e avaliação do modelo baseline
│   └── validate_contracts.py      # Demonstração do contrato Pandera (válido vs. corrompido)
├── src/
│   └── credit_risk/               # Pacote Python principal
│       ├── __init__.py
│       ├── config.py              # Configurações dinâmicas via Pydantic Settings
│       ├── contracts.py           # Contratos de dados estritos com Pandera
│       ├── data.py                # Download com streaming, amostragem e Parquet
│       ├── drift.py               # Simulação de Data/Concept Drift e relatórios Evidently AI
│       ├── fairness.py            # Auditoria de vieses (disparate impact, negação indevida)
│       ├── model.py               # Pipeline Scikit-Learn (treino, avaliação e inferência)
│       └── observability.py       # Logging centralizado, saúde de contratos e MLflow Tracking
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py              # Teste inicial de ambiente e versão
│   ├── test_contracts.py          # Testes dos contratos Pandera e bloqueio de anomalias
│   ├── test_data.py               # Testes de carga, amostragem e particionamento
│   ├── test_drift.py              # Testes de simulação de Data/Concept Drift e relatórios Evidently AI
│   ├── test_fairness.py           # Testes da auditoria de vieses
│   ├── test_model.py              # Testes de treino, avaliação, inferência e joblib
│   ├── test_observability.py      # Testes de logging, auditoria de contratos, infraestrutura e MLflow
│   └── test_pipeline_metrics.py   # Regressão das métricas de produção do orquestrador
├── .gitignore
├── .python-version                # 3.12
├── Makefile                       # Automação (install, lint, test, download, validate, train, predict, drift, fairness-ablation, pipeline)
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

### 6. Simulação de Data Drift, Concept Drift e Auditoria de Vieses
```bash
# Simula o choque econômico (features) e a mudança na relação features -> inadimplência (rótulos),
# gerando três relatórios HTML em reports/:
#   drift_report.html             -> Data Drift por coluna (K-S numéricas, PSI categóricas)
#   model_performance_report.html -> Concept Drift (ROC-AUC, matriz de confusão, drift do alvo)
#   fairness_report.html          -> disparate impact e negação indevida por grupo
make drift

# Experimento de mitigação: retreina sem atributos ligados à idade e compara AUC x disparate impact
make fairness-ablation
```

### 7. Pipeline Ponta a Ponta e Observabilidade (MLflow)
```bash
# Executar todo o fluxo (padrão: cenário 'drift'; gera logs/pipeline.log e run no MLflow)
make pipeline

# Testar cenários específicos no pipeline e nos logs centralizados:
make pipeline-clean       # Produção estável (sem drift, status SUCCESS)
make pipeline-drift       # Data + Concept Drift (WARNING por coluna, queda de ROC-AUC, drift do alvo)
make pipeline-corrupted   # Lote corrompido (bloqueio Pandera com log ERROR e run FAILED)

# Abrir a interface web do MLflow para inspecionar métricas, logs e artefatos (http://127.0.0.1:5000)
make mlflow-ui

# Expor métricas para coleta do Prometheus / Grafana em formato OpenMetrics (http://localhost:8000/metrics)
make prometheus-metrics

# Subir a stack completa de Monitoramento em Tempo Real (Prometheus + Grafana Provisionado):
make monitoring-up     # Prometheus na porta 9090 e Grafana na porta 3000 (admin/admin)
make monitoring-down   # Para os containers de monitoramento
```

### 8. Execução via Docker (Opcional)
O projeto também conta com suporte completo a contêineres Docker (`Dockerfile` e `docker-compose.yml`) para quem preferir executar as interfaces web (MLflow na porta 5000, painel de relatórios na porta 8080 ou a stack Prometheus/Grafana) e os pipelines de forma isolada. Os targets correspondentes estão documentados diretamente no [Makefile](Makefile) (`make docker-build`, `make docker-up`, `make docker-down`, etc.).

### 9. Validação do Código (Lint e Testes)
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

O pipeline unificado (`make pipeline`) integra em um único fluxo o **logging estruturado** (`logs/pipeline.log`), as validações do **Pandera**, os testes estatísticos do **Evidently AI** e o **MLflow Tracking**. A cada execução, um run com nome versionado por cenário e timestamp (`credit-risk-<cenário>-<timestamp>`) centraliza parâmetros, métricas de saúde/latência, Concept Drift (performance com rótulos reais), fairness e infraestrutura (memória/CPU via `psutil`), além de artefatos: o modelo `.joblib`, os relatórios `drift_report.html`, `model_performance_report.html` e `fairness_report.html` e o próprio log da execução. A métrica `total_alerts_count` consolida todos os alertas em um único indicador.

Veja o detalhamento completo e as evidências visuais dos três cenários (`clean`, `drift` e `corrupted`) no MLflow UI em **[docs/observability_evidence.md](docs/observability_evidence.md)**.

---

## Governança, Privacy by Design e Conformidade com a LGPD

A camada de sustentação do modelo foi projetada segundo os fundamentos de **Privacy by Design (PbD)**, **Privacy by Default (PbDf)** e os princípios da **Lei Geral de Proteção de Dados (LGPD — Lei nº 13.709/2018)**, garantindo que proteção de dados e prevenção de vieses façam parte da arquitetura desde a ingestão até o monitoramento contínuo.

### 1. Tratamento de PII, Minimização (*Privacy by Default*) e Dados Sintéticos
* **Minimização na origem (Princípio da Necessidade):** Dos ~122 atributos originais do dataset *Home Credit Default Risk*, o sistema ingere por padrão apenas **9 variáveis estritamente necessárias** (`DEFAULT_COLUMNS` em [`src/credit_risk/config.py`](src/credit_risk/config.py)), descartando antes mesmo da persistência em Parquet qualquer dado pessoal sensível (origem racial/étnica, convicção religiosa, saúde, biometria) e identificadores nominais diretos.
* **Pseudonimização técnica:** O campo `SK_ID_CURR` atua exclusivamente como chave técnica pseudonimizada para validar unicidade no contrato de entrada (Pandera), sendo **removido** antes do treinamento, da inferência (`RISK_SCORE`) e da análise estatística de *drift* (`src/credit_risk/drift.py`).
* **Prevenção de discriminação algorítmica:** Excluir atributos sensíveis é necessário, mas **não suficiente**: a auditoria contínua de vieses mostrou impacto adverso por faixa etária mesmo assim (ver [seção 5](#5-mitigação-de-vieses-princípio-da-não-discriminação--art-6º-ix)).
* **Uso de dados sintéticos e simulação baseada em regras:** A simulação de choque econômico (`simulate_economic_shock`) e os lotes corrompidos de teste (`build_corrupted_batch` e suíte `tests/`) operam sobre perturbações estatísticas e regras de negócio sem expor dados pessoais reais em ambientes de validação.

### 2. Mapeamento dos 7 Princípios de Privacy by Design na Arquitetura

| Princípio de PbD (Ann Cavoukian) | Aplicação Concreta neste Repositório |
| :--- | :--- |
| **1. Proativo, não reativo; preventivo, não corretivo** | Contratos **Pandera** (`src/credit_risk/contracts.py`) bloqueiam anomalias antes da inferência; **Evidently AI** e `logs/pipeline.log` emitem alertas `WARNING`/`ERROR` antes da degradação silenciosa. |
| **2. Privacidade como padrão (*Privacy by Default*)** | Subconjunto mínimo de 9 colunas ativado por padrão em `Settings` e exclusão automática de `SK_ID_CURR` e `TARGET` nas etapas analíticas. |
| **3. Privacidade incorporada ao design** | Validação de schema, minimização e rastreabilidade fazem parte do fluxo obrigatório (`scripts/run_pipeline.py`), não de um controle externo opcional. |
| **4. Funcionalidade total (soma positiva)** | A minimização e a pseudonimização preservam o poder discriminativo do modelo baseline (`HistGradientBoostingClassifier`) sem expor PII desnecessária. |
| **5. Segurança de ponta a ponta (ciclo de vida)** | Controle explícito desde a ingestão (`data/raw/`), partições tratadas (`data/reference/`), inferência (`data/production/`) até o descarte de artefatos temporários. |
| **6. Visibilidade e transparência** | Documentação aberta em [`docs/model_card.md`](docs/model_card.md), [`docs/observability_metrics.md`](docs/observability_metrics.md), relatórios em `reports/` (drift, performance e fairness) e runs auditáveis no **MLflow**. |
| **7. Respeito pela privacidade do usuário** | Saída estruturada (`RISK_SCORE` + `TARGET_PRED`) e logs auditáveis viabilizam explicabilidade e revisão humana de decisões automatizadas. |

### 3. Base Legal e Direitos dos Titulares (LGPD)
* **Base Legal Principal — Proteção do Crédito (art. 7º, X):** O tratamento de dados financeiros e contratuais (renda, valor do crédito, anuidade, vínculo de renda, escolaridade e faixa etária) para a decisão de crédito fundamenta-se na hipótese de **proteção do crédito**, que autoriza a análise de risco financeiro, a concessão responsável e o combate à inadimplência.
* **Bases Complementares:** **execução de contrato ou procedimentos preliminares** a pedido do titular (art. 7º, V) no fluxo de contratação; **cumprimento de obrigação legal ou regulatória** (art. 7º, II) para a guarda de registros exigida pelo regulador; e **legítimo interesse do controlador** (art. 7º, IX, c/c art. 10) apenas para o monitoramento de qualidade e de drift do modelo, com teste de balanceamento documentado e uso exclusivo de dados já minimizados.
* **Dados sensíveis (art. 11):** Não são tratados. Nenhuma variável de origem racial/étnica, religião, saúde, biometria ou orientação sexual é ingerida (art. 5º, II).
* **Princípios aplicados (art. 6º):** finalidade e adequação (uso exclusivo para risco de crédito), necessidade (9 de ~122 colunas), qualidade dos dados (contratos Pandera), transparência (Model Card e relatórios), segurança, prevenção (alertas antes da degradação), **não discriminação** (auditoria de vieses) e responsabilização e prestação de contas (MLflow e logs).
* **Qualidade dos Dados e *Accountability*:** Os contratos estritos do Pandera garantem exatidão, clareza e integridade na entrada, enquanto o MLflow (`mlruns/`) e os logs centralizados (`logs/pipeline.log`) mantêm trilha de auditoria para prestação de contas a reguladores.
* **Decisões Automatizadas (art. 20):** O titular pode solicitar a **revisão de decisões tomadas unicamente com base em tratamento automatizado** e tem direito a informações claras sobre os critérios utilizados (§1º). O `RISK_SCORE` contínuo, as features documentadas no *Model Card* e o run do MLflow (versão do modelo, métricas e relatórios) permitem explicar cada decisão. Negações em grupos com alerta de fairness passam por **revisão humana obrigatória**.
* **Direitos dos Titulares (art. 18):** confirmação e acesso, correção de dados inexatos, anonimização/bloqueio/eliminação de dados desnecessários, portabilidade e informação sobre compartilhamento. O canal de atendimento é o **Encarregado (DPO)** (art. 41).

### 4. Ciclo de Vida, Retenção e Descarte de Dados

A eliminação ao fim do tratamento segue o art. 16 da LGPD. A conservação após a finalidade só ocorre para cumprimento de obrigação legal/regulatória (art. 16, I) ou de forma anonimizada (art. 16, IV). Os prazos abaixo são a política proposta para a operação real; no ambiente acadêmico, os dados são públicos e já pseudonimizados na origem.

| Ativo / Diretório | Natureza | Finalidade | Prazo de Retenção | Descarte |
| :--- | :--- | :--- | :--- | :--- |
| `data/raw/` | Dados brutos de origem (122 colunas) | Extração inicial e amostragem | **Até 7 dias**, apenas durante a preparação | Exclusão segura logo após gerar os Parquets minimizados; nunca versionado (`.gitignore`). |
| `data/reference/` | Dados minimizados (9 colunas, Parquet) | Treino e baseline estatístico | **Vigência do modelo + 5 anos** (reprodutibilidade e auditoria de modelo; guarda de registros de operações por 5 anos, Lei 9.613/1998, art. 10) | Eliminação ou anonimização ao fim do prazo. |
| `data/production/` (com scores) | Lotes de inferência e rótulos observados | Escoragem, Concept Drift e auditoria de vieses | **Até 90 dias** na forma identificável (janela de maturação do rótulo FPD 90) | Após consolidar as métricas no MLflow, os registros são anonimizados (remoção de `SK_ID_CURR`) ou eliminados. |
| Propostas **negadas** | Dados de proponentes sem contrato | Atendimento a revisão (art. 20) e contestação | **Até 5 anos** (prazo prescricional do CDC, art. 27) | Eliminação ao fim do prazo; uso vedado para outras finalidades. |
| `logs/pipeline.log` e `mlruns/` | Telemetria e trilha de auditoria (art. 37) | Rastreabilidade de falhas, latência, rejeições e alertas | **5 anos** para métricas agregadas; **90 dias** para logs operacionais | Contêm apenas contagens, estatísticas e nomes de colunas, sem registros individuais em texto plano; não versionados no Git. |
| `models/` e `reports/` | Artefatos derivados (`.joblib`, `.html`) | Inferência e evidência estatística agregada | **Vigência do modelo + 5 anos** | Apenas parâmetros e distribuições agregadas (sem PII individual); arquivados após obsolescência. |

### 5. Mitigação de Vieses (Princípio da Não Discriminação — art. 6º, IX)

A LGPD veda o tratamento para fins discriminatórios ilícitos ou abusivos. A idade não é dado sensível pela LGPD, mas exige cuidado (Estatuto da Pessoa Idosa e risco de exclusão de jovens do crédito). Por isso, o pipeline executa uma **auditoria de vieses a cada execução** ([`src/credit_risk/fairness.py`](src/credit_risk/fairness.py)), com resultados no log (`[fairness]`), no MLflow (`fairness_di_*`, `fairness_alerts_count`) e em [`reports/fairness_report.html`](reports/fairness_report.html).

* **Métricas:** *disparate impact* (menor ÷ maior taxa de aprovação entre grupos; alerta `< 0,80`, regra dos quatro quintos) e *gap* de negação indevida de bons pagadores (alerta `> 10 p.p.`), por faixa etária, escolaridade e tipo de renda.
* **Achado:** o baseline **não atende à regra dos 4/5**. Na referência, a faixa 18–25 anos tem aprovação de 30,8%, contra 84,2% na faixa 51+ (DI = 0,365), e 69% dos bons pagadores jovens são negados indevidamente.
* **Experimento de mitigação** (`make fairness-ablation`): remover a idade melhora pouco o DI etário (0,365 → 0,441) e piora o AUC (0,622 → 0,593). Remover também o tempo de emprego chega a 0,584, mas transfere o impacto adverso para escolaridade e tipo de renda. A conclusão é que *fairness through unawareness* não resolve o problema, porque as variáveis restantes funcionam como **proxies**.
* **Controles adotados:**
  1. **Revisão humana obrigatória** das negações nos grupos com alerta (art. 20). O modelo atua como apoio à decisão, não como decisão final.
  2. **Gate de promoção:** um novo modelo só vai para produção com DI `>= 0,80` em todos os atributos auditados, ou com justificativa de necessidade de negócio aprovada pelo comitê de risco de modelo e registrada no MLflow.
  3. **Próxima versão:** mitigação *in-processing* ou *post-processing* (ex.: restrições de paridade com `fairlearn`, reponderação de amostras), validada contra o trade-off com AUC.
  4. **Monitoramento contínuo:** o DI em produção é acompanhado a cada lote, porque o drift muda a composição dos grupos (o DI etário passou de 0,365 para 0,521 após o choque).

### 6. Papéis, Segurança e Prestação de Contas
* **Encarregado (DPO, art. 41):** canal dos titulares e da ANPD; aprova mudanças de finalidade ou de features.
* **Relatório de Impacto (RIPD, art. 38):** exigível para o tratamento de alto risco (decisão automatizada de crédito). Insumos já produzidos: Model Card, relatórios de fairness e drift, e esta política de retenção.
* **Segurança (art. 46):** dados e logs fora do Git, acesso restrito aos diretórios `data/` e `mlruns/`, artefatos sem PII individual.
* **Incidentes (art. 48):** vazamento de lotes de produção é comunicado à ANPD e aos titulares; os logs do pipeline permitem identificar a execução e o lote afetados.

---

## Análise Causal do Drift e Plano Prescritivo de Sustentação

### 1. Diagnóstico Causal do Drift Observado
O choque simulado tem dois mecanismos causais distintos, e o pipeline mede cada um com uma ferramenta diferente:

```text
Inflação / juros altos ──► renda real ↓, crédito e parcela ↑, novos perfis ──► DATA DRIFT (P(X) muda)
          │                                                                     detectado por K-S / PSI
          └──► superendividamento da classe média formal ─┐
               programas de renegociação para jovens ──────┴──► CONCEPT DRIFT (P(y|X) muda)
                                                                só visível com rótulos reais
```

#### 1.1 Data Drift (mudança nas entradas)
Na simulação (`make drift` / `make pipeline-drift`), o **Evidently AI** detecta *Dataset Drift* global em **6 de 9 colunas monitoradas (66,7%)**:

* **Causas econômicas e estruturais simuladas:**
  * **Perda de poder aquisitivo (`AMT_INCOME_TOTAL`, K-S `p = 0.0`):** redução média de 25% na renda declarada, comprimindo a capacidade de pagamento da base.
  * **Aumento de alavancagem (`AMT_CREDIT` e `AMT_ANNUITY`, K-S `p = 0.0`):** elevação média de 35% no valor financiado e de 30% na parcela anual, ampliando o comprometimento de renda.
  * **Rejuvenescimento e mudança de perfil laboral (`AGE_YEARS` K-S `p = 0.0` e `NAME_INCOME_TYPE` PSI `= 0.5735`):** migração de 25% dos proponentes para categorias de maior vulnerabilidade econômica (*Commercial associate* / *Unemployed*).
* **Impacto nas predições (`RISK_SCORE`, K-S `p = 0.0097`):** embora a taxa agregada de classificação binária (`TARGET_PRED`) varie pouco no corte fixo (`0.5`), a distribuição contínua das probabilidades (`RISK_SCORE`) sofre deslocamento estatisticamente significativo — sinal clássico de **degradação silenciosa** antes mesmo de os rótulos reais de inadimplência (*ground truth*) chegarem com atraso.

#### 1.2 Concept Drift (mudança na relação features → inadimplência)
Quando os rótulos reais chegam (simulados por `simulate_concept_drift`), o relatório [`reports/model_performance_report.html`](reports/model_performance_report.html) mostra a degradação que os testes de distribuição não revelam:

| Métrica | Referência | Produção | Variação |
| :--- | ---: | ---: | ---: |
| ROC-AUC | 0,6224 | 0,5052 | **−0,117** (alerta: queda > 0,05 e AUC < 0,60) |
| Recall (inadimplentes capturados) | 53,6% | 34,8% | −18,8 p.p. |
| Inadimplência **observada** | 8,08% | 12,88% | +4,8 p.p. (teste Z do alvo: p ≈ 0) |
| Inadimplência **predita** (`TARGET_PRED`) | 37,8% | 36,0% | −1,8 p.p. |

* **Mecanismos simulados:** (1) bons pagadores de perfil historicamente estável (assalariados, servidores e aposentados com ensino superior ou ≥ 5 anos de emprego) passam a inadimplir quando a parcela compromete mais de 15% da renda; (2) parte dos inadimplentes jovens ou sem vínculo formal regulariza a dívida por meio de programas de renegociação.
* **Por que é silencioso:** a inadimplência real sobe 4,8 p.p. enquanto a predita **cai**. O modelo continua aprovando justamente o perfil que passou a inadimplir. Sem rótulos, nenhum teste nas entradas detectaria isso, porque `EMPLOYED_YEARS` e `NAME_EDUCATION_TYPE` não têm drift.
* **Isolando as causas:** só o choque nas features (rótulos inalterados) leva o AUC a 0,596. A queda adicional até 0,505 vem da mudança em P(y|X), o que comprova Concept Drift além do Data Drift.

### 2. Plano Prescritivo de Sustentação
As ações operacionais vinculam diretamente os alertas definidos em [`docs/observability_metrics.md`](docs/observability_metrics.md) à tomada de decisão de engenharia e risco:

1. **Bloqueio preventivo na ingestão (`rejection_rate > 0%`):**
   * *Ação:* O contrato Pandera interrompe automaticamente o pipeline (`pipeline_status = 0`), grava os detalhes no log centralizado (`ERROR`) e registra a falha no MLflow, impedindo que dados nulos, negativos ou fora de domínio contaminem o score de crédito.
2. **Alarme de *Drift* Moderado (`drift_share` entre 25% e 50%):**
   * *Ação:* Inspecionar `reports/drift_report.html`, recalibrar o ponto de corte (*threshold*) sobre `RISK_SCORE` junto à área de política de crédito e aumentar a frequência de monitoramento dos atrasos reais de pagamento (FPD/SPD).
3. **Gatilho de Retreinamento e Recalibração (`drift_share >= 0.50`, `performance_alert = 1` ou `target_drift = 1`):**
   * *Ação:* Acionar *retreino programado* com janela temporal mais recente (pós-choque, já rotulada), comparar o novo modelo candidato no MLflow contra o baseline atual (`production_roc_auc` e `production_recall`) e aplicar o *gate* de fairness (DI `>= 0,80`) antes de promover o novo artefato `.joblib` para produção.
4. **Alerta de viés (`fairness_alerts_count > 0`):**
   * *Ação:* Encaminhar as negações dos grupos afetados para revisão humana e reportar ao comitê de risco de modelo, com o `fairness_report.html` do run como evidência.
5. **Estabilidade da infraestrutura (`process_memory_mb`, `system_cpu_percent`, latências):**
   * *Ação:* Se os SLOs forem violados de forma recorrente, reduzir o tamanho do lote ou escalar o recurso computacional antes que o atraso afete a janela de decisão de crédito.

### 3. Lições Aprendidas
* **Contratos de dados não detectam drift:** o lote com choque econômico passa em todas as regras do Pandera. Qualidade e estabilidade estatística são camadas complementares.
* **Data Drift não é sinônimo de queda de performance, e vice-versa:** o Concept Drift só aparece com rótulos. Em produção real, monitorar `RISK_SCORE` funciona como alerta antecipado até o rótulo maturar.
* **A taxa predita engana:** a inadimplência predita ficou estável enquanto a observada subiu quase 60%.
* **Remover a variável sensível não elimina o viés:** proxies preservam o impacto adverso, então a auditoria precisa ser contínua e acompanhada de controles de processo (revisão humana e gate de promoção).
