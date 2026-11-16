# Métricas de Observabilidade, SLOs e Planos de Ação

Este documento define as métricas operacionais, de qualidade de dados, desempenho de modelo e *drift* estatístico rastreadas pelo pipeline em `logs/pipeline.log` e centralizadas no **MLflow** (`experimento: credit-risk-observability`).

---

## 1. Catálogo de Métricas e Planos de Ação

### 1.1 Saúde Operacional do Pipeline

| Métrica (MLflow) | Descrição | Origem | Threshold / SLO | Severidade | Plano de Ação (Runbook) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pipeline_status` | Status final da execução (`1 = SUCCESS`, `0 = FAILED`). | `run_pipeline.py` | `== 1` | Crítico | Consultar `logs/pipeline.log` (anexado na aba *Artifacts* do run no MLflow) para identificar a etapa e exceção; corrigir o lote ou dependência e reexecutar `make pipeline`. |
| `total_latency_seconds` | Tempo total ponta a ponta da execução do pipeline (s). | `time.perf_counter()` | `< 60s` (lote 20k) | Aviso | Decompor pelas sub-métricas de latência por etapa (`ingestion_`, `validation_`, `training_`, `drift_`) para isolar o gargalo de I/O, treino ou renderização. |
| `ingestion_latency_seconds` | Tempo de leitura/particionamento dos dados de referência (s). | `run_pipeline.py` | `< 15s` | Aviso | Verificar disponibilidade da fonte remota, taxa de leitura em disco e uso de cache em Parquet. |
| `validation_latency_seconds` | Tempo de validação dos contratos Pandera (s). | `run_pipeline.py` | `< 5s` | Aviso | Avaliar crescimento inesperado no volume do lote de entrada ou necessidade de otimização de checagens. |
| `training_latency_seconds` | Tempo de treino e avaliação do modelo baseline (s). | `run_pipeline.py` | `< 20s` | Aviso | Verificar contenção de CPU ou aumento da cardinalidade/tamanho da amostra. |
| `drift_latency_seconds` | Tempo de cálculo estatístico e geração do HTML no Evidently AI (s). | `run_pipeline.py` | `< 20s` | Aviso | Confirmar tamanho da amostra avaliada e manter `plotly<6.0` para evitar serialização lenta. |
| `total_records_received` | Volume total de registros processados na execução (referência + produção). | `run_pipeline.py` | `> 0` (esperado ~24.000) | Crítico | Caso caia abruptamente ou seja `0`, investigar quebra no job de extração/ingestão antes de liberar inferência. |

---

### 1.2 Qualidade e Integridade de Dados (Contratos Pandera)

| Métrica (MLflow) | Descrição | Origem | Threshold / SLO | Severidade | Plano de Ação (Runbook) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `valid_records` | Total de registros que passaram nos contratos Pandera. | `observability.py` | `== total_records_received` | Aviso | Comparar com `rejected_records` para dimensionar impacto na cobertura de escoragem. |
| `rejected_records` | Quantidade de registros com ao menos uma violação de contrato. | `audit_contract_health` | `== 0` | Crítico | Inspecionar no log centralizado as colunas e regras violadas (`AGE_YEARS`, `AMT_INCOME_TOTAL`, `SK_ID_CURR`, etc.) e acionar o time produtor do dado. |
| `rejection_rate` | Proporção de registros rejeitados (`rejected_records / total_records_received`). | `audit_contract_health` | `0%` (Aviso `> 0%`, Crítico `>= 5%`) | Crítico | O contrato Pandera bloqueia automaticamente o lote corrompido (`pipeline_status = 0`), impedindo inferência e contaminação de métricas em produção. |

---

### 1.3 Desempenho do Modelo Baseline

| Métrica (MLflow) | Descrição | Origem | Threshold / SLO | Severidade | Plano de Ação (Runbook) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `roc_auc` | Área sob a curva ROC no conjunto de teste de referência. | `evaluate_model` | `>= 0.60` | Aviso | Se cair abaixo do mínimo histórico, revisar engenharia de atributos, balanceamento de classes ou hiperparâmetros. |
| `recall` | Taxa de captura da classe inadimplente (`TARGET = 1`). | `evaluate_model` | `>= 0.50` | Aviso | Avaliar ajuste do ponto de corte (*threshold*) de decisão ou calibração de probabilidades para evitar perda financeira por falsos negativos. |
| `precision` | Precisão nas predições de inadimplência. | `evaluate_model` | Monitoramento contínuo | Informativo | Balancear com `recall` conforme apetite de risco da operação de crédito. |
| `f1` | Média harmônica entre `precision` e `recall`. | `evaluate_model` | Monitoramento contínuo | Informativo | Acompanhar tendência entre versões de treino no MLflow. |

---

### 1.4 Drift Estatístico e Predições (Evidently AI)

| Métrica (MLflow) | Descrição | Origem | Threshold / SLO | Severidade | Plano de Ação (Runbook) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dataset_drift` | Indicador global de *drift* no dataset (`1 = SIM`, `0 = NÃO`). | `Evidently DataDriftPreset` | `== 0` | Crítico | Abrir `reports/drift_report.html` (também salvo como artefato no MLflow), identificar variáveis afetadas no log (`[drift]`) e avaliar janela de retreinamento. |
| `drifted_columns_count` | Número absoluto de colunas com mudança estatística significativa (K-S ou PSI). | `Evidently DataDriftPreset` | `< 50%` das colunas | Aviso / Crítico | Analisar se o deslocamento decorre de choque macroeconômico (ex.: queda de `AMT_INCOME_TOTAL`, alta de `AMT_CREDIT`) ou mudança de coleta. |
| `drift_share` | Fração de colunas com *drift* (`drifted_columns_count / total_monitored_columns`). | `Evidently DataDriftPreset` | `< 0.50` | Crítico | Quando `>= 0.50`, acionar revisão de política de crédito, recalibração de corte de risco e retreinamento programado com dados recentes. |
| `reference_default_rate` | Taxa de inadimplência predita no lote de referência. | `predict_credit_risk` | Baseline | Informativo | Referência comparativa para monitorar deslocamento das decisões do modelo. |
| `production_default_rate` | Taxa de inadimplência predita no lote de produção. | `predict_credit_risk` | Desvio `< 5 p.p.` vs. referência | Aviso | Caso haja desvio relevante ou *Prediction Drift*, investigar combinação de *covariate shift* e calibração de score (`RISK_SCORE`). |

---

## 2. Simulação de Cenários para Teste de Observabilidade

O pipeline permite validar a reação dos logs centralizados e do MLflow em três cenários distintos:

```bash
# Cenário 1: Produção estável (sem drift, status SUCCESS)
make pipeline-clean

# Cenário 2: Choque macroeconômico simulado (alertas WARNING por coluna no log + drift no MLflow)
make pipeline-drift

# Cenário 3: Lote corrompido (bloqueio Pandera com log ERROR + run FAILED no MLflow)
make pipeline-corrupted
```
