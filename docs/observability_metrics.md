# Métricas de Observabilidade, SLOs e Planos de Ação

Este documento define as métricas operacionais, de qualidade de dados, desempenho de modelo e *drift* estatístico rastreadas pelo pipeline em `logs/pipeline.log` e centralizadas no **MLflow** (`experimento: credit-risk-observability`).

Os thresholds e SLOs abaixo são referências documentadas para investigação e resposta. A documentação de um plano de ação não implica sua execução automática pelo pipeline.

---

## 1. Catálogo de Métricas e Planos de Ação

### 1.1 Saúde Operacional do Pipeline

| Métrica (MLflow) | Descrição | Origem | Threshold / SLO | Severidade | Plano de Ação (Runbook) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pipeline_status` | Status final da execução (`1 = SUCCESS`, `0 = FAILED`). | `run_pipeline.py` | `== 1` | Crítico | Consultar `logs/pipeline.log` (anexado na aba *Artifacts* do run no MLflow) para identificar a etapa e exceção; corrigir o lote ou dependência e reexecutar `make pipeline`. |
| `total_latency_seconds` | Tempo total da execução até a consolidação do resumo, antes do envio de métricas e artefatos ao MLflow (s). | `time.perf_counter()` | `< 60s` (amostra de referência de 20k) | Aviso | Decompor pelas sub-métricas de latência por etapa (`ingestion_`, `validation_`, `training_`, `drift_`) para isolar o gargalo de I/O, treino ou renderização. |
| `ingestion_latency_seconds` | Tempo de leitura/particionamento dos dados de referência (s). | `run_pipeline.py` | `< 15s` | Aviso | Verificar disponibilidade da fonte remota, taxa de leitura em disco e uso de cache em Parquet. |
| `validation_latency_seconds` | Tempo de validação dos contratos Pandera (s). | `run_pipeline.py` | `< 5s` | Aviso | Avaliar crescimento inesperado no volume do lote de entrada ou necessidade de otimização de checagens. |
| `training_latency_seconds` | Tempo de treino e avaliação do modelo baseline (s). | `run_pipeline.py` | `< 20s` | Aviso | Verificar contenção de CPU ou aumento da cardinalidade/tamanho da amostra. |
| `drift_latency_seconds` | Tempo de cálculo estatístico e geração do HTML no Evidently AI (s). | `run_pipeline.py` | `< 20s` | Aviso | Confirmar tamanho da amostra avaliada e investigar a geração e serialização do relatório HTML. |
| `total_records_received` | Quantidade de registros recebidos no lote de produção, sem somar treino ou teste de referência. | `run_pipeline.py`, `len(prod_df)` | `> 0` (4.000 no cenário padrão atual; depende do tamanho configurado da amostra) | Crítico | Caso caia abruptamente ou seja `0`, investigar falha anterior à preparação do lote de produção antes de liberar inferência. |

---

### 1.2 Qualidade e Integridade de Dados (Contratos Pandera)

| Métrica (MLflow) | Descrição | Origem | Threshold / SLO | Severidade | Plano de Ação (Runbook) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `valid_records` | Quantidade de registros do lote de produção sem violações identificadas na auditoria; não representa necessariamente registros que receberam predições. | `run_pipeline.py`, `total_records_received - rejected_records` | `== total_records_received` | Aviso | Comparar com `rejected_records` para dimensionar a qualidade do lote; verificar `pipeline_status` para saber se a inferência foi liberada. |
| `rejected_records` | Quantidade de registros do lote de produção com ao menos uma violação de contrato, sem somar múltiplas violações do mesmo registro. | `audit_contract_health` | `== 0` | Crítico | Inspecionar no log centralizado as colunas e regras violadas (`AGE_YEARS`, `AMT_INCOME_TOTAL`, `SK_ID_CURR`, etc.) e acionar o time produtor do dado. |
| `rejection_rate` | Proporção de registros inválidos no lote de produção (`rejected_records / total_records_received`), em escala de 0 a 1; o log apresenta a porcentagem. | `run_pipeline.py`, a partir de `audit_contract_health` | `0%` (Aviso `> 0%`, Crítico `>= 5%`) | Crítico | O contrato Pandera bloqueia o lote corrompido inteiro (`pipeline_status = 0`), impedindo inferência; a severidade documentada não altera essa barreira. |

#### Escopo e interpretação das métricas de produção

Treino e teste continuam sendo validados e descritos nos logs de referência, mas seus registros **não entram no denominador das métricas de qualidade da produção**.

No cenário `corrupted` padrão, os resultados esperados após a correção são:

| Métrica | Valor |
| :--- | ---: |
| `total_records_received` | 4000 |
| `valid_records` | 3994 |
| `rejected_records` | 6 |
| `rejection_rate` | 0.0015 (0,15%) |
| `pipeline_status` | 0 |

Mesmo com 3.994 registros sem violações identificadas, **o lote inteiro de 4.000 registros é bloqueado**. O pipeline não remove apenas os seis registros inválidos nem realiza a inferência dos demais. Sete violações podem corresponder a seis registros distintos, pois um registro pode violar mais de uma regra.

Se a execução falhar antes da preparação do lote de produção, a contagem de produção permanece zero. Runs históricos anteriores à correção preservam os valores antigos; devem ser interpretados conforme a versão do código que os gerou.

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
| `drift_share` | Fração de colunas com *drift* (`drifted_columns_count / total_monitored_columns`). | `Evidently DataDriftPreset` | `< 0.50` | Crítico | Quando `>= 0.50`, acionar revisão de política de crédito e avaliar recalibração e retreinamento com dados recentes, mediante validação e aprovação. |
| `reference_default_rate` | Taxa de inadimplência predita no lote de referência. | `predict_credit_risk` | Baseline | Informativo | Referência comparativa para monitorar deslocamento das decisões do modelo. |
| `production_default_rate` | Taxa de inadimplência predita no lote de produção, não a taxa de inadimplência observada. | `predict_credit_risk` | Desvio `< 5 p.p.` vs. referência | Aviso | Caso haja desvio relevante ou *Prediction Drift*, investigar combinação de *covariate shift* e calibração de score (`RISK_SCORE`). |

As métricas de classificação da seção 1.3 (`roc_auc`, `recall`, `precision`, `f1`) são calculadas sobre o teste de referência. A degradação em produção é medida separadamente na seção 1.5, usando os rótulos reais que chegam com atraso.

---

### 1.5 Concept Drift: Performance em Produção com Rótulos Reais

Na prática, o rótulo de inadimplência (`TARGET`) só é conhecido semanas depois da concessão (ex.: FPD 30/60/90 dias). O pipeline simula essa janela rotulada: o lote de produção carrega o `TARGET` observado, que **nunca é enviado ao modelo** (`predict_credit_risk` o remove) e serve apenas para avaliar as predições já feitas. Relatório visual: `reports/model_performance_report.html` (Evidently `ClassificationPreset` + teste Z do alvo).

| Métrica (MLflow) | Descrição | Origem | Threshold / SLO | Severidade | Plano de Ação (Runbook) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `production_roc_auc` | ROC-AUC do `RISK_SCORE` contra o `TARGET` observado em produção. | `evaluate_predictions` | `>= 0.60` | Crítico | Abaixo do mínimo: abrir `model_performance_report.html`, segmentar erros por perfil e acionar retreino com a janela pós-choque. |
| `roc_auc_drop` | Queda de ROC-AUC (referência − produção). | `run_pipeline.py` | `<= 0.05` | Crítico | Queda relevante **com** `dataset_drift = 0` indica Concept Drift puro (a relação X → y mudou sem mudança nas entradas). |
| `production_recall` / `production_precision` / `production_f1` | Métricas de classificação com rótulos reais de produção. | `evaluate_predictions` | `recall >= 0.50` | Aviso | Recall em queda significa inadimplentes sendo aprovados: revisar *threshold* com a área de risco. |
| `reference_observed_default_rate` / `production_observed_default_rate` | Inadimplência **observada** (não predita) em cada lote. | `TARGET` | Desvio `< 3 p.p.` | Aviso | Comparar com `production_default_rate` (predita): se a observada sobe e a predita não, o modelo está subestimando o risco. |
| `target_drift` | Drift estatístico na distribuição do alvo real (`1 = SIM`). | Evidently `ColumnDriftMetric` (teste Z) | `== 0` | Crítico | Recalibrar probabilidades e reavaliar a política de crédito. |
| `performance_alert` | `1` se `production_roc_auc < 0.60` ou `roc_auc_drop > 0.05`. | `log_concept_drift_details` | `== 0` | Crítico | Gatilho formal de retreino (ver plano prescritivo no README). |

### 1.6 Fairness: Auditoria de Vieses por Grupo

Relatório visual: `reports/fairness_report.html`. Os atributos de grupo (faixa etária derivada de `AGE_YEARS`, escolaridade e tipo de renda) são usados **apenas para auditoria**.

| Métrica (MLflow) | Descrição | Origem | Threshold / SLO | Severidade | Plano de Ação (Runbook) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fairness_di_age_reference` / `fairness_di_age_production` | *Disparate impact* por faixa etária: menor taxa de aprovação ÷ maior taxa de aprovação. | `run_fairness_audit` | `>= 0.80` (regra dos 4/5) | Crítico | Encaminhar negações do grupo menos aprovado para revisão humana (LGPD art. 20) e bloquear a promoção de novos modelos que piorem o indicador. |
| `fairness_di_education_production` | *Disparate impact* por escolaridade em produção. | `run_fairness_audit` | `>= 0.80` | Aviso | Investigar se a escolaridade atua como proxy de renda ou região. |
| `fairness_di_income_type_production` | *Disparate impact* por tipo de renda em produção. | `run_fairness_audit` | `>= 0.80` | Aviso | Avaliar política específica para trabalhadores informais. |
| `fairness_alerts_count` | Quantidade de pares (lote, atributo) com DI `< 0.80` ou gap de negação indevida `> 10 p.p.`. | `count_fairness_alerts` | `== 0` | Aviso | Consolidar no comitê de risco de modelo. |

### 1.7 Estabilidade da Infraestrutura e Consolidação de Alertas

| Métrica (MLflow) | Descrição | Origem | Threshold / SLO | Severidade | Plano de Ação (Runbook) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `process_memory_mb` | Memória residente (RSS) do processo do pipeline ao final da execução. | `psutil` | `< 1024 MB` | Aviso | Verificar crescimento da amostra e vazamentos em DataFrames intermediários. |
| `process_cpu_seconds` | Tempo de CPU (usuário + sistema) consumido pelo processo. | `psutil` | `< 60 s` | Aviso | Comparar com `total_latency_seconds` para distinguir gargalo de CPU de gargalo de I/O. |
| `system_cpu_percent` | Uso médio de CPU do host durante a execução. | `psutil` | `< 90%` | Aviso | Contenção de recursos com outros processos: reagendar o job ou escalar a máquina. |
| `system_memory_percent` | Uso de memória do host ao final da execução. | `psutil` | `< 90%` | Aviso | Risco de *swap*/OOM: reduzir paralelismo ou o tamanho do lote. |
| `total_alerts_count` | Soma de colunas com drift + `target_drift` + `performance_alert` + `fairness_alerts_count` + (`rejected_records > 0`). | `run_pipeline.py` | Tendência estável | Informativo | Indicador único para painéis e para a priorização de investigação. |

#### Valores de referência por cenário (execução local, amostra de 20k)

| Métrica | `clean` | `drift` | `corrupted` |
| :--- | ---: | ---: | ---: |
| `pipeline_status` | 1 | 1 | 0 |
| `drift_share` | 0.0 | 0.667 | — |
| `production_roc_auc` | 0.6224 | 0.5052 | — |
| `roc_auc_drop` | 0.0 | 0.1172 | — |
| `production_observed_default_rate` | 0.0808 | 0.1288 | — |
| `target_drift` | 0 | 1 | — |
| `performance_alert` | 0 | 1 | — |
| `fairness_di_age_production` | 0.365 | 0.521 | — |
| `total_alerts_count` | 6 | 14 | 1 |

No cenário `clean`, os 6 alertas são de fairness: o viés vem do próprio modelo baseline, não do drift (ver a seção de mitigação de vieses no README).

---

## 2. Simulação de Cenários para Teste de Observabilidade

O pipeline permite validar a reação dos logs centralizados e do MLflow em três cenários distintos:

```bash
# Cenário 1: Produção estável (sem drift, status SUCCESS)
make pipeline-clean

# Cenário 2: Choque macroeconômico simulado (alertas WARNING por coluna no log + drift no MLflow)
make pipeline-drift

# Cenário 3: Lote corrompido (bloqueio Pandera com log ERROR + pipeline FAILED no MLflow)
make pipeline-corrupted
```

O status operacional é indicado por `pipeline_status` e pela tag `status` enviada ao MLflow. A tag `FAILED` não deve ser confundida com o status nativo do run do MLflow, que pode estar finalizado normalmente mesmo quando o pipeline registrou uma falha controlada.
