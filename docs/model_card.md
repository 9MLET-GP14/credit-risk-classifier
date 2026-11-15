# Model Card: Credit Risk Baseline Classifier

## Descrição do Modelo

**Nome:** `credit-risk-baseline-classifier`  
**Arquitetura Vencedora:** `ColumnTransformer` (Imputação + Padronização + One-Hot Encoding) + `HistGradientBoostingClassifier` com ponderação de classes (`class_weight="balanced"`)  
**Framework:** Scikit-Learn  
**Artefato:** `models/baseline_model.joblib`  

Modelo de classificação binária voltado para a avaliação de risco de inadimplência na concessão de crédito (*Credit Scoring*), atuando como modelo baseline para a camada de sustentação, contratos de dados (Pandera), detecção de *Data/Concept Drift* (Evidently AI) e observabilidade (MLflow).

### Diagrama da Arquitetura e Pipeline

```text
Dados Tabulares Brutos (7 Features Financeiras e Demográficas)
  │
  ├──► Numéricas (AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AGE_YEARS, EMPLOYED_YEARS)
  │      └──► SimpleImputer(strategy="median") ──► StandardScaler()
  │
  └──► Categóricas (NAME_INCOME_TYPE, NAME_EDUCATION_TYPE)
         └──► SimpleImputer(strategy="most_frequent") ──► OneHotEncoder(handle_unknown="ignore", sparse_output=False)
  │
  ▼
HistGradientBoostingClassifier (class_weight="balanced", max_iter=200, random_state=42)
  │
  ▼
Predição Binária (0: Adimplente | 1: Inadimplente) + Probabilidade [0, 1] ──► models/baseline_model.joblib
```

---

## Comparativo de Resultados dos Modelos

Ambos os modelos foram treinados e avaliados sob exatamente as mesmas condições: **15.998 amostras de treino** e **4.000 amostras de teste** estratificadas pela variável alvo `TARGET` (taxa de inadimplência de 8,08%).

| Modelo | `roc_auc` | `recall` | `precision` | `f1` | `accuracy` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`logistic_regression`** (`class_weight="balanced"`, `max_iter=1000`) | **0,6239** | **0,5944** | 0,1095 | 0,1849 | 0,5767 |
| **`hist_gradient_boosting`** *(Vencedor / Padrão)* | 0,6224 | 0,5356 | **0,1145** | **0,1887** | **0,6280** |

### Justificativa da Escolha do Modelo Vencedor

O **`HistGradientBoostingClassifier`** foi selecionado como modelo padrão em `src/credit_risk/config.py` pelos seguintes motivos:
1. **Melhor equilíbrio geral e acurácia:** Obteve maior **Acurácia** (`0,6280` vs `0,5767`), maior **Precision** (`0,1145` vs `0,1095`) e maior **F1-Score** (`0,1887` vs `0,1849`), mantendo o **ROC-AUC** praticamente empatado (`0,6224` vs `0,6239`).
2. **Capacidade não-linear:** Modelos baseados em árvores capturam interações não-lineares entre renda, valor do crédito, idade e tempo de emprego sem necessidade de engenharia manual de polinômios.
3. **Flexibilidade:** A arquitetura via `config.py` mantém a `LogisticRegression` imediatamente disponível caso o negócio decida priorizar estritamente o *Recall* (`0,5944`) em cenários de política de crédito mais conservadora.

---

## Documentação Técnica do Modelo Vencedor (`HistGradientBoostingClassifier`)

### Hiperparâmetros Configurados

| Componente | Parâmetro | Valor | Descrição |
| :--- | :--- | :---: | :--- |
| **Imputação Numérica** | `strategy` | `"median"` | Imputação robusta a *outliers* de renda/crédito |
| **Escalonamento** | `StandardScaler` | Padrão | Padronização com média 0 e desvio-padrão 1 |
| **Imputação Categórica** | `strategy` | `"most_frequent"` | Preenchimento pela moda |
| **Codificação Categórica** | `handle_unknown` | `"ignore"` | Evita falhas caso categorias inéditas surjam em produção |
| **Codificação Categórica** | `sparse_output` | `False` | Matriz densa exigida pelo `HistGradientBoostingClassifier` |
| **Classificador** | `class_weight` | `"balanced"` | Compensa o desbalanceamento natural (~8,08% de inadimplentes) |
| **Classificador** | `max_iter` | `200` | Número máximo de árvores/iterações de boosting |
| **Classificador** | `random_state` | `42` | Garantia de reprodutibilidade |

---

## Dados de Treinamento e Features

| Propriedade | Detalhe |
| :--- | :--- |
| **Dataset Original** | [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (`application_train.csv`) |
| **Subconjunto Configurado** | 20.000 amostras estratificadas (conforme [ADR 0001](adr/0001-use-configurable-dataset-subset.md)) |
| **Divisão de Dados** | 15.998 amostras de Treino (`reference_train.parquet`) / 4.000 amostras de Teste (`reference_test.parquet`) |
| **Alvo (`TARGET`)** | `0`: Pagamento em dia (91,92%) \| `1`: Dificuldade de pagamento / Inadimplência (8,08%) |
| **Features Numéricas (5)** | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AGE_YEARS`, `EMPLOYED_YEARS` |
| **Features Categóricas (2)** | `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE` |

---

## Uso Previsto, Limitações e Considerações Éticas

* **Caso de uso principal:** Apoio à decisão de concessão de crédito e laboratório de validação para monitoramento contínuo de *Data Drift*, *Concept Drift* e contratos de qualidade em produção.
* **Fora do escopo:** Concessão ou negação automática de crédito em ambiente financeiro real sem revisão de políticas de risco e sem a adição de variáveis de histórico de bureau externo (`EXT_SOURCE_*`).
* **Limitações conhecidas:** O teto de `~0,62` de ROC-AUC decorre da escolha intencional de um subconjunto enxuto de 7 variáveis socioeconômicas básicas, priorizando agilidade e interpretabilidade para o foco de observabilidade do projeto.
* **Mitigação de Vieses e Privacidade:** Variáveis potencialmente discriminatórias ou identificadores diretos (`SK_ID_CURR`) não são utilizados como features do modelo. O uso de dados pessoais segue os princípios de minimização e proteção ao crédito descritos na documentação de governança (LGPD).

