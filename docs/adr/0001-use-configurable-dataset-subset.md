# 1. Uso de Subconjunto Configurável do Dataset Home Credit Default Risk

* **Status:** Accepted
* **Date:** 2026-11-15

## Context

O dataset completo *Home Credit Default Risk* (`application_train.csv`) possui 307.511 registros e 122 colunas (~165 MB em CSV bruto), com diversas variáveis esparsas ou de baixa relevância direta para um modelo baseline de risco de crédito.

Processar todo o conjunto de dados em cada ciclo de desenvolvimento, validação de contratos e execução de testes estatísticos de drift elevaria o consumo de memória RAM e o tempo de execução dos pipelines locais e de CI/CD, sem agregar ganhos proporcionais ao foco principal do projeto (observabilidade, contratos de dados, detecção de drift e governança).

## Decision

Decidimos trabalhar por padrão com um subconjunto estratificado e configurável do dataset, definido por variáveis em `src/credit_risk/config.py`:

* **`sample_size`:** `20.000` registros (amostragem estratificada pela variável alvo `TARGET`), mantendo representatividade estatística e superando com folga o requisito mínimo de 5.000 amostras.
* **`selected_columns`:** `9` atributos financeiros e demográficos centrais (`SK_ID_CURR`, `TARGET`, `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, `NAME_EDUCATION_TYPE`, `NAME_INCOME_TYPE`).

Esses parâmetros são gerenciados via `pydantic-settings`, permitindo alteração imediata via variáveis de ambiente ou arquivo `.env` sem modificar o código-fonte.

## Consequences

### Positive
* **Agilidade na Experimentação e CI/CD:** Treinamento, validação de contratos (Pandera) e geração de relatórios (Evidently AI) executam em poucos segundos.
* **Eficiência de Recursos:** Baixo consumo de memória RAM e disco, permitindo execução fluida em ambientes com recursos computacionais limitados.
* **Flexibilidade:** Possibilidade de escalar para o dataset completo (`SAMPLE_SIZE=None`) ou adicionar novas colunas alterando apenas a configuração.

### Trade-offs
* O modelo baseline não explora todas as 122 features originais da competição, o que é aceitável dado que o objetivo do projeto é a camada de sustentação, confiabilidade e observabilidade em produção.

