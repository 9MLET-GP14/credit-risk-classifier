"""Simulação de choque econômico em produção e detecção estatística de drift com Evidently AI."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset, DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report


def simulate_economic_shock(
    reference_df: pd.DataFrame,
    random_state: int = 42,
    keep_target: bool = False,
) -> pd.DataFrame:
    """Simula degradação temporal (choque macroeconômico) preservando o contrato Pandera.

    Altera intencionalmente a distribuição de múltiplas features importantes:
    - Queda real de renda (~25% em média) simulando cenário inflacionário.
    - Aumento de crédito demandado (~35%) e anuidade (~30%) simulando maior endividamento.
    - Rejuvenescimento do perfil de solicitantes (-4 anos em média, mínimo 18).
    - Aumento de solicitantes sem vínculo tradicional ('Commercial associate' / 'Unemployed').

    Args:
        reference_df: Dataset limpo de referência (sem drift).
        random_state: Semente para reprodutibilidade.
        keep_target: Se True, preserva a coluna TARGET (rótulos que chegam com atraso),
            permitindo aplicar ``simulate_concept_drift`` em seguida.

    Returns:
        DataFrame de produção degradado pronto para inferência.
    """
    rng = np.random.default_rng(random_state)
    prod_df = reference_df.copy()
    if not keep_target:
        prod_df = prod_df.drop(columns=["TARGET"], errors="ignore")

    if "AMT_INCOME_TOTAL" in prod_df.columns:
        income_factor = rng.normal(loc=0.75, scale=0.08, size=len(prod_df)).clip(min=0.35)
        prod_df["AMT_INCOME_TOTAL"] = (prod_df["AMT_INCOME_TOTAL"] * income_factor).clip(
            lower=1000.0
        )

    if "AMT_CREDIT" in prod_df.columns:
        credit_factor = rng.normal(loc=1.35, scale=0.12, size=len(prod_df)).clip(min=0.90)
        prod_df["AMT_CREDIT"] = (prod_df["AMT_CREDIT"] * credit_factor).clip(lower=1000.0)

    if "AMT_ANNUITY" in prod_df.columns:
        annuity_factor = rng.normal(loc=1.30, scale=0.10, size=len(prod_df)).clip(min=0.85)
        prod_df["AMT_ANNUITY"] = (prod_df["AMT_ANNUITY"] * annuity_factor).clip(lower=500.0)

    if "AGE_YEARS" in prod_df.columns:
        age_shift = rng.integers(-6, -1, size=len(prod_df))
        prod_df["AGE_YEARS"] = (prod_df["AGE_YEARS"] + age_shift).clip(lower=18, upper=100)

    if "NAME_INCOME_TYPE" in prod_df.columns:
        # 25% da base migra para categorias de maior vulnerabilidade econômica
        shift_mask = rng.random(size=len(prod_df)) < 0.25
        replacement_types = rng.choice(
            ["Commercial associate", "Unemployed"],
            size=shift_mask.sum(),
            p=[0.7, 0.3],
        )
        prod_df.loc[shift_mask, "NAME_INCOME_TYPE"] = replacement_types

    return prod_df.reset_index(drop=True)


def simulate_concept_drift(
    production_df: pd.DataFrame,
    target_column: str = "TARGET",
    random_state: int = 42,
    new_default_prob: float = 0.30,
    relief_prob: float = 0.40,
) -> pd.DataFrame:
    """Simula Concept Drift alterando a relação P(TARGET | X) nos rótulos de produção.

    Os rótulos representam a inadimplência real observada com atraso (ex.: FPD 30/60/90).
    Duas mudanças estruturais pós-choque invertem padrões que o modelo aprendeu:

    1. **Superendividamento da classe média formal:** perfis historicamente de baixo risco
       (assalariados/servidores/aposentados com ensino superior ou >= 5 anos de emprego)
       passam a inadimplir quando a parcela compromete mais de 15% da renda, efeito da
       inflação e dos juros sobre contratos longos.
    2. **Programas de renegociação/transferência de renda:** parte dos inadimplentes jovens
       (< 30 anos) ou sem vínculo formal regulariza a dívida.

    As features não são alteradas aqui: o drift está apenas na relação features → alvo,
    por isso não é detectável por testes de distribuição (K-S/PSI) nas entradas.

    Args:
        production_df: Lote de produção contendo a coluna alvo.
        target_column: Nome da coluna alvo.
        random_state: Semente para reprodutibilidade.
        new_default_prob: Probabilidade de um bom pagador do segmento afetado inadimplir.
        relief_prob: Probabilidade de um inadimplente do segmento beneficiado regularizar.

    Returns:
        Cópia do lote com a coluna alvo modificada.
    """
    if target_column not in production_df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' ausente: Concept Drift requer rótulos.")

    rng = np.random.default_rng(random_state + 1)
    df = production_df.copy()
    target = df[target_column].astype(int).to_numpy().copy()

    payment_burden = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    stable_profile = df["NAME_INCOME_TYPE"].isin(["Working", "State servant", "Pensioner"]) & (
        (df["NAME_EDUCATION_TYPE"] == "Higher education") | (df["EMPLOYED_YEARS"] >= 5)
    )
    overindebted = (stable_profile & (payment_burden > 0.15)).to_numpy()
    relieved = (
        (df["AGE_YEARS"] < 30) | df["NAME_INCOME_TYPE"].isin(["Unemployed", "Commercial associate"])
    ).to_numpy()

    new_defaults = overindebted & (target == 0) & (rng.random(len(df)) < new_default_prob)
    cured = relieved & (target == 1) & (rng.random(len(df)) < relief_prob)
    target[new_defaults] = 1
    target[cured] = 0

    df[target_column] = target
    return df


def generate_performance_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    report_path: Path,
    target_column: str = "TARGET",
    score_column: str = "RISK_SCORE",
) -> dict[str, Any]:
    """Gera relatório Evidently de Concept Drift: qualidade de classificação e drift do alvo.

    Compara a performance do modelo (ROC-AUC, Recall, Precision, matriz de confusão) e a
    distribuição do alvo real entre referência e produção rotulada.

    Returns:
        Resumo com o resultado do teste de drift do alvo (``target_drift``).
    """
    columns = [target_column, score_column]
    ref = reference_df[columns].copy()
    prod = production_df[columns].copy()
    for frame in (ref, prod):
        frame[target_column] = frame[target_column].astype(int)

    column_mapping = ColumnMapping(target=target_column, prediction=score_column, pos_label=1)
    report = Report(
        metrics=[
            ClassificationPreset(),
            ColumnDriftMetric(column_name=target_column, stattest="z"),
        ]
    )
    report.run(reference_data=ref, current_data=prod, column_mapping=column_mapping)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_path))

    summary: dict[str, Any] = {
        "target_drift": False,
        "target_drift_score": 1.0,
        "report_path": str(report_path),
    }
    for metric in report.as_dict().get("metrics", []):
        if metric.get("metric") == "ColumnDriftMetric":
            result = metric.get("result", {})
            summary["target_drift"] = bool(result.get("drift_detected", False))
            summary["target_drift_score"] = round(float(result.get("drift_score", 1.0)), 6)
    return summary


def generate_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    report_path: Path,
    id_column: str = "SK_ID_CURR",
    target_column: str = "TARGET",
    prediction_column: str = "TARGET_PRED",
) -> dict[str, Any]:
    """Compara datasets de referência e produção com Evidently AI e exporta relatório HTML.

    Args:
        reference_df: Dataset de referência (preferencialmente com predições).
        production_df: Dataset de produção simulado (preferencialmente com predições).
        report_path: Caminho de saída do arquivo HTML do Evidently.
        id_column: Coluna de identificador único (removida da análise estatística).
        target_column: Coluna alvo real (removida caso não exista em produção).
        prediction_column: Coluna de predição do modelo para capturar Prediction Drift.

    Returns:
        Dicionário estruturado com o resumo estatístico do drift detectado.
    """
    drop_cols = [id_column, target_column]
    ref_clean = reference_df.drop(columns=drop_cols, errors="ignore").copy()
    prod_clean = production_df.drop(columns=drop_cols, errors="ignore").copy()

    # Garante alinhamento exato das colunas presentes em ambos os datasets
    common_cols = [c for c in ref_clean.columns if c in prod_clean.columns]
    ref_clean = ref_clean[common_cols].copy()
    prod_clean = prod_clean[common_cols].copy()

    metrics: list[Any] = [DataDriftPreset(num_stattest="ks", cat_stattest="psi")]
    column_mapping = ColumnMapping()

    if prediction_column in common_cols:
        # Converte 0/1 para rótulos textuais para renderizar corretamente no eixo categórico
        label_map = {0: "Adimplente (0)", 1: "Inadimplente (1)"}
        if pd.api.types.is_numeric_dtype(ref_clean[prediction_column]):
            ref_clean[prediction_column] = (
                ref_clean[prediction_column]
                .map(label_map)
                .fillna(ref_clean[prediction_column].astype(str))
            )
            prod_clean[prediction_column] = (
                prod_clean[prediction_column]
                .map(label_map)
                .fillna(prod_clean[prediction_column].astype(str))
            )
        column_mapping.prediction = prediction_column
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)
    report.run(
        reference_data=ref_clean,
        current_data=prod_clean,
        column_mapping=column_mapping,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_path))

    return _extract_drift_summary(report.as_dict(), report_path=report_path)


def _extract_drift_summary(report_dict: dict[str, Any], report_path: Path) -> dict[str, Any]:
    """Extrai métricas principais do dicionário gerado pelo Evidently Report."""
    dataset_drift = False
    number_of_columns = 0
    number_of_drifted_columns = 0
    share_of_drifted_columns = 0.0
    drifted_columns: list[str] = []
    column_stats: dict[str, dict[str, Any]] = {}

    for metric in report_dict.get("metrics", []):
        metric_name = metric.get("metric")
        result = metric.get("result", {})

        if metric_name == "DatasetDriftMetric":
            dataset_drift = bool(result.get("dataset_drift", False))
            number_of_columns = int(result.get("number_of_columns", 0))
            number_of_drifted_columns = int(result.get("number_of_drifted_columns", 0))
            share_of_drifted_columns = float(result.get("share_of_drifted_columns", 0.0))

        elif metric_name == "DataDriftTable":
            drift_by_columns = result.get("drift_by_columns", {})
            for col_name, col_info in drift_by_columns.items():
                is_drifted = bool(col_info.get("drift_detected", False))
                if is_drifted:
                    drifted_columns.append(col_name)
                column_stats[col_name] = {
                    "drift_detected": is_drifted,
                    "stattest_name": col_info.get("stattest_name"),
                    "drift_score": round(float(col_info.get("drift_score", 0.0)), 6),
                }

    return {
        "dataset_drift": dataset_drift,
        "number_of_columns": number_of_columns,
        "number_of_drifted_columns": number_of_drifted_columns,
        "share_of_drifted_columns": round(share_of_drifted_columns, 4),
        "drifted_columns": sorted(drifted_columns),
        "column_stats": column_stats,
        "report_path": str(report_path),
    }
