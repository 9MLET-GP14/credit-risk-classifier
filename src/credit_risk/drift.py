"""Simulação de choque econômico em produção e detecção estatística de drift com Evidently AI."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report


def simulate_economic_shock(
    reference_df: pd.DataFrame,
    random_state: int = 42,
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

    Returns:
        DataFrame de produção degradado (sem coluna TARGET) pronto para inferência.
    """
    rng = np.random.default_rng(random_state)
    prod_df = reference_df.drop(columns=["TARGET"], errors="ignore").copy()

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
