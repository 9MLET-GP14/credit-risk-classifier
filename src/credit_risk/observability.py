"""Observabilidade centralizada: logging estruturado, saúde de contratos e MLflow Tracking."""

import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pandera.pandas as pa

from credit_risk.contracts import format_contract_errors, validate_data

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class PipelineHealthMetrics:
    """Métricas consolidadas de saúde, qualidade de dados, modelo e drift do pipeline."""

    total_records_received: int
    valid_records: int
    rejected_records: int
    rejection_rate: float
    roc_auc: float
    recall: float
    precision: float
    f1: float
    dataset_drift: bool
    drifted_columns_count: int
    total_monitored_columns: int
    drift_share: float
    reference_default_rate: float
    production_default_rate: float
    pipeline_status: int = 1
    total_latency_seconds: float = 0.0
    ingestion_latency_seconds: float = 0.0
    validation_latency_seconds: float = 0.0
    training_latency_seconds: float = 0.0
    drift_latency_seconds: float = 0.0

    def to_mlflow_metrics(self) -> dict[str, float]:
        """Converte todas as métricas para valores numéricos aceitos pelo MLflow."""
        raw = asdict(self)
        return {k: float(int(v) if isinstance(v, bool) else v) for k, v in raw.items()}


def get_logger(
    name: str = "credit_risk",
    level: int = logging.INFO,
    log_file: Path | None = None,
    log_mode: str = "a",
) -> logging.Logger:
    """Retorna um logger centralizado com timestamps e níveis padronizados."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        resolved_path = log_file.resolve()
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == resolved_path
            for h in logger.handlers
        )
        if not has_file_handler:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(resolved_path, mode=log_mode, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
            logger.addHandler(file_handler)

    return logger


def validate_with_observability(
    df: pd.DataFrame,
    is_training: bool,
    logger: logging.Logger,
    stage_name: str = "ingestão",
) -> pd.DataFrame:
    """Executa o contrato Pandera registrando sucesso (INFO) ou bloqueio detalhado (ERROR)."""
    try:
        validated = validate_data(df, is_training=is_training, lazy=True)
        logger.info(
            "[%s] Contrato Pandera aprovado: %d registro(s) íntegro(s).",
            stage_name,
            len(validated),
        )
        return validated
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "[%s] Contrato Pandera bloqueou o lote (%d registro(s)):\n%s",
            stage_name,
            len(df),
            format_contract_errors(exc),
        )
        raise


def audit_contract_health(
    df: pd.DataFrame,
    is_training: bool,
    logger: logging.Logger,
    stage_name: str = "auditoria",
) -> dict[str, int | float | bool]:
    """Audita um lote contra o contrato Pandera sem interromper o fluxo, retornando taxas."""
    total = len(df)
    try:
        validate_data(df, is_training=is_training, lazy=True)
        logger.info("[%s] Auditoria limpa: 0 rejeições em %d registro(s).", stage_name, total)
        return {
            "total_records": total,
            "valid_records": total,
            "rejected_records": 0,
            "rejection_rate": 0.0,
            "passed": True,
        }
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        if "index" in failure_cases.columns and not failure_cases["index"].dropna().empty:
            rejected = int(failure_cases["index"].dropna().nunique())
        else:
            rejected = min(total, len(failure_cases)) if total > 0 else 0
        rejected = min(rejected, total)
        rejection_rate = (rejected / total) if total > 0 else 0.0
        logger.warning(
            "[%s] Anomalias detectadas: %d/%d registro(s) rejeitado(s) (%.2f%%).\n%s",
            stage_name,
            rejected,
            total,
            rejection_rate * 100,
            format_contract_errors(exc),
        )
        return {
            "total_records": total,
            "valid_records": total - rejected,
            "rejected_records": rejected,
            "rejection_rate": round(rejection_rate, 6),
            "passed": False,
        }


def log_drift_details(summary: dict[str, Any], logger: logging.Logger) -> None:
    """Registra no log centralizado os resultados globais e por coluna do Evidently AI."""
    if summary.get("dataset_drift"):
        logger.warning(
            "[drift] Dataset Drift DETECTADO: %d/%d colunas (%.1f%%).",
            summary.get("number_of_drifted_columns", 0),
            summary.get("number_of_columns", 0),
            float(summary.get("share_of_drifted_columns", 0.0)) * 100,
        )
    else:
        logger.info(
            "[drift] Nenhum Dataset Drift global detectado (%d/%d colunas).",
            summary.get("number_of_drifted_columns", 0),
            summary.get("number_of_columns", 0),
        )

    column_stats = summary.get("column_stats", {})
    for col_name, col_info in column_stats.items():
        is_drifted = bool(col_info.get("drift_detected", False))
        log_fn = logger.warning if is_drifted else logger.info
        status = "DRIFT" if is_drifted else "OK"
        log_fn(
            "[drift]   %-20s | %-5s | teste=%s | score/p=%.6f",
            col_name,
            status,
            col_info.get("stattest_name"),
            float(col_info.get("drift_score", 0.0)),
        )


def log_pipeline_run_to_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    health_metrics: PipelineHealthMetrics,
    model_path: Path | None = None,
    drift_report_path: Path | None = None,
    log_file_path: Path | None = None,
    tags: dict[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """Registra parâmetros, métricas de saúde/drift, tags e artefatos de uma execução no MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(health_metrics.to_mlflow_metrics())

        if model_path is not None and model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="model")

        if drift_report_path is not None and drift_report_path.exists():
            mlflow.log_artifact(str(drift_report_path), artifact_path="reports")

        run_id = run.info.run_id
        if logger is not None:
            logger.info(
                "Run registrado no MLflow (experiment='%s', run_id='%s', uri='%s').",
                experiment_name,
                run_id,
                tracking_uri,
            )
            for handler in logger.handlers:
                handler.flush()

        if log_file_path is not None and log_file_path.exists():
            mlflow.log_artifact(str(log_file_path), artifact_path="logs")

        return run_id
