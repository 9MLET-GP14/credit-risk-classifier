"""Observabilidade centralizada: logging estruturado, saúde de contratos e MLflow Tracking."""

import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pandera.pandas as pa
import psutil
from prometheus_client import CollectorRegistry, Gauge, write_to_textfile

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
    # Concept Drift: performance em produção com rótulos reais (ground truth com atraso)
    production_roc_auc: float = 0.0
    production_recall: float = 0.0
    production_precision: float = 0.0
    production_f1: float = 0.0
    roc_auc_drop: float = 0.0
    reference_observed_default_rate: float = 0.0
    production_observed_default_rate: float = 0.0
    target_drift: bool = False
    performance_alert: bool = False
    # Fairness: disparate impact por atributo (referência e produção)
    fairness_di_age_reference: float = 1.0
    fairness_di_age_production: float = 1.0
    fairness_di_education_production: float = 1.0
    fairness_di_income_type_production: float = 1.0
    fairness_alerts_count: int = 0
    # Consolidação de alertas (drift por coluna + alvo + performance + fairness + rejeições)
    total_alerts_count: int = 0
    # Infraestrutura: consumo de recursos do processo e do host
    process_memory_mb: float = 0.0
    process_cpu_seconds: float = 0.0
    system_cpu_percent: float = 0.0
    system_memory_percent: float = 0.0

    def to_mlflow_metrics(self) -> dict[str, float]:
        """Converte todas as métricas para valores numéricos aceitos pelo MLflow."""
        raw = asdict(self)
        return {k: float(int(v) if isinstance(v, bool) else v) for k, v in raw.items()}


def collect_infrastructure_metrics() -> dict[str, float]:
    """Coleta métricas de estabilidade da infraestrutura (processo atual e host).

    ``system_cpu_percent`` mede o uso desde a chamada anterior a ``psutil.cpu_percent``;
    chame ``prime_cpu_monitor()`` no início do pipeline para cobrir toda a execução.
    """
    process = psutil.Process()
    cpu_times = process.cpu_times()
    return {
        "process_memory_mb": round(process.memory_info().rss / 1024**2, 2),
        "process_cpu_seconds": round(cpu_times.user + cpu_times.system, 3),
        "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
        "system_memory_percent": float(psutil.virtual_memory().percent),
    }


def prime_cpu_monitor() -> None:
    """Inicia a janela de medição de CPU do host usada por ``collect_infrastructure_metrics``."""
    psutil.cpu_percent(interval=None)


def log_concept_drift_details(
    reference_metrics: dict[str, float],
    production_metrics: dict[str, float],
    target_summary: dict[str, Any],
    logger: logging.Logger,
    min_roc_auc: float = 0.60,
    max_auc_drop: float = 0.05,
) -> bool:
    """Registra a degradação de performance em produção e retorna se há alerta."""
    auc_drop = reference_metrics["roc_auc"] - production_metrics["roc_auc"]
    alert = production_metrics["roc_auc"] < min_roc_auc or auc_drop > max_auc_drop
    log_fn = logger.warning if alert else logger.info
    log_fn(
        "[concept] ROC-AUC referência=%.4f -> produção=%.4f (queda=%.4f) | "
        "Recall %.4f -> %.4f | %s",
        reference_metrics["roc_auc"],
        production_metrics["roc_auc"],
        auc_drop,
        reference_metrics["recall"],
        production_metrics["recall"],
        "ALERTA DE DEGRADAÇÃO" if alert else "performance estável",
    )
    target_fn = logger.warning if target_summary.get("target_drift") else logger.info
    target_fn(
        "[concept] Drift do alvo real (TARGET): %s | teste=z | p=%.6f",
        "DETECTADO" if target_summary.get("target_drift") else "não detectado",
        float(target_summary.get("target_drift_score", 1.0)),
    )
    return bool(alert)


def log_fairness_details(results: dict[str, dict[str, dict]], logger: logging.Logger) -> None:
    """Registra no log centralizado os indicadores de viés por dataset e atributo."""
    for dataset, by_column in results.items():
        for column, summary in by_column.items():
            log_fn = logger.warning if summary["alert"] else logger.info
            log_fn(
                "[fairness] %-10s | %-20s | disparate_impact=%.3f | gap_negação=%.3f | "
                "menos aprovado=%s | %s",
                dataset,
                column,
                summary["disparate_impact"],
                summary["false_denial_gap"],
                summary.get("least_approved_group", "-"),
                "ALERTA" if summary["alert"] else "OK",
            )


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
    extra_report_paths: list[Path] | None = None,
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

        for extra_path in extra_report_paths or []:
            if extra_path.exists():
                mlflow.log_artifact(str(extra_path), artifact_path="reports")

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


def export_metrics_to_prometheus(
    health_metrics: PipelineHealthMetrics,
    output_path: Path,
    scenario: str = "drift",
    logger: logging.Logger | None = None,
) -> None:
    """Exporta métricas de saúde, drift e infra em formato OpenMetrics (.prom)."""
    registry = CollectorRegistry()

    # Gauge geral de status (1=sucesso, 0=falha)
    g_status = Gauge(
        "credit_risk_pipeline_status",
        "Status de execução do pipeline (1=sucesso, 0=falha)",
        ["scenario"],
        registry=registry,
    )
    g_status.labels(scenario=scenario).set(float(health_metrics.pipeline_status))

    # Métricas de volumetria e qualidade (Pandera)
    g_records_received = Gauge(
        "credit_risk_total_records_received",
        "Total de registros recebidos no lote de produção",
        ["scenario"],
        registry=registry,
    )
    g_records_received.labels(scenario=scenario).set(float(health_metrics.total_records_received))

    g_rejected_records = Gauge(
        "credit_risk_rejected_records",
        "Registros rejeitados pelo contrato de dados Pandera",
        ["scenario"],
        registry=registry,
    )
    g_rejected_records.labels(scenario=scenario).set(float(health_metrics.rejected_records))

    g_rejection_rate = Gauge(
        "credit_risk_rejection_rate",
        "Taxa de rejeição do lote pelo contrato Pandera",
        ["scenario"],
        registry=registry,
    )
    g_rejection_rate.labels(scenario=scenario).set(float(health_metrics.rejection_rate))

    # Métricas de latência
    g_latency = Gauge(
        "credit_risk_pipeline_latency_seconds",
        "Latência total de execução do pipeline em segundos",
        ["scenario"],
        registry=registry,
    )
    g_latency.labels(scenario=scenario).set(float(health_metrics.total_latency_seconds))

    # Métricas de Drift (Evidently AI)
    g_drift_share = Gauge(
        "credit_risk_drift_share",
        "Proporção de colunas com Data Drift detectado pelo Evidently AI",
        ["scenario"],
        registry=registry,
    )
    g_drift_share.labels(scenario=scenario).set(float(health_metrics.drift_share))

    g_drift_cols = Gauge(
        "credit_risk_drifted_columns_count",
        "Quantidade de colunas com drift estatístico",
        ["scenario"],
        registry=registry,
    )
    g_drift_cols.labels(scenario=scenario).set(float(health_metrics.drifted_columns_count))

    g_target_drift = Gauge(
        "credit_risk_target_drift",
        "Indica se houve drift na taxa de inadimplência observada (Concept Drift)",
        ["scenario"],
        registry=registry,
    )
    g_target_drift.labels(scenario=scenario).set(float(health_metrics.target_drift))

    # Métricas de Modelo e Degradação
    g_roc_auc = Gauge(
        "credit_risk_model_roc_auc",
        "ROC-AUC do modelo em teste de referência",
        ["scenario"],
        registry=registry,
    )
    g_roc_auc.labels(scenario=scenario).set(float(health_metrics.roc_auc))

    g_prod_roc_auc = Gauge(
        "credit_risk_production_roc_auc",
        "ROC-AUC do modelo em produção com rótulos reais observados",
        ["scenario"],
        registry=registry,
    )
    g_prod_roc_auc.labels(scenario=scenario).set(float(health_metrics.production_roc_auc))

    g_roc_auc_drop = Gauge(
        "credit_risk_roc_auc_drop",
        "Queda absoluta no ROC-AUC entre referência e produção",
        ["scenario"],
        registry=registry,
    )
    g_roc_auc_drop.labels(scenario=scenario).set(float(health_metrics.roc_auc_drop))

    # Métricas de Governança e Fairness
    g_fairness_alerts = Gauge(
        "credit_risk_fairness_alerts_count",
        "Total de alertas de viés (Disparate Impact < 0.80) disparados",
        ["scenario"],
        registry=registry,
    )
    g_fairness_alerts.labels(scenario=scenario).set(float(health_metrics.fairness_alerts_count))

    g_total_alerts = Gauge(
        "credit_risk_total_alerts_count",
        "Total de alertas consolidados do pipeline",
        ["scenario"],
        registry=registry,
    )
    g_total_alerts.labels(scenario=scenario).set(float(health_metrics.total_alerts_count))

    # Métricas de Infraestrutura
    g_mem = Gauge(
        "credit_risk_process_memory_mb",
        "Uso de memória RAM pelo processo em MB",
        ["scenario"],
        registry=registry,
    )
    g_mem.labels(scenario=scenario).set(float(health_metrics.process_memory_mb))

    g_cpu = Gauge(
        "credit_risk_system_cpu_percent",
        "Uso de CPU do host percentual",
        ["scenario"],
        registry=registry,
    )
    g_cpu.labels(scenario=scenario).set(float(health_metrics.system_cpu_percent))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_to_textfile(str(output_path), registry)

    if logger is not None:
        logger.info("Métricas exportadas para o Prometheus Textfile em '%s'.", output_path)
