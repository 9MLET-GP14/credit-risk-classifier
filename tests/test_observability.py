"""Testes unitários para o módulo de observabilidade, logging e MLflow Tracking."""

import logging
from pathlib import Path

import mlflow
import pandas as pd
import pandera.pandas as pa
import pytest
from mlflow.tracking import MlflowClient

from credit_risk.observability import (
    PipelineHealthMetrics,
    audit_contract_health,
    get_logger,
    log_drift_details,
    log_pipeline_run_to_mlflow,
    validate_with_observability,
)


def _sample_valid_df() -> pd.DataFrame:
    """Cria um pequeno lote válido compatível com os contratos Pandera."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1001, 1002, 1003, 1004],
            "TARGET": [0, 1, 0, 0],
            "AMT_INCOME_TOTAL": [120000.0, 85000.0, 150000.0, 95000.0],
            "AMT_CREDIT": [350000.0, 280000.0, 400000.0, 210000.0],
            "AMT_ANNUITY": [18000.0, 14000.0, 22000.0, 11000.0],
            "AGE_YEARS": [35, 42, 29, 51],
            "EMPLOYED_YEARS": [6.0, 10.0, 3.5, 15.0],
            "NAME_INCOME_TYPE": ["Working", "Commercial associate", "State servant", "Pensioner"],
            "NAME_EDUCATION_TYPE": [
                "Higher education",
                "Secondary / secondary special",
                "Higher education",
                "Secondary / secondary special",
            ],
        }
    )


def test_get_logger_avoids_duplicate_handlers_and_writes_file(tmp_path: Path) -> None:
    """Garante criação de logger sem handlers duplicados e escrita em arquivo opcional."""
    log_path = tmp_path / "logs" / "pipeline.log"
    logger_name = "credit_risk.test_unique_logger"

    logger_a = get_logger(name=logger_name, level=logging.INFO, log_file=log_path)
    logger_b = get_logger(name=logger_name, level=logging.INFO, log_file=log_path)

    assert logger_a is logger_b
    assert len(logger_a.handlers) == 2

    logger_a.info("Mensagem de teste estruturada.")
    for handler in logger_a.handlers:
        handler.flush()
        if isinstance(handler, logging.FileHandler):
            handler.close()

    assert log_path.exists()
    assert "Mensagem de teste estruturada." in log_path.read_text(encoding="utf-8")


def test_validate_with_observability_passes_valid_and_logs_error_on_invalid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Valida lote íntegro com INFO e registra ERROR ao bloquear lote corrompido."""
    logger = get_logger("credit_risk.test_validate")
    logger.propagate = True
    valid_df = _sample_valid_df()

    with caplog.at_level(logging.INFO, logger=logger.name):
        validated = validate_with_observability(valid_df, is_training=True, logger=logger)

    assert len(validated) == len(valid_df)
    assert any("Contrato Pandera aprovado" in rec.message for rec in caplog.records)

    corrupted_df = valid_df.copy()
    corrupted_df.loc[0, "AGE_YEARS"] = 14

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=logger.name):
        with pytest.raises(pa.errors.SchemaErrors):
            validate_with_observability(corrupted_df, is_training=True, logger=logger)

    assert any(rec.levelno == logging.ERROR for rec in caplog.records)


def test_audit_contract_health_computes_rejection_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Auditoria deve retornar 0% em lote limpo e taxa proporcional em lote corrompido."""
    logger = get_logger("credit_risk.test_audit")
    logger.propagate = True
    df = _sample_valid_df()

    clean_audit = audit_contract_health(df, is_training=True, logger=logger)
    assert clean_audit["passed"] is True
    assert clean_audit["rejected_records"] == 0
    assert clean_audit["rejection_rate"] == 0.0

    corrupted = df.copy()
    corrupted.loc[0, "AGE_YEARS"] = 12
    corrupted.loc[1, "AMT_INCOME_TOTAL"] = -500.0

    with caplog.at_level(logging.WARNING, logger=logger.name):
        bad_audit = audit_contract_health(corrupted, is_training=True, logger=logger)

    assert bad_audit["passed"] is False
    assert bad_audit["rejected_records"] == 2
    assert bad_audit["rejection_rate"] == 0.5
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_log_pipeline_run_to_mlflow_persists_params_metrics_and_artifacts(tmp_path: Path) -> None:
    """Garante registro completo de parâmetros, métricas e artefatos em run do MLflow."""
    tracking_uri = (tmp_path / "mlruns").resolve().as_uri()
    model_file = tmp_path / "baseline_model.joblib"
    report_file = tmp_path / "drift_report.html"
    model_file.write_bytes(b"dummy-model-bytes")
    report_file.write_text("<html><body>Drift OK</body></html>", encoding="utf-8")

    health = PipelineHealthMetrics(
        total_records_received=4000,
        valid_records=4000,
        rejected_records=0,
        rejection_rate=0.0,
        roc_auc=0.72,
        recall=0.61,
        precision=0.19,
        f1=0.29,
        dataset_drift=True,
        drifted_columns_count=6,
        total_monitored_columns=9,
        drift_share=0.6667,
        reference_default_rate=0.37,
        production_default_rate=0.35,
    )

    log_file = tmp_path / "pipeline.log"
    log_file.write_text("2026-11-16 | INFO | test | ok\n", encoding="utf-8")

    run_id = log_pipeline_run_to_mlflow(
        tracking_uri=tracking_uri,
        experiment_name="test-credit-risk",
        run_name="unit-test-run",
        params={"model_type": "hist_gradient_boosting", "sample_size": 20000},
        health_metrics=health,
        model_path=model_file,
        drift_report_path=report_file,
        log_file_path=log_file,
        tags={"status": "SUCCESS", "scenario": "drift"},
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["model_type"] == "hist_gradient_boosting"
    assert run.data.tags["status"] == "SUCCESS"
    assert run.data.metrics["roc_auc"] == pytest.approx(0.72)
    assert run.data.metrics["dataset_drift"] == 1.0
    assert run.data.metrics["drifted_columns_count"] == 6.0

    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id, tracking_uri=tracking_uri)
    artifact_paths = {a.path for a in artifacts}
    assert "model" in artifact_paths
    assert "reports" in artifact_paths
    assert "logs" in artifact_paths


def test_log_drift_details_logs_warning_and_info(caplog: pytest.LogCaptureFixture) -> None:
    """Garante registro integrado de drift global e por coluna no logger centralizado."""
    logger = get_logger("credit_risk.test_drift_log")
    logger.propagate = True

    summary = {
        "dataset_drift": True,
        "number_of_drifted_columns": 1,
        "number_of_columns": 2,
        "share_of_drifted_columns": 0.5,
        "column_stats": {
            "AMT_INCOME_TOTAL": {
                "stattest_name": "K-S p_value",
                "drift_score": 0.0,
                "drift_detected": True,
            },
            "EMPLOYED_YEARS": {
                "stattest_name": "K-S p_value",
                "drift_score": 1.0,
                "drift_detected": False,
            },
        },
    }

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_drift_details(summary, logger=logger)

    assert any(
        rec.levelno == logging.WARNING and "AMT_INCOME_TOTAL" in rec.message
        for rec in caplog.records
    )
    assert any(
        rec.levelno == logging.INFO and "EMPLOYED_YEARS" in rec.message for rec in caplog.records
    )


def test_collect_infrastructure_metrics_returns_positive_resources() -> None:
    from credit_risk.observability import collect_infrastructure_metrics, prime_cpu_monitor

    prime_cpu_monitor()
    infra = collect_infrastructure_metrics()
    assert set(infra) == {
        "process_memory_mb",
        "process_cpu_seconds",
        "system_cpu_percent",
        "system_memory_percent",
    }
    assert infra["process_memory_mb"] > 0
    assert 0 <= infra["system_memory_percent"] <= 100


def test_log_concept_drift_details_flags_degradation(caplog) -> None:
    from credit_risk.observability import get_logger, log_concept_drift_details

    logger = get_logger("credit_risk.test_concept")
    logger.propagate = True
    ref = {"roc_auc": 0.62, "recall": 0.53}
    with caplog.at_level(logging.WARNING, logger="credit_risk.test_concept"):
        assert log_concept_drift_details(ref, {"roc_auc": 0.50, "recall": 0.35}, {}, logger)
    assert "ALERTA DE DEGRADAÇÃO" in caplog.text
    assert not log_concept_drift_details(ref, {"roc_auc": 0.61, "recall": 0.52}, {}, logger)
