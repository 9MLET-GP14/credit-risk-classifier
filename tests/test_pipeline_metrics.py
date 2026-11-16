"""Regressão das métricas de produção no orquestrador, sem download nem treino real."""

import argparse
import importlib.util
import logging
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pandas as pd
import pytest

from credit_risk.config import Settings


@pytest.fixture
def pipeline_module() -> ModuleType:
    """Importa o script sem executar o bloco __main__."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_metrics_under_test", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_reference(rows: int) -> pd.DataFrame:
    """Cria um lote pequeno de atributos válidos, com IDs únicos."""
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(1, rows + 1),
            "AMT_INCOME_TOTAL": [120000.0] * rows,
            "AMT_CREDIT": [300000.0] * rows,
            "AMT_ANNUITY": [18000.0] * rows,
            "AGE_YEARS": [35] * rows,
            "EMPLOYED_YEARS": [5.0] * rows,
            "NAME_INCOME_TYPE": ["Working"] * rows,
            "NAME_EDUCATION_TYPE": ["Higher education"] * rows,
            "TARGET": [i % 2 for i in range(rows)],
        }
    )


@pytest.fixture
def pipeline_environment(
    pipeline_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, Settings, Mock]:
    """Isola arquivos, treino, inferência, relatório e tracking externos ao teste."""
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        mlruns_dir=tmp_path / "mlruns",
        logs_dir=tmp_path / "logs",
    )
    monkeypatch.setattr(pipeline_module, "Settings", lambda: settings)
    logger = logging.getLogger("credit_risk.test_pipeline_metrics")
    monkeypatch.setattr(pipeline_module, "get_logger", Mock(return_value=logger))
    monkeypatch.setattr(pipeline_module, "train_baseline_model", Mock(return_value=object()))
    monkeypatch.setattr(
        pipeline_module,
        "evaluate_model",
        Mock(return_value={"roc_auc": 0.7, "recall": 0.6, "precision": 0.5, "f1": 0.55}),
    )
    monkeypatch.setattr(pipeline_module, "save_model_artifact", Mock())

    def predict_stub(df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Devolve scores fixos para testar o fluxo, não a qualidade do modelo."""
        result = df.drop(columns=[settings.target_column], errors="ignore").copy()
        result["TARGET_PRED"] = 0
        result["RISK_SCORE"] = 0.1
        return result

    monkeypatch.setattr(pipeline_module, "predict_credit_risk", Mock(side_effect=predict_stub))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", Mock())
    monkeypatch.setattr(
        pipeline_module,
        "generate_drift_report",
        Mock(
            return_value={
                "dataset_drift": False,
                "number_of_drifted_columns": 0,
                "number_of_columns": 9,
                "share_of_drifted_columns": 0.0,
            }
        ),
    )
    monkeypatch.setattr(pipeline_module, "log_drift_details", Mock())
    tracking = Mock(return_value="regression-test-run")
    monkeypatch.setattr(pipeline_module, "log_pipeline_run_to_mlflow", tracking)
    return pipeline_module, settings, tracking


@pytest.mark.parametrize("batch_size", [10, 4000])
@pytest.mark.parametrize("scenario", ["clean", "drift", "corrupted"])
def test_pipeline_metrics_use_only_production_batch(
    pipeline_environment: tuple[ModuleType, Settings, Mock],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    scenario: str,
    batch_size: int,
) -> None:
    """Treino e teste não podem inflar o denominador das rejeições de produção."""
    pipeline, settings, tracking = pipeline_environment
    train_df = _valid_reference(120)
    test_df = _valid_reference(batch_size)
    for path in (settings.reference_train_path, settings.reference_test_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(pd, "read_parquet", Mock(side_effect=[train_df, test_df]))
    monkeypatch.setattr(
        pipeline,
        "parse_args",
        Mock(
            return_value=argparse.Namespace(
                scenario=scenario, refresh_data=False, run_name="metrics-regression"
            )
        ),
    )

    with caplog.at_level(logging.INFO, logger="credit_risk.test_pipeline_metrics"):
        if scenario == "corrupted":
            with pytest.raises(SystemExit) as exc_info:
                pipeline.main()
            assert exc_info.value.code == 1
            # Somente a referência recebeu scores; a inferência em produção foi bloqueada.
            assert pipeline.predict_credit_risk.call_count == 1
            pipeline.generate_drift_report.assert_not_called()
        else:
            pipeline.main()
            assert pipeline.predict_credit_risk.call_count == 2

    tracking.assert_called_once()
    health = tracking.call_args.kwargs["health_metrics"]
    rejected = 6 if scenario == "corrupted" else 0
    expected_rate = rejected / batch_size
    assert health.total_records_received == batch_size
    assert health.valid_records == batch_size - rejected
    assert health.rejected_records == rejected
    assert health.rejection_rate == pytest.approx(expected_rate)
    assert health.pipeline_status == (0 if scenario == "corrupted" else 1)
    assert tracking.call_args.kwargs["tags"]["status"] == (
        "FAILED" if scenario == "corrupted" else "SUCCESS"
    )
    assert f"Registros={batch_size} | Rejeitados={rejected} ({expected_rate * 100:.2f}%)" in (
        caplog.text
    )


def test_failure_before_production_logs_zero_received_records(
    pipeline_environment: tuple[ModuleType, Settings, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falha de ingestão deve registrar zero produção, sem variável não inicializada."""
    pipeline, _, tracking = pipeline_environment
    monkeypatch.setattr(
        pipeline,
        "parse_args",
        Mock(
            return_value=argparse.Namespace(
                scenario="clean", refresh_data=False, run_name="ingestion-failure"
            )
        ),
    )
    monkeypatch.setattr(
        pipeline, "download_raw_dataset", Mock(side_effect=OSError("Falha de ingestão simulada"))
    )
    with pytest.raises(SystemExit) as exc_info:
        pipeline.main()
    assert exc_info.value.code == 1
    tracking.assert_called_once()
    health = tracking.call_args.kwargs["health_metrics"]
    assert health.total_records_received == 0
    assert health.valid_records == 0
    assert health.rejected_records == 0
    assert health.rejection_rate == 0.0
    assert health.pipeline_status == 0
