"""Testes unitários para o treinamento, avaliação e persistência do modelo baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.model import (
    ModelType,
    evaluate_model,
    load_model_artifact,
    predict_credit_risk,
    save_model_artifact,
    train_baseline_model,
)


def _create_synthetic_credit_data(n_rows: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": rng.normal(120_000, 30_000, size=n_rows).clip(min=20_000),
            "AMT_CREDIT": rng.normal(350_000, 100_000, size=n_rows).clip(min=50_000),
            "AMT_ANNUITY": rng.normal(20_000, 5_000, size=n_rows).clip(min=2_000),
            "AGE_YEARS": rng.integers(20, 65, size=n_rows),
            "EMPLOYED_YEARS": rng.uniform(0.0, 20.0, size=n_rows).round(1),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Higher education", "Secondary / secondary special"], size=n_rows
            ),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate"], size=n_rows),
        }
    )
    risk_score = (X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"]) > 3.2
    y = pd.Series(risk_score.astype(int), name="TARGET")
    return X, y


@pytest.mark.parametrize("model_type", ["logistic_regression", "hist_gradient_boosting"])
def test_train_baseline_model_predicts_valid_probabilities(model_type: ModelType) -> None:
    """Garante que ambos os classificadores treinam e retornam probabilidades entre 0 e 1."""
    X, y = _create_synthetic_credit_data(n_rows=100)
    pipeline = train_baseline_model(X, y, model_type=model_type, random_state=42)

    probas = pipeline.predict_proba(X)
    preds = pipeline.predict(X)

    assert probas.shape == (100, 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    assert set(np.unique(preds)).issubset({0, 1})


def test_train_baseline_model_rejects_invalid_model_type() -> None:
    """Garante erro claro quando um model_type não suportado é informado."""
    X, y = _create_synthetic_credit_data(n_rows=40)
    with pytest.raises(ValueError, match="Tipo de modelo inválido"):
        train_baseline_model(X, y, model_type="invalid_model")  # type: ignore[arg-type]


def test_evaluate_model_returns_expected_metrics() -> None:
    """Valida se todas as métricas de risco de crédito são calculadas no intervalo [0, 1]."""
    X, y = _create_synthetic_credit_data(n_rows=120)
    X_train, X_test = X.iloc[:90], X.iloc[90:]
    y_train, y_test = y.iloc[:90], y.iloc[90:]

    pipeline = train_baseline_model(X_train, y_train, random_state=42)
    metrics = evaluate_model(pipeline, X_test, y_test)

    expected_keys = {"roc_auc", "recall", "precision", "f1", "accuracy"}
    assert set(metrics.keys()) == expected_keys
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_save_and_load_model_artifact_preserves_predictions(tmp_path: Path) -> None:
    """Garante que salvar e recarregar via joblib preserva exatamente as predições."""
    X, y = _create_synthetic_credit_data(n_rows=80)
    pipeline = train_baseline_model(X, y, random_state=42)

    artifact_path = tmp_path / "models" / "baseline_model.joblib"
    saved_path = save_model_artifact(pipeline, artifact_path)
    loaded_pipeline = load_model_artifact(saved_path)

    assert saved_path.exists()
    np.testing.assert_allclose(
        pipeline.predict_proba(X),
        loaded_pipeline.predict_proba(X),
    )


def test_predict_credit_risk_generates_scores_and_ignores_target(tmp_path: Path) -> None:
    """Valida que predict_credit_risk remove TARGET, preserva ID e gera RISK_SCORE válido."""
    X, y = _create_synthetic_credit_data(n_rows=60)
    pipeline = train_baseline_model(X, y, random_state=42)
    artifact_path = save_model_artifact(pipeline, tmp_path / "model.joblib")

    input_df = X.copy()
    input_df["SK_ID_CURR"] = range(1000, 1060)
    input_df["TARGET"] = y

    preds_df = predict_credit_risk(df=input_df, model_path=artifact_path)

    assert "TARGET" not in preds_df.columns
    assert "SK_ID_CURR" in preds_df.columns
    assert "TARGET_PRED" in preds_df.columns
    assert "RISK_SCORE" in preds_df.columns
    assert set(preds_df["TARGET_PRED"].unique()).issubset({0, 1})
    assert ((preds_df["RISK_SCORE"] >= 0.0) & (preds_df["RISK_SCORE"] <= 1.0)).all()


def test_predict_credit_risk_raises_when_model_missing(tmp_path: Path) -> None:
    """Deve levantar FileNotFoundError claro caso o artefato não exista."""
    X, _ = _create_synthetic_credit_data(n_rows=10)
    with pytest.raises(FileNotFoundError, match="Artefato do modelo não encontrado"):
        predict_credit_risk(df=X, model_path=tmp_path / "missing.joblib")
