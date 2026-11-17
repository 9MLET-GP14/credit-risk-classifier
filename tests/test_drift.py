"""Testes unitários para simulação de choque econômico e detecção de drift com Evidently AI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.contracts import validate_data
from credit_risk.drift import (
    generate_drift_report,
    generate_performance_report,
    simulate_concept_drift,
    simulate_economic_shock,
)


def _create_reference_sample(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(1, n_rows + 1),
            "TARGET": rng.choice([0, 1], size=n_rows, p=[0.9, 0.1]),
            "AMT_INCOME_TOTAL": rng.normal(150_000, 30_000, size=n_rows).clip(min=25_000),
            "AMT_CREDIT": rng.normal(350_000, 80_000, size=n_rows).clip(min=45_000),
            "AMT_ANNUITY": rng.normal(18_000, 4_000, size=n_rows).clip(min=3_000),
            "AGE_YEARS": rng.integers(25, 65, size=n_rows),
            "EMPLOYED_YEARS": rng.uniform(1.0, 20.0, size=n_rows).round(1),
            "NAME_INCOME_TYPE": rng.choice(["Working", "State servant"], size=n_rows),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Higher education", "Secondary / secondary special"], size=n_rows
            ),
            "TARGET_PRED": rng.choice([0, 1], size=n_rows, p=[0.7, 0.3]),
        }
    )


def test_simulate_economic_shock_preserves_contract_and_shifts_distributions() -> None:
    """O dataset degradado deve respeitar o contrato Pandera e alterar renda e crédito."""
    ref_df = _create_reference_sample(n_rows=300)
    prod_df = simulate_economic_shock(ref_df, random_state=42)

    assert "TARGET" not in prod_df.columns
    assert len(prod_df) == len(ref_df)
    # Deve continuar passando pelo contrato estrito do Pandera (degradação silenciosa!)
    validated = validate_data(prod_df, is_training=False)
    assert len(validated) == len(prod_df)

    # Renda deve cair e crédito deve subir significativamente
    assert prod_df["AMT_INCOME_TOTAL"].mean() < ref_df["AMT_INCOME_TOTAL"].mean() * 0.85
    assert prod_df["AMT_CREDIT"].mean() > ref_df["AMT_CREDIT"].mean() * 1.20


def test_generate_drift_report_creates_html_and_detects_drifted_features(tmp_path: Path) -> None:
    """Garante geração do relatório HTML do Evidently e detecção de drift nas features."""
    ref_df = _create_reference_sample(n_rows=200)
    prod_df = simulate_economic_shock(ref_df, random_state=42)
    prod_df["TARGET_PRED"] = 1  # força drift na predição também

    report_path = tmp_path / "reports" / "drift_report.html"
    summary = generate_drift_report(
        reference_df=ref_df,
        production_df=prod_df,
        report_path=report_path,
    )

    assert report_path.exists()
    assert report_path.stat().st_size > 10_000
    assert summary["dataset_drift"] is True
    assert summary["number_of_drifted_columns"] >= 2
    assert "AMT_INCOME_TOTAL" in summary["drifted_columns"]
    assert "AMT_CREDIT" in summary["drifted_columns"]


def test_simulate_concept_drift_changes_labels_but_not_features() -> None:
    """Concept Drift altera apenas P(TARGET|X): features idênticas, rótulos diferentes."""
    ref_df = _create_reference_sample(n_rows=500)
    shocked = simulate_economic_shock(ref_df, random_state=42, keep_target=True)
    assert "TARGET" in shocked.columns

    drifted = simulate_concept_drift(shocked, random_state=42)
    feature_cols = [c for c in shocked.columns if c != "TARGET"]
    pd.testing.assert_frame_equal(drifted[feature_cols], shocked[feature_cols])
    assert (drifted["TARGET"] != shocked["TARGET"]).sum() > 0
    assert drifted["TARGET"].isin([0, 1]).all()


def test_simulate_concept_drift_requires_target() -> None:
    ref_df = _create_reference_sample(n_rows=50).drop(columns=["TARGET"])
    with pytest.raises(ValueError, match="Concept Drift requer rótulos"):
        simulate_concept_drift(ref_df)


def test_generate_performance_report_detects_target_drift(tmp_path: Path) -> None:
    """Relatório de Concept Drift deve ser gerado e acusar mudança na taxa de inadimplência."""
    rng = np.random.default_rng(0)
    ref_df = pd.DataFrame(
        {"TARGET": rng.choice([0, 1], 1000, p=[0.92, 0.08]), "RISK_SCORE": rng.random(1000)}
    )
    prod_df = pd.DataFrame(
        {"TARGET": rng.choice([0, 1], 1000, p=[0.75, 0.25]), "RISK_SCORE": rng.random(1000)}
    )
    report_path = tmp_path / "performance.html"
    summary = generate_performance_report(ref_df, prod_df, report_path=report_path)
    assert report_path.exists()
    assert summary["target_drift"] is True
    assert summary["target_drift_score"] < 0.05
