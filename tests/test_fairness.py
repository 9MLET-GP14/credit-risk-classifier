"""Testes da auditoria de vieses (fairness) por grupo."""

from pathlib import Path

import pandas as pd

from credit_risk.fairness import (
    add_age_band,
    compute_group_fairness,
    count_fairness_alerts,
    run_fairness_audit,
    summarize_fairness,
)


def _scored_sample() -> pd.DataFrame:
    """Jovens são negados com frequência muito maior que os demais (viés intencional)."""
    young = pd.DataFrame(
        {"AGE_YEARS": [22] * 100, "TARGET": [0] * 90 + [1] * 10, "TARGET_PRED": [1] * 70 + [0] * 30}
    )
    older = pd.DataFrame(
        {"AGE_YEARS": [55] * 100, "TARGET": [0] * 90 + [1] * 10, "TARGET_PRED": [1] * 10 + [0] * 90}
    )
    return pd.concat([young, older], ignore_index=True)


def test_add_age_band_assigns_expected_bands() -> None:
    df = add_age_band(pd.DataFrame({"AGE_YEARS": [18, 25, 26, 40, 51, 90]}))
    assert df["AGE_BAND"].tolist() == ["18-25", "18-25", "26-35", "36-50", "51+", "51+"]


def test_group_fairness_and_disparate_impact_flag_bias() -> None:
    df = add_age_band(_scored_sample())
    table = compute_group_fairness(df, "AGE_BAND")
    young = table.set_index("group").loc["18-25"]
    assert young["approval_rate"] == 0.30
    assert young["false_denial_rate"] == 70 / 90

    summary = summarize_fairness(table)
    assert summary["disparate_impact"] == round(0.30 / 0.90, 4)
    assert summary["least_approved_group"] == "18-25"
    assert summary["alert"] is True


def test_summary_ignores_small_groups_and_passes_without_bias() -> None:
    table = pd.DataFrame(
        {"group": ["a", "b", "tiny"], "n": [100, 100, 5], "approval_rate": [0.8, 0.75, 0.1]}
    )
    summary = summarize_fairness(table, min_group_size=30)
    assert summary["disparate_impact"] == round(0.75 / 0.8, 4)
    assert summary["alert"] is False


def test_run_fairness_audit_writes_html_and_counts_alerts(tmp_path: Path) -> None:
    report = tmp_path / "fairness.html"
    results = run_fairness_audit({"produção": _scored_sample()}, ["AGE_BAND"], report_path=report)
    assert report.exists()
    assert "Relatório de Fairness" in report.read_text(encoding="utf-8")
    assert count_fairness_alerts(results) == 1
