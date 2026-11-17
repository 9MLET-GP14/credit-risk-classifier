"""Auditoria de vieses (fairness) das decisões de crédito por grupo demográfico/socioeconômico.

Métricas por grupo:
- ``approval_rate``: proporção de propostas aprovadas (``TARGET_PRED == 0``).
- ``false_denial_rate``: bons pagadores negados indevidamente (FPR sobre ``TARGET == 0``).
- ``recall``: inadimplentes reais capturados pelo modelo (TPR sobre ``TARGET == 1``).

Métricas agregadas:
- ``disparate_impact``: menor taxa de aprovação / maior taxa de aprovação entre grupos.
  Abaixo de 0,80 viola a "regra dos quatro quintos" (referência usual de impacto adverso).
- ``false_denial_gap``: diferença máxima de negação indevida entre grupos (igualdade de
  oportunidade para bons pagadores). Acima de 0,10 (10 p.p.) gera alerta.
"""

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

AGE_BAND_COLUMN = "AGE_BAND"
AGE_BAND_EDGES = [17, 25, 35, 50, 120]
AGE_BAND_LABELS = ["18-25", "26-35", "36-50", "51+"]

DISPARATE_IMPACT_THRESHOLD = 0.80
FALSE_DENIAL_GAP_THRESHOLD = 0.10
MIN_GROUP_SIZE = 30


def add_age_band(df: pd.DataFrame, age_column: str = "AGE_YEARS") -> pd.DataFrame:
    """Adiciona a coluna ``AGE_BAND`` (faixa etária) usada apenas na auditoria de vieses."""
    result = df.copy()
    result[AGE_BAND_COLUMN] = pd.cut(
        result[age_column], bins=AGE_BAND_EDGES, labels=AGE_BAND_LABELS
    ).astype(str)
    return result


def compute_group_fairness(
    df: pd.DataFrame,
    group_column: str,
    pred_column: str = "TARGET_PRED",
    target_column: str = "TARGET",
) -> pd.DataFrame:
    """Calcula taxas de aprovação, negação indevida e recall para cada grupo."""
    has_target = target_column in df.columns
    rows: list[dict[str, Any]] = []

    for group, group_df in df.groupby(group_column, observed=True):
        preds = group_df[pred_column].astype(int)
        row: dict[str, Any] = {
            "group": str(group),
            "n": int(len(group_df)),
            "approval_rate": float((preds == 0).mean()),
        }
        if has_target:
            target = group_df[target_column].astype(int)
            good, bad = target == 0, target == 1
            row["observed_default_rate"] = float(target.mean())
            row["false_denial_rate"] = float((preds[good] == 1).mean()) if good.any() else None
            row["recall"] = float((preds[bad] == 1).mean()) if bad.any() else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def summarize_fairness(group_table: pd.DataFrame, min_group_size: int = MIN_GROUP_SIZE) -> dict:
    """Resume a tabela por grupo em métricas de impacto adverso e alertas."""
    eligible = group_table[group_table["n"] >= min_group_size]
    if eligible.empty:
        return {"disparate_impact": 1.0, "false_denial_gap": 0.0, "alert": False}

    max_approval = eligible["approval_rate"].max()
    disparate_impact = (
        float(eligible["approval_rate"].min() / max_approval) if max_approval > 0 else 1.0
    )

    false_denial_gap = 0.0
    if "false_denial_rate" in eligible.columns:
        rates = eligible["false_denial_rate"].dropna()
        if not rates.empty:
            false_denial_gap = float(rates.max() - rates.min())

    least_approved = eligible.loc[eligible["approval_rate"].idxmin(), "group"]
    return {
        "disparate_impact": round(disparate_impact, 4),
        "false_denial_gap": round(false_denial_gap, 4),
        "least_approved_group": str(least_approved),
        "alert": bool(
            disparate_impact < DISPARATE_IMPACT_THRESHOLD
            or false_denial_gap > FALSE_DENIAL_GAP_THRESHOLD
        ),
    }


def run_fairness_audit(
    datasets: dict[str, pd.DataFrame],
    group_columns: list[str],
    report_path: Path | None = None,
    pred_column: str = "TARGET_PRED",
    target_column: str = "TARGET",
) -> dict[str, dict[str, dict]]:
    """Audita cada dataset (ex.: referência e produção) por atributo e gera relatório HTML.

    Args:
        datasets: Mapeamento ``nome -> DataFrame`` com predições (e, se houver, rótulos reais).
        group_columns: Atributos de agrupamento. ``AGE_BAND`` é derivado de ``AGE_YEARS``.
        report_path: Caminho opcional do relatório HTML.

    Returns:
        ``{dataset: {atributo: resumo}}``, onde cada resumo inclui a tabela por grupo.
    """
    results: dict[str, dict[str, dict]] = {}
    for name, df in datasets.items():
        prepared = add_age_band(df) if AGE_BAND_COLUMN in group_columns else df
        results[name] = {}
        for column in group_columns:
            table = compute_group_fairness(prepared, column, pred_column, target_column)
            summary = summarize_fairness(table)
            summary["table"] = table
            results[name][column] = summary

    if report_path is not None:
        _write_html_report(results, report_path)
    return results


def count_fairness_alerts(results: dict[str, dict[str, dict]]) -> int:
    """Conta quantos pares (dataset, atributo) dispararam alerta de viés."""
    return sum(int(s["alert"]) for by_column in results.values() for s in by_column.values())


def _write_html_report(results: dict[str, dict[str, dict]], report_path: Path) -> None:
    """Gera um relatório HTML autocontido com as tabelas de fairness por grupo."""
    sections: list[str] = []
    for dataset, by_column in results.items():
        for column, summary in by_column.items():
            status = "ALERTA" if summary["alert"] else "OK"
            css = "alert" if summary["alert"] else "ok"
            table_html = (
                summary["table"]
                .rename(
                    columns={
                        "group": "Grupo",
                        "n": "N",
                        "approval_rate": "Taxa de aprovação",
                        "observed_default_rate": "Inadimplência observada",
                        "false_denial_rate": "Negação indevida (FPR)",
                        "recall": "Recall (TPR)",
                    }
                )
                .to_html(index=False, float_format=lambda v: f"{v:.2%}", na_rep="-")
            )
            sections.append(
                f"<section><h2>{escape(dataset)} · {escape(column)} "
                f"<span class='{css}'>{status}</span></h2>"
                f"<p>Disparate impact: <b>{summary['disparate_impact']:.3f}</b> "
                f"(mínimo {DISPARATE_IMPACT_THRESHOLD}) · Gap de negação indevida: "
                f"<b>{summary['false_denial_gap']:.2%}</b> "
                f"(máximo {FALSE_DENIAL_GAP_THRESHOLD:.0%}) · Grupo menos aprovado: "
                f"<b>{escape(str(summary.get('least_approved_group', '-')))}</b></p>"
                f"{table_html}</section>"
            )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>Relatório de Fairness</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem auto;max-width:960px;color:#1f2933}}
table{{border-collapse:collapse;width:100%;margin-bottom:1rem}}
th,td{{border:1px solid #d9e2ec;padding:.4rem .6rem;text-align:right}}
th{{background:#f0f4f8}} td:first-child,th:first-child{{text-align:left}}
.alert{{background:#fde8e8;color:#9b1c1c;padding:.1rem .5rem;border-radius:4px}}
.ok{{background:#def7ec;color:#03543f;padding:.1rem .5rem;border-radius:4px}}
</style></head><body>
<h1>Relatório de Fairness — Decisões de Crédito</h1>
<p>Grupos com menos de {MIN_GROUP_SIZE} registros são exibidos, mas não entram no cálculo
das métricas agregadas. Os atributos de grupo são usados apenas para auditoria.</p>
{"".join(sections)}
</body></html>"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
