"""Script CLI: simulação de Data/Concept Drift, relatórios Evidently AI e auditoria de vieses."""

import pandas as pd

from credit_risk.config import Settings
from credit_risk.drift import (
    generate_drift_report,
    generate_performance_report,
    simulate_concept_drift,
    simulate_economic_shock,
)
from credit_risk.fairness import run_fairness_audit
from credit_risk.model import evaluate_predictions, predict_credit_risk


def main() -> None:
    """Simula o choque econômico, mede drift, degradação de performance e vieses."""
    settings = Settings()
    target = settings.target_column

    print(f"[1/6] Carregando dataset de referência ({settings.reference_test_path})...")
    ref_df = pd.read_parquet(settings.reference_test_path)
    ref_with_preds = predict_credit_risk(
        df=ref_df,
        model_path=settings.baseline_model_path,
        id_column=settings.id_column,
        target_column=target,
    )
    ref_with_preds[target] = ref_df[target].to_numpy()

    print("[2/6] Simulando Dataset de Produção (Data Drift + Concept Drift)...")
    prod_df = simulate_economic_shock(ref_df, random_state=settings.random_state, keep_target=True)
    prod_df = simulate_concept_drift(
        prod_df, target_column=target, random_state=settings.random_state
    )

    print("[3/6] Executando inferência (com contrato Pandera) sobre o lote de produção...")
    prod_with_preds = predict_credit_risk(
        df=prod_df,
        model_path=settings.baseline_model_path,
        id_column=settings.id_column,
        target_column=target,
    )
    prod_with_preds[target] = prod_df[target].to_numpy()
    settings.production_shock_path.parent.mkdir(parents=True, exist_ok=True)
    prod_with_preds.to_parquet(settings.production_shock_path, index=False)

    print(f"[4/6] Relatório de Data Drift (Evidently AI) em {settings.drift_report_path}...")
    summary = generate_drift_report(
        reference_df=ref_with_preds,
        production_df=prod_with_preds,
        report_path=settings.drift_report_path,
        id_column=settings.id_column,
        target_column=target,
    )

    print(f"[5/6] Relatório de Concept Drift em {settings.performance_report_path}...")
    target_summary = generate_performance_report(
        reference_df=ref_with_preds,
        production_df=prod_with_preds,
        report_path=settings.performance_report_path,
        target_column=target,
    )
    ref_metrics = evaluate_predictions(
        ref_with_preds[target], ref_with_preds["TARGET_PRED"], ref_with_preds["RISK_SCORE"]
    )
    prod_metrics = evaluate_predictions(
        prod_with_preds[target], prod_with_preds["TARGET_PRED"], prod_with_preds["RISK_SCORE"]
    )

    print(f"[6/6] Auditoria de vieses em {settings.fairness_report_path}...")
    fairness = run_fairness_audit(
        datasets={"referência": ref_with_preds, "produção": prod_with_preds},
        group_columns=settings.fairness_group_columns,
        report_path=settings.fairness_report_path,
        target_column=target,
    )

    print("\n--- Data Drift (Evidently AI) ---")
    print(f"Dataset Drift Detectado: {'SIM' if summary['dataset_drift'] else 'NÃO'}")
    print(
        f"Colunas com Drift: {summary['number_of_drifted_columns']} de "
        f"{summary['number_of_columns']} ({summary['share_of_drifted_columns'] * 100:.1f}%)"
    )
    print(
        f"Taxa de Inadimplência Predita: Referência {ref_with_preds['TARGET_PRED'].mean():.2%} "
        f"-> Produção {prod_with_preds['TARGET_PRED'].mean():.2%}"
    )
    for col, stats in summary["column_stats"].items():
        status = "DRIFT" if stats["drift_detected"] else "OK"
        print(
            f"  - {col:22s} | {status:5s} | teste={stats['stattest_name']} "
            f"| score/p={stats['drift_score']}"
        )

    print("\n--- Concept Drift (performance com rótulos reais) ---")
    for metric in ("roc_auc", "recall", "precision", "f1"):
        print(f"  - {metric:9s}: {ref_metrics[metric]:.4f} -> {prod_metrics[metric]:.4f}")
    print(
        f"Inadimplência observada: {ref_with_preds[target].mean():.2%} -> "
        f"{prod_with_preds[target].mean():.2%} | Drift do alvo: "
        f"{'SIM' if target_summary['target_drift'] else 'NÃO'} "
        f"(p={target_summary['target_drift_score']})"
    )

    print("\n--- Fairness (disparate impact; alerta < 0.80) ---")
    for dataset, by_column in fairness.items():
        for column, result in by_column.items():
            flag = "ALERTA" if result["alert"] else "OK"
            print(
                f"  - {dataset:10s} | {column:20s} | DI={result['disparate_impact']:.3f} "
                f"| gap negação={result['false_denial_gap']:.3f} | {flag}"
            )


if __name__ == "__main__":
    main()
