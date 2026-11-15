"""Script CLI para simulação de drift em produção e geração de relatório Evidently AI."""

import pandas as pd

from credit_risk.config import Settings
from credit_risk.drift import generate_drift_report, simulate_economic_shock
from credit_risk.model import predict_credit_risk


def main() -> None:
    """Executa simulação de choque econômico, inferência e relatório estatístico de drift."""
    settings = Settings()

    print(f"[1/4] Carregando dataset de referência ({settings.reference_test_path})...")
    ref_df = pd.read_parquet(settings.reference_test_path)
    ref_with_preds = predict_credit_risk(
        df=ref_df,
        model_path=settings.baseline_model_path,
        id_column=settings.id_column,
        target_column=settings.target_column,
    )

    print("[2/4] Simulando Dataset de Produção com choque econômico...")
    prod_df = simulate_economic_shock(ref_df, random_state=settings.random_state)

    print("[3/4] Executando inferência (com contrato Pandera) sobre o lote de produção...")
    prod_with_preds = predict_credit_risk(
        df=prod_df,
        model_path=settings.baseline_model_path,
        id_column=settings.id_column,
        target_column=settings.target_column,
    )
    settings.production_shock_path.parent.mkdir(parents=True, exist_ok=True)
    prod_with_preds.to_parquet(settings.production_shock_path, index=False)

    print(f"[4/4] Gerando relatório estatístico Evidently AI em {settings.drift_report_path}...")
    summary = generate_drift_report(
        reference_df=ref_with_preds,
        production_df=prod_with_preds,
        report_path=settings.drift_report_path,
        id_column=settings.id_column,
        target_column=settings.target_column,
    )

    ref_default_rate = ref_with_preds["TARGET_PRED"].mean() * 100
    prod_default_rate = prod_with_preds["TARGET_PRED"].mean() * 100

    print("\n--- Resumo da Análise de Drift (Evidently AI) ---")
    print(f"Dataset Drift Detectado: {'SIM' if summary['dataset_drift'] else 'NÃO'}")
    print(
        f"Colunas com Drift: {summary['number_of_drifted_columns']} de "
        f"{summary['number_of_columns']} ({summary['share_of_drifted_columns'] * 100:.1f}%)"
    )
    print(
        f"Taxa de Inadimplência Predita: Referência {ref_default_rate:.2f}% "
        f"-> Produção {prod_default_rate:.2f}%"
    )
    print("Detalhamento por coluna:")
    for col, stats in summary["column_stats"].items():
        status = "DRIFT" if stats["drift_detected"] else "OK"
        print(
            f"  - {col:22s} | {status:5s} | teste={stats['stattest_name']} "
            f"| score/p={stats['drift_score']}"
        )
    print(f"\nRelatório HTML salvo em: {summary['report_path']}")


if __name__ == "__main__":
    main()
