"""Script CLI para inferência em lote de risco de crédito."""

import argparse
from pathlib import Path

import pandas as pd

from credit_risk.config import Settings
from credit_risk.model import predict_credit_risk


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando para a inferência em lote."""
    parser = argparse.ArgumentParser(
        description="Executa inferência em lote usando o modelo treinado de risco de crédito."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Caminho do arquivo de entrada (.parquet ou .csv). Padrão: reference_test.parquet.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Caminho do arquivo de saída (.parquet ou .csv). Padrão: predictions.parquet.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Caminho do modelo .joblib. Padrão: models/baseline_model.joblib.",
    )
    return parser.parse_args()


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Formato de entrada não suportado: {path.suffix}. Use .parquet ou .csv.")


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Formato de saída não suportado: {path.suffix}. Use .parquet ou .csv.")


def main() -> None:
    """Carrega dados de entrada, gera predições e salva os scores em disco."""
    args = parse_args()
    settings = Settings()

    input_path = args.input_path or settings.reference_test_path
    output_path = args.output_path or settings.predictions_path
    model_path = args.model_path or settings.baseline_model_path

    print(f"[1/3] Carregando dados para inferência de: {input_path}")
    input_df = _load_dataframe(input_path)

    print(f"[2/3] Executando predições com o modelo: {model_path}")
    predictions_df = predict_credit_risk(
        df=input_df,
        model_path=model_path,
        id_column=settings.id_column,
        target_column=settings.target_column,
    )

    print(f"[3/3] Salvando {len(predictions_df)} predições em: {output_path}")
    _save_dataframe(predictions_df, output_path)

    default_rate = predictions_df["TARGET_PRED"].mean() * 100
    avg_score = predictions_df["RISK_SCORE"].mean()
    print(
        f"Concluído! Inadimplência predita: {default_rate:.2f}% | Risk score médio: {avg_score:.4f}"
    )


if __name__ == "__main__":
    main()
