"""Script CLI para treinamento e avaliação do modelo baseline de risco de crédito."""

import argparse

import pandas as pd

from credit_risk.config import Settings
from credit_risk.contracts import validate_data
from credit_risk.model import evaluate_model, save_model_artifact, train_baseline_model


def parse_args() -> argparse.Namespace:
    """Lê argumentos opcionais de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treina e avalia o modelo baseline de risco de crédito."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic_regression", "hist_gradient_boosting"],
        default=None,
        help="Sobrescreve o modelo configurado em Settings.",
    )
    return parser.parse_args()


def main() -> None:
    """Carrega o dataset de referência, valida o contrato, treina e salva o artefato."""
    args = parse_args()
    settings = Settings()
    model_type = args.model_type or settings.model_type

    print(f"[1/5] Carregando partições de referência de {settings.reference_dir}...")
    train_df = pd.read_parquet(settings.reference_train_path)
    test_df = pd.read_parquet(settings.reference_test_path)

    print("[2/5] Validando contratos de dados com Pandera...")
    train_df = validate_data(train_df, is_training=True)
    test_df = validate_data(test_df, is_training=True)

    drop_cols = [c for c in (settings.id_column, settings.target_column) if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[settings.target_column]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[settings.target_column]

    print(f"[3/5] Treinando '{model_type}' (features: {list(X_train.columns)})...")
    pipeline = train_baseline_model(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        random_state=settings.random_state,
    )

    print("[4/5] Avaliando desempenho no conjunto de teste de referência...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    for name, value in metrics.items():
        print(f"  - {name}: {value:.4f}")

    print(f"[5/5] Salvando artefato em {settings.baseline_model_path}...")
    save_model_artifact(pipeline, settings.baseline_model_path)
    print("Treinamento concluído com sucesso!")


if __name__ == "__main__":
    main()
