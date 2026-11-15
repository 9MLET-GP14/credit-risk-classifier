"""Script para validação de contratos de dados Pandera (padrão limpo e simulação corrompida)."""

import argparse
import sys

import pandas as pd
import pandera.pandas as pa

from credit_risk.config import Settings
from credit_risk.contracts import format_contract_errors, validate_data


def parse_args() -> argparse.Namespace:
    """Lê argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Valida datasets de crédito contra os contratos de dados Pandera."
    )
    parser.add_argument(
        "--simulate-corruption",
        action="store_true",
        help="Injeta anomalias intencionais para demonstrar o contrato bloqueando a ingestão.",
    )
    return parser.parse_args()


def build_corrupted_batch(clean_sample: pd.DataFrame) -> pd.DataFrame:
    """Injeta anomalias intencionais para demonstrar o bloqueio de ingestão pelo contrato."""
    corrupted = clean_sample.drop(columns=["TARGET"], errors="ignore").copy().reset_index(drop=True)

    # Regra 1 violada: menor de idade (< 18 anos)
    corrupted.loc[0, "AGE_YEARS"] = 15
    # Regra 2 violada: renda nula e negativa
    corrupted.loc[1, "AMT_INCOME_TOTAL"] = None
    corrupted.loc[2, "AMT_INCOME_TOTAL"] = -1500.0
    # Regra 3 violada: crédito negativo
    corrupted.loc[3, "AMT_CREDIT"] = -20000.0
    # Regra 4 violada: ID duplicado
    corrupted.loc[4, "SK_ID_CURR"] = corrupted.loc[0, "SK_ID_CURR"]
    # Regra 5 violada: categoria desconhecida
    corrupted.loc[5, "NAME_INCOME_TYPE"] = "Unknown_Category"

    return corrupted


def main() -> None:
    """Executa a validação do dataset limpo por padrão ou a simulação corrompida se solicitado."""
    args = parse_args()
    settings = Settings()

    if not args.simulate_corruption:
        print(f"Validando partições de referência em: {settings.reference_dir}...")
        train_df = pd.read_parquet(settings.reference_train_path)
        test_df = pd.read_parquet(settings.reference_test_path)

        validated_train = validate_data(train_df, is_training=True)
        validated_test = validate_data(test_df, is_training=True)

        print(
            f"  -> APROVADO! Treino ({len(validated_train)} registros) e Teste "
            f"({len(validated_test)} registros) 100% em conformidade com o contrato Pandera."
        )
        return

    print("Simulando ingestão de lote corrompido (anomalias intencionais)...")
    test_sample = pd.read_parquet(settings.reference_test_path).head(10)
    corrupted_df = build_corrupted_batch(test_sample)

    try:
        validate_data(corrupted_df, is_training=False)
        print("ERRO INESPERADO: O lote corrompido não foi bloqueado!")
        sys.exit(1)
    except pa.errors.SchemaErrors as exc:
        print("  -> BLOQUEADO COM SUCESSO! O contrato Pandera interceptou a ingestão:")
        print(format_contract_errors(exc))


if __name__ == "__main__":
    main()
