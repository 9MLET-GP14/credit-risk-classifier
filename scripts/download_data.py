"""Script CLI para download, amostragem e geração do dataset de referência."""

import argparse

from credit_risk.config import Settings
from credit_risk.data import (
    download_raw_dataset,
    load_and_sample_data,
    preprocess_reference_data,
    split_and_save_reference,
)


def parse_args() -> argparse.Namespace:
    """Lê argumentos opcionais de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Baixa e prepara o dataset de referência do Home Credit Default Risk."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sobrescreve o tamanho da amostra configurado em Settings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força o re-download mesmo que o arquivo bruto já exista.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa o fluxo completo de aquisição e particionamento dos dados."""
    args = parse_args()
    settings = Settings()

    sample_size = args.sample_size if args.sample_size is not None else settings.sample_size

    print(f"[1/4] Verificando dataset bruto em: {settings.raw_data_path}")
    raw_path = download_raw_dataset(
        url=settings.raw_data_url,
        dest_path=settings.raw_data_path,
        force=args.force,
    )

    print(
        f"[2/4] Carregando {len(settings.selected_columns)} colunas "
        f"(amostra configurada: {sample_size or 'completo'})..."
    )
    sampled_df = load_and_sample_data(
        filepath=raw_path,
        sample_size=sample_size,
        columns=settings.selected_columns,
        target_column=settings.target_column,
        random_state=settings.random_state,
    )

    print("[3/4] Aplicando pré-processamento para dataset de referência...")
    clean_df = preprocess_reference_data(sampled_df)

    print("[4/4] Salvando partições Parquet em data/reference/...")
    train_df, test_df = split_and_save_reference(
        df=clean_df,
        train_path=settings.reference_train_path,
        test_path=settings.reference_test_path,
        target_column=settings.target_column,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    target_rate = clean_df[settings.target_column].mean() * 100
    print(
        f"Concluído! Treino: {len(train_df)} linhas | Teste: {len(test_df)} linhas | "
        f"Taxa de inadimplência (TARGET=1): {target_rate:.2f}%"
    )


if __name__ == "__main__":
    main()
