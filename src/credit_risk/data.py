"""Módulo de download, amostragem e particionamento do dataset de crédito."""

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def download_raw_dataset(url: str, dest_path: Path, force: bool = False) -> Path:
    """Baixa o dataset bruto em chunks para evitar alto consumo de memória.

    Args:
        url: URL pública do arquivo CSV.
        dest_path: Caminho de destino local.
        force: Se True, sobrescreve o arquivo mesmo que já exista.

    Returns:
        Caminho do arquivo baixado.
    """
    if dest_path.exists() and not force:
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")

    with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as out_file:
        while chunk := response.read(1024 * 1024):
            out_file.write(chunk)

    tmp_path.replace(dest_path)
    return dest_path


def load_and_sample_data(
    filepath: Path,
    sample_size: int | None = 20_000,
    columns: list[str] | None = None,
    target_column: str = "TARGET",
    random_state: int = 42,
) -> pd.DataFrame:
    """Carrega apenas as colunas configuradas e extrai uma amostra estratificada.

    Args:
        filepath: Caminho do CSV bruto.
        sample_size: Quantidade desejada de linhas (None para carregar tudo).
        columns: Lista de colunas a carregar (None para carregar todas).
        target_column: Nome da coluna alvo para amostragem estratificada.
        random_state: Semente para reprodutibilidade.

    Returns:
        DataFrame carregado e amostrado.
    """
    df = pd.read_csv(filepath, usecols=columns)

    if sample_size is not None and sample_size < len(df):
        if target_column in df.columns:
            df, _ = train_test_split(
                df,
                train_size=sample_size,
                stratify=df[target_column],
                random_state=random_state,
            )
        else:
            df = df.sample(n=sample_size, random_state=random_state)

    return df.reset_index(drop=True)


def preprocess_reference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza atributos derivados e remove nulos do dataset de referência.

    Args:
        df: DataFrame amostrado com colunas originais do Home Credit.

    Returns:
        DataFrame limpo e pronto para contratos de dados e modelagem.
    """
    clean_df = df.copy()

    if "DAYS_BIRTH" in clean_df.columns:
        clean_df["AGE_YEARS"] = (-clean_df["DAYS_BIRTH"] // 365).astype(int)
        clean_df = clean_df.drop(columns=["DAYS_BIRTH"])

    if "DAYS_EMPLOYED" in clean_df.columns:
        # No Home Credit, o valor sentinela 365243 indica ausência de emprego formal.
        employed_days = clean_df["DAYS_EMPLOYED"].replace(365243, 0).clip(upper=0)
        clean_df["EMPLOYED_YEARS"] = (-employed_days / 365.25).round(1)
        clean_df = clean_df.drop(columns=["DAYS_EMPLOYED"])

    clean_df = clean_df.dropna().drop_duplicates(
        subset=["SK_ID_CURR"] if "SK_ID_CURR" in clean_df.columns else None
    )
    return clean_df.reset_index(drop=True)


def split_and_save_reference(
    df: pd.DataFrame,
    train_path: Path,
    test_path: Path,
    target_column: str = "TARGET",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o dataset de referência em treino/teste e salva em formato Parquet.

    Args:
        df: DataFrame limpo de referência.
        train_path: Caminho Parquet de saída para treino.
        test_path: Caminho Parquet de saída para teste.
        target_column: Coluna alvo para estratificação.
        test_size: Fração reservada para teste.
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla (train_df, test_df).
    """
    stratify_col = df[target_column] if target_column in df.columns else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    return train_df, test_df
