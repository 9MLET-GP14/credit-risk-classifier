"""Testes unitários para configuração, download, amostragem e preparação de dados."""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from credit_risk.config import Settings
from credit_risk.data import (
    download_raw_dataset,
    load_and_sample_data,
    preprocess_reference_data,
    split_and_save_reference,
)


def _create_sample_csv(path: Path, n_rows: int = 100) -> Path:
    df = pd.DataFrame(
        {
            "SK_ID_CURR": range(1, n_rows + 1),
            "TARGET": [0] * int(n_rows * 0.8) + [1] * (n_rows - int(n_rows * 0.8)),
            "AMT_INCOME_TOTAL": [50000.0 + i * 100 for i in range(n_rows)],
            "AMT_CREDIT": [200000.0 + i * 500 for i in range(n_rows)],
            "AMT_ANNUITY": [12000.0 + i * 50 for i in range(n_rows)],
            "DAYS_BIRTH": [-10000 - i * 30 for i in range(n_rows)],
            "DAYS_EMPLOYED": [365243 if i % 10 == 0 else -1500 - i for i in range(n_rows)],
            "NAME_EDUCATION_TYPE": ["Higher education"] * n_rows,
            "NAME_INCOME_TYPE": ["Working"] * n_rows,
            "EXTRA_UNUSED_COL": ["ignore"] * n_rows,
        }
    )
    df.to_csv(path, index=False)
    return path


def test_settings_defaults_and_env_override(monkeypatch) -> None:
    """Valida defaults de configuração e capacidade de sobrescrita via env vars."""
    default_settings = Settings()
    assert default_settings.sample_size == 20_000
    assert "TARGET" in default_settings.selected_columns

    monkeypatch.setenv("SAMPLE_SIZE", "5000")
    monkeypatch.setenv("TEST_SIZE", "0.3")
    custom_settings = Settings()
    assert custom_settings.sample_size == 5000
    assert custom_settings.test_size == 0.3


def test_load_and_sample_data_filters_columns_and_samples(tmp_path: Path) -> None:
    """Garante projeção estrita de colunas e amostragem estratificada exata."""
    csv_path = _create_sample_csv(tmp_path / "raw.csv", n_rows=100)
    selected_cols = ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL"]

    sampled = load_and_sample_data(
        filepath=csv_path,
        sample_size=50,
        columns=selected_cols,
        target_column="TARGET",
        random_state=42,
    )

    assert len(sampled) == 50
    assert list(sampled.columns) == selected_cols
    assert sampled["TARGET"].mean() == 0.2


def test_preprocess_reference_data_creates_age_and_employed_years(tmp_path: Path) -> None:
    """Valida cálculo de idade, tratamento da sentinela 365243 e remoção de nulos."""
    csv_path = _create_sample_csv(tmp_path / "raw.csv", n_rows=20)
    df = pd.read_csv(csv_path)
    df.loc[1, "AMT_ANNUITY"] = None  # força 1 linha nula que deve ser limpa

    clean_df = preprocess_reference_data(df)

    assert "DAYS_BIRTH" not in clean_df.columns
    assert "DAYS_EMPLOYED" not in clean_df.columns
    assert "AGE_YEARS" in clean_df.columns
    assert "EMPLOYED_YEARS" in clean_df.columns
    assert (clean_df["AGE_YEARS"] >= 18).all()
    assert (clean_df["EMPLOYED_YEARS"] >= 0).all()
    assert len(clean_df) == 19


def test_split_and_save_reference_persists_parquet(tmp_path: Path) -> None:
    """Valida salvamento das partições de treino e teste em Parquet."""
    csv_path = _create_sample_csv(tmp_path / "raw.csv", n_rows=50)
    clean_df = preprocess_reference_data(pd.read_csv(csv_path))

    train_path = tmp_path / "ref" / "train.parquet"
    test_path = tmp_path / "ref" / "test.parquet"

    train_df, test_df = split_and_save_reference(
        df=clean_df,
        train_path=train_path,
        test_path=test_path,
        test_size=0.2,
        random_state=42,
    )

    assert train_path.exists()
    assert test_path.exists()
    assert len(train_df) == 40
    assert len(test_df) == 10


def test_download_raw_dataset_skips_if_exists(tmp_path: Path) -> None:
    """Não deve baixar novamente caso o arquivo bruto já exista e force=False."""
    dest = tmp_path / "existing.csv"
    dest.write_text("a,b\n1,2\n")

    with patch("credit_risk.data.urllib.request.urlopen") as mock_urlopen:
        result = download_raw_dataset("https://example.com/data.csv", dest, force=False)
        assert result == dest
        mock_urlopen.assert_not_called()


def test_download_raw_dataset_streams_when_missing(tmp_path: Path) -> None:
    """Deve realizar streaming e salvar arquivo quando não existir localmente."""
    dest = tmp_path / "nested" / "data.csv"
    mock_response = MagicMock()
    mock_response.__enter__.return_value = BytesIO(b"col1,col2\n10,20\n")

    with patch("credit_risk.data.urllib.request.urlopen", return_value=mock_response):
        result = download_raw_dataset("https://example.com/data.csv", dest, force=False)

    assert result == dest
    assert dest.read_bytes() == b"col1,col2\n10,20\n"
