"""Testes unitários para contratos de dados com Pandera."""

import pandas as pd
import pandera.pandas as pa
import pytest

from credit_risk.contracts import format_contract_errors, validate_data


def _create_valid_batch(n_rows: int = 5, with_target: bool = True) -> pd.DataFrame:
    data = {
        "SK_ID_CURR": list(range(100, 100 + n_rows)),
        "AMT_INCOME_TOTAL": [80000.0 + i * 1000 for i in range(n_rows)],
        "AMT_CREDIT": [250000.0 + i * 5000 for i in range(n_rows)],
        "AMT_ANNUITY": [15000.0 + i * 500 for i in range(n_rows)],
        "AGE_YEARS": [25 + i for i in range(n_rows)],
        "EMPLOYED_YEARS": [2.5 + i for i in range(n_rows)],
        "NAME_INCOME_TYPE": ["Working"] * n_rows,
        "NAME_EDUCATION_TYPE": ["Higher education"] * n_rows,
    }
    if with_target:
        data["TARGET"] = [0 if i % 2 == 0 else 1 for i in range(n_rows)]
    return pd.DataFrame(data)


def test_validate_data_accepts_valid_training_and_inference_batches() -> None:
    """Dados conformes devem passar tanto no modo treino quanto no modo inferência."""
    train_df = _create_valid_batch(with_target=True)
    inf_df = _create_valid_batch(with_target=False)

    validated_train = validate_data(train_df, is_training=True)
    validated_inf = validate_data(inf_df, is_training=False)

    assert len(validated_train) == 5
    assert len(validated_inf) == 5


def test_validate_data_blocks_underage_client() -> None:
    """Deve bloquear cliente com idade menor que 18 anos."""
    df = _create_valid_batch(with_target=False)
    df.loc[0, "AGE_YEARS"] = 17

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        validate_data(df, is_training=False)

    summary = format_contract_errors(exc_info.value)
    assert "AGE_YEARS" in summary


def test_validate_data_blocks_null_or_non_positive_income() -> None:
    """Deve bloquear renda nula ou menor/igual a zero."""
    df = _create_valid_batch(with_target=False)
    df.loc[0, "AMT_INCOME_TOTAL"] = None
    df.loc[1, "AMT_INCOME_TOTAL"] = 0.0

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        validate_data(df, is_training=False)

    summary = format_contract_errors(exc_info.value)
    assert "AMT_INCOME_TOTAL" in summary


def test_validate_data_blocks_duplicate_ids_and_invalid_category() -> None:
    """Deve bloquear IDs duplicados e categorias fora do domínio permitido."""
    df = _create_valid_batch(with_target=False)
    df.loc[1, "SK_ID_CURR"] = df.loc[0, "SK_ID_CURR"]
    df.loc[2, "NAME_INCOME_TYPE"] = "Alien_Category"

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        validate_data(df, is_training=False)

    summary = format_contract_errors(exc_info.value)
    assert "SK_ID_CURR" in summary
    assert "NAME_INCOME_TYPE" in summary


def test_validate_data_requires_binary_target_in_training_mode() -> None:
    """No modo treino, TARGET deve existir e conter apenas 0 ou 1."""
    df = _create_valid_batch(with_target=True)
    df.loc[0, "TARGET"] = 2

    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(df, is_training=True)
