"""Contratos de dados estritos com Pandera para proteção de treino e inferência."""

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

ALLOWED_INCOME_TYPES = [
    "Businessman",
    "Commercial associate",
    "Maternity leave",
    "Pensioner",
    "State servant",
    "Student",
    "Unemployed",
    "Working",
]

ALLOWED_EDUCATION_TYPES = [
    "Academic degree",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Secondary / secondary special",
]


class CreditFeaturesContract(pa.DataFrameModel):
    """Contrato de qualidade para as features de entrada (modo inferência)."""

    SK_ID_CURR: Series[int] = pa.Field(ge=1, unique=True, nullable=False, coerce=True)
    AMT_INCOME_TOTAL: Series[float] = pa.Field(gt=0.0, nullable=False, coerce=True)
    AMT_CREDIT: Series[float] = pa.Field(gt=0.0, nullable=False, coerce=True)
    AMT_ANNUITY: Series[float] = pa.Field(gt=0.0, nullable=False, coerce=True)
    AGE_YEARS: Series[int] = pa.Field(ge=18, le=100, nullable=False, coerce=True)
    EMPLOYED_YEARS: Series[float] = pa.Field(ge=0.0, le=80.0, nullable=False, coerce=True)
    NAME_INCOME_TYPE: Series[str] = pa.Field(isin=ALLOWED_INCOME_TYPES, nullable=False)
    NAME_EDUCATION_TYPE: Series[str] = pa.Field(isin=ALLOWED_EDUCATION_TYPES, nullable=False)

    class Config:
        """Permite colunas extras caso o usuário configure novas features no config.py."""

        strict = False
        coerce = True


class CreditTrainingContract(CreditFeaturesContract):
    """Contrato de qualidade completo para treino (features + coluna alvo binária)."""

    TARGET: Series[int] = pa.Field(isin=[0, 1], nullable=False, coerce=True)


def validate_data(
    df: pd.DataFrame,
    is_training: bool = False,
    lazy: bool = True,
) -> pd.DataFrame:
    """Valida um DataFrame contra o contrato de dados de crédito.

    Args:
        df: DataFrame a ser validado.
        is_training: Se True, exige e valida também a coluna TARGET.
        lazy: Se True, coleta todas as violações antes de levantar SchemaErrors.

    Returns:
        DataFrame validado e com tipos coerentes com o contrato.

    Raises:
        pandera.errors.SchemaErrors: Caso existam violações quando lazy=True.
        pandera.errors.SchemaError: Caso exista violação quando lazy=False.
    """
    schema = CreditTrainingContract if is_training else CreditFeaturesContract
    return schema.validate(df, lazy=lazy)


def format_contract_errors(exc: pa.errors.SchemaErrors) -> str:
    """Formata os erros coletados pelo Pandera em um resumo legível para logs/CLI."""
    failure_cases = exc.failure_cases
    if failure_cases.empty:
        return str(exc)

    summary_lines = [f"Total de violações encontradas: {len(failure_cases)}"]
    grouped = failure_cases.groupby(["column", "check"], dropna=False).size()
    for (column, check), count in grouped.items():
        summary_lines.append(f"  - Coluna '{column}' | Regra '{check}': {count} ocorrência(s)")
    return "\n".join(summary_lines)
