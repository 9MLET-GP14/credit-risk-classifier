"""Configurações centralizadas e flexíveis do projeto via Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_COLUMNS: list[str] = [
    "SK_ID_CURR",
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
]


class Settings(BaseSettings):
    """Parâmetros configuráveis de dados, amostragem e caminhos do projeto."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Fonte dos dados (Home Credit Default Risk - application_train)
    raw_data_url: str = "https://huggingface.co/datasets/jamirc/home_credit_default_risk/resolve/main/application_tr.csv"

    # Diretórios principais
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")

    # Nomes de arquivos
    raw_filename: str = "application_train.csv"
    reference_train_filename: str = "reference_train.parquet"
    reference_test_filename: str = "reference_test.parquet"

    # Parâmetros de amostragem e seleção de colunas
    sample_size: int | None = 20_000
    selected_columns: list[str] = DEFAULT_COLUMNS
    target_column: str = "TARGET"
    test_size: float = 0.2
    random_state: int = 42

    @property
    def raw_dir(self) -> Path:
        """Diretório para dados brutos."""
        return self.data_dir / "raw"

    @property
    def reference_dir(self) -> Path:
        """Diretório para o dataset de referência limpo."""
        return self.data_dir / "reference"

    @property
    def production_dir(self) -> Path:
        """Diretório para o dataset simulado de produção."""
        return self.data_dir / "production"

    @property
    def raw_data_path(self) -> Path:
        """Caminho completo do arquivo CSV bruto."""
        return self.raw_dir / self.raw_filename

    @property
    def reference_train_path(self) -> Path:
        """Caminho completo do dataset de treino de referência."""
        return self.reference_dir / self.reference_train_filename

    @property
    def reference_test_path(self) -> Path:
        """Caminho completo do dataset de teste de referência."""
        return self.reference_dir / self.reference_test_filename


settings = Settings()
