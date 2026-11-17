"""Configurações centralizadas e flexíveis do projeto via Pydantic Settings."""

from pathlib import Path
from typing import Literal

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
    mlruns_dir: Path = Path("mlruns")
    logs_dir: Path = Path("logs")

    # Nomes de arquivos
    raw_filename: str = "application_train.csv"
    reference_train_filename: str = "reference_train.parquet"
    reference_test_filename: str = "reference_test.parquet"
    baseline_model_filename: str = "baseline_model.joblib"
    predictions_filename: str = "predictions.parquet"
    production_shock_filename: str = "production_shock.parquet"
    drift_report_filename: str = "drift_report.html"
    performance_report_filename: str = "model_performance_report.html"
    fairness_report_filename: str = "fairness_report.html"
    pipeline_log_filename: str = "pipeline.log"

    # Parâmetros de amostragem, seleção de colunas e modelo
    sample_size: int | None = 20_000
    selected_columns: list[str] = DEFAULT_COLUMNS
    id_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"
    test_size: float = 0.2
    random_state: int = 42
    model_type: Literal["logistic_regression", "hist_gradient_boosting"] = "hist_gradient_boosting"

    # Observabilidade e MLflow Tracking
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "credit-risk-observability"

    # Thresholds de alerta de Concept Drift e auditoria de vieses
    min_production_roc_auc: float = 0.60
    max_roc_auc_drop: float = 0.05
    fairness_group_columns: list[str] = ["AGE_BAND", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"]

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

    @property
    def baseline_model_path(self) -> Path:
        """Caminho completo do artefato do modelo baseline."""
        return self.models_dir / self.baseline_model_filename

    @property
    def predictions_path(self) -> Path:
        """Caminho completo do arquivo de predições em lote."""
        return self.production_dir / self.predictions_filename

    @property
    def production_shock_path(self) -> Path:
        """Caminho completo do dataset de produção com choque econômico simulado."""
        return self.production_dir / self.production_shock_filename

    @property
    def drift_report_path(self) -> Path:
        """Caminho completo do relatório HTML de drift gerado pelo Evidently AI."""
        return self.reports_dir / self.drift_report_filename

    @property
    def performance_report_path(self) -> Path:
        """Relatório Evidently de Concept Drift (qualidade de classificação e drift do alvo)."""
        return self.reports_dir / self.performance_report_filename

    @property
    def fairness_report_path(self) -> Path:
        """Relatório HTML da auditoria de vieses por grupo."""
        return self.reports_dir / self.fairness_report_filename

    @property
    def resolved_mlflow_tracking_uri(self) -> str:
        """URI de tracking do MLflow (padrão local em file:///.../mlruns)."""
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return self.mlruns_dir.resolve().as_uri()

    @property
    def pipeline_log_path(self) -> Path:
        """Caminho completo do arquivo de log centralizado do pipeline."""
        return self.logs_dir / self.pipeline_log_filename


settings = Settings()
