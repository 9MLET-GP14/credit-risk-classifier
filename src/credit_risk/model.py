"""Treinamento, avaliação e persistência do modelo baseline de risco de crédito."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ModelType = Literal["logistic_regression", "hist_gradient_boosting"]


def _create_classifier(
    model_type: ModelType,
    random_state: int,
) -> LogisticRegression | HistGradientBoostingClassifier:
    """Instancia o classificador configurado para risco de crédito."""
    if model_type == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=200,
            random_state=random_state,
        )
    raise ValueError(
        f"Tipo de modelo inválido: '{model_type}'. "
        "Escolha 'logistic_regression' ou 'hist_gradient_boosting'."
    )


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_type: ModelType = "hist_gradient_boosting",
    random_state: int = 42,
) -> Pipeline:
    """Constrói o pipeline Scikit-Learn de pré-processamento e classificação.

    Args:
        numeric_features: Lista de nomes das colunas numéricas.
        categorical_features: Lista de nomes das colunas categóricas.
        model_type: Algoritmo classificador ('logistic_regression' ou 'hist_gradient_boosting').
        random_state: Semente para reprodutibilidade.

    Returns:
        Pipeline Scikit-Learn ainda não treinado.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    classifier = _create_classifier(model_type=model_type, random_state=random_state)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: ModelType = "hist_gradient_boosting",
    random_state: int = 42,
) -> Pipeline:
    """Infere tipos de features e treina o modelo baseline configurado.

    Args:
        X_train: DataFrame com as features de treino (sem colunas de ID/alvo).
        y_train: Série binária com o alvo de inadimplência.
        model_type: Algoritmo classificador ('logistic_regression' ou 'hist_gradient_boosting').
        random_state: Semente para reprodutibilidade.

    Returns:
        Pipeline treinado.
    """
    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_type=model_type,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calcula métricas de classificação relevantes para risco de crédito.

    Args:
        pipeline: Modelo treinado.
        X_test: DataFrame com features de teste.
        y_test: Série binária com os rótulos reais.

    Returns:
        Dicionário com roc_auc, recall, precision, f1 e accuracy.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }


def save_model_artifact(pipeline: Pipeline, path: Path) -> Path:
    """Serializa e salva o pipeline treinado em disco via joblib.

    Args:
        pipeline: Modelo treinado.
        path: Caminho de destino do arquivo .joblib.

    Returns:
        Caminho do arquivo salvo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model_artifact(path: Path) -> Pipeline:
    """Carrega um pipeline serializado do disco.

    Args:
        path: Caminho do arquivo .joblib.

    Returns:
        Pipeline Scikit-Learn carregado.
    """
    return joblib.load(path)
