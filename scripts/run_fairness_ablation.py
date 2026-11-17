"""Experimento de mitigação de viés: retreina o modelo removendo atributos ligados à idade.

Compara ROC-AUC e disparate impact por grupo para avaliar se "fairness through unawareness"
(simplesmente remover a variável sensível) reduz o impacto adverso, ou se proxies persistem.
"""

import pandas as pd

from credit_risk.config import Settings
from credit_risk.fairness import run_fairness_audit
from credit_risk.model import evaluate_predictions, train_baseline_model

ABLATIONS: dict[str, list[str]] = {
    "baseline (todas as features)": [],
    "sem AGE_YEARS": ["AGE_YEARS"],
    "sem AGE_YEARS e EMPLOYED_YEARS": ["AGE_YEARS", "EMPLOYED_YEARS"],
}


def main() -> None:
    """Treina cada variante e imprime uma tabela comparativa."""
    settings = Settings()
    train_df = pd.read_parquet(settings.reference_train_path)
    test_df = pd.read_parquet(settings.reference_test_path)
    base_drop = [settings.id_column, settings.target_column]

    rows = []
    for name, removed in ABLATIONS.items():
        model = train_baseline_model(
            train_df.drop(columns=base_drop + removed),
            train_df[settings.target_column],
            model_type=settings.model_type,
            random_state=settings.random_state,
        )
        features = test_df.drop(columns=base_drop + removed)
        scored = test_df.copy()
        scored["TARGET_PRED"] = model.predict(features)
        scores = model.predict_proba(features)[:, 1]
        metrics = evaluate_predictions(
            scored[settings.target_column], scored["TARGET_PRED"], scores
        )
        audit = run_fairness_audit({"teste": scored}, settings.fairness_group_columns)["teste"]
        rows.append(
            {
                "Variante": name,
                "ROC-AUC": round(metrics["roc_auc"], 4),
                **{f"DI {col}": audit[col]["disparate_impact"] for col in audit},
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
