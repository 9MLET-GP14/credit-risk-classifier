"""Pipeline ponta a ponta: Pandera -> Treino -> Data/Concept Drift -> Fairness -> MLflow."""

import argparse
import sys
import time
from datetime import datetime

import pandas as pd
import pandera.pandas as pa

from credit_risk.config import Settings
from credit_risk.contracts import build_corrupted_batch
from credit_risk.data import (
    download_raw_dataset,
    load_and_sample_data,
    preprocess_reference_data,
    split_and_save_reference,
)
from credit_risk.drift import (
    generate_drift_report,
    generate_performance_report,
    simulate_concept_drift,
    simulate_economic_shock,
)
from credit_risk.fairness import count_fairness_alerts, run_fairness_audit
from credit_risk.model import (
    evaluate_model,
    evaluate_predictions,
    predict_credit_risk,
    save_model_artifact,
    train_baseline_model,
)
from credit_risk.observability import (
    PipelineHealthMetrics,
    audit_contract_health,
    collect_infrastructure_metrics,
    get_logger,
    log_concept_drift_details,
    log_drift_details,
    log_fairness_details,
    log_pipeline_run_to_mlflow,
    prime_cpu_monitor,
    validate_with_observability,
)


def parse_args() -> argparse.Namespace:
    """Lê argumentos opcionais para execução e simulação de cenários no pipeline."""
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo de sustentação, drift e observabilidade MLflow."
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Reprocessa os Parquets de referência a partir do CSV bruto.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["clean", "drift", "corrupted"],
        default="drift",
        help=(
            "Cenário de produção: 'clean' (sem drift), 'drift' (choque econômico com Data "
            "Drift + Concept Drift, padrão) ou 'corrupted' (lote com anomalias para testar "
            "bloqueio do Pandera)."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nome customizado do run registrado no MLflow.",
    )
    return parser.parse_args()


def main() -> None:
    """Orquestra todas as etapas do pipeline e registra métricas e logs no MLflow."""
    args = parse_args()
    settings = Settings()
    logger = get_logger("credit_risk.pipeline", log_file=settings.pipeline_log_path, log_mode="w")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"credit-risk-{args.scenario}-{timestamp}"

    prime_cpu_monitor()
    pipeline_start = time.perf_counter()
    ingestion_latency = 0.0
    validation_latency = 0.0
    training_latency = 0.0
    drift_latency = 0.0

    # As métricas de volume e rejeição consideram somente o lote de produção.
    production_total_records = 0
    rejected_records = 0
    rejection_rate = 0.0
    eval_metrics: dict[str, float] = {"roc_auc": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0}
    drift_summary: dict[str, object] = {
        "dataset_drift": False,
        "number_of_drifted_columns": 0,
        "number_of_columns": 0,
        "share_of_drifted_columns": 0.0,
    }
    ref_default_rate = 0.0
    prod_default_rate = 0.0
    prod_metrics: dict[str, float] = {"roc_auc": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0}
    ref_observed_default_rate = 0.0
    prod_observed_default_rate = 0.0
    target_summary: dict[str, object] = {"target_drift": False}
    performance_alert = False
    fairness_results: dict[str, dict[str, dict]] = {}
    pipeline_status = 1

    params = {
        "scenario": args.scenario,
        "model_type": settings.model_type,
        "sample_size": settings.sample_size if settings.sample_size is not None else "full",
        "selected_columns_count": len(settings.selected_columns),
        "selected_columns": ",".join(settings.selected_columns),
        "test_size": settings.test_size,
        "random_state": settings.random_state,
    }

    logger.info(
        "Iniciando pipeline ponta a ponta (cenário='%s', log='%s').",
        args.scenario,
        settings.pipeline_log_path,
    )

    try:
        # 1. Ingestão e particionamento de referência
        t0 = time.perf_counter()
        if (
            args.refresh_data
            or not settings.reference_train_path.exists()
            or not settings.reference_test_path.exists()
        ):
            logger.info("[ingestão] Preparando dataset de referência a partir dos dados brutos...")
            download_raw_dataset(url=settings.raw_data_url, dest_path=settings.raw_data_path)
            raw_df = load_and_sample_data(
                filepath=settings.raw_data_path,
                columns=settings.selected_columns,
                sample_size=settings.sample_size,
                random_state=settings.random_state,
            )
            clean_df = preprocess_reference_data(raw_df)
            train_df, test_df = split_and_save_reference(
                df=clean_df,
                train_path=settings.reference_train_path,
                test_path=settings.reference_test_path,
                target_column=settings.target_column,
                test_size=settings.test_size,
                random_state=settings.random_state,
            )
        else:
            logger.info("[ingestão] Carregando partições de referência existentes em Parquet...")
            train_df = pd.read_parquet(settings.reference_train_path)
            test_df = pd.read_parquet(settings.reference_test_path)
        ingestion_latency = round(time.perf_counter() - t0, 4)
        logger.info(
            "[ingestão] Concluída em %.4fs (treino=%d, teste=%d).",
            ingestion_latency,
            len(train_df),
            len(test_df),
        )

        # 2. Validação de Contratos Pandera nas partições de referência
        t0 = time.perf_counter()
        train_df = validate_with_observability(
            train_df, is_training=True, logger=logger, stage_name="referência-treino"
        )
        test_df = validate_with_observability(
            test_df, is_training=True, logger=logger, stage_name="referência-teste"
        )
        validation_latency = round(time.perf_counter() - t0, 4)

        # 3. Treinamento e avaliação do modelo baseline
        t0 = time.perf_counter()
        drop_cols = [
            c for c in (settings.id_column, settings.target_column) if c in train_df.columns
        ]
        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df[settings.target_column]
        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df[settings.target_column]

        logger.info("[treino] Treinando modelo baseline (%s)...", settings.model_type)
        model = train_baseline_model(
            X_train=X_train,
            y_train=y_train,
            model_type=settings.model_type,
            random_state=settings.random_state,
        )
        eval_metrics = evaluate_model(pipeline=model, X_test=X_test, y_test=y_test)
        save_model_artifact(model, settings.baseline_model_path)
        training_latency = round(time.perf_counter() - t0, 4)
        logger.info(
            "[treino] Concluído em %.4fs | ROC-AUC=%.4f | Recall=%.4f | artefato='%s'.",
            training_latency,
            eval_metrics["roc_auc"],
            eval_metrics["recall"],
            settings.baseline_model_path,
        )

        # 4. Preparação do lote de produção conforme o cenário selecionado
        ref_with_preds = predict_credit_risk(
            df=test_df,
            model_path=settings.baseline_model_path,
            id_column=settings.id_column,
            target_column=settings.target_column,
        )
        ref_default_rate = float(ref_with_preds["TARGET_PRED"].mean())
        ref_with_preds[settings.target_column] = test_df[settings.target_column].to_numpy()

        # Os lotes 'clean' e 'drift' carregam o TARGET observado com atraso (ground truth),
        # usado apenas para medir Concept Drift; a inferência nunca recebe o alvo.
        if args.scenario == "clean":
            logger.info("[produção] Cenário 'clean': utilizando lote de produção sem choque.")
            prod_df = test_df.copy()
        elif args.scenario == "drift":
            logger.info(
                "[produção] Cenário 'drift': simulando choque macroeconômico (Data Drift) "
                "e mudança na relação features -> inadimplência (Concept Drift)."
            )
            prod_df = simulate_economic_shock(
                test_df, random_state=settings.random_state, keep_target=True
            )
            prod_df = simulate_concept_drift(
                prod_df, target_column=settings.target_column, random_state=settings.random_state
            )
        else:
            logger.warning("[produção] Cenário 'corrupted': injetando anomalias no lote.")
            prod_df = build_corrupted_batch(test_df)

        # Auditoria e barreira estrita do contrato Pandera no lote de produção
        t_val_prod = time.perf_counter()
        production_total_records = len(prod_df)
        prod_audit = audit_contract_health(
            prod_df, is_training=False, logger=logger, stage_name="produção-auditoria"
        )
        rejected_records = int(prod_audit["rejected_records"])
        rejection_rate = (
            rejected_records / production_total_records if production_total_records > 0 else 0.0
        )
        validate_with_observability(
            prod_df, is_training=False, logger=logger, stage_name="produção-barreira"
        )
        validation_latency = round(validation_latency + (time.perf_counter() - t_val_prod), 4)

        prod_with_preds = predict_credit_risk(
            df=prod_df,
            model_path=settings.baseline_model_path,
            id_column=settings.id_column,
            target_column=settings.target_column,
        )
        prod_default_rate = float(prod_with_preds["TARGET_PRED"].mean())
        if settings.target_column in prod_df.columns:
            prod_with_preds[settings.target_column] = prod_df[settings.target_column].to_numpy()
        settings.production_shock_path.parent.mkdir(parents=True, exist_ok=True)
        prod_with_preds.to_parquet(settings.production_shock_path, index=False)

        # 5. Análise estatística de Drift com Evidently AI integrada ao log
        t0 = time.perf_counter()
        logger.info(
            "[drift] Executando Evidently AI e gerando relatório em '%s'...",
            settings.drift_report_path,
        )
        drift_summary = generate_drift_report(
            reference_df=ref_with_preds,
            production_df=prod_with_preds,
            report_path=settings.drift_report_path,
            id_column=settings.id_column,
            target_column=settings.target_column,
        )
        log_drift_details(drift_summary, logger=logger)

        # 6. Concept Drift: performance real em produção (rótulos com atraso)
        target = settings.target_column
        if target in prod_with_preds.columns:
            ref_metrics = evaluate_predictions(
                ref_with_preds[target], ref_with_preds["TARGET_PRED"], ref_with_preds["RISK_SCORE"]
            )
            prod_metrics = evaluate_predictions(
                prod_with_preds[target],
                prod_with_preds["TARGET_PRED"],
                prod_with_preds["RISK_SCORE"],
            )
            ref_observed_default_rate = float(ref_with_preds[target].mean())
            prod_observed_default_rate = float(prod_with_preds[target].mean())
            target_summary = generate_performance_report(
                reference_df=ref_with_preds,
                production_df=prod_with_preds,
                report_path=settings.performance_report_path,
                target_column=target,
            )
            performance_alert = log_concept_drift_details(
                ref_metrics,
                prod_metrics,
                target_summary,
                logger=logger,
                min_roc_auc=settings.min_production_roc_auc,
                max_auc_drop=settings.max_roc_auc_drop,
            )

        # 7. Auditoria de vieses (fairness) por faixa etária, escolaridade e tipo de renda
        fairness_results = run_fairness_audit(
            datasets={"referência": ref_with_preds, "produção": prod_with_preds},
            group_columns=settings.fairness_group_columns,
            report_path=settings.fairness_report_path,
            target_column=target,
        )
        log_fairness_details(fairness_results, logger=logger)
        drift_latency = round(time.perf_counter() - t0, 4)
        logger.info("[drift] Análise de drift, concept drift e fairness em %.4fs.", drift_latency)

    except pa.errors.SchemaErrors:
        pipeline_status = 0
        logger.error("Pipeline interrompido pelo contrato Pandera (cenário='%s').", args.scenario)
    except Exception as exc:
        pipeline_status = 0
        logger.exception("Falha inesperada durante execução do pipeline: %s", exc)
    finally:
        total_latency = round(time.perf_counter() - pipeline_start, 4)
        status_label = "SUCCESS" if pipeline_status == 1 else "FAILED"
        infra = collect_infrastructure_metrics()
        fairness_alerts = count_fairness_alerts(fairness_results)

        def _di(dataset: str, column: str) -> float:
            return float(
                fairness_results.get(dataset, {}).get(column, {}).get("disparate_impact", 1.0)
            )

        target_drift = bool(target_summary.get("target_drift", False))
        total_alerts = (
            int(drift_summary["number_of_drifted_columns"])
            + int(target_drift)
            + int(performance_alert)
            + fairness_alerts
            + int(rejected_records > 0)
        )
        logger.info(
            "[infra] Memória do processo=%.1f MB | CPU do processo=%.2fs | CPU host=%.1f%% | "
            "Memória host=%.1f%% | Alertas totais=%d",
            infra["process_memory_mb"],
            infra["process_cpu_seconds"],
            infra["system_cpu_percent"],
            infra["system_memory_percent"],
            total_alerts,
        )
        logger.info(
            "[resumo] Status=%s | Latência total=%.4fs | Registros=%d | Rejeitados=%d (%.2f%%)",
            status_label,
            total_latency,
            production_total_records,
            rejected_records,
            rejection_rate * 100,
        )

        health = PipelineHealthMetrics(
            total_records_received=production_total_records,
            valid_records=max(0, production_total_records - rejected_records),
            rejected_records=rejected_records,
            rejection_rate=round(rejection_rate, 6),
            roc_auc=float(eval_metrics["roc_auc"]),
            recall=float(eval_metrics["recall"]),
            precision=float(eval_metrics["precision"]),
            f1=float(eval_metrics["f1"]),
            dataset_drift=bool(drift_summary["dataset_drift"]),
            drifted_columns_count=int(drift_summary["number_of_drifted_columns"]),
            total_monitored_columns=int(drift_summary["number_of_columns"]),
            drift_share=float(drift_summary["share_of_drifted_columns"]),
            reference_default_rate=round(ref_default_rate, 6),
            production_default_rate=round(prod_default_rate, 6),
            pipeline_status=pipeline_status,
            total_latency_seconds=total_latency,
            ingestion_latency_seconds=ingestion_latency,
            validation_latency_seconds=validation_latency,
            training_latency_seconds=training_latency,
            drift_latency_seconds=drift_latency,
            production_roc_auc=float(prod_metrics["roc_auc"]),
            production_recall=float(prod_metrics["recall"]),
            production_precision=float(prod_metrics["precision"]),
            production_f1=float(prod_metrics["f1"]),
            roc_auc_drop=round(float(eval_metrics["roc_auc"]) - float(prod_metrics["roc_auc"]), 6)
            if prod_metrics["roc_auc"]
            else 0.0,
            reference_observed_default_rate=round(ref_observed_default_rate, 6),
            production_observed_default_rate=round(prod_observed_default_rate, 6),
            target_drift=target_drift,
            performance_alert=performance_alert,
            fairness_di_age_reference=_di("referência", "AGE_BAND"),
            fairness_di_age_production=_di("produção", "AGE_BAND"),
            fairness_di_education_production=_di("produção", "NAME_EDUCATION_TYPE"),
            fairness_di_income_type_production=_di("produção", "NAME_INCOME_TYPE"),
            fairness_alerts_count=fairness_alerts,
            total_alerts_count=total_alerts,
            **infra,
        )

        run_id = log_pipeline_run_to_mlflow(
            tracking_uri=settings.resolved_mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            run_name=run_name,
            params=params,
            health_metrics=health,
            model_path=settings.baseline_model_path if pipeline_status == 1 else None,
            drift_report_path=settings.drift_report_path if pipeline_status == 1 else None,
            log_file_path=settings.pipeline_log_path,
            extra_report_paths=(
                [settings.performance_report_path, settings.fairness_report_path]
                if pipeline_status == 1
                else None
            ),
            tags={"status": status_label, "scenario": args.scenario},
            logger=logger,
        )
        logger.info("Pipeline finalizado (%s). MLflow Run ID: %s", status_label, run_id)

    if pipeline_status == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
