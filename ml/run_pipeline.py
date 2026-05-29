"""
run_pipeline.py — ShiftMetrics ML Pipeline
Orquestador end-to-end: corre todas las rondas en orden.

Rondas:
  1. Baselines (MajorityClass + ThresholdBSR)
  2. Logistic Regression (grid 18 configs)
  3. XGBoost (Optuna 50 trials)
  4. LightGBM challenger (Optuna 50 trials)
  5. Champion/Challenger selection + McNemar test
  6. Calibración del champion (Platt + isotonic, holdout cal 2015)
  7. LOPO-CV del champion GBM (hiperparametros congelados)
  8. SHAP explicabilidad (global + local + PDPs)
  9. Drift analysis (temporal + simulado)
 10. Threshold selection (F2-opt + ROI) + Bootstrap CI
"""

import time
import json
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.tracking
import joblib

from config import (
    FEATURE_COLS, TARGET_COL,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
    OPTUNA_N_TRIALS, MODEL_REGISTRY_NAME, LOPO_PROJECTS,
)
from feature_store import get_feature_matrix
from metrics import compute_metrics, metrics_table, bootstrap_ci
from baselines import run_baselines
from train_lr import train_logistic_regression, run_lopo_cv
from train_gbm import tune_xgboost, tune_lightgbm, select_champion, run_lopo_cv_gbm
from calibration import calibrate_champion
from shap_analysis import run_shap_analysis
from drift import run_drift_analysis
from threshold import run_threshold_analysis


def parse_args():
    p = argparse.ArgumentParser(description="ShiftMetrics ML Pipeline")
    p.add_argument("--n-trials",     type=int,   default=OPTUNA_N_TRIALS,
                   help="Optuna trials por modelo GBM (default: OPTUNA_N_TRIALS en config)")
    p.add_argument("--skip-lr",      action="store_true", help="Saltar ronda 2 (LR)")
    p.add_argument("--skip-shap",    action="store_true", help="Saltar ronda SHAP")
    p.add_argument("--skip-drift",   action="store_true", help="Saltar analisis de drift")
    p.add_argument("--champion-out", type=str,   default="/tmp/champion_calibrated.pkl",
                   help="Path donde guardar el champion calibrado")
    p.add_argument("--results-out",  type=str,   default="/tmp/leaderboard.csv",
                   help="Path donde guardar el leaderboard final")
    return p.parse_args()


def main():
    args = parse_args()

    print("ShiftMetrics — Full ML Pipeline")
    print(f"  Experiment : {MLFLOW_EXPERIMENT}")
    print(f"  Trials     : {args.n_trials}")
    print("-" * 55)

    t0_total = time.time()

    # 0. Feature matrix (4-way temporal split)
    print("\n[0/10] Cargando feature matrix desde BigQuery...")
    train, cal, val, test, df_full = get_feature_matrix()
    print(f"  Train={len(train):,} | Cal={len(cal):,} | Val={len(val):,} | Test={len(test):,}")

    # Pool pre-val para LOPO-CV (train 2000-2014 + cal 2015 = 2000-2015)
    train_lopo = pd.concat([train, cal], ignore_index=True)

    all_leaderboard = []

    # 1. Baselines
    print("\n[1/10] Ronda 1 - Baselines...")
    t0 = time.time()
    baseline_results = run_baselines(train, val, test)
    all_leaderboard.extend(baseline_results)
    print(f"  Completado en {time.time()-t0:.0f}s")

    # 2. Logistic Regression
    if not args.skip_lr:
        print("\n[2/10] Ronda 2 - Logistic Regression grid (18 configs)...")
        t0 = time.time()
        lr_pipeline, lr_metrics, lr_results_df = train_logistic_regression(train, val)
        all_leaderboard.append({**lr_metrics, "label": "Best LR"})

        print("\n  LOPO-CV con mejor LR (pool 2000-2015)...")
        lopo_lr_df = run_lopo_cv(lr_pipeline, train_lopo)
        print(lopo_lr_df[["project", "f2", "recall", "precision"]].to_string(index=False))
        print(f"  Completado en {time.time()-t0:.0f}s")
    else:
        print("\n[2/10] Ronda 2 - LR omitida (--skip-lr)")
        lr_pipeline = None

    # 3. XGBoost
    print(f"\n[3/10] Ronda 3 - XGBoost Optuna ({args.n_trials} trials)...")
    t0 = time.time()
    xgb_model, xgb_params, xgb_optuna_f2 = tune_xgboost(train, val, n_trials=args.n_trials)
    all_leaderboard.append({"label": "XGBoost_best", "f2": xgb_optuna_f2})
    print(f"  Completado en {time.time()-t0:.0f}s | Optuna val F2={xgb_optuna_f2:.4f}")

    # 4. LightGBM
    print(f"\n[4/10] Ronda 4 - LightGBM Optuna ({args.n_trials} trials, challenger)...")
    t0 = time.time()
    lgbm_model, lgbm_params, lgbm_optuna_f2 = tune_lightgbm(train, val, n_trials=args.n_trials)
    all_leaderboard.append({"label": "LightGBM_best", "f2": lgbm_optuna_f2})
    print(f"  Completado en {time.time()-t0:.0f}s | Optuna val F2={lgbm_optuna_f2:.4f}")

    # 5. Champion/Challenger selection + McNemar test
    print("\n[5/10] Ronda 5 - Champion/Challenger decision...")
    t0 = time.time()
    champion, champion_name, all_results = select_champion(
        xgb_model, xgb_optuna_f2,
        lgbm_model, lgbm_optuna_f2,
        val, test,
    )
    champion_params = xgb_params if champion_name == "XGBoost" else lgbm_params
    print(f"  Champion: {champion_name} | Completado en {time.time()-t0:.0f}s")

    # 6. Calibración — sobre cal (2015), holdout limpio
    print(f"\n[6/10] Ronda 6 - Calibracion del champion ({champion_name})...")
    print(f"  Fitting sobre cal (2015) — holdout nunca visto por el modelo refit")
    t0 = time.time()
    cal_model, cal_method, cal_metrics = calibrate_champion(
        champion, champion_name, cal, val, test
    )
    print(f"  Metodo ganador: {cal_method} | Completado en {time.time()-t0:.0f}s")

    joblib.dump(cal_model, args.champion_out)
    print(f"  Champion guardado en {args.champion_out}")

    # 7. LOPO-CV del champion GBM (hiperparametros congelados, pool 2000-2015)
    # threshold=0.5: mide generalizacion del procedimiento de entrenamiento (base model).
    # La evaluacion al operating point de deployment (calibrado, t_f2) se hace en ronda 10.
    print(f"\n[7/10] Ronda 7 - LOPO-CV champion GBM ({champion_name}, hiperparametros congelados)...")
    t0 = time.time()
    lopo_gbm_df = run_lopo_cv_gbm(champion_params, champion_name, train_lopo, threshold=0.5)
    print(lopo_gbm_df[["project", "f2", "recall", "precision"]].to_string(index=False))
    print(f"  Media F2 LOPO: {lopo_gbm_df['f2'].mean():.4f} "
          f"(std={lopo_gbm_df['f2'].std():.4f}) [base model, t=0.5]")
    print(f"  Completado en {time.time()-t0:.0f}s")

    # Log LOPO-GBM a MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"lopo_cv_{champion_name}"):
        mlflow.set_tag("model_role", "lopo_cv")
        mlflow.set_tag("champion_name", champion_name)
        mlflow.set_tag("threshold_note", "t=0.5_base_model_generalizability")
        mlflow.log_metric("lopo_f2_mean", lopo_gbm_df["f2"].mean())
        mlflow.log_metric("lopo_f2_std",  lopo_gbm_df["f2"].std())
        for _, row in lopo_gbm_df.iterrows():
            mlflow.log_metric(f"lopo_f2_{row['project']}", row["f2"])

    # 8. SHAP + PDPs
    if not args.skip_shap:
        print(f"\n[8/10] Ronda 8 - SHAP + PDPs ({champion_name})...")
        t0 = time.time()
        shap_results = run_shap_analysis(cal_model, champion_name, val, test)
        print(f"  Top features: {shap_results['top5_features']}")
        print(f"  Completado en {time.time()-t0:.0f}s")
    else:
        print("\n[8/10] Ronda 8 - SHAP omitida (--skip-shap)")

    # 9. Drift
    if not args.skip_drift:
        print(f"\n[9/10] Ronda 9 - Drift analysis ({champion_name})...")
        t0 = time.time()
        drift_results = run_drift_analysis(
            cal_model, champion_name, train_lopo, val, test, df_full
        )
        print(f"  Features drifted: {drift_results['ks_report']['drift_flag'].sum()}")
        print(f"  Completado en {time.time()-t0:.0f}s")
    else:
        print("\n[9/10] Ronda 9 - Drift omitida (--skip-drift)")

    # 10. Threshold
    print(f"\n[10/10] Ronda 10 - Threshold selection ({champion_name})...")
    t0 = time.time()
    threshold_results = run_threshold_analysis(cal_model, champion_name, val, test)
    t_f2  = threshold_results["threshold_f2opt"]
    t_roi = threshold_results["threshold_roi"]
    print(f"  F2-optimal threshold : {t_f2:.3f}")
    print(f"  ROI threshold        : {t_roi:.3f}")
    print(f"  Completado en {time.time()-t0:.0f}s")

    # Evaluacion por proyecto del modelo calibrado al threshold de deployment
    # Complementa el LOPO-CV (base model, t=0.5) con el operating point real:
    # modelo calibrado + t_f2 seleccionado en val — sin reentrenamiento por fold.
    print(f"\n  Eval por proyecto (calibrado, t={t_f2:.3f}) — operating point de deployment:")
    lopo_cal_rows = []
    for proj in LOPO_PROJECTS:
        X_proj = train_lopo.loc[train_lopo["project"] == proj, FEATURE_COLS].values
        y_proj = train_lopo.loc[train_lopo["project"] == proj, TARGET_COL].values
        if len(X_proj) == 0:
            continue
        prob_proj = cal_model.predict_proba(X_proj)[:, 1]
        pred_proj = (prob_proj >= t_f2).astype(int)
        m = compute_metrics(y_proj, pred_proj, prob_proj,
                             threshold=t_f2, label=f"lopo_cal_{proj}")
        lopo_cal_rows.append({**m, "project": proj})
    lopo_cal_df = pd.DataFrame(lopo_cal_rows)
    print(lopo_cal_df[["project", "f2", "recall", "precision"]].to_string(index=False))
    print(f"  Media F2 (calibrado): {lopo_cal_df['f2'].mean():.4f} "
          f"(std={lopo_cal_df['f2'].std():.4f}) [calibrado, t={t_f2:.3f}]")

    # Log evaluacion calibrada por proyecto a MLflow
    with mlflow.start_run(run_name=f"lopo_cal_{champion_name}"):
        mlflow.set_tag("model_role", "lopo_cal_eval")
        mlflow.set_tag("champion_name", champion_name)
        mlflow.set_tag("threshold_note", f"t={t_f2:.3f}_calibrated_deployment_point")
        mlflow.log_metric("lopo_cal_f2_mean", lopo_cal_df["f2"].mean())
        mlflow.log_metric("lopo_cal_f2_std",  lopo_cal_df["f2"].std())
        for _, row in lopo_cal_df.iterrows():
            mlflow.log_metric(f"lopo_cal_f2_{row['project']}", row["f2"])

    # Leaderboard final con bootstrap CIs
    print("\nLeaderboard final (test, threshold F2-opt)")
    print("-" * 55)

    X_val,  y_val  = val[FEATURE_COLS].values,  val[TARGET_COL].values
    X_test, y_test = test[FEATURE_COLS].values, test[TARGET_COL].values

    prob_v = cal_model.predict_proba(X_val)[:, 1]
    prob_t = cal_model.predict_proba(X_test)[:, 1]

    mv = compute_metrics(y_val,  (prob_v >= t_f2).astype(int), prob_v,  threshold=t_f2)
    mt = compute_metrics(y_test, (prob_t >= t_f2).astype(int), prob_t, threshold=t_f2)

    # Flagging rate — fraccion de sprints que el modelo marca como en riesgo.
    # Un t=0.05 post-calibracion puede ser agresivo; monitorear en produccion.
    flagging_rate = float((prob_t >= t_f2).mean())
    print(f"  Flagging rate (test): {flagging_rate:.1%}"
          + (" — WARN: >85%, revisar threshold o recalibrar" if flagging_rate > 0.85 else ""))

    # Bootstrap CI sobre test (1000 muestras) — requerimiento de rigor estadistico
    print(f"\n  Computando bootstrap CI (n=1000) sobre test set...")
    ci = bootstrap_ci(y_test, prob_t, threshold=t_f2, n_bootstrap=1000)

    model_label = f"Champion ({champion_name}, {cal_method}, t={t_f2:.2f})"
    final_leaderboard = [{
        "model":         model_label,
        "val_f2":        mv["f2"],
        "val_recall":    mv["recall"],
        "val_precision": mv["precision"],
        "val_pr_auc":    mv["pr_auc"],
        "val_brier":     mv["brier"],
        "test_f2":       mt["f2"],
        "test_f2_ci":    f"[{ci['f2_ci_lo']:.4f}, {ci['f2_ci_hi']:.4f}]",
        "test_recall":   mt["recall"],
        "test_recall_ci": f"[{ci['recall_ci_lo']:.4f}, {ci['recall_ci_hi']:.4f}]",
        "test_precision": mt["precision"],
        "test_brier":    mt["brier"],
        "test_brier_ci": f"[{ci['brier_ci_lo']:.4f}, {ci['brier_ci_hi']:.4f}]",
        "test_pr_auc":   mt["pr_auc"],
    }]

    lb_df = pd.DataFrame(final_leaderboard)
    print(lb_df[["model", "val_f2", "test_f2", "test_f2_ci",
                  "test_recall", "test_brier"]].to_string(index=False))
    lb_df.to_csv(args.results_out, index=False)
    print(f"\nLeaderboard guardado en {args.results_out}")

    # Log CI a MLflow
    with mlflow.start_run(run_name=f"final_eval_{champion_name}"):
        mlflow.set_tag("model_role", "final_evaluation")
        mlflow.set_tag("champion_name", champion_name)
        mlflow.log_metric("test_f2",          mt["f2"])
        mlflow.log_metric("test_f2_ci_lo",    ci["f2_ci_lo"])
        mlflow.log_metric("test_f2_ci_hi",    ci["f2_ci_hi"])
        mlflow.log_metric("test_recall",       mt["recall"])
        mlflow.log_metric("test_recall_ci_lo", ci["recall_ci_lo"])
        mlflow.log_metric("test_recall_ci_hi", ci["recall_ci_hi"])
        mlflow.log_metric("test_precision",    mt["precision"])
        mlflow.log_metric("test_brier",        mt["brier"])
        mlflow.log_metric("test_pr_auc",       mt["pr_auc"])
        mlflow.log_metric("lopo_f2_mean",      lopo_gbm_df["f2"].mean())
        mlflow.log_metric("lopo_f2_std",       lopo_gbm_df["f2"].std())
        mlflow.log_metric("lopo_cal_f2_mean",  lopo_cal_df["f2"].mean())
        mlflow.log_metric("lopo_cal_f2_std",   lopo_cal_df["f2"].std())
        mlflow.log_metric("flagging_rate_test", flagging_rate)
        mlflow.log_metric("val_f2",            mv["f2"])
        mlflow.log_metric("val_brier",         mv["brier"])
        mlflow.log_param("threshold_f2opt",    t_f2)
        mlflow.log_param("threshold_roi",      t_roi)
        mlflow.log_param("calibration_method", cal_method)

    # Promover champion a alias "production" en MLflow Model Registry.
    # Usa la API de aliases (MLflow >= 2.9) — la API de stages esta deprecada.
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "champion")
        client.set_registered_model_alias(
            name=MODEL_REGISTRY_NAME,
            alias="production",
            version=version.version,
        )
        print(f"\nModel Registry: {MODEL_REGISTRY_NAME} v{version.version} alias 'production' set")
    except Exception as e:
        print(f"\nModel Registry alias update: {e}")

    # Resumen JSON de configuracion final
    summary = {
        "champion":            champion_name,
        "calibration_method":  cal_method,
        "calibration_set":     "2015 (dedicated holdout, not in refit training)",
        "threshold_f2opt":     t_f2,
        "threshold_roi":       t_roi,
        "val_f2":              mv["f2"],
        "val_brier":           mv["brier"],
        "test_f2":             mt["f2"],
        "test_f2_95ci":        [ci["f2_ci_lo"], ci["f2_ci_hi"]],
        "test_recall":         mt["recall"],
        "test_recall_95ci":    [ci["recall_ci_lo"], ci["recall_ci_hi"]],
        "test_precision":      mt["precision"],
        "test_brier":          mt["brier"],
        "test_brier_95ci":     [ci["brier_ci_lo"], ci["brier_ci_hi"]],
        "test_pr_auc":         mt["pr_auc"],
        "lopo_f2_mean":          round(float(lopo_gbm_df["f2"].mean()), 4),
        "lopo_f2_std":           round(float(lopo_gbm_df["f2"].std()),  4),
        "lopo_by_project":       lopo_gbm_df[["project", "f2", "recall"]].to_dict("records"),
        "lopo_cal_f2_mean":      round(float(lopo_cal_df["f2"].mean()), 4),
        "lopo_cal_f2_std":       round(float(lopo_cal_df["f2"].std()),  4),
        "lopo_cal_by_project":   lopo_cal_df[["project", "f2", "recall"]].to_dict("records"),
        "flagging_rate_test":    round(flagging_rate, 4),
        "mcnemar_in_registry":   "see champion_* MLflow run",
        "champion_artifact":     args.champion_out,
        "model_registry":        MODEL_REGISTRY_NAME,
    }

    summary_path = "/tmp/pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResumen guardado en {summary_path}")

    elapsed = time.time() - t0_total
    print(f"\nPipeline completo en {elapsed/60:.1f} minutos")
    print(f"Champion      : {champion_name} | Calibracion: {cal_method}")
    print(f"Threshold     : F2-opt={t_f2:.3f} | ROI={t_roi:.3f}")
    print(f"Test F2       : {mt['f2']:.4f} "
          f"(95% CI [{ci['f2_ci_lo']:.4f}, {ci['f2_ci_hi']:.4f}])")
    print(f"Test Recall   : {mt['recall']:.4f} "
          f"(95% CI [{ci['recall_ci_lo']:.4f}, {ci['recall_ci_hi']:.4f}])")
    print(f"LOPO-CV F2    : {lopo_gbm_df['f2'].mean():.4f} "
          f"+/- {lopo_gbm_df['f2'].std():.4f} [base model, t=0.5]")
    print(f"LOPO Cal F2   : {lopo_cal_df['f2'].mean():.4f} "
          f"+/- {lopo_cal_df['f2'].std():.4f} [calibrado, t={t_f2:.3f}]")
    print(f"Flagging rate : {flagging_rate:.1%} (test)")


if __name__ == "__main__":
    main()
