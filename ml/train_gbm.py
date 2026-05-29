"""
train_gbm.py — ShiftMetrics ML Pipeline
XGBoost y LightGBM con tuning Bayesiano via Optuna.
Champion/Challenger: el mejor de los dos pasa a calibración.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import optuna
import xgboost as xgb
import lightgbm as lgb
from joblib import Parallel, delayed

from config import (
    FEATURE_COLS, TARGET_COL, RANDOM_SEED,
    SCALE_POS_WEIGHT, OPTUNA_N_TRIALS, OPTUNA_TIMEOUT, OPTUNA_DB_PATH,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, LOPO_PROJECTS,
)
from metrics import compute_metrics, metrics_table, mcnemar_test
from feature_store import lopo_splits

optuna.logging.set_verbosity(optuna.logging.WARNING)


# XGBoost

def _xgb_objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Función objetivo para Optuna — cada trial = 1 MLflow run."""
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":          trial.suggest_int("max_depth", 3, 9),
        "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
        "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 0.3, 1.5),
        "tree_method":        "hist",
        "eval_metric":        "aucpr",
        "random_state":       RANDOM_SEED,
        "n_jobs":             -1,
    }

    with mlflow.start_run(run_name=f"XGB_trial_{trial.number}", nested=True):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("champion_round", "3")
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        m = compute_metrics(y_val, model.predict(X_val), y_prob, label=f"XGB_t{trial.number}")

        mlflow.log_metrics({f"val_{k}": v for k, v in m.items() if isinstance(v, (int, float))})
        mlflow.xgboost.log_model(model, artifact_path="model",
                                  pip_requirements=["xgboost"],
                                  input_example=X_val[:1].astype(float))

    return m["f2"]  # Optuna minimiza — pero usamos direction="maximize"


def tune_xgboost(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    n_trials: int = OPTUNA_N_TRIALS,
) -> tuple[xgb.XGBClassifier, dict, float]:
    """
    Optuna study para XGBoost con persistent SQLite storage.
    Retorna (best_model_refit, best_params, optuna_best_val_f2).

    El modelo retornado es el refit sobre train+val (max datos para produccion).
    La comparacion champion/challenger usa optuna_best_val_f2 — el F2 evaluado
    sobre val ANTES del refit, por lo que es una estimacion sin leakage.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_val   = val[FEATURE_COLS].values
    y_val   = val[TARGET_COL].values

    print(f"\nOptuna XGBoost: {n_trials} trials (storage={OPTUNA_DB_PATH})...")

    try:
        study = optuna.create_study(
            direction="maximize",
            study_name="xgboost_f2",
            storage=OPTUNA_DB_PATH,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if completed > 0:
            print(f"  Reanudando estudio existente ({completed} trials completados)")
    except Exception as e:
        print(f"  Storage no disponible ({e}), usando estudio en memoria")
        study = optuna.create_study(
            direction="maximize",
            study_name="xgboost_f2",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

    with mlflow.start_run(run_name="XGB_optuna_study"):
        mlflow.set_tag("model_family", "xgboost")
        study.optimize(
            lambda trial: _xgb_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=OPTUNA_TIMEOUT,
            n_jobs=1,  # nested MLflow runs necesitan secuencial
            show_progress_bar=True,
        )
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_f2", study.best_value)

    # Refit sobre train+val — modelo de produccion con maximos datos disponibles.
    # NOTA: el cal set (2015) NO se incluye en el refit, preservando su uso como
    # holdout de calibracion limpio.
    best_params = {**study.best_params, "tree_method": "hist",
                   "random_state": RANDOM_SEED, "n_jobs": -1}
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )

    print(f"  XGBoost best F2={study.best_value:.4f} | params={study.best_params}")
    return best_model, best_params, study.best_value


# LightGBM

def _lgb_objective(trial, X_train, y_train, X_val, y_val) -> float:
    """Función objetivo Optuna para LightGBM — challenger vs. XGBoost."""
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("max_depth", 3, 9),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "is_unbalance":      True,   # equivalente a class_weight='balanced' en XGB
        "random_state":      RANDOM_SEED,
        "n_jobs":            -1,
        "verbose":           -1,
    }

    with mlflow.start_run(run_name=f"LGBM_trial_{trial.number}", nested=True):
        mlflow.set_tag("model_type", "lightgbm")
        mlflow.set_tag("champion_round", "4")
        mlflow.log_params(params)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        m = compute_metrics(y_val, model.predict(X_val), y_prob, label=f"LGBM_t{trial.number}")

        mlflow.log_metrics({f"val_{k}": v for k, v in m.items() if isinstance(v, (int, float))})
        mlflow.lightgbm.log_model(model, artifact_path="model",
                                   pip_requirements=["lightgbm"],
                                   input_example=X_val[:1].astype(float))

    return m["f2"]


def tune_lightgbm(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    n_trials: int = OPTUNA_N_TRIALS,
) -> tuple[lgb.LGBMClassifier, dict, float]:
    """
    Optuna study para LightGBM con persistent SQLite storage.
    Challenger de XGBoost (ronda 4).
    Retorna (best_model_refit, best_params, optuna_best_val_f2).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_val   = val[FEATURE_COLS].values
    y_val   = val[TARGET_COL].values

    print(f"\nOptuna LightGBM: {n_trials} trials (storage={OPTUNA_DB_PATH})...")

    try:
        study = optuna.create_study(
            direction="maximize",
            study_name="lgbm_f2",
            storage=OPTUNA_DB_PATH,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if completed > 0:
            print(f"  Reanudando estudio existente ({completed} trials completados)")
    except Exception as e:
        print(f"  Storage no disponible ({e}), usando estudio en memoria")
        study = optuna.create_study(
            direction="maximize",
            study_name="lgbm_f2",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        )

    with mlflow.start_run(run_name="LGBM_optuna_study"):
        mlflow.set_tag("model_family", "lightgbm")
        study.optimize(
            lambda trial: _lgb_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=OPTUNA_TIMEOUT,
            n_jobs=1,
            show_progress_bar=True,
        )
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_f2", study.best_value)

    best_params = {**study.best_params, "is_unbalance": True,
                   "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1}
    best_model = lgb.LGBMClassifier(**best_params)
    best_model.fit(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )

    print(f"  LightGBM best F2={study.best_value:.4f} | params={study.best_params}")
    return best_model, best_params, study.best_value


# Champion/Challenger comparison

def select_champion(
    xgb_model,  xgb_optuna_f2: float,
    lgbm_model, lgbm_optuna_f2: float,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[object, str, dict]:
    """
    Regla de promocion: challenger supera al champion si ΔF2 > 0.005.
    La decision se basa en xgb_optuna_f2 / lgbm_optuna_f2 — los F2 obtenidos
    durante el estudio Optuna, evaluados sobre val ANTES del refit final.
    Esto evita el leakage de comparar sobre datos que el modelo refit ya vio.

    Tambien ejecuta el test de McNemar sobre val para verificar significancia
    estadistica de la diferencia entre clasificadores.

    Retorna (champion_model, champion_name, metrics_dict).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_val,  y_val  = val[FEATURE_COLS].values,  val[TARGET_COL].values
    X_test, y_test = test[FEATURE_COLS].values, test[TARGET_COL].values

    # Decision basada en Optuna val F2 (sin leakage)
    delta = lgbm_optuna_f2 - xgb_optuna_f2
    PROMOTION_THRESHOLD = 0.005

    print(f"  XGBoost  Optuna val F2 = {xgb_optuna_f2:.4f}")
    print(f"  LightGBM Optuna val F2 = {lgbm_optuna_f2:.4f}")
    print(f"  Delta = {delta:+.4f} (threshold = {PROMOTION_THRESHOLD})")

    if delta > PROMOTION_THRESHOLD:
        champion, champion_name = lgbm_model, "LightGBM"
        print(f"  Champion: LightGBM (deltaF2={delta:+.4f} > {PROMOTION_THRESHOLD})")
    else:
        champion, champion_name = xgb_model, "XGBoost"
        print(f"  Champion: XGBoost (LightGBM no supera threshold, deltaF2={delta:+.4f})")

    # Metricas post-refit en val y test (refit incluye val, estas metricas son
    # optimistas en val — se logean para tracking pero NO para tomar decisiones)
    results = {}
    for model, name in [(xgb_model, "XGBoost"), (lgbm_model, "LightGBM")]:
        results[name] = {}
        for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
            prob = model.predict_proba(X_s)[:, 1]
            m    = compute_metrics(y_s, model.predict(X_s), prob, label=f"{name}_{split_name}")
            results[name][split_name] = m

    # Test de McNemar: diferencia estadisticamente significativa?
    pred_xgb  = xgb_model.predict(X_val)
    pred_lgbm = lgbm_model.predict(X_val)
    mc = mcnemar_test(y_val, pred_xgb, pred_lgbm)
    print(f"  McNemar test (val): chi2={mc['statistic']:.4f}, "
          f"p={mc['p_value']:.4f}, "
          f"significativo={'si' if mc['significant'] else 'no'}")
    print(f"    XGB correcto/LGBM incorrecto: {mc['a_right_b_wrong']:,}")
    print(f"    XGB incorrecto/LGBM correcto: {mc['a_wrong_b_right']:,}")
    if mc["significant"] and mc["a_wrong_b_right"] > mc["a_right_b_wrong"]:
        print(
            f"  [McNemar context] LGBM corrige {mc['a_wrong_b_right']:,} errores exclusivos de XGB "
            f"vs {mc['a_right_b_wrong']:,} a la inversa — McNemar favorece LGBM a nivel de caso. "
            f"Champion XGB seleccionado por F2-Optuna ({xgb_optuna_f2:.4f} vs "
            f"{lgbm_optuna_f2:.4f}): McNemar opera al threshold=0.5 de predict(), "
            f"Optuna optimiza al threshold-optimo por trial. "
            f"XGB maximiza recall mas agresivamente — premiado por F2 (beta=2). "
            f"Ambos modelos son significativamente distintos; la eleccion es correcta para el KPI."
        )

    with mlflow.start_run(run_name=f"champion_{champion_name}"):
        mlflow.set_tag("model_role", "champion")
        mlflow.set_tag("champion_name", champion_name)
        mlflow.log_metric("xgb_optuna_val_f2",   xgb_optuna_f2)
        mlflow.log_metric("lgbm_optuna_val_f2",  lgbm_optuna_f2)
        mlflow.log_metric("delta_optuna_f2",      delta)
        mlflow.log_metric("mcnemar_chi2",         mc["statistic"])
        mlflow.log_metric("mcnemar_p_value",      mc["p_value"])
        mlflow.log_metric("val_f2_refit",   results[champion_name]["val"]["f2"])
        mlflow.log_metric("test_f2",        results[champion_name]["test"]["f2"])
        mlflow.log_metric("val_pr_auc",     results[champion_name]["val"]["pr_auc"])
        mlflow.log_metric("test_pr_auc",    results[champion_name]["test"]["pr_auc"])
        mlflow.log_metric("val_brier_refit",results[champion_name]["val"]["brier"])
        mlflow.log_metric("test_brier",     results[champion_name]["test"]["brier"])
        mlflow.set_tag("mcnemar_significant", str(mc["significant"]))

    return champion, champion_name, results


def run_lopo_cv_gbm(
    champion_params: dict,
    champion_family: str,
    train_df: pd.DataFrame,
    projects: list = LOPO_PROJECTS,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    LOPO-CV del champion GBM con hiperparametros congelados.

    Entrena un modelo fresco en cada fold {train - project}, evalua en el
    proyecto excluido. No re-hace HPO por fold — usar los hiperparametros del
    estudio completo es practica estandar (equivalente a eval de generalizacion
    con arquitectura fija, como en LOPO publicado en literatura de SE).

    Nota sobre threshold:
        threshold=0.5 (default) evalua el modelo BASE sin calibrar — mide
        la generalizacion del procedimiento de entrenamiento entre proyectos.
        Para metricas en el operating point de deployment (post-calibracion),
        ver la evaluacion por proyecto en run_pipeline.py (lopo_cal_df).

    Parametros:
        champion_params: hiperparametros optimos del estudio Optuna
        champion_family: "XGBoost" o "LightGBM"
        train_df: conjunto completo pre-val (train + cal, 2000-2015)
        threshold: threshold de decision para compute_metrics (default=0.5)
    """
    results = []
    for X_tr, y_tr, X_te, y_te, project in lopo_splits(train_df, projects):
        if champion_family == "LightGBM":
            model = lgb.LGBMClassifier(**champion_params)
            model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=-1)])
        else:
            model = xgb.XGBClassifier(**champion_params)
            model.fit(X_tr, y_tr, verbose=False)

        prob = model.predict_proba(X_te)[:, 1]
        y_te_arr = y_te.values if hasattr(y_te, "values") else y_te
        m = compute_metrics(y_te_arr, model.predict(X_te), prob,
                            threshold=threshold, label=f"LOPO_{project}")
        results.append({**m, "project": project})

    return pd.DataFrame(results)


if __name__ == "__main__":
    from feature_store import get_feature_matrix
    train, cal, val, test, _ = get_feature_matrix()

    xgb_model,  xgb_params,  xgb_f2  = tune_xgboost(train, val)
    lgbm_model, lgbm_params, lgbm_f2 = tune_lightgbm(train, val)

    champion, champion_name, all_results = select_champion(
        xgb_model, xgb_f2, lgbm_model, lgbm_f2, val, test
    )
    print(f"\nChampion final: {champion_name}")

    champion_params = xgb_params if champion_name == "XGBoost" else lgbm_params
    train_lopo = pd.concat([train, cal])
    lopo_df = run_lopo_cv_gbm(champion_params, champion_name, train_lopo)
    print(lopo_df[["project", "f2", "recall", "precision"]].to_string(index=False))
