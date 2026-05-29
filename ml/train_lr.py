"""
train_lr.py — ShiftMetrics ML Pipeline
Logistic Regression con grid search sobre C, penalty y solver.
Challenger de ronda 2 vs. mejor baseline.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTE

from config import (
    FEATURE_COLS, TARGET_COL, RANDOM_SEED, F_BETA,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, LOPO_PROJECTS,
)
from metrics import compute_metrics, metrics_table, optimal_f2_threshold
from feature_store import lopo_splits


# Grid de hiperparametros: 6 valores de C × 2 penalizaciones = 12 sin SMOTE + 3 con SMOTE = 15 total
PARAM_GRID = [
    {"C": c, "penalty": p, "solver": s, "smote": sm}
    for c  in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    for p, s in [("l2", "lbfgs"), ("l1", "saga")]
    for sm in [False]
] + [
    # SMOTE solo con l2 para no sobrecargar el grid
    {"C": c, "penalty": "l2", "solver": "lbfgs", "smote": True}
    for c in [0.1, 1.0, 10.0]
]


def _build_pipeline(C: float, penalty: str, solver: str) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight="balanced",
            max_iter=2000,   # saga+L1 con C alto requiere >1000 iteraciones
            random_state=RANDOM_SEED,
        )),
    ])


def _run_one_experiment(
    params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val:   pd.DataFrame,
    y_val:   np.ndarray,
) -> dict:
    """Entrena un set de hiperparámetros y retorna métricas. MLflow logging interno."""
    C, penalty, solver, use_smote = (
        params["C"], params["penalty"], params["solver"], params["smote"]
    )
    run_name = f"LR_C{C}_{penalty}_smote{use_smote}"

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("champion_round", "2")
        mlflow.log_params({"C": C, "penalty": penalty, "solver": solver, "smote": use_smote})

        pipe = _build_pipeline(C, penalty, solver)

        X_tr_fit, y_tr_fit = X_train, y_train
        if use_smote:
            # SMOTE solo en train, nunca val/test; imputar antes porque SMOTE requiere sin NaN
            from sklearn.impute import SimpleImputer as SI
            imp = SI(strategy="median").fit(X_train)
            X_imp = pd.DataFrame(imp.transform(X_train), columns=FEATURE_COLS)
            sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
            X_tr_fit, y_tr_fit = sm.fit_resample(X_imp, y_train)
            # Reemplazar imputer del pipe porque los datos ya estan imputados
            pipe.steps[0] = ("imputer", SimpleImputer(strategy="median"))

        pipe.fit(X_tr_fit, y_tr_fit)

        y_prob_val  = pipe.predict_proba(X_val)[:, 1]
        y_pred_val  = pipe.predict(X_val)
        m = compute_metrics(y_val, y_pred_val, y_prob_val, label=run_name)

        mlflow.log_metrics({f"val_{k}": v for k, v in m.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(
            pipe, artifact_path="model",
            registered_model_name=None,
            pip_requirements=["scikit-learn", "imbalanced-learn"],
            input_example=X_val.iloc[:1].astype(float),
        )

        return {**m, "label": run_name, "pipeline": pipe, "params": params}


def run_lopo_cv(
    best_pipeline: Pipeline,
    train_df: pd.DataFrame,
    projects: list[str] = LOPO_PROJECTS,
) -> pd.DataFrame:
    """
    LOPO-CV sobre los 5 proyectos seleccionados. Corre en paralelo (joblib).
    """
    def _lopo_fold(X_tr, y_tr, X_te, y_te, project):
        import copy
        p = copy.deepcopy(best_pipeline)
        p.fit(X_tr, y_tr)
        prob = p.predict_proba(X_te)[:, 1]
        y_te_arr = y_te.values if hasattr(y_te, "values") else y_te
        m = compute_metrics(y_te_arr, p.predict(X_te), prob, label=f"LOPO_{project}")
        return {**m, "project": project}

    results = Parallel(n_jobs=5)(
        delayed(_lopo_fold)(X_tr, y_tr, X_te, y_te, proj)
        for X_tr, y_tr, X_te, y_te, proj in lopo_splits(train_df, projects)
    )
    return pd.DataFrame(results)


def train_logistic_regression(
    train: pd.DataFrame,
    val:   pd.DataFrame,
) -> tuple[Pipeline, dict, pd.DataFrame]:
    """
    Corre el grid completo de LR. Retorna (best_pipeline, best_metrics, all_results_df).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL].values
    X_val   = val[FEATURE_COLS]
    y_val   = val[TARGET_COL].values

    print(f"\nEntrenando {len(PARAM_GRID)} configuraciones de LR...")
    all_results = []

    with mlflow.start_run(run_name="LR_grid_search"):
        mlflow.set_tag("model_family", "logistic_regression")

        # MLflow nested runs requieren ejecucion secuencial
        for params in PARAM_GRID:
            result = _run_one_experiment(params, X_train, y_train, X_val, y_val)
            all_results.append(result)
            print(f"  {result['label']}: F2={result['f2']:.4f} PR-AUC={result['pr_auc']:.4f}")

    # Seleccionar mejor por F2 en val
    best = max(all_results, key=lambda x: x["f2"])
    print(f"\nMejor LR: {best['label']} -> F2={best['f2']:.4f}")

    results_df = metrics_table([{k: v for k, v in r.items() if k != "pipeline"}
                                 for r in all_results])
    return best["pipeline"], best, results_df


if __name__ == "__main__":
    from feature_store import get_feature_matrix
    train, cal, val, test, _ = get_feature_matrix()
    pipe, best_metrics, results_df = train_logistic_regression(train, val)
    print("\n=== LR Grid Results (Val) ===")
    print(results_df[["label", "f2", "pr_auc", "recall", "precision", "brier"]].head(10).to_string(index=False))

    print("\nRunning LOPO-CV con mejor LR (pool 2000-2015)...")
    import pandas as pd
    train_lopo = pd.concat([train, cal], ignore_index=True)
    lopo_df = run_lopo_cv(pipe, train_lopo)
    print(lopo_df[["project", "f2", "recall", "precision"]].to_string(index=False))
