"""
shap_analysis.py — ShiftMetrics ML Pipeline
Explicabilidad con SHAP: global (beeswarm + bar) y 3 casos locales.
Requiere el modelo calibrado final (champion).
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.inspection import PartialDependenceDisplay

from config import (
    FEATURE_COLS, TARGET_COL,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
)


# Global SHAP

def compute_shap_values(
    model,
    X: np.ndarray,
    model_type: str = "tree",
) -> shap.Explanation:
    """
    Calcula SHAP values. TreeExplainer para XGBoost/LightGBM,
    KernelExplainer para modelos no-árbol (LR calibrada).
    """
    if model_type == "tree":
        # Intenta TreeExplainer; si el modelo es CalibratedClassifierCV,
        # extrae el estimador base interno
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer(X)
        except Exception:
            # CalibratedClassifierCV envuelve el modelo — extraer estimador base
            base = getattr(model, "estimator", model)
            explainer = shap.TreeExplainer(base)
            sv = explainer(X)
    else:
        background = shap.sample(X, 100, random_state=42)
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1], background
        )
        sv = explainer(X[:200])  # KernelExplainer es lento — muestreo

    return sv


def plot_global_importance(
    shap_values: shap.Explanation,
    feature_names: list[str],
    save_dir: str = "/tmp",
) -> list[str]:
    """
    Genera beeswarm + bar plots de importancia global.
    Retorna paths de los archivos generados.
    """
    paths = []

    # Beeswarm (interacción valores-efecto)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Feature Importance — Beeswarm", fontsize=13)
    plt.tight_layout()
    beeswarm_path = f"{save_dir}/shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(beeswarm_path)

    # Bar plot (media |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("SHAP Feature Importance — Mean |SHAP|", fontsize=13)
    plt.tight_layout()
    bar_path = f"{save_dir}/shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(bar_path)

    # Tabla de importancia global
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    csv_path = f"{save_dir}/shap_importance.csv"
    imp_df.to_csv(csv_path, index=False)
    paths.append(csv_path)

    return paths


# Local cases

def _find_case(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    case_type: str,      # "tp", "tn", "fp", "fn"
    project_col: np.ndarray | None = None,
    project_filter: str | None = None,
) -> int:
    """
    Busca el índice del caso más representativo (máxima confianza para TP/TN,
    máxima confianza para FP/FN).
    """
    mask_dict = {
        "tp": (y_true == 1) & (y_pred == 1),
        "tn": (y_true == 0) & (y_pred == 0),
        "fp": (y_true == 0) & (y_pred == 1),
        "fn": (y_true == 1) & (y_pred == 0),
    }
    mask = mask_dict[case_type]

    if project_filter is not None and project_col is not None:
        project_mask = np.array([p == project_filter for p in project_col])
        mask = mask & project_mask

    if not mask.any():
        mask = mask_dict[case_type]  # fallback sin filtro de proyecto

    indices = np.where(mask)[0]
    if len(indices) == 0:
        return 0

    # Más confiado: más extremo en probabilidad
    probs_subset = y_prob[indices]
    if case_type in ("tp", "fp"):
        best = indices[np.argmax(probs_subset)]
    else:
        best = indices[np.argmin(probs_subset)]
    return int(best)


def plot_local_case(
    shap_values: shap.Explanation,
    idx: int,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    case_label: str,
    save_path: str,
) -> None:
    """Waterfall plot para un caso individual."""
    sv_i = shap_values[idx]
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(sv_i, max_display=12, show=False)
    plt.title(
        f"SHAP Local — {case_label}\n"
        f"True={int(y_true[idx])} | P(defecto)={y_prob[idx]:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# Partial Dependence Plots

def plot_pdp(
    model,
    X: np.ndarray,
    feature_names: list[str],
    top_features_idx: list[int],
    save_path: str,
    n_features: int = 5,
) -> None:
    """
    Partial Dependence Plots (PDPs) para las top N features segun SHAP.

    SHAP dice CUANTO importa cada feature; el PDP dice COMO afecta —
    la relacion funcional entre la feature y la probabilidad predicha.
    Juntos proveen una imagen completa de la explicabilidad del modelo.

    Usa grid_resolution=50 para curvas suaves sin costo computacional excesivo.
    Para CalibratedClassifierCV extrae el estimador base para compatibilidad.
    """
    n = min(n_features, len(top_features_idx))
    feat_indices = list(top_features_idx[:n])

    # Extraer estimador base si el modelo es un wrapper de calibracion
    base_model = model
    if hasattr(model, "estimator"):
        base_model = model.estimator
    elif hasattr(model, "calibrated_classifiers_"):
        try:
            base_model = model.calibrated_classifiers_[0].estimator
        except (IndexError, AttributeError):
            base_model = model

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    PartialDependenceDisplay.from_estimator(
        base_model,
        X,
        features=feat_indices,
        feature_names=feature_names,
        ax=axes,
        kind="average",
        grid_resolution=50,
        percentiles=(0.05, 0.95),
        line_kw={"color": "steelblue", "lw": 2},
    )

    for ax, feat_idx in zip(axes, feat_indices):
        ax.set_title(feature_names[feat_idx], fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Partial Dependence Plots — Top SHAP Features\n"
        "ShiftMetrics Sprint Defect Prediction",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# Main entry point

def run_shap_analysis(
    model,
    champion_name: str,
    val:  pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """
    Pipeline SHAP completo: global + 3 casos locales.
    Loguea todos los artefactos a MLflow.
    Retorna dict con paths de los artefactos generados.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_test   = test[FEATURE_COLS].values
    y_test   = test[TARGET_COL].values
    y_prob   = model.predict_proba(X_test)[:, 1]
    y_pred   = model.predict(X_test)
    projects = test["project"].values if "project" in test.columns else None

    model_type = "tree" if champion_name in ("XGBoost", "LightGBM") else "kernel"

    print(f"\n[SHAP] Computando SHAP values para {champion_name} "
          f"sobre test set ({len(X_test):,} filas)...")
    shap_values = compute_shap_values(model, X_test, model_type=model_type)

    with mlflow.start_run(run_name=f"shap_{champion_name}"):
        mlflow.set_tag("model_role", "explainability")
        mlflow.set_tag("champion_name", champion_name)

        # Global importance
        global_paths = plot_global_importance(shap_values, FEATURE_COLS)
        for p in global_paths:
            mlflow.log_artifact(p, artifact_path="shap/global")

        # 3 casos locales
        cases = [
            ("tp", "HTTPCLIENT", "TP — HTTPCLIENT sprint correctly flagged"),
            ("tn", "IO",         "TN — IO sprint correctly cleared"),
            ("fn", "MATH",       "FN — MATH sprint missed defect (high risk)"),
        ]

        local_paths = []
        for case_type, project, label in cases:
            idx = _find_case(y_test, y_pred, y_prob, case_type, projects, project)
            save_path = f"/tmp/shap_local_{case_type}_{project}.png"
            plot_local_case(shap_values, idx, y_test, y_prob, label, save_path)
            mlflow.log_artifact(save_path, artifact_path="shap/local")
            local_paths.append(save_path)
            print(f"  [{case_type.upper()} / {project}] idx={idx}, "
                  f"P={y_prob[idx]:.3f}, true={int(y_test[idx])}")

        # Top-5 features por SHAP en MLflow params
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top5 = np.argsort(mean_abs)[::-1][:5]
        for rank, fi in enumerate(top5, 1):
            mlflow.log_param(f"shap_top{rank}", FEATURE_COLS[fi])
            mlflow.log_metric(f"shap_importance_top{rank}", float(mean_abs[fi]))

        # PDPs para las top-5 features SHAP
        print(f"\n[SHAP] Generando Partial Dependence Plots para top-5 features...")
        pdp_path = "/tmp/shap_pdp_top5.png"
        plot_pdp(model, X_test, FEATURE_COLS, list(top5), pdp_path)
        mlflow.log_artifact(pdp_path, artifact_path="shap/pdp")
        print(f"  PDPs guardados en {pdp_path}")

    return {
        "global_plots": global_paths,
        "local_plots":  local_paths,
        "top5_features": [FEATURE_COLS[i] for i in top5],
        "pdp_plot":     pdp_path,
    }


if __name__ == "__main__":
    import joblib
    from feature_store import get_feature_matrix

    train, cal, val, test, _ = get_feature_matrix()

    try:
        model = joblib.load("/tmp/champion_calibrated.pkl")
        name  = "LightGBM"
    except FileNotFoundError:
        from train_gbm import tune_lightgbm
        model, _, _ = tune_lightgbm(train, val, n_trials=3)
        name = "LightGBM"

    results = run_shap_analysis(model, name, val, test)
    print(f"\nTop 5 features: {results['top5_features']}")
