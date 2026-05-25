"""
calibration.py — ShiftMetrics SI7009
Platt scaling + isotonic regression sobre el champion model.
Compara Brier antes/después y genera reliability diagrams.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from config import (
    FEATURE_COLS, TARGET_COL, RANDOM_SEED,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MODEL_REGISTRY_NAME,
)
from metrics import compute_metrics


# Reliability diagram

def plot_reliability_diagram(
    y_true: np.ndarray,
    prob_before: np.ndarray,
    prob_after:  np.ndarray,
    method: str,
    save_path: str,
) -> None:
    """
    Curva de calibración (reliability diagram) antes vs. después.
    n_bins=10 equilibra granularidad con estabilidad estadística.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, probs, title in [
        (axes[0], prob_before, "Before calibration"),
        (axes[1], prob_after,  f"After calibration ({method})"),
    ]:
        fraction_of_positives, mean_predicted = calibration_curve(
            y_true, probs, n_bins=10, strategy="uniform"
        )
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_predicted, fraction_of_positives, "s-", label="Model")
        brier = brier_score_loss(y_true, probs)
        ax.set_title(f"{title}\nBrier = {brier:.4f}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Reliability Diagram — ShiftMetrics Sprint Defect", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# Calibration wrapper

def calibrate_champion(
    champion_model,
    champion_name: str,
    cal:  pd.DataFrame,
    val:  pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[object, str, dict]:
    """
    Aplica Platt scaling e isotonic regression al champion.

    Uso de splits:
      cal  -> fitting de los calibradores (cv="prefit"). El modelo refit fue
              entrenado sobre train+val, por lo que cal (2015) es un holdout
              limpio — ninguno de los datos de cal estuvo en el entrenamiento.
              Esto elimina el triple-dipping que inflaba el Brier en val.
      val  -> evaluacion honesta del Brier post-calibracion (threshold selection)
      test -> evaluacion final intocable

    Compara sigmoid vs isotonic en cal-Brier. El ganador se registra en el
    MLflow Model Registry como version "Staging".

    Retorna (calibrated_model, method_winner, metrics_dict).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_cal  = cal[FEATURE_COLS].values
    y_cal  = cal[TARGET_COL].values
    X_val  = val[FEATURE_COLS].values
    y_val  = val[TARGET_COL].values
    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_COL].values

    # Probabilidades del modelo sin calibrar sobre los tres splits
    prob_cal_raw  = champion_model.predict_proba(X_cal)[:, 1]
    prob_val_raw  = champion_model.predict_proba(X_val)[:, 1]
    prob_test_raw = champion_model.predict_proba(X_test)[:, 1]
    brier_raw_cal  = brier_score_loss(y_cal,  prob_cal_raw)
    brier_raw_val  = brier_score_loss(y_val,  prob_val_raw)
    brier_raw_test = brier_score_loss(y_test, prob_test_raw)

    print(f"\n[Calibration] Champion: {champion_name}")
    print(f"  Brier (cal,  raw) = {brier_raw_cal:.4f}  [fitting set, 2015]")
    print(f"  Brier (val,  raw) = {brier_raw_val:.4f}  [eval set, 2016-2018]")
    print(f"  Brier (test, raw) = {brier_raw_test:.4f}  [final eval, 2019-2021]")

    results = {}

    with mlflow.start_run(run_name=f"calibration_{champion_name}") as run:
        mlflow.set_tag("model_role", "calibration")
        mlflow.set_tag("champion_name", champion_name)
        mlflow.set_tag("cal_strategy", "dedicated_holdout_2015")
        mlflow.log_metric("brier_cal_raw",  brier_raw_cal)
        mlflow.log_metric("brier_val_raw",  brier_raw_val)
        mlflow.log_metric("brier_test_raw", brier_raw_test)

        for method in ["sigmoid", "isotonic"]:
            # Fit sobre cal — holdout limpio (nunca visto por el modelo refit)
            cal_model = CalibratedClassifierCV(
                estimator=champion_model,
                method=method,
                cv="prefit",
            )
            cal_model.fit(X_cal, y_cal)

            prob_cal_cal  = cal_model.predict_proba(X_cal)[:, 1]
            prob_val_cal  = cal_model.predict_proba(X_val)[:, 1]
            prob_test_cal = cal_model.predict_proba(X_test)[:, 1]

            brier_cal_cal  = brier_score_loss(y_cal,  prob_cal_cal)
            brier_cal_val  = brier_score_loss(y_val,  prob_val_cal)
            brier_cal_test = brier_score_loss(y_test, prob_test_cal)

            m_cal  = compute_metrics(y_cal,  cal_model.predict(X_cal),  prob_cal_cal,  label=f"{method}_cal")
            m_val  = compute_metrics(y_val,  cal_model.predict(X_val),  prob_val_cal,  label=f"{method}_val")
            m_test = compute_metrics(y_test, cal_model.predict(X_test), prob_test_cal, label=f"{method}_test")

            mlflow.log_metric(f"brier_cal_{method}",  brier_cal_cal)
            mlflow.log_metric(f"brier_val_{method}",  brier_cal_val)
            mlflow.log_metric(f"brier_test_{method}", brier_cal_test)
            mlflow.log_metric(f"f2_cal_{method}",     m_cal["f2"])
            mlflow.log_metric(f"f2_val_{method}",     m_val["f2"])
            mlflow.log_metric(f"f2_test_{method}",    m_test["f2"])

            delta_cal = brier_raw_cal - brier_cal_cal
            print(f"  [{method:>9}] Brier cal={brier_cal_cal:.4f} (delta={delta_cal:+.4f}) "
                  f"| val={brier_cal_val:.4f} | test={brier_cal_test:.4f} "
                  f"| F2 val={m_val['f2']:.4f}")

            # Reliability diagram sobre cal (set de fitting — refleja calibracion real)
            diagram_path = f"/tmp/reliability_{method}.png"
            plot_reliability_diagram(
                y_cal, prob_cal_raw, prob_cal_cal, method, diagram_path
            )
            mlflow.log_artifact(diagram_path, artifact_path="plots")

            results[method] = {
                "model":      cal_model,
                "brier_cal":  brier_cal_cal,
                "brier_val":  brier_cal_val,
                "brier_test": brier_cal_test,
                "f2_cal":     m_cal["f2"],
                "f2_val":     m_val["f2"],
                "f2_test":    m_test["f2"],
            }

        # Seleccionar por menor Brier en cal (el set de fitting — evita optimizar
        # sobre val que podria tener sesgo de HPO)
        best_method = min(results, key=lambda k: results[k]["brier_cal"])
        best_brier  = results[best_method]["brier_cal"]

        if best_brier >= brier_raw_cal - 1e-6:
            print("  Calibracion no mejora Brier en cal, se retiene modelo raw")
            winner_model  = champion_model
            winner_method = "none (raw)"
        else:
            winner_model  = results[best_method]["model"]
            winner_method = best_method
            print(f"  Ganador: {best_method} | Brier cal={best_brier:.4f} "
                  f"| val={results[best_method]['brier_val']:.4f} "
                  f"| test={results[best_method]['brier_test']:.4f}")

        mlflow.set_tag("calibration_winner", winner_method)
        mlflow.log_metric("brier_cal_winner",  results[best_method]["brier_cal"])
        mlflow.log_metric("brier_val_winner",  results[best_method]["brier_val"])
        mlflow.log_metric("brier_test_winner", results[best_method]["brier_test"])

        mlflow.sklearn.log_model(winner_model, artifact_path="calibrated_model",
                                  pip_requirements=["scikit-learn"],
                                  input_example=X_cal[:1].astype(float))

        # Registrar en MLflow Model Registry con alias (API >= 2.9)
        try:
            model_uri = f"runs:/{run.info.run_id}/calibrated_model"
            mv = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
            print(f"  Model Registry: {MODEL_REGISTRY_NAME} v{mv.version} creada")

            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name=MODEL_REGISTRY_NAME,
                alias="champion",
                version=mv.version,
            )
            print(f"  Alias 'champion' -> v{mv.version}")
        except Exception as e:
            print(f"  Model Registry no disponible: {e}")

    winner_metrics = {
        "brier_cal_raw":  brier_raw_cal,
        "brier_val_raw":  brier_raw_val,
        "brier_test_raw": brier_raw_test,
        **{f"brier_cal_{m}":  results[m]["brier_cal"]  for m in results},
        **{f"brier_val_{m}":  results[m]["brier_val"]  for m in results},
        **{f"brier_test_{m}": results[m]["brier_test"] for m in results},
        **{f"f2_val_{m}":     results[m]["f2_val"]     for m in results},
    }

    return winner_model, winner_method, winner_metrics


if __name__ == "__main__":
    from feature_store import get_feature_matrix
    from train_gbm import tune_xgboost, tune_lightgbm, select_champion

    train, cal, val, test, _ = get_feature_matrix()

    xgb_model,  xgb_params,  xgb_f2  = tune_xgboost(train, val, n_trials=5)
    lgbm_model, lgbm_params, lgbm_f2 = tune_lightgbm(train, val, n_trials=5)

    champion, champion_name, _ = select_champion(
        xgb_model, xgb_f2, lgbm_model, lgbm_f2, val, test
    )

    cal_model, method, cal_metrics = calibrate_champion(
        champion, champion_name, cal, val, test
    )
    print(f"\nCalibracion final: {method}")
    for k, v in cal_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
