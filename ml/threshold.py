"""
threshold.py — ShiftMetrics ML Pipeline
Selección de threshold operacional: F2-óptimo + ROI (cost_fn=3xcost_fp).
PR curve, curva F2 vs threshold, análisis de impacto por umbral.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import mlflow

from config import (
    FEATURE_COLS, TARGET_COL, F_BETA,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
)
from metrics import compute_metrics, optimal_f2_threshold, roi_threshold


# Threshold sweep completo
def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evalúa F2, precision, recall, y business_value en el rango de thresholds.
    Business value: tp - 1.0*fp - 3.0*fn (FN cuesta 3x FP).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    from sklearn.metrics import fbeta_score, precision_score, recall_score
    from sklearn.metrics import confusion_matrix as cm

    rows = []
    for t in thresholds:
        pred_t = (y_prob >= t).astype(int)
        tn, fp, fn, tp = cm(y_true, pred_t, labels=[0, 1]).ravel()
        f2_val = fbeta_score(y_true, pred_t, beta=F_BETA, zero_division=0)
        prec   = precision_score(y_true, pred_t, zero_division=0)
        rec    = recall_score(y_true, pred_t, zero_division=0)
        bv     = tp - 1.0 * fp - 3.0 * fn
        rows.append({
            "threshold":      round(float(t), 3),
            "f2":             round(float(f2_val), 4),
            "precision":      round(float(prec), 4),
            "recall":         round(float(rec), 4),
            "business_value": int(bv),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    return pd.DataFrame(rows)


# Plots
def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_f2:   float,
    t_roi:  float,
    save_path: str,
) -> None:
    """PR curve con marcadores para el threshold F2-óptimo y ROI-óptimo."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # Encontrar puntos en la curva para los dos thresholds
    def _pr_at_threshold(t):
        idx = np.searchsorted(thresholds, t)
        idx = min(idx, len(precision) - 2)
        return precision[idx], recall[idx]

    p_f2,  r_f2  = _pr_at_threshold(t_f2)
    p_roi, r_roi = _pr_at_threshold(t_roi)

    plt.figure(figsize=(9, 6))
    plt.plot(recall, precision, lw=2, label=f"PR Curve (AUC={pr_auc:.3f})")
    plt.scatter(r_f2,  p_f2,  s=120, zorder=5, color="green",
                label=f"F2-opt threshold={t_f2:.2f} (R={r_f2:.2f}, P={p_f2:.2f})")
    plt.scatter(r_roi, p_roi, s=120, zorder=5, color="orange", marker="^",
                label=f"ROI threshold={t_roi:.2f} (R={r_roi:.2f}, P={p_roi:.2f})")
    plt.axhline(y_true.mean(), color="gray", linestyle="--",
                alpha=0.5, label=f"Baseline (pos rate={y_true.mean():.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — ShiftMetrics Sprint Defect")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_tradeoff(
    sweep_df: pd.DataFrame,
    t_f2: float,
    t_roi: float,
    save_path: str,
) -> None:
    """Curvas de F2, precision, recall y business_value vs threshold."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(sweep_df["threshold"], sweep_df["f2"],        label="F2",        color="blue")
    ax1.plot(sweep_df["threshold"], sweep_df["precision"],  label="Precision", color="green",  linestyle="--")
    ax1.plot(sweep_df["threshold"], sweep_df["recall"],     label="Recall",    color="orange", linestyle="--")
    ax1.axvline(t_f2,  color="blue",   linestyle=":", alpha=0.7, label=f"F2-opt={t_f2:.2f}")
    ax1.axvline(t_roi, color="purple", linestyle=":", alpha=0.7, label=f"ROI={t_roi:.2f}")
    ax1.set_ylabel("Score")
    ax1.set_title("Threshold Trade-off — ShiftMetrics Sprint Defect")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(sweep_df["threshold"], sweep_df["business_value"], color="purple")
    ax2.axvline(t_f2,  color="blue",   linestyle=":", alpha=0.7)
    ax2.axvline(t_roi, color="purple", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Business Value (tp - fp - 3·fn)")
    ax2.set_title("Business Value vs Threshold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# Main entry point
def run_threshold_analysis(
    model,
    champion_name: str,
    val:  pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """
    Análisis completo de threshold sobre val (selección) y test (evaluación).
    Retorna dict con thresholds seleccionados y métricas en ambos splits.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_val,  y_val  = val[FEATURE_COLS].values,  val[TARGET_COL].values
    X_test, y_test = test[FEATURE_COLS].values, test[TARGET_COL].values

    prob_val  = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    # Selección de thresholds en VAL (nunca en test)
    t_f2,  f2_best  = optimal_f2_threshold(y_val, prob_val)
    t_roi, roi_best = roi_threshold(y_val, prob_val, cost_fn=3.0, cost_fp=1.0)

    print(f"\n[Threshold] Champion: {champion_name}")
    print(f"  F2-optimal (val): threshold={t_f2:.3f}, F2={f2_best:.4f}")
    print(f"  ROI-optimal (val): threshold={t_roi:.3f}, BizValue={roi_best:.0f}")

    with mlflow.start_run(run_name=f"threshold_{champion_name}"):
        mlflow.set_tag("model_role", "threshold_selection")
        mlflow.set_tag("champion_name", champion_name)
        mlflow.log_param("threshold_f2",  t_f2)
        mlflow.log_param("threshold_roi", t_roi)

        # Métricas en val y test para ambos thresholds
        for split_name, X_s, y_s, prob_s in [
            ("val",  X_val,  y_val,  prob_val),
            ("test", X_test, y_test, prob_test),
        ]:
            for thresh_name, thresh_val in [("f2opt", t_f2), ("roi", t_roi)]:
                pred = (prob_s >= thresh_val).astype(int)
                m    = compute_metrics(y_s, pred, prob_s, threshold=thresh_val,
                                       label=f"{thresh_name}_{split_name}")
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{split_name}_{thresh_name}_{k}", v)
                print(f"  [{split_name} / {thresh_name}={thresh_val:.2f}] "
                      f"F2={m['f2']:.4f} | Recall={m['recall']:.4f} | Prec={m['precision']:.4f}")

        # Sweep y plots (sobre val)
        sweep_df = threshold_sweep(y_val, prob_val)
        sweep_csv = "/tmp/threshold_sweep.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        mlflow.log_artifact(sweep_csv, artifact_path="threshold")

        pr_path = "/tmp/pr_curve.png"
        plot_pr_curve(y_val, prob_val, t_f2, t_roi, pr_path)
        mlflow.log_artifact(pr_path, artifact_path="threshold")

        tradeoff_path = "/tmp/threshold_tradeoff.png"
        plot_threshold_tradeoff(sweep_df, t_f2, t_roi, tradeoff_path)
        mlflow.log_artifact(tradeoff_path, artifact_path="threshold")

        # Métricas en test con threshold F2-óptimo (el que se usará en producción)
        pred_test_f2opt = (prob_test >= t_f2).astype(int)
        m_test_final    = compute_metrics(y_test, pred_test_f2opt, prob_test, threshold=t_f2)
        mlflow.log_metric("test_final_f2",      m_test_final["f2"])
        mlflow.log_metric("test_final_recall",  m_test_final["recall"])
        mlflow.log_metric("test_final_prec",    m_test_final["precision"])
        mlflow.log_metric("test_final_pr_auc",  m_test_final["pr_auc"])

    return {
        "threshold_f2opt": t_f2,
        "threshold_roi":   t_roi,
        "f2_at_t_f2opt":   f2_best,
        "roi_at_t_roi":    roi_best,
        "test_metrics_f2opt": m_test_final,
        "plots": [pr_path, tradeoff_path],
    }


if __name__ == "__main__":
    from feature_store import get_feature_matrix
    from train_gbm import tune_xgboost

    train, val, test, _ = get_feature_matrix()
    model, _ = tune_xgboost(train, val, n_trials=3)

    results = run_threshold_analysis(model, "XGBoost", val, test)
    print(f"\nThresholds seleccionados:")
    print(f"  F2-optimal: {results['threshold_f2opt']:.3f}")
    print(f"  ROI:        {results['threshold_roi']:.3f}")
    print(f"\nTest metrics (threshold F2-opt):")
    for k, v in results["test_metrics_f2opt"].items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}")
