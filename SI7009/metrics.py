"""
metrics.py — ShiftMetrics SI7009
Suite de métricas unificada. Una sola función para evaluar cualquier modelo.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (
    fbeta_score, precision_score, recall_score,
    average_precision_score, brier_score_loss,
    roc_auc_score, confusion_matrix,
)
from config import F_BETA


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict:
    """
    Calcula toda la suite de métricas requeridas por SI7009.

    Retorna dict con:
      f2, precision, recall, pr_auc, roc_auc, brier, tn, fp, fn, tp
    """
    y_pred_thresh = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh, labels=[0, 1]).ravel()

    return {
        "label":     label,
        "threshold": threshold,
        # Metrica primaria (F_BETA=2: recall pesa 2x mas que precision)
        "f2":        round(fbeta_score(y_true, y_pred_thresh, beta=F_BETA, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred_thresh, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred_thresh, zero_division=0), 4),
        # Métricas de ranking (threshold-independent)
        "pr_auc":    round(average_precision_score(y_true, y_prob), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        # Calibración
        "brier":     round(brier_score_loss(y_true, y_prob), 4),
        # Matriz de confusión
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def metrics_table(results: list[dict]) -> pd.DataFrame:
    """
    Formatea lista de dicts de métricas en DataFrame ordenado por F2 desc.
    """
    df = pd.DataFrame(results)
    cols_order = ["label", "f2", "pr_auc", "recall", "precision", "roc_auc", "brier",
                  "tp", "fp", "fn", "tn", "threshold"]
    df = df[[c for c in cols_order if c in df.columns]]
    return df.sort_values("f2", ascending=False).reset_index(drop=True)


def optimal_f2_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Busca el threshold que maximiza F2 en el conjunto dado.
    Retorna (threshold_optimo, f2_maximo).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    f2s = [
        fbeta_score(y_true, (y_prob >= t).astype(int), beta=F_BETA, zero_division=0)
        for t in thresholds
    ]
    best_idx = np.argmax(f2s)
    return round(thresholds[best_idx], 3), round(f2s[best_idx], 4)


def roi_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 3.0,
    cost_fp: float = 1.0,
) -> tuple[float, float]:
    """
    Threshold óptimo según modelo de ROI de negocio.
    Default: cost(FN) = 3x cost(FP).
    Un defecto escapado cuesta 3x mas que una revision innecesaria.

    Retorna (threshold_optimo, business_value_maximo).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_val = 0.5, -np.inf
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t, labels=[0, 1]).ravel()
        # Valor de negocio = TP ganados - FP (revisiones falsas) - FN (defectos escapados)
        value = tp - cost_fp * fp - cost_fn * fn
        if value > best_val:
            best_val, best_t = value, t
    return round(best_t, 3), round(best_val, 1)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Intervalos de confianza al 95% via bootstrap (n=1000 muestras con reemplazo).
    Reporta CI para F2, recall, precision, PR-AUC y Brier score.

    Necesario para publicar cualquier resultado de ML con rigor estadístico:
    una métrica puntual sin incertidumbre no permite comparaciones honestas.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    records: dict[str, list] = {k: [] for k in ("f2", "recall", "precision", "pr_auc", "brier")}

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m = compute_metrics(
            y_true[idx],
            (y_prob[idx] >= threshold).astype(int),
            y_prob[idx],
            threshold=threshold,
        )
        for key in records:
            records[key].append(m[key])

    ci = {}
    for key, vals in records.items():
        arr = np.array(vals)
        ci[f"{key}_ci_lo"] = round(float(np.percentile(arr, 2.5)),  4)
        ci[f"{key}_ci_hi"] = round(float(np.percentile(arr, 97.5)), 4)

    return ci


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict:
    """
    Test de McNemar para comparar dos clasificadores binarios.
    H0: ambos clasificadores tienen la misma tasa de error.
    Usa aproximación chi-cuadrado con corrección de continuidad de Edwards.

    Interpretación:
      p < 0.05 -> diferencia estadísticamente significativa entre modelos
      La corrección de continuidad es conservadora pero apropiada para n pequeño.
    """
    a_right_b_wrong = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    a_wrong_b_right = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    discordant = a_right_b_wrong + a_wrong_b_right

    if discordant == 0:
        return {
            "statistic": 0.0, "p_value": 1.0, "significant": False,
            "a_right_b_wrong": 0, "a_wrong_b_right": 0,
            "note": "modelos identicos en todas las predicciones",
        }

    stat  = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / discordant
    p_val = float(1.0 - chi2.cdf(stat, df=1))

    return {
        "statistic":      round(float(stat), 4),
        "p_value":        round(p_val, 4),
        "significant":    p_val < 0.05,
        "a_right_b_wrong": a_right_b_wrong,   # A correcto, B incorrecto
        "a_wrong_b_right": a_wrong_b_right,   # A incorrecto, B correcto
    }
