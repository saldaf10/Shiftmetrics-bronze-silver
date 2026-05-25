"""
baselines.py — ShiftMetrics SI7009
Dos baselines que todo modelo debe superar en F2.

Baseline 1 — MajorityClassifier:  predice siempre la clase mayoritaria (1).
  F2 esperado: 0.923  (70.45% positivos x recall=1.0)
  Este es el baseline MÁS DIFÍCIL de superar con F2 en datasets desbalanceados.

Baseline 2 — ThresholdBSRClassifier: predice 1 si bug_story_ratio > umbral.
  Umbral óptimo determinado en train set (no en val/test).
  Simula la heurística de negocio de un Tech Lead.
"""

import numpy as np
import pandas as pd
import mlflow
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from metrics import compute_metrics, optimal_f2_threshold
from config import TARGET_COL, FEATURE_COLS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT


# Baseline 1: Clase Mayoritaria

class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Siempre predice la clase mayoritaria del train set."""

    def fit(self, X, y):
        self.majority_class_ = int(np.bincount(y).argmax())
        self.majority_rate_  = float(np.mean(y == self.majority_class_))
        return self

    def predict(self, X):
        check_is_fitted(self)
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X):
        check_is_fitted(self)
        # Probabilidad constante = tasa de la clase mayoritaria
        p = self.majority_rate_
        proba = np.full((len(X), 2), [1 - p, p] if self.majority_class_ == 1 else [p, 1 - p])
        return proba


# Baseline 2: Heuristica BSR

class ThresholdBSRClassifier(BaseEstimator, ClassifierMixin):
    """
    Predice 1 si bug_story_ratio > threshold.
    El threshold óptimo se busca en F2 sobre el train set.
    Cuando bug_story_ratio es NaN, predice la clase mayoritaria del train.
    """

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold    # None -> buscar óptimo en fit()

    def fit(self, X, y):
        # Extraer columna BSR (puede venir como DataFrame o numpy)
        bsr = self._get_bsr(X)
        self.majority_class_ = int(np.bincount(y.astype(int)).argmax())

        if self.threshold is None:
            # Búsqueda del umbral óptimo en F2 sobre el train set
            valid_mask   = ~np.isnan(bsr)
            bsr_valid    = bsr[valid_mask]
            y_valid      = np.array(y)[valid_mask]
            candidates   = np.percentile(bsr_valid, np.arange(10, 91, 5))
            best_t, best_f2 = candidates[0], -1
            for t in candidates:
                preds = (bsr_valid >= t).astype(int)
                from sklearn.metrics import fbeta_score
                f2 = fbeta_score(y_valid, preds, beta=2.0, zero_division=0)
                if f2 > best_f2:
                    best_f2, best_t = f2, t
            self.threshold_ = float(best_t)
            print(f"  ThresholdBSR: umbral optimo = {self.threshold_:.2f}, F2_train = {best_f2:.4f}")
        else:
            self.threshold_ = float(self.threshold)
        return self

    def predict(self, X):
        check_is_fitted(self)
        bsr  = self._get_bsr(X)
        pred = np.where(np.isnan(bsr), self.majority_class_, (bsr >= self.threshold_).astype(int))
        return pred

    def predict_proba(self, X):
        check_is_fitted(self)
        bsr  = self._get_bsr(X)
        # Probabilidad monotónica con BSR: sigmoid(bsr - threshold)
        bsr_filled = np.where(np.isnan(bsr), self.threshold_, bsr)
        prob_pos   = 1 / (1 + np.exp(-(bsr_filled - self.threshold_)))
        return np.column_stack([1 - prob_pos, prob_pos])

    @staticmethod
    def _get_bsr(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            col = "log_bug_story_ratio" if "log_bug_story_ratio" in X.columns else "bug_story_ratio"
            return X[col].values
        # Si es numpy array, asume que log_bug_story_ratio está en la posición correcta
        return X[:, FEATURE_COLS.index("log_bug_story_ratio")]


# Runner con MLflow

def run_baselines(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> list[dict]:
    """
    Entrena y evalúa los dos baselines. Loguea cada uno a MLflow.
    Retorna lista de dicts de métricas para el leaderboard.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL].values
    X_val   = val[FEATURE_COLS]
    y_val   = val[TARGET_COL].values
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL].values

    results = []

    # Baseline 1: MajorityClassifier
    with mlflow.start_run(run_name="baseline_majority_class"):
        mlflow.set_tag("model_type", "baseline")
        mlflow.set_tag("champion_round", "1")

        clf = MajorityClassifier()
        clf.fit(X_train, y_train)

        for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
            m = compute_metrics(y_s, clf.predict(X_s), clf.predict_proba(X_s)[:, 1],
                                label=f"majority_{split_name}")
            mlflow.log_metrics({f"{split_name}_{k}": v for k, v in m.items()
                                 if isinstance(v, (int, float))})
            if split_name == "val":
                results.append({**m, "label": "Baseline MajorityClass"})

        print(f"  Baseline MajorityClass: Val F2={results[-1]['f2']:.4f}, "
              f"PR-AUC={results[-1]['pr_auc']:.4f}")

    # Baseline 2: ThresholdBSR
    for tag, threshold_val in [("auto", None), ("p25", 0.96), ("p75", 6.83)]:
        with mlflow.start_run(run_name=f"baseline_heuristic_bsr_{tag}"):
            mlflow.set_tag("model_type", "baseline")
            mlflow.log_param("threshold_mode", tag)

            clf2 = ThresholdBSRClassifier(threshold=threshold_val)
            clf2.fit(X_train, y_train)
            mlflow.log_param("threshold_value", clf2.threshold_)

            for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
                m = compute_metrics(y_s, clf2.predict(X_s), clf2.predict_proba(X_s)[:, 1],
                                    label=f"heuristic_bsr_{tag}_{split_name}")
                mlflow.log_metrics({f"{split_name}_{k}": v for k, v in m.items()
                                     if isinstance(v, (int, float))})
                if split_name == "val":
                    results.append({**m, "label": f"Baseline BSR (thresh={clf2.threshold_:.2f})"})

            print(f"  Baseline BSR ({tag}): Val F2={results[-1]['f2']:.4f}")

    return results


if __name__ == "__main__":
    from feature_store import get_feature_matrix
    train, val, test, _ = get_feature_matrix()
    results = run_baselines(train, val, test)

    from metrics import metrics_table
    print("\n=== Leaderboard Baselines ===")
    print(metrics_table(results).to_string(index=False))
