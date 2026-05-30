"""utils/data.py — Carga de artefactos: MLflow primero, local como fallback.

Intenta conectarse al MLflow Server real del proyecto. Si no puede
(red, credenciales, timeout), usa los parquets y pickles locales.
Esto permite que el dashboard funcione tanto en produccion como offline.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ─── Config MLflow ────────────────────────────────────────────
MLFLOW_TRACKING_URI  = "https://mlflow-server-919593201130.us-central1.run.app"
MLFLOW_EXPERIMENT    = "shiftmetrics-sprint-defect"
MODEL_REGISTRY_NAME  = "ShiftMetrics-DefectoEscapado"

# Run IDs canonicos (del pipeline ejecutado)
CANONICAL_RUNS = {
    "final_eval":          "90d0407fefeb449a8b99ea3c6dc3f8dc",
    "calibration":         "355c43bed63b4c7e96d9afd37eacbd4b",
    "shap":                "0e6c9b9597cf4023a0b9a1976fd6c05f",
    "drift":               "d536b2fdbac74a798ad0a2e8b5081b4e",
    "threshold":           "4f398f29acdb4d92b5710aef391b08ed",
    "champion_selection":  "b81e450774924dfc94cb82bd173cc2ef",
    "lopo_cv":             "cc338b317d73472c96cfaa1b9f6d0a86",
}

# Fallback local
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ─── Estado de conexion (compartido en session_state) ─────────
def _init_mlflow_status():
    """Inicializa el estado de conexion MLflow en session_state."""
    if "mlflow_status" not in st.session_state:
        st.session_state["mlflow_status"] = {
            "connected": False,
            "source": "local",
            "error": "",
            "client": None,
        }


def _get_mlflow_client():
    """Intenta conectar al MLflow server. Cachea el resultado."""
    _init_mlflow_status()
    status = st.session_state["mlflow_status"]

    if status["client"] is not None:
        return status["client"]

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        # Test de conexion: pedir el experimento
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if exp is None:
            # Intentar listar experimentos como fallback
            client.search_experiments(max_results=1)
        status["connected"] = True
        status["source"] = "mlflow"
        status["client"] = client
        status["error"] = ""
        return client
    except Exception as e:
        status["connected"] = False
        status["source"] = "local"
        status["error"] = str(e)[:200]
        return None


def get_data_source() -> dict:
    """Retorna el estado de conexion para mostrar en el UI."""
    _init_mlflow_status()
    _get_mlflow_client()  # intenta conectar si no lo ha hecho
    return st.session_state["mlflow_status"]


# ─── Metricas del modelo ─────────────────────────────────────

@st.cache_data(ttl=300)
def cargar_metricas() -> dict:
    """Metricas del champion. MLflow primero, local fallback."""
    client = _get_mlflow_client()

    if client is not None:
        try:
            run = client.get_run(CANONICAL_RUNS["final_eval"])
            metrics = run.data.metrics
            params = run.data.params
            tags = run.data.tags

            def m(name, default=0.0):
                v = metrics.get(name)
                return float(v) if v is not None else default

            result = {
                "f2":              m("test_f2"),
                "f1":              m("test_f1", 0.0),
                "precision":       m("test_precision"),
                "recall":          m("test_recall"),
                "roc_auc":         m("test_roc_auc", 0.0),
                "pr_auc":          m("test_pr_auc"),
                "brier":           m("test_brier"),
                "flagging_rate":   m("flagging_rate_test"),
                "val_f2":          m("val_f2"),
                "val_brier":       m("val_brier"),
                "lopo_f2_mean":    m("lopo_f2_mean"),
                "lopo_f2_std":     m("lopo_f2_std"),
                "lopo_cal_f2_mean":m("lopo_cal_f2_mean"),
                "lopo_cal_f2_std": m("lopo_cal_f2_std"),
                "test_f2_ci_lo":   m("test_f2_ci_lo"),
                "test_f2_ci_hi":   m("test_f2_ci_hi"),
                "test_recall_ci_lo": m("test_recall_ci_lo"),
                "test_recall_ci_hi": m("test_recall_ci_hi"),
                "threshold":       float(params.get("threshold_f2opt", 0.220)),
                "threshold_roi":   float(params.get("threshold_roi", 0.220)),
                "calibration_method": params.get("calibration_method", "isotonic"),
                "modelo_familia":  f"{tags.get('champion_name', 'XGBoost')} (calibrado {params.get('calibration_method', 'isotonic')})",
                "registry_name":   MODEL_REGISTRY_NAME,
                "registry_version": "v4",
                "fecha_evaluacion": run.info.end_time,
                "n_test":          int(m("n_test", 6683)),
                "source":          "mlflow",
                "run_id":          run.info.run_id,
            }
            # Formatear fecha si viene como timestamp
            if isinstance(result["fecha_evaluacion"], (int, float)):
                import datetime
                result["fecha_evaluacion"] = datetime.datetime.fromtimestamp(
                    result["fecha_evaluacion"] / 1000).strftime("%Y-%m-%d %H:%M")

            return result
        except Exception as e:
            pass  # caer al fallback local

    # Fallback local
    local_path = DATA_DIR / "model_metrics.json"
    if local_path.exists():
        met = json.loads(local_path.read_text())
        met["source"] = "local"
        return met
    return {"source": "error", "error": "Sin metricas disponibles"}


# ─── Champion model ──────────────────────────────────────────

@st.cache_resource
def cargar_champion() -> dict:
    """Modelo champion. Intenta MLflow Registry, luego local pkl."""
    client = _get_mlflow_client()

    if client is not None:
        try:
            import mlflow.sklearn
            # Intentar cargar con alias "champion" o "production"
            for alias in ["champion", "production"]:
                try:
                    uri = f"models:/{MODEL_REGISTRY_NAME}@{alias}"
                    model = mlflow.sklearn.load_model(uri)
                    return {
                        "model": model,
                        "feature_cols": _get_feature_cols(),
                        "threshold": 0.220,
                        "source": f"mlflow ({alias})",
                    }
                except Exception:
                    continue

            # Intentar por version numerica
            versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                uri = f"models:/{MODEL_REGISTRY_NAME}/{latest.version}"
                model = mlflow.sklearn.load_model(uri)
                return {
                    "model": model,
                    "feature_cols": _get_feature_cols(),
                    "threshold": 0.220,
                    "source": f"mlflow (v{latest.version})",
                }
        except Exception:
            pass

    # Fallback local
    local_path = DATA_DIR / "champion.pkl"
    if local_path.exists():
        with open(local_path, "rb") as fh:
            data = pickle.load(fh)
        data["source"] = "local"
        return data

    return {"model": None, "source": "no disponible", "error": "Sin modelo"}


def _get_feature_cols():
    """Feature cols del config del proyecto."""
    return [
        "num_bugs_sprint", "num_stories_sprint", "num_tasks_sprint",
        "total_issues_sprint",
        "log_avg_cycle_time", "log_bug_story_ratio", "log_total_issues",
        "sprint_year", "sprint_month_sin", "sprint_month_cos",
        "deploy_frequency_weekly", "change_failure_rate",
        "bsr_missing", "cycle_missing", "dora_missing",
        "bugs_per_issue", "log_cycle_x_bsr",
    ]


# ─── Datos locales (snapshots pre-computados) ────────────────
# Estos siempre se cargan de local porque son snapshots historicos
# que no cambian entre corridas del pipeline.

@st.cache_data(ttl=3600)
def cargar_snapshot() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "test_snapshot.parquet")


@st.cache_data(ttl=3600)
def cargar_shap_global() -> pd.DataFrame:
    """SHAP global. Intenta bajar del run de SHAP en MLflow, luego local."""
    client = _get_mlflow_client()
    if client is not None:
        try:
            run = client.get_run(CANONICAL_RUNS["shap"])
            metrics = run.data.metrics
            # Reconstruir ranking SHAP desde metricas logeadas
            shap_items = []
            for key, val in metrics.items():
                if key.startswith("shap_importance_top"):
                    rank = int(key.replace("shap_importance_top", ""))
                    feature_key = f"shap_top{rank}"
                    feature_name = run.data.params.get(feature_key, f"feature_{rank}")
                    shap_items.append({
                        "feature": feature_name,
                        "importancia": float(val),
                    })
            if shap_items:
                df = pd.DataFrame(shap_items).sort_values("importancia", ascending=False)
                # Agregar explicaciones
                desc = _get_feature_descriptions()
                df["explicacion"] = df["feature"].map(desc)
                return df.reset_index(drop=True)
        except Exception:
            pass

    return pd.read_parquet(DATA_DIR / "shap_global.parquet")


@st.cache_data(ttl=3600)
def cargar_drift_psi() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "drift_psi.parquet")


@st.cache_data(ttl=3600)
def cargar_drift_yearly() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "drift_yearly.parquet")


@st.cache_data(ttl=3600)
def cargar_reliability() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "reliability_curve.parquet")


def _get_feature_descriptions() -> dict[str, str]:
    return {
        "num_bugs_sprint": "Mas bugs en el sprint = mas probabilidad de que alguno escape.",
        "log_avg_cycle_time": "Ciclos largos amplian la ventana de exposicion.",
        "bugs_per_issue": "Alta densidad de bugs indica sprint correctivo.",
        "cycle_missing": "Sin dato de ciclo = procesos inmaduros.",
        "log_total_issues": "Sprints grandes tienen mas superficie de error.",
        "log_bug_story_ratio": "Proporcion alta bugs/stories = deuda tecnica.",
        "sprint_year": "Captura la mejora historica de procesos.",
        "num_stories_sprint": "Velocidad del sprint.",
        "log_cycle_x_bsr": "Bugs cronicos que tardan en resolverse.",
        "bsr_missing": "Sin ratio bug/story = sin instrumentacion.",
        "total_issues_sprint": "Tamano absoluto del sprint.",
        "num_tasks_sprint": "Composicion de carga.",
        "deploy_frequency_weekly": "Frecuencia de despliegue (DORA).",
        "change_failure_rate": "Tasa de cambios fallidos (DORA).",
        "sprint_month_sin": "Estacionalidad.",
        "sprint_month_cos": "Componente ciclica del mes.",
        "dora_missing": "Sin metricas DORA.",
    }


def feature_descripcion() -> dict[str, str]:
    """Nombres legibles de features para el dashboard."""
    return {
        "num_bugs_sprint": "Bugs en el sprint",
        "num_stories_sprint": "Stories en el sprint",
        "num_tasks_sprint": "Tasks en el sprint",
        "total_issues_sprint": "Total de issues",
        "log_avg_cycle_time": "Tiempo de ciclo (log)",
        "log_bug_story_ratio": "Ratio bugs/stories (log)",
        "log_total_issues": "Tamano del sprint (log)",
        "sprint_year": "Ano del sprint",
        "sprint_month_sin": "Mes (ciclica seno)",
        "sprint_month_cos": "Mes (ciclica coseno)",
        "deploy_frequency_weekly": "Despliegues por semana",
        "change_failure_rate": "Tasa de cambios fallidos",
        "bsr_missing": "Sin ratio bugs/stories",
        "cycle_missing": "Sin tiempo de ciclo",
        "dora_missing": "Sin metricas DORA",
        "bugs_per_issue": "Densidad de bugs",
        "log_cycle_x_bsr": "Bugs cronicos (cycle x ratio)",
    }
