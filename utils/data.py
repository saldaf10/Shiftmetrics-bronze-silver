"""utils/data.py — Carga de artefactos: MLflow REST API primero, local fallback.

Usa la REST API de MLflow directamente (sin el SDK pesado) para
cargar metricas y parametros. El modelo se carga de local porque
descargar un pickle via REST no es practico en Streamlit Cloud.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

# ─── Config MLflow ────────────────────────────────────────────
MLFLOW_BASE_URL      = "https://mlflow-server-919593201130.us-central1.run.app"
MLFLOW_API           = f"{MLFLOW_BASE_URL}/api/2.0/mlflow"
MODEL_REGISTRY_NAME  = "ShiftMetrics-DefectoEscapado"
REQUEST_TIMEOUT      = 8  # segundos

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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ─── REST API helpers ─────────────────────────────────────────

def _mlflow_get(endpoint: str, params: dict | None = None) -> dict | None:
    """GET a la REST API de MLflow. Retorna None si falla."""
    try:
        url = f"{MLFLOW_API}/{endpoint}"
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=120)
def _fetch_run(run_id: str) -> dict | None:
    """Obtener un run por ID via REST."""
    data = _mlflow_get("runs/get", {"run_id": run_id})
    if data and "run" in data:
        return data["run"]
    return None


def _run_metrics(run: dict) -> dict[str, float]:
    """Extraer metricas de un run como dict {nombre: valor}."""
    metrics = {}
    for m in run.get("data", {}).get("metrics", []):
        metrics[m["key"]] = float(m["value"])
    return metrics


def _run_params(run: dict) -> dict[str, str]:
    """Extraer parametros de un run."""
    params = {}
    for p in run.get("data", {}).get("params", []):
        params[p["key"]] = p["value"]
    return params


def _run_tags(run: dict) -> dict[str, str]:
    """Extraer tags de un run."""
    tags = {}
    for t in run.get("data", {}).get("tags", []):
        if not t["key"].startswith("mlflow."):
            tags[t["key"]] = t["value"]
    return tags


# ─── Estado de conexion ──────────────────────────────────────

@st.cache_data(ttl=60)
def get_data_source() -> dict:
    """Testea la conexion al MLflow server."""
    try:
        resp = requests.get(
            f"{MLFLOW_API}/experiments/search",
            params={"max_results": 1},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            return {"connected": True, "source": "mlflow", "error": "",
                    "url": MLFLOW_BASE_URL}
    except Exception as e:
        return {"connected": False, "source": "local",
                "error": str(e)[:150]}
    return {"connected": False, "source": "local", "error": "HTTP error"}


# ─── Metricas del modelo ─────────────────────────────────────

@st.cache_data(ttl=300)
def cargar_metricas() -> dict:
    """Metricas del champion. MLflow REST primero, local fallback."""

    run = _fetch_run(CANONICAL_RUNS["final_eval"])
    if run is not None:
        metrics = _run_metrics(run)
        params  = _run_params(run)
        tags    = _run_tags(run)

        def m(name, default=0.0):
            return metrics.get(name, default)

        end_time = run.get("info", {}).get("end_time")
        fecha = ""
        if end_time:
            import datetime
            fecha = datetime.datetime.fromtimestamp(
                int(end_time) / 1000).strftime("%Y-%m-%d %H:%M")

        return {
            "f2":              m("test_f2"),
            "precision":       m("test_precision"),
            "recall":          m("test_recall"),
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
            "threshold":       float(params.get("threshold_f2opt", "0.220")),
            "calibration_method": params.get("calibration_method", "isotonic"),
            "modelo_familia":  f"{tags.get('champion_name', 'XGBoost')} (calibrado {params.get('calibration_method', 'isotonic')})",
            "registry_name":   MODEL_REGISTRY_NAME,
            "registry_version": "v4",
            "fecha_evaluacion": fecha,
            "n_test":          6683,
            "source":          "mlflow",
            "run_id":          CANONICAL_RUNS["final_eval"],
        }

    # Fallback local
    local_path = DATA_DIR / "model_metrics.json"
    if local_path.exists():
        met = json.loads(local_path.read_text())
        met["source"] = "local"
        return met
    return {"source": "error", "error": "Sin metricas"}


# ─── SHAP global ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def cargar_shap_global() -> pd.DataFrame:
    """Ranking SHAP. Intenta MLflow REST, luego local."""
    run = _fetch_run(CANONICAL_RUNS["shap"])
    if run is not None:
        metrics = _run_metrics(run)
        params  = _run_params(run)
        items = []
        for key, val in metrics.items():
            if key.startswith("shap_importance_top"):
                rank = key.replace("shap_importance_top", "")
                feat = params.get(f"shap_top{rank}", f"feature_{rank}")
                items.append({"feature": feat, "importancia": val})
        if items:
            df = pd.DataFrame(items).sort_values("importancia", ascending=False)
            desc = _get_feature_descriptions()
            df["explicacion"] = df["feature"].map(desc)
            return df.reset_index(drop=True)

    return pd.read_parquet(DATA_DIR / "shap_global.parquet")


# ─── Champion model (siempre local — no se puede bajar pickle via REST) ──

@st.cache_resource
def cargar_champion() -> dict:
    """Modelo champion. Siempre de local pkl."""
    local_path = DATA_DIR / "champion.pkl"
    if local_path.exists():
        with open(local_path, "rb") as fh:
            data = pickle.load(fh)
        data["source"] = "local (pkl)"
        return data
    return {"model": None, "source": "no disponible"}


# ─── Datos locales (snapshots) ────────────────────────────────

@st.cache_data(ttl=3600)
def cargar_snapshot() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "test_snapshot.parquet")

@st.cache_data(ttl=3600)
def cargar_drift_psi() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "drift_psi.parquet")

@st.cache_data(ttl=3600)
def cargar_drift_yearly() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "drift_yearly.parquet")

@st.cache_data(ttl=3600)
def cargar_reliability() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "reliability_curve.parquet")


# ─── Descripciones ────────────────────────────────────────────

def _get_feature_descriptions() -> dict[str, str]:
    return {
        "num_bugs_sprint": "Mas bugs = mas probabilidad de escape.",
        "log_avg_cycle_time": "Ciclos largos amplian la ventana de exposicion.",
        "bugs_per_issue": "Alta densidad de bugs = sprint correctivo.",
        "cycle_missing": "Sin dato de ciclo = proceso inmaduro.",
        "log_total_issues": "Sprints grandes = mas superficie de error.",
        "log_bug_story_ratio": "Proporcion alta bugs/stories = deuda tecnica.",
        "sprint_year": "Captura mejora historica de procesos.",
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
