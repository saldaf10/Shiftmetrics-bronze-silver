"""utils/data.py — Carga de artefactos del dashboard ShiftMetrics."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_data(ttl=3600)
def cargar_snapshot() -> pd.DataFrame:
    """Snapshot de sprints del split de prueba con probabilidad y drivers."""
    return pd.read_parquet(DATA_DIR / "test_snapshot.parquet")


@st.cache_data(ttl=3600)
def cargar_shap_global() -> pd.DataFrame:
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


@st.cache_data(ttl=3600)
def cargar_metricas() -> dict:
    return json.loads((DATA_DIR / "model_metrics.json").read_text())


@st.cache_resource
def cargar_champion() -> dict:
    """Modelo entrenado + metadatos. Cache de recurso, no de datos."""
    with open(DATA_DIR / "champion.pkl", "rb") as fh:
        return pickle.load(fh)


def feature_descripcion() -> dict[str, str]:
    """Nombres legibles de features para el dashboard."""
    return {
        "num_bugs_sprint": "Bugs en el sprint",
        "num_stories_sprint": "Stories en el sprint",
        "num_tasks_sprint": "Tasks en el sprint",
        "total_issues_sprint": "Total de issues",
        "log_avg_cycle_time": "Tiempo de ciclo (log)",
        "log_bug_story_ratio": "Ratio bugs/stories (log)",
        "log_total_issues": "Tamaño del sprint (log)",
        "sprint_year": "Año del sprint",
        "sprint_month_sin": "Mes (componente cíclica seno)",
        "sprint_month_cos": "Mes (componente cíclica coseno)",
        "deploy_frequency_weekly": "Despliegues por semana",
        "change_failure_rate": "Tasa de cambios fallidos",
        "bsr_missing": "Sin ratio bugs/stories",
        "cycle_missing": "Sin tiempo de ciclo",
        "dora_missing": "Sin métricas DORA",
        "bugs_per_issue": "Densidad de bugs",
        "log_cycle_x_bsr": "Bugs crónicos (cycle × ratio)",
    }
