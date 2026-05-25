"""
config.py — ShiftMetrics SI7009
Configuración centralizada. Un único lugar para cambiar parámetros.
"""

import os

# GCP
GCP_PROJECT  = "shiftmetrics-analytics"
BQ_TABLE     = "shiftmetrics-analytics.shiftmetrics_gold.sprint_features"
GCS_BUCKET   = "gs://shiftmetrics-bronze"

# MLflow
MLFLOW_TRACKING_URI = "https://mlflow-server-919593201130.us-central1.run.app"
MLFLOW_EXPERIMENT   = "shiftmetrics-sprint-defect"
ARTIFACT_ROOT       = "gs://shiftmetrics-bronze/mlruns/"

# Reproducibilidad
RANDOM_SEED = 42

# Target
TARGET_COL = "defecto_escapado"

# Split temporal (4-way split — calibración independiente del set de HPO)
# Train: 2000-2014 -> ~25,400 filas (HPO training)
# Cal:   2015      -> ~1,700  filas (calibración — el modelo refit NO usa este año)
# Val:   2016-2018 -> 12,941  filas (HPO evaluation + threshold selection)
# Test:  2019-2021 ->  6,683  filas (evaluación final, intocable)
TRAIN_END_YEAR = 2014
CAL_END_YEAR   = 2015   # holdout exclusivo para calibración de probabilidades
VAL_END_YEAR   = 2018

# Features (por grupo, basado en cobertura real del Gold)
# Grupo 1: conteos Jira — cobertura 100%
COUNT_FEATURES = [
    "num_bugs_sprint",
    "num_stories_sprint",
    "num_tasks_sprint",
    "total_issues_sprint",
]

# Grupo 2: features derivadas con transformación log1p — cobertura 37-92%
# Distribuciones reales: cycle_time p50=90d, p99=2341d -> log-normal confirmado
# bug_story_ratio p50=2.5, p99=48 -> heavy right-skew
LOG_FEATURES = [
    "avg_cycle_time_days",    # 91.8% cobertura
    "bug_story_ratio",        # 37.5% cobertura
    "total_issues_sprint",    # 100% cobertura (redundante transformado)
]

# Grupo 3: temporal — sin transformación
TEMPORAL_FEATURES = [
    "sprint_year",         # captura drift secular real (cycle_time −96% en 13 años)
    "sprint_month_sin",    # encoding cíclico del mes
    "sprint_month_cos",
]

# Grupo 4: DORA — 9% cobertura; incluir con indicador de ausencia
DORA_FEATURES = [
    "deploy_frequency_weekly",
    "change_failure_rate",
]

# Indicadores de ausencia (la ausencia misma puede ser señal)
MISSING_INDICATORS = [
    "bsr_missing",    # 62.5% ausente — sprints sin mix de tipos de issue
    "cycle_missing",  # 8.2% ausente
    "dora_missing",   # 91% ausente
]

# Feature set principal para entrenamiento
FEATURE_COLS = (
    COUNT_FEATURES
    + ["log_avg_cycle_time", "log_bug_story_ratio", "log_total_issues"]
    + TEMPORAL_FEATURES
    + DORA_FEATURES
    + MISSING_INDICATORS
    + ["bugs_per_issue", "log_cycle_x_bsr"]  # interacciones
)

# LOPO-CV: proyectos seleccionados por balance y tamaño (data-driven)
# Excluidos: FOP (98.4% pos), LUCENE (98.7%), GROOVY (98.6%) — no informativos
LOPO_PROJECTS = ["HTTPCLIENT", "IO", "MATH", "MYFACES", "NET"]

# Desbalance: 70.45% positivo -> moderado
# neg/pos = 12631/30116 ≈ 0.42
SCALE_POS_WEIGHT = 12631 / 30116   # para XGBoost

# Métricas
F_BETA = 2.0    # penaliza FN 2x más que FP — recall > precision en este dominio

# Optuna
OPTUNA_N_TRIALS  = 50
OPTUNA_N_JOBS    = 4    # trials paralelos en joblib
OPTUNA_TIMEOUT   = 600  # segundos
# Persistent storage: set OPTUNA_DB_PATH env var para ruta personalizada en el Workbench
OPTUNA_DB_PATH   = os.getenv("OPTUNA_DB_PATH", "sqlite:///optuna_shiftmetrics.db")

# MLflow Model Registry
MODEL_REGISTRY_NAME = "ShiftMetrics-DefectoEscapado"
