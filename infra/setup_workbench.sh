#!/bin/bash
# setup_workbench.sh — ShiftMetrics ML Pipeline
# Ejecutar UNA VEZ en el Vertex AI Workbench después de crear la instancia.
# Instala dependencias, configura GCP auth, y sube/descarga el código.
#
# Uso:
#   bash setup_workbench.sh
#
# Prerequisitos:
#   - gcloud auth login (ya hecho en la instancia o via ADC)
#   - El bucket gs://shiftmetrics-bronze existe

set -euo pipefail

PROJECT_ID="shiftmetrics-analytics"
BUCKET="gs://shiftmetrics-bronze"
ML_LOCAL_DIR="$HOME/shiftmetrics/ml"
GCS_CODE_PATH="${BUCKET}/code/ml"

echo "============================================================"
echo "  ShiftMetrics ML Pipeline — Workbench Setup"
echo "============================================================"

# ── 1. Auth y proyecto ────────────────────────────────────────────────────────
echo "[1/5] Configurando GCP project..."
gcloud config set project "${PROJECT_ID}"

# ── 2. Crear directorio de trabajo ────────────────────────────────────────────
echo "[2/5] Creando directorio de trabajo..."
mkdir -p "${ML_LOCAL_DIR}"

# ── 3. Instalar dependencias ──────────────────────────────────────────────────
echo "[3/5] Instalando dependencias Python..."

pip install --quiet --upgrade pip

pip install --quiet \
    google-cloud-bigquery==3.27.0 \
    google-cloud-bigquery-storage==2.27.0 \
    google-cloud-storage==2.19.0 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    xgboost==2.1.3 \
    lightgbm \
    optuna==4.1.0 \
    shap==0.46.0 \
    mlflow==2.18.0 \
    imbalanced-learn==0.12.4 \
    scipy \
    matplotlib \
    joblib \
    pyarrow \
    db-dtypes

echo "  ✅ Dependencias instaladas"

# ── 4. Bajar código desde GCS ─────────────────────────────────────────────────
echo "[4/5] Descargando código desde GCS..."

if gsutil ls "${GCS_CODE_PATH}/" &>/dev/null; then
    gsutil -m cp -r "${GCS_CODE_PATH}/*" "${ML_LOCAL_DIR}/"
    echo "  ✅ Código descargado desde ${GCS_CODE_PATH}"
else
    echo "  ⚠️  Código no encontrado en ${GCS_CODE_PATH}"
    echo "     Sube el código primero desde tu máquina local:"
    echo "     bash infra/upload_ml.sh"
fi

# ── 5. Variables de entorno ────────────────────────────────────────────────────
echo "[5/5] Configurando variables de entorno..."

MLFLOW_URI="https://mlflow-server-919593201130.us-central1.run.app"

cat >> "$HOME/.bashrc" << EOF

# ShiftMetrics ML Pipeline
export MLFLOW_TRACKING_URI="${MLFLOW_URI}"
export GCP_PROJECT="${PROJECT_ID}"
export PYTHONPATH="${ML_LOCAL_DIR}:\$PYTHONPATH"
EOF

echo "  ✅ Variables configuradas en ~/.bashrc"
echo ""
echo "============================================================"
echo "  Setup completo."
echo "  Recarga el shell: source ~/.bashrc"
echo "  Ejecuta el pipeline: cd ${ML_LOCAL_DIR} && python run_pipeline.py"
echo "============================================================"
